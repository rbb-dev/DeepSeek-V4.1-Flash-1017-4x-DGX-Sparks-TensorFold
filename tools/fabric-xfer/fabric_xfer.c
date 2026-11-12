/* fabric-xfer: send / recv / selftest. See README.md. Control plane is a plain TCP socket (see the README for why). */
#include "fx.h"

#include <arpa/inet.h>
#include <errno.h>
#include <getopt.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define GiB ((uint64_t)1 << 30)
#define MAXR 8

typedef struct {
    int threads, port, size_only, no_direct, readback, verify, gid_index, ntargets, npat;
    double ring_gib, progress_s; int chunk_mib;
    int token_fd;
    const char *allow;
    const char *label, *manifest, *local, *bind, *addrs, *dest;
    const char *pat[8];
    struct { char name[64]; char addrs[128]; } target[FX_MAX_PEERS];
} opts;

static char g_label[96];
static void logcb(void *ctx, const char *m)
{
    (void)ctx;
    printf("[%s] %s\n", g_label, m);
    fflush(stdout);
}

static int parse_opts(int argc, char **argv, opts *o)
{
    memset(o, 0, sizeof *o);
    o->token_fd = -1;
    o->threads = 8; o->port = 5400; o->gid_index = -1; o->ring_gib = -1; o->chunk_mib = 16; o->progress_s = 1.0;
    static const struct option lo[] = {
        { "ring-gib", 1, 0, 1 }, { "chunk-mib", 1, 0, 2 }, { "no-direct", 0, 0, 3 }, { "readback", 0, 0, 4 },
        { "verify-existing", 0, 0, 5 }, { "manifest", 1, 0, 6 }, { "gid-index", 1, 0, 7 }, { "engine-pattern", 1, 0, 8 },
        { "local", 1, 0, 9 }, { "target", 1, 0, 10 }, { "bind", 1, 0, 11 }, { "addrs", 1, 0, 12 }, { "dest", 1, 0, 13 },
        { "token-fd", 1, 0, 14 }, { "allow", 1, 0, 15 }, { "progress-s", 1, 0, 16 }, { "port", 1, 0, 'p' }, { 0, 0, 0, 0 } };
    int c;
    while ((c = getopt_long(argc, argv, "n:p:l:s", lo, NULL)) != -1) {
        switch (c) {
        case 'n': o->threads = atoi(optarg); break;
        case 'p': o->port = atoi(optarg); break;
        case 'l': o->label = optarg; break;
        case 's': o->size_only = 1; break;
        case 1: o->ring_gib = atof(optarg); break;
        case 2: o->chunk_mib = atoi(optarg); break;
        case 3: o->no_direct = 1; break;
        case 4: o->readback = 1; break;
        case 5: o->verify = 1; break;
        case 6: o->manifest = optarg; break;
        case 7: o->gid_index = atoi(optarg); break;
        case 8: if (o->npat < 8) o->pat[o->npat++] = optarg; break;
        case 9: o->local = optarg; break;
        case 10: {
            char *eq = strchr(optarg, '=');
            if (!eq || o->ntargets >= FX_MAX_PEERS) return -1;
            snprintf(o->target[o->ntargets].name, sizeof o->target[0].name, "%.*s", (int)(eq - optarg), optarg);
            snprintf(o->target[o->ntargets].addrs, sizeof o->target[0].addrs, "%s", eq + 1);
            o->ntargets++;
            break;
        }
        case 11: o->bind = optarg; break;
        case 12: o->addrs = optarg; break;
        case 13: o->dest = optarg; break;
        case 16: o->progress_s = atof(optarg); if (o->progress_s < 0.1) return -1; break;   /* the sender's progress line every N s; receivers every 5 N s */
        case 14: o->token_fd = atoi(optarg); break;
        case 15: o->allow = optarg; break;
        default: return -1;
        }
    }
    if (o->threads < 1 || o->threads > 99 || o->chunk_mib < 1) return -1;
    return 0;
}

static int split_addrs(const char *list, char out[MAXR][32])
{
    int n = 0;
    char buf[256];
    snprintf(buf, sizeof buf, "%s", list);
    for (char *s = strtok(buf, ","); s; s = strtok(NULL, ",")) {
        if (n >= MAXR || fx_check_ipv4(s)) return -1;
        snprintf(out[n++], 32, "%s", s);
    }
    return n;
}

static int resolve_rails(const char *list, int gid_index, fx_rail *rails)
{
    char a[MAXR][32];
    int n = split_addrs(list, a);
    if (n < 1) { fprintf(stderr, "ERROR: bad address list '%s'\n", list); return -1; }
    for (int i = 0; i < n; i++) {
        if (fx_find_gid(a[i], "/sys/class/infiniband", &rails[i])) { fprintf(stderr, "ERROR: no RoCE v2 GID holds %s\n", a[i]); return -1; }
        if (gid_index >= 0) rails[i].gid_index = gid_index;
    }
    return n;
}

static uint64_t ring_budget(const opts *o, void (*lg)(void *, const char *))
{
    static const char *const defpat[] = { "dsv41", "tensorfold", "glm53" };
    FILE *f = fopen("/proc/meminfo", "r");
    char text[8192] = "";
    if (f) { size_t n = fread(text, 1, sizeof text - 1, f); text[n] = 0; fclose(f); }
    int64_t avail = fx_parse_meminfo(text);
    if (avail < 0) { fprintf(stderr, "ERROR: cannot read MemAvailable\n"); return 0; }
    char names[512], why[256], line[1024];
    int eng = o->npat ? fx_detect_engines(o->pat, o->npat, names, sizeof names) : fx_detect_engines(defpat, 3, names, sizeof names);
    uint64_t ring = fx_memory_policy((uint64_t)avail, eng > 0, o->ring_gib, why, sizeof why);
    snprintf(line, sizeof line, "memory: MemAvailable %.1f GiB; ring %.1f GiB (%s)%s%s", (double)avail / GiB, (double)ring / GiB, why,
             eng > 0 ? "; engines: " : "", eng > 0 ? names : "");
    if (eng < 0) strncat(line, "; docker ps unavailable: engine presence unknown", sizeof line - strlen(line) - 1);
    lg(NULL, line);
    struct rlimit rl;
    if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY && rl.rlim_cur < ring) {
        snprintf(line, sizeof line, "WARNING: RLIMIT_MEMLOCK is %.1f GiB, below the ring (%.1f GiB): registering it will fail unless this user has unlimited locked memory", (double)rl.rlim_cur / GiB, (double)ring / GiB);
        lg(NULL, line);
    }
    return ring;
}

typedef struct { fx_rail rails[MAXR]; int nr, npeers; char err[FX_ERRLEN]; } vctx;
static fx_transport *mk_verbs(void *c)
{
    vctx *v = c;
    fx_transport *t = fx_verbs_new(v->rails, v->nr, v->npeers, v->err, sizeof v->err);
    if (!t) fprintf(stderr, "ERROR: %s\n", v->err);
    return t;
}

/* the one-time token: 32..128 hex characters, read once from a file descriptor (a pipe), never from the command line */
static char g_token[130];
static int read_token(int fd)
{
    if (fd < 0) return -1;
    size_t n = 0;
    ssize_t r;
    while (n < sizeof g_token - 1 && (r = read(fd, g_token + n, sizeof g_token - 1 - n)) > 0) n += (size_t)r;
    while (n && (g_token[n - 1] == '\n' || g_token[n - 1] == '\r')) n--;
    g_token[n] = 0;
    if (n < 32) return -1;
    for (size_t i = 0; i < n; i++) if (!((g_token[i] >= '0' && g_token[i] <= '9') || (g_token[i] >= 'a' && g_token[i] <= 'f'))) return -1;
    return 0;
}

static void fill_cfg(const opts *o, fx_cfg *cfg, uint64_t ring)
{
    memset(cfg, 0, sizeof *cfg);
    cfg->chunk = (size_t)o->chunk_mib << 20; cfg->ring_bytes = ring; cfg->io_threads = o->threads; cfg->direct = !o->no_direct;
    cfg->readback = o->readback; cfg->verify_existing = o->verify; cfg->dry = o->size_only; cfg->manifest = o->manifest;
    cfg->progress_s = o->progress_s; cfg->log = o->size_only ? NULL : logcb; cfg->token = g_token;
}

static int connect_retry(const char *ip, int port)
{
    for (int i = 0; i < 600; i++) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in sa = { .sin_family = AF_INET, .sin_port = htons((uint16_t)port) };
        inet_pton(AF_INET, ip, &sa.sin_addr);
        if (connect(fd, (struct sockaddr *)&sa, sizeof sa) == 0) {
            int one = 1;
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
            return fd;
        }
        close(fd);
        struct timespec ts = { 0, 200000000 };
        nanosleep(&ts, NULL);
    }
    return -1;
}

static int do_send(int argc, char **argv)
{
    opts o;
    if (parse_opts(argc, argv, &o) || optind + 1 != argc || !o.local || !o.ntargets) {
        fprintf(stderr, "usage: fabric-xfer send [-n THREADS] [-p PORT] [-l LABEL] [-s] [--ring-gib G] [--chunk-mib M] [--no-direct] [--readback]\n"
                        "         [--verify-existing] [--manifest PATH] [--gid-index I] --token-fd N --local ADDR_A,ADDR_B --target NAME=ADDR_A,ADDR_B ... SRC_DIR < paths\n"
                        "       a line of paths is PATH (to every target) or N[,N...]<TAB>PATH (to those targets, 1 = the first --target)\n");
        return 2;
    }
    snprintf(g_label, sizeof g_label, "fabric-xfer %s", o.label ? o.label : o.target[0].name);
    if (read_token(o.token_fd)) { fprintf(stderr, "ERROR: --token-fd N must name a descriptor holding the one-time token (32 to 128 hex characters)\n"); return 2; }
    const char *src = argv[optind];
    /* a line is a path, or "N[,N...]<TAB>path": the path goes only to those targets (1 = the first --target), so one run can carry
       the files every worker needs and each worker's own (a tab cannot be in a path: control characters are refused) */
    char **paths = NULL; int np = 0, routed = 0;
    uint64_t *masks = NULL, all = (1ull << o.ntargets) - 1;
    char *line = NULL; size_t cap = 0; ssize_t len;
    while ((len = getline(&line, &cap, stdin)) > 0) {
        while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = 0;
        if (!len) continue;
        char *path = line, *tab = strchr(line, '\t');
        uint64_t m = all;
        if (tab) {
            *tab = 0; path = tab + 1; m = 0; routed = 1;
            for (char *sv = NULL, *tok = strtok_r(line, ",", &sv); tok; tok = strtok_r(NULL, ",", &sv)) {
                char *end; long k = strtol(tok, &end, 10);
                if (*end || k < 1 || k > o.ntargets) { fprintf(stderr, "ERROR: a route names target '%s'; the targets are 1 to %d\n", tok, o.ntargets); return 2; }
                m |= 1ull << (k - 1);
            }
            if (!m || !*path) { fprintf(stderr, "ERROR: a routed line needs targets and a path\n"); return 2; }
        }
        paths = realloc(paths, sizeof(char *) * (np + 1)); masks = realloc(masks, sizeof(uint64_t) * (np + 1));
        masks[np] = m; paths[np++] = strdup(path);
    }
    free(line);
    if (!np) { if (o.size_only) puts("0"); else logcb(NULL, "no paths given: nothing to send"); return 0; }
    signal(SIGPIPE, SIG_IGN);
    fx_cfg cfg;
    uint64_t ring = 0;
    if (!o.size_only) { ring = ring_budget(&o, logcb); if (!ring) return 1; }
    fill_cfg(&o, &cfg, ring);
    if (routed) cfg.path_masks = masks;
    vctx vc;
    memset(&vc, 0, sizeof vc);
    vc.npeers = o.ntargets;
    if (!o.size_only) {
        vc.nr = resolve_rails(o.local, o.gid_index, vc.rails);
        if (vc.nr < 1) return 1;
        char l[1024] = "rails:";
        for (int i = 0; i < vc.nr; i++) { char t[128]; snprintf(t, sizeof t, " %s:%d gid %d (%s)", vc.rails[i].ibdev, vc.rails[i].port, vc.rails[i].gid_index, vc.rails[i].ip); strncat(l, t, sizeof l - strlen(l) - 1); }
        logcb(NULL, l);
    }
    fx_peer peers[FX_MAX_PEERS];
    fx_ctrl ctrls[FX_MAX_PEERS];
    struct timespec tc0;
    clock_gettime(CLOCK_MONOTONIC, &tc0);
    for (int i = 0; i < o.ntargets; i++) {
        char a[MAXR][32];
        int n = split_addrs(o.target[i].addrs, a);
        if (n < 1 || (!o.size_only && n != vc.nr)) { fprintf(stderr, "ERROR: target %s needs %d addresses\n", o.target[i].name, vc.nr); return 2; }
        int fd = connect_retry(a[0], o.port);
        if (fd < 0) { fprintf(stderr, "ERROR: cannot connect to %s (%s:%d)\n", o.target[i].name, a[0], o.port); return 1; }
        fx_ctrl_init(&ctrls[i], fd);
        memset(&peers[i], 0, sizeof peers[i]);
        snprintf(peers[i].name, sizeof peers[i].name, "%s", o.target[i].name);
        peers[i].ctrl = &ctrls[i];
    }
    struct timespec tc1;
    clock_gettime(CLOCK_MONOTONIC, &tc1);
    double t_connect = (double)(tc1.tv_sec - tc0.tv_sec) + (double)(tc1.tv_nsec - tc0.tv_nsec) * 1e-9;
    fx_send_res *res = calloc(1, sizeof *res);
    int rc = fx_run_sender(src, paths, np, peers, o.ntargets, mk_verbs, &vc, &cfg, res);
    for (int i = 0; i < o.ntargets; i++) close(ctrls[i].fd);
    if (rc) { fprintf(stderr, "ERROR: %s\n", res->err); return 1; }
    if (o.size_only) { printf("%llu\n", (unsigned long long)res->would_send); return 0; }
    if (res->would_send == 0) { logcb(NULL, "nothing to send: every file is in place at its size"); return 0; }
    char l[512];
    snprintf(l, sizeof l, "%.1f GB to send in total, %.1f GB sent in %.1f s: %.2f GB/s over %d receiver(s)", (double)res->would_send / 1e9,
             (double)res->total_bytes / 1e9, res->seconds, (double)res->total_bytes / (res->seconds > 1e-9 ? res->seconds : 1e-9) / 1e9, o.ntargets);
    logcb(NULL, l);
    {
        double steady = res->t_last > res->t_first ? (double)res->total_bytes / (res->t_last - res->t_first) / 1e9 : 0;
        snprintf(l, sizeof l, "timing (s): connect to receivers %.2f (this includes waiting for ssh to start them) | inventory + plan %.2f | receivers allocate+register their rings %.2f | ring allocated+registered here %.2f | queue pairs + receivers ready %.2f | first write %.2f | last write done %.2f | all verdicts in %.2f | sender total %.2f",
                 t_connect, res->t_plan, res->t_rings - res->t_plan, res->t_alloc - res->t_rings, res->t_ready - res->t_alloc, res->t_first, res->t_last, res->t_done, res->seconds);
        logcb(NULL, l);
        snprintf(l, sizeof l, "rate between first write and last write done: %.2f GB/s (all receivers together); data-phase share of the sender's total: %.0f %%",
                 steady, res->seconds > 0 ? 100.0 * (res->t_last - res->t_first) / res->seconds : 0.0);
        logcb(NULL, l);
    }
    for (int i = 0; i < res->npeers; i++) {
        snprintf(l, sizeof l, "%s: %d files verified by per-chunk CRC-32C, %d failed, %.1f GB sent, %.1f GB written", res->peers[i].name, res->peers[i].files_ok, res->peers[i].files_failed, (double)res->peers[i].bytes / 1e9, (double)res->peers[i].written / 1e9);
        logcb(NULL, l);
    }
    for (int i = 0; i < res->nfail; i++) fprintf(stderr, "ERROR: %s\n", res->fail[i]);
    return res->ok ? 0 : 1;
}

static int do_recv(int argc, char **argv)
{
    opts o;
    if (parse_opts(argc, argv, &o) || !o.bind || !o.addrs || !o.dest || !o.allow) {
        fprintf(stderr, "usage: fabric-xfer recv --bind ADDR --port P --addrs A,B --dest DIR --token-fd N --allow IP,IP,... [options]\n");
        return 2;
    }
    {
        char al[512], *sv = NULL;
        snprintf(al, sizeof al, "%s", o.allow);
        for (char *t = strtok_r(al, ",", &sv); t; t = strtok_r(NULL, ",", &sv))
            if (fx_check_ipv4(t)) { fprintf(stderr, "ERROR: --allow takes IPv4 addresses, not '%s'\n", t); return 2; }
    }
    char host[64] = "";
    gethostname(host, sizeof host);
    snprintf(g_label, sizeof g_label, "recv %s", host);
    if (read_token(o.token_fd)) { fprintf(stderr, "ERROR: --token-fd N must name a descriptor holding the one-time token (32 to 128 hex characters)\n"); return 2; }
    signal(SIGPIPE, SIG_IGN);
    char dest[4096];
    const char *home = getenv("HOME");
    if (fx_expand_dest(o.dest, home ? home : "/", dest, sizeof dest)) { fprintf(stderr, "ERROR: bad --dest %s\n", o.dest); return 2; }
    uint64_t ring = ring_budget(&o, logcb);
    if (!ring) return 1;
    vctx vc;
    memset(&vc, 0, sizeof vc);
    vc.npeers = 1;
    vc.nr = resolve_rails(o.addrs, o.gid_index, vc.rails);
    if (vc.nr < 1) return 1;
    int ls = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in sa = { .sin_family = AF_INET, .sin_port = htons((uint16_t)o.port) };
    inet_pton(AF_INET, o.bind, &sa.sin_addr);
    if (bind(ls, (struct sockaddr *)&sa, sizeof sa) || listen(ls, 1)) { fprintf(stderr, "ERROR: cannot listen on %s:%d: %s\n", o.bind, o.port, strerror(errno)); return 1; }
    puts("FX-READY");
    fflush(stdout);
    /* only the sender's expected addresses get past accept (a stranger is closed on, and we keep listening); the first one that does is
     * the one connection: after it nothing listens any more */
    char peer[64];
    int fd = fx_accept_allowed(ls, o.allow, 120000, peer, sizeof peer);
    close(ls);
    if (fd < 0) { fprintf(stderr, "ERROR: no sender connected from an allowed address within 120 s\n"); return 1; }
    { char l[128]; snprintf(l, sizeof l, "connection from %s", peer); logcb(NULL, l); }
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    fx_ctrl c;
    fx_ctrl_init(&c, fd);
    fx_cfg cfg;
    fill_cfg(&o, &cfg, ring);
    fx_recv_res *res = calloc(1, sizeof *res);
    int rc = fx_run_receiver(&c, mk_verbs, &vc, dest, &cfg, ring, res);
    if (rc) fprintf(stderr, "ERROR: %s\n", res->err);
    for (int i = 0; i < res->nfail; i++) fprintf(stderr, "ERROR: %s\n", res->fail[i]);
    return rc || res->files_failed ? 1 : 0;
}

static double bench(int sw, const uint8_t *buf, size_t n, int reps)
{
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    uint32_t x = 0;
    for (int i = 0; i < reps; i++) x ^= sw ? fx_crc32c_sw(0, buf, n) : fx_crc32c(0, buf, n);
    clock_gettime(CLOCK_MONOTONIC, &b);
    double s = (double)(b.tv_sec - a.tv_sec) + (double)(b.tv_nsec - a.tv_nsec) * 1e-9;
    if (x == 0xdeadbeef) puts("");
    return (double)n * reps / s / 1e9;
}

static int do_selftest(int argc, char **argv)
{
    int rc = 0;
    const char *addrs[MAXR]; int na = 0;
    for (int i = 2; i < argc; i++) if (!strcmp(argv[i], "--addr") && i + 1 < argc && na < MAXR) addrs[na++] = argv[++i];
    printf("crc32c: %s\n", fx_crc_hw_active() ? "ARMv8 CRC32C instructions (known-answer and table cross-check passed)" : "software slice-by-8 (no hardware path in use)");
    size_t n = 64u << 20;
    uint8_t *buf = malloc(n);
    for (size_t i = 0; i < n; i++) buf[i] = (uint8_t)(i * 2654435761u >> 24);
    printf("crc32c speed, one thread: dispatch %.2f GB/s, software table %.2f GB/s (needs to beat the disk, ~5.5 GB/s, over the I/O threads)\n", bench(0, buf, n, 4), bench(1, buf, n, 2));
    free(buf);
    FILE *f = fopen("/proc/meminfo", "r");
    char text[8192] = "";
    if (f) { size_t k = fread(text, 1, sizeof text - 1, f); text[k] = 0; fclose(f); }
    char names[512];
    static const char *const pat[] = { "dsv41", "tensorfold", "glm53" };
    int eng = fx_detect_engines(pat, 3, names, sizeof names);
    printf("MemAvailable %.1f GiB; engines: %s\n", (double)fx_parse_meminfo(text) / GiB, eng < 0 ? "docker ps unavailable" : eng ? names : "none seen");
    struct rlimit rl;
    if (getrlimit(RLIMIT_MEMLOCK, &rl) == 0) printf("RLIMIT_MEMLOCK: %s\n", rl.rlim_cur == RLIM_INFINITY ? "unlimited" : "limited (registering a big ring will fail)");
    for (int i = 0; i < na; i++) {
        fx_rail r;
        char out[512];
        if (fx_find_gid(addrs[i], "/sys/class/infiniband", &r)) { printf("FAIL: no RoCE v2 GID holds %s\n", addrs[i]); rc = 1; continue; }
        printf("%s: %s port %d gid index %d netdev %s\n", addrs[i], r.ibdev, r.port, r.gid_index, r.netdev);
        if (fx_verbs_probe(&r, out, sizeof out)) { printf("FAIL: %s\n", out); rc = 1; } else printf("  %s\n", out);
    }
    return rc;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: fabric-xfer send|recv|selftest ...\n"); return 2; }
    if (!strcmp(argv[1], "send")) return do_send(argc - 1, argv + 1);
    if (!strcmp(argv[1], "recv")) return do_recv(argc - 1, argv + 1);
    if (!strcmp(argv[1], "selftest")) return do_selftest(argc, argv);
    fprintf(stderr, "unknown mode %s\n", argv[1]);
    return 2;
}
