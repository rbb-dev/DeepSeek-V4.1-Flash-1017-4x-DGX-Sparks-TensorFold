/* Protocol tests: the real sender and receiver engines, the real control channel (socketpairs), the RDMA layer replaced
 * by the in-process loopback wire. Correctness only. */
#include "../fx.h"
#include "t.h"
#include <dirent.h>
#include <fcntl.h>
#include <ftw.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#define CH 65536u
#define TOKEN "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

static char tmp[256], srcd[300];
static int partials;

static void rmrf(const char *p) { char c[800]; snprintf(c, sizeof c, "rm -rf '%s'", p); if (system(c)) {} }
static void mkdirs(const char *p) { char c[800]; snprintf(c, sizeof c, "mkdir -p '%s'", p); if (system(c)) {} }

static uint64_t rng = 88172645463325252ull;
static uint8_t rnd8(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint8_t)(rng >> 24); }

static void make_file(const char *root, const char *rel, size_t size)
{
    char p[700];
    snprintf(p, sizeof p, "%s/%s", root, rel);
    char d[700]; snprintf(d, sizeof d, "%s", p);
    *strrchr(d, '/') = 0;
    mkdirs(d);
    FILE *f = fopen(p, "w");
    uint8_t *b = malloc(size ? size : 1);
    for (size_t i = 0; i < size; i++) b[i] = rnd8();
    fwrite(b, 1, size, f);
    fclose(f);
    free(b);
}

static uint8_t *slurp(const char *root, const char *rel, size_t *n)
{
    char p[700];
    snprintf(p, sizeof p, "%s/%s", root, rel);
    FILE *f = fopen(p, "r");
    if (!f) { *n = (size_t)-1; return NULL; }
    fseek(f, 0, SEEK_END);
    *n = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(*n ? *n : 1);
    if (fread(b, 1, *n, f) != *n) {}
    fclose(f);
    return b;
}

static int same(const char *a, const char *b, const char *rel)
{
    size_t na, nb;
    uint8_t *x = slurp(a, rel, &na), *y = slurp(b, rel, &nb);
    int ok = x && y && na == nb && memcmp(x, y, na) == 0;
    free(x); free(y);
    return ok;
}
static int exists(const char *root, const char *rel)
{
    char p[700]; snprintf(p, sizeof p, "%s/%s", root, rel);
    return access(p, F_OK) == 0;
}

static int count_partials_cb(const char *path, const struct stat *sb, int flag, struct FTW *ftw)
{
    (void)sb; (void)flag; (void)ftw;
    size_t n = strlen(path);
    if (n > 8 && !strcmp(path + n - 8, ".partial")) partials++;
    return 0;
}
static int no_partials(const char *dir) { partials = 0; nftw(dir, count_partials_cb, 8, FTW_PHYS); return partials == 0; }

/* ---- harness */
typedef struct { fx_ctrl c; fx_transport *tr; char dest[400]; fx_cfg cfg; uint64_t ring; fx_recv_res res; int rc; pthread_t th; } rarg;
static fx_transport *ident(void *ctx) { return ctx; }
static void *recv_main(void *a)
{
    rarg *r = a;
    r->rc = fx_run_receiver(&r->c, ident, r->tr, r->dest, &r->cfg, r->ring, &r->res);
    close(r->c.fd);
    return NULL;
}

typedef struct { int s_slots, r_slots, readback, verify, dry; const char *manifest; fx_loopnet *net; const char *stoken, *rtoken; int no_stoken; const uint64_t *masks; } opt;
static int run(char **paths, int np, char (*dests)[400], int nd, const opt *o, fx_send_res *sres, rarg *rr)
{
    fx_loopnet *net = o->net ? o->net : fx_loopnet_new(2);
    fx_peer peers[8];
    fx_ctrl sc[8];
    fx_cfg cfg = { .token = TOKEN, .chunk = CH, .io_threads = 3, .direct = 1, .readback = o->readback, .verify_existing = o->verify, .dry = o->dry,
                   .manifest = o->manifest, .progress_s = 0, .ring_bytes = (uint64_t)o->s_slots * CH, .path_masks = o->masks };
    for (int i = 0; i < nd; i++) {
        int sv[2];
        socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
        fx_ctrl_init(&sc[i], sv[0]);
        snprintf(peers[i].name, sizeof peers[i].name, "r%d", i);
        peers[i].ctrl = &sc[i];
        memset(&rr[i], 0, sizeof rr[i]);
        fx_ctrl_init(&rr[i].c, sv[1]);
        rr[i].tr = fx_loopnet_transport(net);
        snprintf(rr[i].dest, sizeof rr[i].dest, "%s", dests[i]);
        rr[i].cfg = cfg;
        if (o->rtoken) rr[i].cfg.token = o->rtoken;
        rr[i].ring = (uint64_t)o->r_slots * CH;
        pthread_create(&rr[i].th, NULL, recv_main, &rr[i]);
    }
    fx_transport *st = fx_loopnet_transport(net);
    if (o->stoken) cfg.token = o->stoken;
    if (o->no_stoken) cfg.token = NULL;
    int rc = fx_run_sender(srcd, paths, np, peers, nd, ident, st, &cfg, sres);
    for (int i = 0; i < nd; i++) { pthread_join(rr[i].th, NULL); close(sc[i].fd); fx_ctrl_free(&sc[i]); fx_ctrl_free(&rr[i].c); }
    if (!o->net) fx_loopnet_free(net);
    return rc;
}

static void fresh(void)
{
    rmrf(tmp);
    mkdirs(tmp);
    snprintf(srcd, sizeof srcd, "%s/src", tmp);
    mkdirs(srcd);
}
static void dest_of(char (*d)[400], int i) { snprintf(d[i], 400, "%s/d%d", tmp, i); }

static const opt O4 = { .s_slots = 4, .r_slots = 4 };

static void test_sizes_tails_recycling(void)
{
    fresh();
    struct { const char *n; size_t s; } f[] = { { "empty", 0 }, { "one", 1 }, { "a/b/c4095", 4095 }, { "x4096", 4096 }, { "x4097", 4097 },
                                               { "tail", 3 * CH + 123 }, { "big", 20 * CH }, { "exact", 2 * CH }, { "suid", 5 } };
    char *paths[9]; size_t total = 0, chunks = 0;
    for (int i = 0; i < 9; i++) { make_file(srcd, f[i].n, f[i].s); paths[i] = (char *)f[i].n; total += f[i].s; chunks += (f[i].s + CH - 1) / CH; }
    { char ps[700]; snprintf(ps, sizeof ps, "%s/suid", srcd); chmod(ps, 06755); }
    chmod("/dev/null", 0666);
    char p1[700]; snprintf(p1, sizeof p1, "%s/one", srcd);
    chmod(p1, 0640);
    struct timespec ts[2] = { { 1700000000, 5 }, { 1700000001, 123456789 } };
    utimensat(AT_FDCWD, p1, ts, 0);
    char d[1][400]; dest_of(d, 0);
    fx_send_res sres; rarg rr[1];
    fx_loopnet *net = fx_loopnet_new(2);
    opt o = O4; o.net = net;
    CHECK(run(paths, 9, d, 1, &o, &sres, rr) == 0);
    CHECK(sres.ok && sres.nfail == 0);
    for (int i = 0; i < 9; i++) CHECK(same(srcd, d[0], f[i].n));
    CHECK(no_partials(d[0]));
    struct stat st;
    char dp[800]; snprintf(dp, sizeof dp, "%s/one", d[0]);
    CHECK(stat(dp, &st) == 0 && (st.st_mode & 07777) == 0640 && st.st_mtim.tv_sec == 1700000001 && st.st_mtim.tv_nsec == 123456789);
    CHECK(chunks > 4 * 3);                                         /* many more chunks than slots: slots were recycled */
    CHECK(sres.peers[0].files_ok == 9 && sres.peers[0].bytes == total && rr[0].res.files_ok == 9);
    snprintf(dp, sizeof dp, "%s/suid", d[0]);
    CHECK(stat(dp, &st) == 0 && (st.st_mode & 07777) == 0755);      /* setuid, setgid and sticky never land */
    CHECK(fx_loopnet_rail_bytes(net, 0) > 0 && fx_loopnet_rail_bytes(net, 1) > 0);   /* both rails carried data */
    CHECK(fx_loopnet_rail_bytes(net, 0) + fx_loopnet_rail_bytes(net, 1) == total);
    CHECK(fx_loopnet_violations(net) == 0);
    fx_loopnet_free(net);
}

static void test_nothing_dry_wrongsize_missing(void)
{
    fresh();
    make_file(srcd, "a", 100); make_file(srcd, "b", 5000);
    char *paths[] = { "a", "b" };
    char d[1][400]; dest_of(d, 0);
    mkdirs(d[0]);
    for (int i = 0; i < 2; i++) { char c[900]; snprintf(c, sizeof c, "cp %s/%s %s/%s", srcd, paths[i], d[0], paths[i]); if (system(c)) {} }
    fx_send_res sres; rarg rr[1];
    fx_loopnet *net = fx_loopnet_new(2);
    opt o = O4; o.net = net;
    CHECK(run(paths, 2, d, 1, &o, &sres, rr) == 0 && sres.ok && sres.total_bytes == 0 && sres.would_send == 0);
    CHECK(fx_loopnet_rail_bytes(net, 0) + fx_loopnet_rail_bytes(net, 1) == 0);
    fx_loopnet_free(net);

    fresh();
    make_file(srcd, "a", 100); make_file(srcd, "b", 5000);
    dest_of(d, 0);
    o = O4; o.dry = 1;
    CHECK(run(paths, 2, d, 1, &o, &sres, rr) == 0 && sres.would_send == 5100 && !exists(d[0], "a"));

    fresh();
    make_file(srcd, "w", 3 * CH);
    char *wp[] = { "w" };
    dest_of(d, 0);
    mkdirs(d[0]);
    { char p[700]; snprintf(p, sizeof p, "%s/w", d[0]); FILE *f = fopen(p, "w"); fputs("short", f); fclose(f); }
    CHECK(run(wp, 1, d, 1, &O4, &sres, rr) == 0 && sres.ok && same(srcd, d[0], "w"));

    fresh();
    make_file(srcd, "a", 10);
    char *mp[] = { "a", "nope" };
    dest_of(d, 0);
    CHECK(run(mp, 2, d, 1, &O4, &sres, rr) == -1 && strstr(sres.err, "nope"));
    CHECK(!exists(d[0], "a"));
}

static int slow_hook(int fd, const void *b, size_t n, uint64_t off) { usleep(2000); return fx_pwrite_real(fd, b, n, off); }

static void test_slot_protocol(void)
{
    fresh();
    make_file(srcd, "f", 30 * CH + 7);
    char *paths[] = { "f" };
    char d[1][400]; dest_of(d, 0);
    fx_send_res sres; rarg rr[1];
    fx_loopnet *net = fx_loopnet_new(2);
    opt o = { .s_slots = 6, .r_slots = 3, .net = net };
    fx_write_hook = slow_hook;
    int rc = run(paths, 1, d, 1, &o, &sres, rr);
    fx_write_hook = NULL;
    CHECK(rc == 0 && sres.ok && same(srcd, d[0], "f"));
    CHECK(fx_loopnet_violations(net) == 0);          /* no slot written before its credit came back; no credit for a slot not out */
    CHECK(fx_loopnet_max_busy(net) == 3);            /* never beyond the receiver's depth, and the depth was used */
    fx_loopnet_free(net);
}

static void test_fanout(void)
{
    fresh();
    struct { const char *n; size_t s; } f[] = { { "keep", 2 * CH + 5 }, { "wrong", CH + 1 }, { "fresh", 5 * CH }, { "small", 10 } };
    char *paths[4]; size_t total = 0;
    for (int i = 0; i < 4; i++) { make_file(srcd, f[i].n, f[i].s); paths[i] = (char *)f[i].n; total += f[i].s; }
    char d[3][400];
    for (int i = 0; i < 3; i++) dest_of(d, i);
    mkdirs(d[0]); mkdirs(d[1]);
    { char c[900]; snprintf(c, sizeof c, "cp %s/keep %s/keep", srcd, d[0]); if (system(c)) {} }      /* receiver 0 holds keep, whole */
    { char p[800]; snprintf(p, sizeof p, "%s/wrong", d[1]); FILE *fh = fopen(p, "w"); fputs("x", fh); fclose(fh); }   /* receiver 1: wrong size */
    fx_send_res sres; rarg rr[3];
    CHECK(run(paths, 4, d, 3, &O4, &sres, rr) == 0 && sres.ok);
    for (int i = 0; i < 3; i++) for (int k = 0; k < 4; k++) CHECK(same(srcd, d[i], f[k].n));
    CHECK(sres.peers[0].bytes == total - f[0].s && sres.peers[1].bytes == total && sres.peers[2].bytes == total);
    size_t chunks = 0;
    for (int k = 0; k < 4; k++) chunks += (f[k].s + CH - 1) / CH;
    CHECK(sres.chunks_read == chunks);               /* read once: each needed chunk one time, not once a receiver */
    for (int i = 0; i < 3; i++) CHECK(sres.peers[i].written == sres.peers[i].bytes);   /* every byte sent was written (credited) */
}

/* one sender, files routed to some receivers only (one run for every worker's share: shared files to all, each worker's own to it) */
static void test_targets(void)
{
    fresh();
    /* to all three; to receiver 1 only; to receivers 0 and 2 */
    struct { const char *n; size_t s; uint64_t m; } f[] = { { "all", 3 * CH + 7, 7 }, { "one", 2 * CH, 2 }, { "two", CH + 9, 5 } };
    char *paths[3]; uint64_t masks[3];
    for (int i = 0; i < 3; i++) { make_file(srcd, f[i].n, f[i].s); paths[i] = (char *)f[i].n; masks[i] = f[i].m; }
    char d[3][400];
    for (int i = 0; i < 3; i++) dest_of(d, i);
    opt o = O4; o.masks = masks;
    fx_send_res sres; rarg rr[3];
    CHECK(run(paths, 3, d, 3, &o, &sres, rr) == 0 && sres.ok);
    for (int i = 0; i < 3; i++) for (int k = 0; k < 3; k++) CHECK(((f[k].m >> i) & 1) ? same(srcd, d[i], f[k].n) : !exists(d[i], f[k].n));
    CHECK(sres.peers[0].bytes == f[0].s + f[2].s && sres.peers[1].bytes == f[0].s + f[1].s && sres.peers[2].bytes == f[0].s + f[2].s);
    CHECK(sres.would_send == 3 * f[0].s + f[1].s + 2 * f[2].s);
    size_t chunks = 0;
    for (int k = 0; k < 3; k++) chunks += (f[k].s + CH - 1) / CH;
    CHECK(sres.chunks_read == chunks);               /* each needed chunk read once, whoever needs it */
    for (int i = 0; i < 3; i++) CHECK(sres.peers[i].written == sres.peers[i].bytes);
    /* a route naming a receiver the run does not have is refused before anything is sent */
    fresh();
    make_file(srcd, "x", CH);
    char *px[1] = { "x" }; uint64_t bad[1] = { 1ull << 3 };
    opt ob = O4; ob.masks = bad;
    fx_send_res sb; rarg rb[3];
    for (int i = 0; i < 3; i++) dest_of(d, i);
    CHECK(run(px, 1, d, 3, &ob, &sb, rb) == -1 && strstr(sb.err, "receiver"));
    for (int i = 0; i < 3; i++) CHECK(!exists(d[i], "x"));
}

static unsigned corrupt_seq;
static void corrupt_cb(void *arg, unsigned seq, uint8_t *data, size_t len)
{
    (void)arg; (void)len;
    if (seq == corrupt_seq) data[5] ^= 0xFF;
}

static void test_corruption(void)
{
    fresh();
    struct { const char *n; size_t s; } f[] = { { "bad", 3 * CH + 10 }, { "good", 3 * CH }, { "good2", 2 * CH } };
    char *paths[3];
    for (int i = 0; i < 3; i++) { make_file(srcd, f[i].n, f[i].s); paths[i] = (char *)f[i].n; }
    char d[1][400]; dest_of(d, 0);
    fx_send_res sres; rarg rr[1];
    fx_loopnet *net = fx_loopnet_new(2);
    corrupt_seq = 1;                                   /* the second chunk of "bad" */
    fx_loopnet_set_corrupt(net, corrupt_cb, NULL);
    opt o = O4; o.net = net;
    CHECK(run(paths, 3, d, 1, &o, &sres, rr) == 0);
    CHECK(!sres.ok && sres.nfail == 1 && strstr(sres.fail[0], "bad") && strstr(sres.fail[0], "mismatch"));
    CHECK(!exists(d[0], "bad") && !exists(d[0], "bad.partial"));
    CHECK(same(srcd, d[0], "good") && same(srcd, d[0], "good2"));
    CHECK(sres.peers[0].files_failed == 1 && rr[0].res.files_failed == 1 && no_partials(d[0]));
    fx_loopnet_free(net);
}

static _Atomic int wcount;
static int bad_disk_hook(int fd, const void *b, size_t n, uint64_t off)
{
    if (++wcount == 2) {
        uint8_t *c = malloc(n);
        memcpy(c, b, n);
        c[0] ^= 0xFF;
        int r = fx_pwrite_real(fd, c, n, off);
        free(c);
        return r;
    }
    return fx_pwrite_real(fd, b, n, off);
}

static void test_readback_and_verify(void)
{
    fresh();
    make_file(srcd, "f", 3 * CH);
    char *paths[] = { "f" };
    char d[1][400]; dest_of(d, 0);
    fx_send_res sres; rarg rr[1];
    wcount = 0;
    fx_write_hook = bad_disk_hook;
    opt o = O4; o.readback = 1;
    int rc = run(paths, 1, d, 1, &o, &sres, rr);
    fx_write_hook = NULL;
    CHECK(rc == 0 && !sres.ok && !exists(d[0], "f") && strstr(sres.fail[0], "read-back"));

    fresh();
    make_file(srcd, "same", CH + 3); make_file(srcd, "diff", CH + 3);
    char *vp[] = { "same", "diff" };
    dest_of(d, 0);
    mkdirs(d[0]);
    { char c[900]; snprintf(c, sizeof c, "cp %s/same %s/same", srcd, d[0]); if (system(c)) {} }
    make_file(d[0], "diff", CH + 3);                    /* same size, other bytes */
    o = O4; o.verify = 1;
    CHECK(run(vp, 2, d, 1, &o, &sres, rr) == 0 && sres.ok && same(srcd, d[0], "same") && same(srcd, d[0], "diff"));
    CHECK(sres.peers[0].bytes == CH + 3);               /* only "diff" was sent */
}

static void test_manifest(void)
{
    fresh();
    make_file(srcd, "a", 2 * CH + 9);
    make_file(srcd, "b", 0);
    char *paths[] = { "a", "b" };
    char d[1][400]; dest_of(d, 0);
    char man[400]; snprintf(man, sizeof man, "%s/m.json", tmp);
    fx_send_res sres; rarg rr[1];
    opt o = O4; o.manifest = man;
    CHECK(run(paths, 2, d, 1, &o, &sres, rr) == 0 && sres.ok);
    size_t n, an;
    uint8_t *j = slurp(tmp, "m.json", &n), *a = slurp(srcd, "a", &an);
    char *js = malloc(n + 1); memcpy(js, j, n); js[n] = 0;
    char want[64]; snprintf(want, sizeof want, "\"crc32c\":\"%08x\"", fx_crc32c(0, a, an));
    CHECK(strstr(js, want) != NULL);
    CHECK(strstr(js, "\"crc32c\":\"00000000\"") != NULL);        /* the empty file */
    free(js); free(j); free(a);
}


/* ---------------------------------------------------------------------------------------- security */
static int outside_clean(const char *dir) { DIR *d = opendir(dir); int n = 0; struct dirent *e; while ((e = readdir(d))) if (e->d_name[0] != '.') n++; closedir(d); return n; }

typedef struct { fx_ctrl c; fx_transport *tr; char dest[400]; fx_cfg cfg; fx_recv_res res; int rc; } rawr;
static void *raw_recv(void *a)
{
    rawr *r = a;
    r->rc = fx_run_receiver(&r->c, ident, r->tr, r->dest, &r->cfg, 4 * CH, &r->res);
    close(r->c.fd);
    return NULL;
}

/* the receiver's challenge-response: read its nonce, answer with HMAC-SHA256(token, nonce) in HELLO. 0 ok, -1 no challenge came */
static int raw_login(fx_ctrl *s, const char *token)
{
    fx_msg m;
    if (fx_ctrl_recv(s, 5000, &m) || m.type != FX_M_CHALLENGE) return -1;
    fx_rd rd = { m.data, m.len, 0, 0 };
    uint32_t nl;
    const uint8_t *nonce = fx_get_bytes(&rd, &nl);
    uint8_t mac[32];
    fx_hmac_sha256(mac, token, strlen(token), nonce, nl);
    fx_msg_free(&m);
    fx_buf b = { 0 };
    fx_put_u32(&b, FX_PROTO); fx_put_bytes(&b, mac, 32);
    fx_ctrl_send(s, FX_M_HELLO, &b); fx_buf_free(&b);
    return 0;
}

/* A fake sender that speaks the control protocol by hand, sending one path (len bytes, may contain NUL) in LIST or PLAN.
 * Returns 1 if the receiver refused with an error message containing `want` and ended with -1. */
static int raw_attempt(const char *path, size_t plen, int in_list, const char *want)
{
    int sv[2];
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
    fx_loopnet *net = fx_loopnet_new(2);
    rawr r;
    memset(&r, 0, sizeof r);
    fx_ctrl_init(&r.c, sv[1]);
    r.tr = fx_loopnet_transport(net);
    snprintf(r.dest, sizeof r.dest, "%s/dd/d0", tmp);
    mkdirs(r.dest);
    r.cfg = (fx_cfg){ .token = TOKEN, .chunk = CH, .io_threads = 2, .direct = 1 };
    pthread_t th;
    pthread_create(&th, NULL, raw_recv, &r);
    fx_ctrl s;
    fx_ctrl_init(&s, sv[0]);
    fx_buf b = { 0 };
    fx_msg m;
    int ok = 0;
    if (raw_login(&s, TOKEN)) goto done;
    if (fx_ctrl_recv(&s, 5000, &m)) goto done;
    fx_msg_free(&m);
    fx_put_u64(&b, CH); fx_put_u8(&b, 0); fx_put_u32(&b, 1);
    if (in_list) fx_put_bytes(&b, path, plen); else fx_put_str(&b, "fine");     /* the PLAN variant sends a valid LIST first */
    fx_put_u64(&b, 10);
    fx_ctrl_send(&s, FX_M_LIST, &b); fx_buf_free(&b);
    if (fx_ctrl_recv(&s, 5000, &m)) goto done;
    if (!in_list) {
        if (m.type != FX_M_INV) { fx_msg_free(&m); goto done; }
        fx_msg_free(&m);
        fx_put_u64(&b, CH); fx_put_u8(&b, 0); fx_put_u32(&b, 1);
        fx_put_bytes(&b, path, plen); fx_put_u64(&b, 10); fx_put_u32(&b, 0644); fx_put_u64(&b, 0); fx_put_u8(&b, 1);
        fx_ctrl_send(&s, FX_M_PLAN, &b); fx_buf_free(&b);
        if (fx_ctrl_recv(&s, 5000, &m)) goto done;
    }
    if (m.type == FX_M_ERROR) {
        fx_rd rd = { m.data, m.len, 0, 0 };
        char *t = fx_get_str(&rd);
        ok = strstr(t, want) != NULL;
        free(t);
    }
    fx_msg_free(&m);
done:
    close(sv[0]);
    pthread_join(th, NULL);
    fx_ctrl_free(&s); fx_ctrl_free(&r.c);
    fx_buf_free(&b);
    return ok && r.rc == -1;
}

static void test_traversal_refused(void)
{
    fresh();
    char dd[400]; snprintf(dd, sizeof dd, "%s/dd", tmp);
    struct { const char *p; size_t n; } bad[] = {
        { "../esc", 6 }, { "/etc/fx_evil", 12 }, { "a/../../esc", 11 }, { "a/../b", 6 }, { "..", 2 }, { "./a", 3 }, { "a/./b", 5 },
        { ".", 1 }, { "", 0 }, { "a//b", 4 }, { "a/", 2 }, { "a\001b", 3 }, { "a\177b", 3 }, { "a\nb", 3 }, { "a\0b", 3 } };
    for (size_t i = 0; i < sizeof bad / sizeof *bad; i++) {
        CHECK(raw_attempt(bad[i].p, bad[i].n, 1, "malformed plan"));      /* refused in the LIST handler */
        CHECK(raw_attempt(bad[i].p, bad[i].n, 0, "malformed plan"));      /* and in the PLAN handler */
    }
    char longp[4200];
    memset(longp, 'a', sizeof longp);
    CHECK(raw_attempt(longp, 4100, 1, "malformed plan") && raw_attempt(longp, 4100, 0, "malformed plan"));
    CHECK(!exists(dd, "esc") && !exists("/etc", "fx_evil"));
    CHECK(outside_clean(dd) == 1);                    /* only d0, which the receiver was given; nothing else appeared */
    char d0[420]; snprintf(d0, sizeof d0, "%s/d0", dd);
    CHECK(outside_clean(d0) == 0);

    /* the sender refuses to read outside its source folder too */
    make_file(tmp, "esc", 10);
    make_file(srcd, "ok", 10);
    char *paths[] = { "../esc" };
    char d[1][400]; dest_of(d, 0);
    fx_send_res sres; rarg rr[1];
    CHECK(run(paths, 1, d, 1, &O4, &sres, rr) == -1 && strstr(sres.err, "unsafe"));
    CHECK(!exists(d[0], "esc"));
}

/* each scenario alone, so that nothing the receiver does after a refusal can tidy up evidence of an escape */
static void symlink_case(int fallback, int dir_link)
{
    fresh();
    extern int fx_openat2_disabled;
    fx_openat2_disabled = fallback;
    char out[400], victim[500], c[1400];
    snprintf(out, sizeof out, "%s/outside", tmp);
    mkdirs(out);
    make_file(out, "victim", 20);
    size_t vn0; uint8_t *v0 = slurp(out, "victim", &vn0);
    char d[1][400]; dest_of(d, 0);
    mkdirs(d[0]);
    char *paths[1];
    if (dir_link) {                                                   /* a directory symlink in DEST: link -> outside */
        make_file(srcd, "link/pwn", 100);
        snprintf(c, sizeof c, "ln -s %s %s/link", out, d[0]); if (system(c)) {}
        paths[0] = "link/pwn";
    } else {                                                          /* the .partial leaf is a symlink to a file outside */
        make_file(srcd, "f", 100);
        snprintf(victim, sizeof victim, "%s/victim", out);
        snprintf(c, sizeof c, "ln -s %s %s/f.partial", victim, d[0]); if (system(c)) {}
        paths[0] = "f";
    }
    fx_send_res sres; rarg rr[1];
    int rc = run(paths, 1, d, 1, &O4, &sres, rr);
    CHECK(rc != 0 || !sres.ok);                                       /* refused */
    CHECK(!exists(out, "pwn") && !exists(out, "pwn.partial"));        /* nothing was created through the link */
    size_t vn1; uint8_t *v1 = slurp(out, "victim", &vn1);
    CHECK(vn0 == vn1 && v1 && memcmp(v0, v1, vn0) == 0);              /* the file the symlink pointed at was not touched */
    CHECK(outside_clean(out) == 1);
    free(v0); free(v1);
    fx_openat2_disabled = 0;
}

static void test_symlinks_refused(void)
{
    for (int fallback = 0; fallback < 2; fallback++)       /* openat2 where the kernel has it, then the per-component walk */
        for (int dir_link = 0; dir_link < 2; dir_link++) symlink_case(fallback, dir_link);
}

/* a fake sender presents `stok` to a receiver configured with `rtok`; returns 1 if the receiver dropped it without a reply */
static int raw_hello(const char *rtok, const char *stok)
{
    int sv[2];
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
    fx_loopnet *net = fx_loopnet_new(2);
    rawr r;
    memset(&r, 0, sizeof r);
    fx_ctrl_init(&r.c, sv[1]);
    r.tr = fx_loopnet_transport(net);
    snprintf(r.dest, sizeof r.dest, "%s/dd/d0", tmp);
    r.cfg = (fx_cfg){ .token = rtok, .chunk = CH, .io_threads = 2, .direct = 1 };
    pthread_t th;
    pthread_create(&th, NULL, raw_recv, &r);
    fx_ctrl sc;
    fx_ctrl_init(&sc, sv[0]);
    fx_msg m;
    int got = -1;
    if (raw_login(&sc, stok) == 0) {
        got = fx_ctrl_recv(&sc, 3000, &m);                 /* 0 = a reply arrived; -1 = connection closed with nothing said */
        if (got == 0) fx_msg_free(&m);
    }
    close(sv[0]);
    pthread_join(th, NULL);
    fx_ctrl_free(&sc); fx_ctrl_free(&r.c);
    return got == -1 && r.rc == -1 && strstr(r.res.err, "authentication") != NULL;
}

/* two connections must be challenged with different nonces, or a recorded answer could be replayed */
static int challenge_nonce(uint8_t out[32])
{
    int sv[2];
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
    fx_loopnet *net = fx_loopnet_new(2);
    rawr r;
    memset(&r, 0, sizeof r);
    fx_ctrl_init(&r.c, sv[1]);
    r.tr = fx_loopnet_transport(net);
    snprintf(r.dest, sizeof r.dest, "%s/dd/d0", tmp);
    r.cfg = (fx_cfg){ .token = TOKEN, .chunk = CH, .io_threads = 2, .direct = 1 };
    pthread_t th;
    pthread_create(&th, NULL, raw_recv, &r);
    fx_ctrl sc;
    fx_ctrl_init(&sc, sv[0]);
    fx_msg m;
    int ok = 0;
    if (fx_ctrl_recv(&sc, 5000, &m) == 0 && m.type == FX_M_CHALLENGE) {
        fx_rd rd = { m.data, m.len, 0, 0 };
        uint32_t n; const uint8_t *p = fx_get_bytes(&rd, &n);
        if (n == 32) { memcpy(out, p, 32); ok = 1; }
        fx_msg_free(&m);
    }
    close(sv[0]);
    pthread_join(th, NULL);
    fx_ctrl_free(&sc); fx_ctrl_free(&r.c);
    return ok;
}

static void test_auth(void)
{
    fresh();
    uint8_t n1[32], n2[32];
    CHECK(challenge_nonce(n1) && challenge_nonce(n2) && memcmp(n1, n2, 32) != 0);
    CHECK(raw_hello("", ""));                              /* an empty token never authenticates, even against an empty one */
    CHECK(raw_hello(TOKEN, ""));
    CHECK(raw_hello(TOKEN, "wrong"));
    CHECK(!raw_hello(TOKEN, TOKEN));                       /* (the right token is not dropped) */
    make_file(srcd, "a", 100);
    char *paths[] = { "a" };
    char d[1][400]; dest_of(d, 0);
    fx_send_res sres; rarg rr[1];
    opt o = O4; o.rtoken = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";      /* wrong token */
    CHECK(run(paths, 1, d, 1, &o, &sres, rr) == -1 && rr[0].rc == -1 && strstr(rr[0].res.err, "authentication"));
    CHECK(!exists(d[0], "a") && !exists(d[0], "a.partial"));
    o = O4; o.rtoken = TOKEN "0";                                                                     /* longer token, same prefix */
    CHECK(run(paths, 1, d, 1, &o, &sres, rr) == -1 && rr[0].rc == -1);
    o = O4; o.no_stoken = 1;                                                                          /* no token at all */
    CHECK(run(paths, 1, d, 1, &o, &sres, rr) == -1 && !exists(d[0], "a"));
    o = O4; o.rtoken = "";                                                                            /* receiver with an empty token */
    o.stoken = "";
    CHECK(run(paths, 1, d, 1, &o, &sres, rr) == -1 && rr[0].rc == -1 && !exists(d[0], "a"));
    CHECK(fx_ct_equal("abc", 3, "abc", 3) && !fx_ct_equal("abc", 3, "abd", 3) && !fx_ct_equal("abc", 3, "abcd", 4) && !fx_ct_equal("", 0, "x", 1));
    CHECK(run(paths, 1, d, 1, &O4, &sres, rr) == 0 && sres.ok && same(srcd, d[0], "a"));              /* the right token works */
}

/* a transport that adds one bogus write-with-immediate event for the receiver, as a hostile RDMA peer could */
typedef struct { fx_transport t; fx_transport *in; unsigned slot[2], seq[2], bytes[2]; int n, armed, injected; } evil;
static int ev_alloc(fx_transport *t, size_t n, uint8_t **b) { evil *e = t->ctx; return e->in->alloc_ring(e->in, n, b); }
static int ev_cp(fx_transport *t, int p, int d, uint8_t *i, size_t *l) { evil *e = t->ctx; return e->in->create_peer(e->in, p, d, i, l); }
static int ev_conn(fx_transport *t, int p, const uint8_t *i, size_t l) { evil *e = t->ctx; e->armed = 1; return e->in->connect_peer(e->in, p, i, l); }
static int ev_pc(fx_transport *t, int p, int r, uint32_t s) { evil *e = t->ctx; return e->in->post_credit(e->in, p, r, s); }
static int ev_pw(fx_transport *t, int p, int r, size_t a, size_t b, size_t c, uint32_t d, uint64_t f) { evil *e = t->ctx; return e->in->post_write(e->in, p, r, a, b, c, d, f); }
static int ev_poll(fx_transport *t, fx_event *out, int max)
{
    evil *e = t->ctx;
    if (e->armed && !e->injected) {
        e->injected = 1;
        for (int i = 0; i < e->n; i++) out[i] = (fx_event){ FX_E_IMM, 0, i, fx_pack_imm(e->slot[i], e->seq[i]), e->bytes[i] };
        return e->n;
    }
    return e->in->poll(e->in, out, max);
}
static void ev_destroy(fx_transport *t) { evil *e = t->ctx; e->in->destroy(e->in); free(e); }

static fx_transport *mk_evil(fx_transport *in, int n, const unsigned *slot, const unsigned *seq, const unsigned *bytes)
{
    evil *e = calloc(1, sizeof *e);
    e->in = in; e->n = n;
    for (int i = 0; i < n; i++) { e->slot[i] = slot[i]; e->seq[i] = seq[i]; e->bytes[i] = bytes[i]; }
    e->t = (fx_transport){ .ctx = e, .rails = in->rails, .max_slots = in->max_slots, .alloc_ring = ev_alloc, .create_peer = ev_cp, .connect_peer = ev_conn,
                           .post_write = ev_pw, .post_credit = ev_pc, .poll = ev_poll, .destroy = ev_destroy };
    return &e->t;
}

static void test_hostile_immediate(void)
{
    /* a slot beyond the ring; a chunk number beyond the plan; one slot written twice before it was handed back; one chunk twice */
    struct { int n; unsigned slot[2], seq[2], bytes[2]; const char *want; } cases[] = {
        { 1, { 4000, 0 }, { 0, 0 }, { CH, 0 }, "slot 4000" }, { 1, { 0, 0 }, { 100000, 0 }, { CH, 0 }, "beyond the plan" },
        { 2, { 0, 0 }, { 0, 1 }, { CH, CH }, "before it was handed back" }, { 2, { 0, 1 }, { 0, 0 }, { CH, CH }, "twice" },
        { 1, { 0, 0 }, { 0, 0 }, { CH - 1, 0 }, "bytes for chunk" }, { 1, { 0, 0 }, { 0, 0 }, { 0, 0 }, "bytes for chunk" },
        { 1, { 0, 0 }, { 0, 0 }, { CH + 1, 0 }, "bytes for chunk" } };    /* and a write whose length is not the chunk's */
    for (int k = 0; k < 7; k++) {
        fresh();
        make_file(srcd, "f", 8 * CH);
        char *paths[] = { "f" };
        char d[1][400]; dest_of(d, 0);
        fx_loopnet *net = fx_loopnet_new(2);
        int sv[2];
        socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
        fx_ctrl sc;
        fx_ctrl_init(&sc, sv[0]);
        fx_peer peer = { .name = "r0", .ctrl = &sc };
        rarg r;
        memset(&r, 0, sizeof r);
        fx_ctrl_init(&r.c, sv[1]);
        r.tr = mk_evil(fx_loopnet_transport(net), cases[k].n, cases[k].slot, cases[k].seq, cases[k].bytes);
        snprintf(r.dest, sizeof r.dest, "%s", d[0]);
        r.cfg = (fx_cfg){ .token = TOKEN, .chunk = CH, .io_threads = 2, .direct = 1 };
        r.ring = 4 * CH;
        pthread_create(&r.th, NULL, recv_main, &r);
        fx_cfg cfg = { .token = TOKEN, .chunk = CH, .io_threads = 2, .direct = 1, .ring_bytes = 4 * CH };
        fx_send_res sres;
        int rc = fx_run_sender(srcd, paths, 1, &peer, 1, ident, fx_loopnet_transport(net), &cfg, &sres);
        pthread_join(r.th, NULL);
        CHECK(r.rc == -1 && strstr(r.res.err, cases[k].want));
        CHECK(rc == -1 && !exists(d[0], "f"));
        close(sc.fd); fx_ctrl_free(&sc); fx_ctrl_free(&r.c);
    }
}

int main(void)
{
    snprintf(tmp, sizeof tmp, "/tmp/fxengXXXXXX");
    if (!mkdtemp(tmp)) return 2;
    test_sizes_tails_recycling();
    test_nothing_dry_wrongsize_missing();
    test_slot_protocol();
    test_fanout();
    test_targets();
    test_corruption();
    test_readback_and_verify();
    test_manifest();
    test_traversal_refused();
    test_symlinks_refused();
    test_auth();
    test_hostile_immediate();
    rmrf(tmp);
    T_DONE();
}
