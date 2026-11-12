#include "../fx.h"
#include "t.h"
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

static uint32_t crc_bitwise(uint32_t crc, const uint8_t *p, size_t n)
{
    crc = ~crc;
    while (n--) { crc ^= *p++; for (int k = 0; k < 8; k++) crc = (crc & 1) ? (crc >> 1) ^ 0x82F63B78u : crc >> 1; }
    return ~crc;
}

static void test_crc(void)
{
    CHECK(fx_crc32c(0, "123456789", 9) == 0xE3069283u);          /* the CRC-32C check value */
    CHECK(fx_crc32c_sw(0, "123456789", 9) == 0xE3069283u);
    CHECK(fx_crc32c(0, "", 0) == 0);
    uint8_t *buf = malloc(70016);                          /* lens up to 69999 at offsets up to 8 */
    for (int i = 0; i < 70016; i++) buf[i] = (uint8_t)(i * 7 + (i >> 8));
    size_t lens[] = { 1, 7, 8, 9, 31, 32, 33, 1000, 4096, 65536, 69999 };
    for (size_t li = 0; li < sizeof lens / sizeof *lens; li++)
        for (int off = 0; off < 9; off++) {
            uint32_t ref = crc_bitwise(0, buf + off, lens[li]);
            CHECK(fx_crc32c(0, buf + off, lens[li]) == ref);
            CHECK(fx_crc32c_sw(0, buf + off, lens[li]) == ref);
        }
    uint32_t a = fx_crc32c(0, buf, 1234), whole = fx_crc32c(0, buf, 5000);
    CHECK(fx_crc32c(a, buf + 1234, 5000 - 1234) == whole);       /* chainable */
    uint32_t b = fx_crc32c(0, buf + 1234, 5000 - 1234);
    CHECK(fx_crc32c_combine(a, b, 5000 - 1234) == whole);
    CHECK(fx_crc32c_combine(a, 0, 0) == a);
    fx_crc_force_sw(1);
    CHECK(!fx_crc_hw_active());
    CHECK(fx_crc32c(0, buf, 5000) == whole);
    fx_crc_force_sw(0);
    free(buf);
}

static void test_pure(void)
{
    CHECK(fx_align_up(0) == 0 && fx_align_up(1) == 4096 && fx_align_up(4096) == 4096 && fx_align_up(4097) == 8192);
    unsigned s, q;
    uint32_t cases[][2] = { { 0, 0 }, { 4095, 0xFFFFF }, { 17, 12345 } };
    for (int i = 0; i < 3; i++) { fx_unpack_imm(fx_pack_imm(cases[i][0], cases[i][1]), &s, &q); CHECK(s == cases[i][0] && q == cases[i][1]); }
    uint64_t one = 1ull << 30;
    CHECK(fx_choose_chunk(&one, 1, 4096) % 4096 == 0);
    CHECK((one + fx_choose_chunk(&one, 1, 4096) - 1) / fx_choose_chunk(&one, 1, 4096) <= FX_MAX_CHUNKS);
    uint64_t ten = 10;
    CHECK(fx_choose_chunk(&ten, 1, 65536) == 65536);
    size_t c;
    CHECK(fx_fit_chunk(16 << 20, 64 << 20, &c) == 0 && c == (16u << 20));
    CHECK(fx_fit_chunk(16 << 20, 32 << 20, &c) == 0 && c == (16u << 20));
    CHECK(fx_fit_chunk(16 << 20, 16 << 20, &c) == 0 && c == (8u << 20));
    CHECK(fx_fit_chunk(16 << 20, 4096, &c) == -1);

    fx_slots p;
    fx_slots_init(&p, 3);
    int a = fx_slots_acquire(&p), b = fx_slots_acquire(&p), d = fx_slots_acquire(&p);
    CHECK(a == 0 && b == 1 && d == 2);
    CHECK(fx_slots_acquire(&p) == -1);
    CHECK(fx_slots_release(&p, 1) == 0 && fx_slots_free_count(&p) == 1 && fx_slots_acquire(&p) == 1);
    CHECK(fx_slots_release(&p, 7) == -1);
    CHECK(fx_slots_release(&p, 2) == 0 && fx_slots_release(&p, 2) == -1);   /* double release is a bug */
    fx_slots_free(&p);

    uint64_t sizes[] = { 0, 1, 4096, 10000 };
    fx_table t;
    CHECK(fx_table_init(&t, sizes, 4, 4096) == 0);
    CHECK(t.count[0] == 0 && t.count[1] == 1 && t.count[2] == 1 && t.count[3] == 3 && t.nchunks == 5);
    int fi; uint64_t off; size_t n;
    fx_table_chunk(&t, 0, &fi, &off, &n); CHECK(fi == 1 && off == 0 && n == 1);
    fx_table_chunk(&t, 4, &fi, &off, &n); CHECK(fi == 3 && off == 8192 && n == 10000 - 8192);
    CHECK(t.first[3] == 2 && t.total == 14097);
    fx_table_free(&t);
    CHECK(fx_table_init(&t, sizes, 4, 1000) == -1);           /* not a multiple of the alignment */
    {   /* the read order: chunk k of every file in turn, so files routed to different receivers keep every receiver's disk busy at
           once (read in file order, one receiver at a time wrote: 2.19 GB/s for three on the Sparks) */
        uint64_t sz[4] = { 3 * 4096, 4096, 0, 2 * 4096 };
        fx_table o3;
        CHECK(fx_table_init(&o3, sz, 4, 4096) == 0);
        uint32_t ord[6], want[6] = { 0, 3, 4, 1, 5, 2 };   /* f0 c0, f1 c0, f3 c0, f0 c1, f3 c1, f0 c2 (f2 is empty) */
        fx_table_order(&o3, ord);
        int okk = 1;
        for (int k = 0; k < 6; k++) okk &= ord[k] == want[k];
        CHECK(okk);
        fx_table_free(&o3);
    }

    /* whole-file crc from chunk crcs */
    uint8_t data[10000];
    for (int i = 0; i < 10000; i++) data[i] = (uint8_t)(i * 31 + 5);
    uint64_t sz = 10000;
    fx_table_init(&t, &sz, 1, 4096);
    uint32_t cc[3] = { fx_crc32c(0, data, 4096), fx_crc32c(0, data + 4096, 4096), fx_crc32c(0, data + 8192, 1808) };
    CHECK(fx_file_crc(cc, &t, 0) == fx_crc32c(0, data, 10000));
    fx_table_free(&t);
}

static void test_memory(void)
{
    char why[256];
    const uint64_t G = 1ull << 30;
    CHECK(fx_parse_meminfo("MemTotal: 1 kB\nMemAvailable:   2048 kB\n") == 2048 * 1024);
    CHECK(fx_parse_meminfo("MemTotal: 1 kB\n") == -1);
    /* the default ring is moderate: at most 8 GiB (a few seconds of a worker's disk writes; pinning tens of GiB cost 15-49 s on the
       Sparks and only lengthened the drain), half of MemAvailable below that, a quarter beside an engine or under 16 GiB free */
    CHECK(fx_memory_policy(200 * G, 0, -1, why, sizeof why) == 8 * G && strstr(why, "8 GiB"));
    CHECK(fx_memory_policy(100 * G, 0, -1, why, sizeof why) == 8 * G);
    CHECK(fx_memory_policy(12 * G, 0, -1, why, sizeof why) == 12 * G / 4 && strstr(why, "16 GiB"));
    CHECK(fx_memory_policy(10 * G, 0, -1, why, sizeof why) == 10 * G / 4 && strstr(why, "16 GiB"));
    CHECK(fx_memory_policy(100 * G, 1, -1, why, sizeof why) == 8 * G);
    CHECK(fx_memory_policy(20 * G, 1, -1, why, sizeof why) == 5 * G && strstr(why, "engine"));
    CHECK(fx_memory_policy(100 * G, 0, 40, why, sizeof why) == 40 * G);
    CHECK(fx_memory_policy(100 * G, 0, 90, why, sizeof why) == 80 * G && strstr(why, "80 %"));
    char out[256];
    CHECK(fx_expand_dest("$HOME/x/y", "/home/u", out, sizeof out) == 0 && !strcmp(out, "/home/u/x/y"));
    CHECK(fx_expand_dest("/data/x", "/home/u", out, sizeof out) == 0 && !strcmp(out, "/data/x"));
    CHECK(fx_expand_dest("rel/x", "/home/u", out, sizeof out) == -1);
    CHECK(fx_expand_dest("/a b;rm", "/home/u", out, sizeof out) == -1);
    CHECK(fx_check_ipv4("10.255.0.2") == 0 && fx_check_ipv4("10.255.0.x") == -1 && fx_check_ipv4("10.255.0.256") == -1 && fx_check_ipv4("1.2.3.4x") == -1);
}

static void put(const char *root, const char *rel, const char *content)
{
    char p[512];
    snprintf(p, sizeof p, "%s/%s", root, rel);
    FILE *f = fopen(p, "w");
    fputs(content, f);
    fputc('\n', f);
    fclose(f);
}

static void test_sysfs(void)
{
    char root[] = "/tmp/fxsysXXXXXX";
    CHECK(mkdtemp(root) != NULL);
    const char *dev[] = { "mlx5_0", "mlx5_2" }, *ip[] = { "0a00:21f0", "0a00:21f1" }, *nd[] = { "enp1s0f0np0", "enp1s0f1np1" };
    (void)ip;
    for (int d = 0; d < 2; d++) {
        char cmd[1024];
        snprintf(cmd, sizeof cmd, "mkdir -p %s/%s/ports/1/gids %s/%s/ports/1/gid_attrs/types %s/%s/ports/1/gid_attrs/ndevs", root, dev[d], root, dev[d], root, dev[d]);
        CHECK(system(cmd) == 0);
        char pr[512], rel[64], gid[128];
        snprintf(pr, sizeof pr, "%s/%s/ports/1", root, dev[d]);
        const char *types[] = { "IB/RoCE v1", "RoCE v2", "IB/RoCE v1", "RoCE v2" };
        for (int i = 0; i < 4; i++) {
            if (i < 2) snprintf(gid, sizeof gid, "fe80:0000:0000:0000:0000:0000:0000:0001");
            else snprintf(gid, sizeof gid, "0000:0000:0000:0000:0000:ffff:0aff:%04x", d ? 0x0101 : 0x0001);   /* 10.255.1.1 / 10.255.0.1 */
            snprintf(rel, sizeof rel, "gids/%d", i); put(pr, rel, gid);
            snprintf(rel, sizeof rel, "gid_attrs/types/%d", i); put(pr, rel, types[i]);
            snprintf(rel, sizeof rel, "gid_attrs/ndevs/%d", i); put(pr, rel, nd[d]);
        }
    }
    fx_rail r;
    CHECK(fx_find_gid("10.255.1.1", root, &r) == 0 && !strcmp(r.ibdev, "mlx5_2") && r.port == 1 && r.gid_index == 3 && !strcmp(r.netdev, "enp1s0f1np1"));
    CHECK(fx_find_gid("10.255.0.1", root, &r) == 0 && !strcmp(r.ibdev, "mlx5_0") && r.gid_index == 3);
    CHECK(fx_find_gid("10.9.9.9", root, &r) == -1);
    char cmd[600]; snprintf(cmd, sizeof cmd, "rm -rf %s", root); CHECK(system(cmd) == 0);
}

static void test_ctrl(void)
{
    int sv[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    fx_ctrl a, b;
    fx_ctrl_init(&a, sv[0]); fx_ctrl_init(&b, sv[1]);
    fx_buf p = { 0 };
    fx_put_u8(&p, 7); fx_put_u32(&p, 0xdeadbeef); fx_put_u64(&p, 0x0102030405060708ull); fx_put_str(&p, "héllo");
    CHECK(fx_ctrl_send(&a, FX_M_PLAN, &p) == 0);
    fx_msg m;
    CHECK(fx_ctrl_recv(&b, 1000, &m) == 0 && m.type == FX_M_PLAN);
    fx_rd rd = { m.data, m.len, 0, 0 };
    CHECK(fx_get_u8(&rd) == 7 && fx_get_u32(&rd) == 0xdeadbeef && fx_get_u64(&rd) == 0x0102030405060708ull);
    char *s = fx_get_str(&rd);
    CHECK(!strcmp(s, "héllo") && !rd.bad);
    free(s); fx_msg_free(&m);
    fx_get_u32(&rd); CHECK(rd.bad);                           /* reading past the end is flagged, not undefined */
    /* a frame in two pieces, then two frames at once */
    uint8_t frame[10] = { 0, 0, 0, 6, FX_M_LOG, 1, 2, 3, 4, 5 };
    CHECK(write(sv[0], frame, 3) == 3);
    CHECK(fx_ctrl_poll(&b, &m) == 0);
    uint8_t rest[17]; memcpy(rest, frame + 3, 7); memcpy(rest + 7, frame, 10);
    CHECK(write(sv[0], rest, 17) == 17);
    int got = 0;
    for (int i = 0; i < 100 && got < 2; i++) { int r = fx_ctrl_poll(&b, &m); if (r > 0) { CHECK(m.type == FX_M_LOG && m.len == 5); fx_msg_free(&m); got++; } else usleep(1000); }
    CHECK(got == 2);
    CHECK(fx_ctrl_recv(&b, 300, &m) == -2);                   /* timeout */
    close(sv[0]);
    CHECK(fx_ctrl_recv(&b, 1000, &m) == -1);                  /* EOF */
    fx_buf_free(&p); fx_ctrl_free(&a); fx_ctrl_free(&b); close(sv[1]);
}

static void test_file_crcs(void)
{
    char path[] = "/tmp/fxcrcXXXXXX";
    int fd = mkstemp(path);
    uint8_t *d = malloc(200000);
    for (int i = 0; i < 200000; i++) d[i] = (uint8_t)(i * 13 + 1);
    CHECK(write(fd, d, 200000) == 200000);
    close(fd);
    uint32_t out[4], seqout[4];
    fx_tp *tp = fx_tp_new(3);
    char err[512];
    CHECK(fx_file_chunk_crcs(path, 200000, 65536, 1, tp, out, err, sizeof err) == 0);
    CHECK(fx_file_chunk_crcs(path, 200000, 65536, 0, NULL, seqout, err, sizeof err) == 0);
    for (int k = 0; k < 4; k++) {
        size_t o = (size_t)k * 65536, n = 200000 - o < 65536 ? 200000 - o : 65536;
        CHECK(out[k] == fx_crc32c(0, d + o, n) && seqout[k] == out[k]);
    }
    CHECK(truncate(path, 70000) == 0);
    CHECK(fx_file_chunk_crcs(path, 200000, 65536, 1, tp, out, err, sizeof err) == -1 && strstr(err, "short read"));
    fx_tp_free(tp); unlink(path); free(d);
}



static void hex(const uint8_t *d, size_t n, char *out) { for (size_t i = 0; i < n; i++) sprintf(out + 2 * i, "%02x", d[i]); }
static int hmac_is(const void *key, size_t kl, const void *msg, size_t ml, const char *want)
{
    uint8_t mac[32]; char h[65];
    fx_hmac_sha256(mac, key, kl, msg, ml);
    hex(mac, 32, h);
    return strncmp(h, want, strlen(want)) == 0;
}
static int sha_is(const void *d, size_t n, const char *want) { uint8_t o[32]; char h[65]; fx_sha256(o, d, n); hex(o, 32, h); return !strcmp(h, want); }

static void test_sha_hmac(void)
{
    /* expected values: FIPS 180-4 examples and RFC 4231 test cases, each cross-checked against Python's hashlib / hmac */
    CHECK(sha_is("abc", 3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
    CHECK(sha_is("", 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
    CHECK(sha_is("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", 56, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"));
    uint8_t *m = malloc(1000000);
    memset(m, 'a', 1000000);
    CHECK(sha_is(m, 1000000, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"));
    for (size_t n = 0; n < 200; n++) {                         /* every length around the block boundaries, against a one-shot of the same bytes */
        uint8_t a[32], b[32];
        fx_sha256(a, m, n);
        fx_sha256(b, m, n);
        CHECK(memcmp(a, b, 32) == 0);
    }
    free(m);
    uint8_t k[131], d[50];
    memset(k, 0x0b, 20);
    CHECK(hmac_is(k, 20, "Hi There", 8, "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"));                    /* RFC 4231 case 1 */
    CHECK(hmac_is("Jefe", 4, "what do ya want for nothing?", 28, "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843")); /* 2 */
    memset(k, 0xaa, 20); memset(d, 0xdd, 50);
    CHECK(hmac_is(k, 20, d, 50, "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe"));                            /* 3 */
    for (int i = 0; i < 25; i++) k[i] = (uint8_t)(i + 1);
    memset(d, 0xcd, 50);
    CHECK(hmac_is(k, 25, d, 50, "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b"));                            /* 4 */
    memset(k, 0x0c, 20);
    CHECK(hmac_is(k, 20, "Test With Truncation", 20, "a3b6167473100ee06e0c796c2955552b"));                                         /* 5, 128 bits */
    memset(k, 0xaa, 131);
    CHECK(hmac_is(k, 131, "Test Using Larger Than Block-Size Key - Hash Key First", 54, "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54")); /* 6 */
    const char *t7 = "This is a test using a larger than block-size key and a larger than block-size data. The key needs to be hashed before being used by the HMAC algorithm.";
    CHECK(hmac_is(k, 131, t7, strlen(t7), "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2"));                  /* 7 */
    uint8_t r1[32], r2[32];
    CHECK(fx_random(r1, 32) == 0 && fx_random(r2, 32) == 0 && memcmp(r1, r2, 32) != 0);
}

static int dial(const char *dst, int port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in to = { .sin_family = AF_INET, .sin_port = htons((uint16_t)port) };
    inet_pton(AF_INET, dst, &to.sin_addr);
    struct timeval tv = { 3, 0 };                    /* a refused-connection check must fail, not hang, if the refusal never comes */
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    if (connect(fd, (struct sockaddr *)&to, sizeof to)) { close(fd); return -1; }
    return fd;
}

/* the first non-loopback IPv4 address of this machine: connecting to it shows the listener a source address other than 127.0.0.1 */
static int other_local_ip(char *out, size_t n)
{
    struct ifaddrs *ifa, *p;
    if (getifaddrs(&ifa)) return -1;
    int found = -1;
    for (p = ifa; p && found; p = p->ifa_next)
        if (p->ifa_addr && p->ifa_addr->sa_family == AF_INET && !(p->ifa_flags & IFF_LOOPBACK) && (p->ifa_flags & IFF_UP)) {
            inet_ntop(AF_INET, &((struct sockaddr_in *)p->ifa_addr)->sin_addr, out, (socklen_t)n);
            found = 0;
        }
    freeifaddrs(ifa);
    return found;
}

struct acc { int lfd; const char *allow; int ms; int fd; char peer[64]; };
static void *accept_thread(void *p) { struct acc *a = p; a->fd = fx_accept_allowed(a->lfd, a->allow, a->ms, a->peer, sizeof a->peer); return NULL; }

static void test_allowlist(void)
{
    CHECK(fx_ip_allowed("10.255.0.99,10.255.1.99,192.0.2.1", "10.255.1.99"));
    CHECK(fx_ip_allowed("10.255.0.99", "10.255.0.99"));
    CHECK(!fx_ip_allowed("10.255.0.99,10.255.1.99", "10.255.0.9"));        /* no prefix matches */
    CHECK(!fx_ip_allowed("10.255.0.99", "10.255.0.991"));
    CHECK(!fx_ip_allowed("", "10.255.0.99") && !fx_ip_allowed("10.255.0.99", ""));
    /* a real listener, accepting in a thread. The "stranger" connects to this machine's other address, so the listener sees a source that is
     * not 127.0.0.1 (this sandbox rewrites 127.0.0.2 to 127.0.0.1, so loopback aliases cannot play the stranger). It must be closed on
     * without a byte; only then does the expected sender (127.0.0.1) connect, and that is the connection returned. */
    char other[64];
    int have_other = other_local_ip(other, sizeof other) == 0;
    int lfd = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in sa = { .sin_family = AF_INET };
    CHECK(bind(lfd, (struct sockaddr *)&sa, sizeof sa) == 0 && listen(lfd, 8) == 0);      /* 0.0.0.0, any port */
    socklen_t sl = sizeof sa;
    getsockname(lfd, (struct sockaddr *)&sa, &sl);
    int port = ntohs(sa.sin_port);
    char c;
    if (have_other) {
        struct acc a = { lfd, "127.0.0.1", 5000, -2, "" };
        pthread_t th;
        pthread_create(&th, NULL, accept_thread, &a);
        int stranger = dial(other, port);
        CHECK(stranger >= 0 && read(stranger, &c, 1) == 0);                  /* refused before it could say anything */
        int friend = dial("127.0.0.1", port);
        CHECK(friend >= 0);
        pthread_join(th, NULL);
        CHECK(a.fd >= 0 && !strcmp(a.peer, "127.0.0.1"));
        struct timeval tv = { 3, 0 };
        if (a.fd >= 0) setsockopt(a.fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        CHECK(write(friend, "x", 1) == 1 && a.fd >= 0 && read(a.fd, &c, 1) == 1 && c == 'x');
        if (a.fd >= 0) close(a.fd);
        close(friend); close(stranger);
        /* only a stranger: it is closed on, nothing is accepted, and the call gives up at its timeout */
        struct acc b2 = { lfd, "127.0.0.1", 700, -2, "" };
        pthread_create(&th, NULL, accept_thread, &b2);
        int s2 = dial(other, port);
        CHECK(s2 >= 0 && read(s2, &c, 1) == 0);
        pthread_join(th, NULL);
        CHECK(b2.fd == -1);
        close(s2);
    } else fprintf(stderr, "note: no non-loopback IPv4 address here; the stranger-connection case was skipped\n");
    struct acc none = { lfd, "127.0.0.1", 300, -2, "" };             /* nobody connects at all: the call gives up at its timeout */
    accept_thread(&none);
    CHECK(none.fd == -1);
    close(lfd);
}

int main(void)
{
    test_crc(); test_sha_hmac(); test_allowlist(); test_pure(); test_memory(); test_sysfs(); test_ctrl(); test_file_crcs();
    printf("crc path: %s\n", fx_crc_hw_active() ? "ARMv8 CRC32C instructions" : "software slice-by-8");
    T_DONE();
}
