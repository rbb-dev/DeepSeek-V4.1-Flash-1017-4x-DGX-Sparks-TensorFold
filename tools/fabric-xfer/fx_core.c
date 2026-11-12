/* fabric-xfer core: checksums, chunk table, slot pool, memory policy, control channel, thread pool, disk helpers. */
#include "fx.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <time.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <netinet/in.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#if defined(__aarch64__) && defined(__GNUC__)
#include <arm_acle.h>
#include <asm/hwcap.h>
#include <sys/auxv.h>
#define FX_HAVE_HW 1
#ifndef HWCAP_CRC32
#define HWCAP_CRC32 (1 << 7)
#endif
#endif

#define GiB ((uint64_t)1 << 30)
#define POLY_C 0x82F63B78u

/* ------------------------------------------------------------------------------------------------ CRC-32C */
static uint32_t T[8][256];
static pthread_once_t crc_once = PTHREAD_ONCE_INIT;
static int use_hw, force_sw;
static void crc_init(void);

#ifdef FX_HAVE_HW
__attribute__((target("+crc"))) static uint32_t crc_hw(uint32_t crc, const uint8_t *p, size_t n)
{
    crc = ~crc;
    while (n && ((uintptr_t)p & 7)) { crc = __crc32cb(crc, *p++); n--; }
    while (n >= 32) {
        uint64_t a, b, c, d;
        memcpy(&a, p, 8); memcpy(&b, p + 8, 8); memcpy(&c, p + 16, 8); memcpy(&d, p + 24, 8);
        crc = __crc32cd(crc, a); crc = __crc32cd(crc, b); crc = __crc32cd(crc, c); crc = __crc32cd(crc, d);
        p += 32; n -= 32;
    }
    while (n >= 8) { uint64_t a; memcpy(&a, p, 8); crc = __crc32cd(crc, a); p += 8; n -= 8; }
    while (n--) crc = __crc32cb(crc, *p++);
    return ~crc;
}
#endif

uint32_t fx_crc32c_sw(uint32_t crc, const void *buf, size_t len)
{
    pthread_once(&crc_once, crc_init);
    const uint8_t *p = buf;
    crc = ~crc;
    while (len && ((uintptr_t)p & 7)) { crc = (crc >> 8) ^ T[0][(crc ^ *p++) & 0xff]; len--; }
    while (len >= 8) {
        uint64_t v;
        memcpy(&v, p, 8);
        v ^= crc;
        crc = T[7][v & 0xff] ^ T[6][(v >> 8) & 0xff] ^ T[5][(v >> 16) & 0xff] ^ T[4][(v >> 24) & 0xff] ^
              T[3][(v >> 32) & 0xff] ^ T[2][(v >> 40) & 0xff] ^ T[1][(v >> 48) & 0xff] ^ T[0][v >> 56];
        p += 8; len -= 8;
    }
    while (len--) crc = (crc >> 8) ^ T[0][(crc ^ *p++) & 0xff];
    return ~crc;
}

static void crc_init(void)
{
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = (c & 1) ? (c >> 1) ^ POLY_C : c >> 1;
        T[0][i] = c;
    }
    for (uint32_t i = 0; i < 256; i++)
        for (int k = 1; k < 8; k++) T[k][i] = (T[k - 1][i] >> 8) ^ T[0][T[k - 1][i] & 0xff];
#ifdef FX_HAVE_HW
    if (getauxval(AT_HWCAP) & HWCAP_CRC32) {
        /* the instructions are trusted only after a known answer and a cross-check against the table path */
        uint8_t pat[1237];
        for (size_t i = 0; i < sizeof pat; i++) pat[i] = (uint8_t)(i * 131 + 7);
        uint32_t ref = ~0u;
        for (size_t i = 0; i < sizeof pat; i++) ref = (ref >> 8) ^ T[0][(ref ^ pat[i]) & 0xff];
        ref = ~ref;
        use_hw = crc_hw(0, (const uint8_t *)"123456789", 9) == 0xE3069283u && crc_hw(0, pat, sizeof pat) == ref;
    }
#endif
}

uint32_t fx_crc32c(uint32_t crc, const void *buf, size_t len)
{
    pthread_once(&crc_once, crc_init);
#ifdef FX_HAVE_HW
    if (use_hw && !force_sw) return crc_hw(crc, buf, len);
#endif
    return fx_crc32c_sw(crc, buf, len);
}

int fx_crc_hw_active(void)
{
    fx_crc32c(0, "", 0);
    return use_hw && !force_sw;
}

void fx_crc_force_sw(int on) { force_sw = on; }

static uint32_t gf2_times(const uint32_t *mat, uint32_t vec)
{
    uint32_t sum = 0;
    while (vec) { if (vec & 1) sum ^= *mat; vec >>= 1; mat++; }
    return sum;
}
static void gf2_square(uint32_t *sq, const uint32_t *mat)
{
    for (int n = 0; n < 32; n++) sq[n] = gf2_times(mat, mat[n]);
}

uint32_t fx_crc32c_combine(uint32_t crc1, uint32_t crc2, uint64_t len2)
{
    uint32_t even[32], odd[32];
    if (len2 == 0) return crc1;
    odd[0] = POLY_C;
    uint32_t row = 1;
    for (int n = 1; n < 32; n++) { odd[n] = row; row <<= 1; }
    gf2_square(even, odd);
    gf2_square(odd, even);
    do {
        gf2_square(even, odd);
        if (len2 & 1) crc1 = gf2_times(even, crc1);
        len2 >>= 1;
        if (!len2) break;
        gf2_square(odd, even);
        if (len2 & 1) crc1 = gf2_times(odd, crc1);
        len2 >>= 1;
    } while (len2);
    return crc1 ^ crc2;
}

/* ------------------------------------------------------------------------------------------------ helpers */
uint32_t fx_pack_imm(unsigned slot, unsigned seq) { return (slot << 20) | (seq & 0xFFFFFu); }
void fx_unpack_imm(uint32_t imm, unsigned *slot, unsigned *seq) { *slot = (imm >> 20) & 0xFFFu; *seq = imm & 0xFFFFFu; }

size_t fx_choose_chunk(const uint64_t *sizes, int n, size_t chunk)
{
    for (;;) {
        uint64_t c = 0;
        for (int i = 0; i < n; i++) c += (sizes[i] + chunk - 1) / chunk;
        if (c <= FX_MAX_CHUNKS) return chunk;
        chunk *= 2;
    }
}

int fx_fit_chunk(size_t chunk, uint64_t budget, size_t *out)
{
    if (budget >= (uint64_t)FX_MIN_SLOTS * chunk) { *out = chunk; return 0; }
    uint64_t c = budget / FX_MIN_SLOTS / FX_ALIGN * FX_ALIGN;
    if (c >= (1u << 20)) c = c / (1u << 20) * (1u << 20);
    if (c < FX_ALIGN) return -1;
    *out = (size_t)c;
    return 0;
}

int64_t fx_parse_meminfo(const char *text)
{
    const char *p = strstr(text, "MemAvailable:");
    if (!p) return -1;
    return (int64_t)strtoull(p + 13, NULL, 10) * 1024;
}

uint64_t fx_memory_policy(uint64_t avail, int engine, double ring_gib, char *why, size_t wl)
{
    if (ring_gib >= 0) {
        uint64_t want = (uint64_t)(ring_gib * (double)GiB), limit = avail / 5 * 4;
        if (want > limit) {
            snprintf(why, wl, "--ring-gib %g is above 80 %% of MemAvailable (%.1f GiB): using 80 %%", ring_gib, (double)avail / GiB);
            return limit;
        }
        snprintf(why, wl, "--ring-gib %g", ring_gib);
        return want;
    }
    uint64_t b = avail / 2, cap = (uint64_t)FX_DEFAULT_RING_GIB * GiB;
    if (b > cap) b = cap;
    snprintf(why, wl, "50 %% of MemAvailable %.1f GiB, at most %u GiB", (double)avail / GiB, FX_DEFAULT_RING_GIB);
    int low = avail < 16 * GiB;
    if (engine || low) {
        uint64_t q = avail / 4;
        if (q < b) {          /* said only when it is what limits the ring */
            b = q;
            snprintf(why, wl, "25 %% of MemAvailable %.1f GiB at most, because %s%s%s", (double)avail / GiB,
                     engine ? "an engine container is running" : "", engine && low ? " and " : "",
                     low ? "MemAvailable is below 16 GiB" : "");
        }
    }
    return b;
}

int fx_detect_engines(const char *const *pat, int npat, char *names, size_t len)
{
    names[0] = 0;
    FILE *f = popen("docker ps --format '{{.Names}}' 2>/dev/null", "r");
    if (!f) return -1;
    char line[256];
    int count = 0;
    while (fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = 0;
        for (int i = 0; i < npat; i++)
            if (line[0] && strstr(line, pat[i])) {
                if (names[0] && strlen(names) + strlen(line) + 3 < len) strcat(names, ", ");
                if (strlen(names) + strlen(line) + 1 < len) strcat(names, line);
                count++;
                break;
            }
    }
    return pclose(f) == 0 ? count : -1;
}

int fx_expand_dest(const char *dest, const char *home, char *out, size_t outlen)
{
    const char *rest;
    int hm = strncmp(dest, "$HOME/", 6) == 0;
    if (hm) rest = dest + 6;
    else if (dest[0] == '/') rest = dest + 1;
    else return -1;
    for (const char *q = rest; *q; q++)
        if (!(isalnum((unsigned char)*q) || strchr("_./+@=:-", *q))) return -1;
    if (hm) snprintf(out, outlen, "%s/%s", home, rest);
    else snprintf(out, outlen, "%s", dest);
    return 0;
}

int fx_check_ipv4(const char *s)
{
    unsigned a, b, c, d; int n = 0;
    if (sscanf(s, "%u.%u.%u.%u%n", &a, &b, &c, &d, &n) != 4 || s[n]) return -1;
    return (a > 255 || b > 255 || c > 255 || d > 255) ? -1 : 0;
}

int fx_path_ok(const char *p, size_t len)
{
    if (len == 0 || len > FX_PATH_MAX || p[0] == '/') return 0;
    size_t seg = 0;                                   /* start of the current segment */
    for (size_t i = 0; i <= len; i++) {
        if (i < len && ((unsigned char)p[i] < 0x20 || p[i] == 0x7f)) return 0;     /* NUL and control characters */
        if (i == len || p[i] == '/') {
            size_t n = i - seg;
            if (n == 0) return 0;                                                  /* empty segment: "a//b", "a/" */
            if (n == 1 && p[seg] == '.') return 0;
            if (n == 2 && p[seg] == '.' && p[seg + 1] == '.') return 0;
            seg = i + 1;
        }
    }
    return 1;
}

int fx_ct_equal(const void *a, size_t alen, const void *b, size_t blen)
{
    const unsigned char *x = a, *y = b;
    unsigned diff = (unsigned)(alen ^ blen);
    for (size_t i = 0; i < 256; i++) {                 /* always the same number of steps, whatever the contents */
        unsigned char c = i < alen ? x[i] : 0, d = i < blen ? y[i] : 0;
        diff |= (unsigned)(c ^ d);
    }
    return alen <= 256 && blen <= 256 && diff == 0;
}

/* ------------------------------------------------------------------------------------------------ SHA-256, HMAC */
static const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
    0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
    0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
    0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
    0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2 };
#define ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

typedef struct { uint32_t h[8]; uint64_t len; uint8_t buf[64]; size_t n; } sha_ctx;

static void sha_block(sha_ctx *c, const uint8_t *p)
{
    uint32_t w[64], a, b, cc, d, e, f, g, h;
    for (int i = 0; i < 16; i++) w[i] = ((uint32_t)p[4 * i] << 24) | ((uint32_t)p[4 * i + 1] << 16) | ((uint32_t)p[4 * i + 2] << 8) | p[4 * i + 3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ROR(w[i - 15], 7) ^ ROR(w[i - 15], 18) ^ (w[i - 15] >> 3), s1 = ROR(w[i - 2], 17) ^ ROR(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    a = c->h[0]; b = c->h[1]; cc = c->h[2]; d = c->h[3]; e = c->h[4]; f = c->h[5]; g = c->h[6]; h = c->h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + (ROR(e, 6) ^ ROR(e, 11) ^ ROR(e, 25)) + ((e & f) ^ (~e & g)) + K256[i] + w[i];
        uint32_t t2 = (ROR(a, 2) ^ ROR(a, 13) ^ ROR(a, 22)) + ((a & b) ^ (a & cc) ^ (b & cc));
        h = g; g = f; f = e; e = d + t1; d = cc; cc = b; b = a; a = t1 + t2;
    }
    c->h[0] += a; c->h[1] += b; c->h[2] += cc; c->h[3] += d; c->h[4] += e; c->h[5] += f; c->h[6] += g; c->h[7] += h;
}
static void sha_init(sha_ctx *c)
{
    static const uint32_t iv[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    memcpy(c->h, iv, sizeof iv); c->len = 0; c->n = 0;
}
static void sha_update(sha_ctx *c, const void *data, size_t len)
{
    const uint8_t *p = data;
    c->len += len;
    while (len) {
        size_t k = 64 - c->n < len ? 64 - c->n : len;
        memcpy(c->buf + c->n, p, k);
        c->n += k; p += k; len -= k;
        if (c->n == 64) { sha_block(c, c->buf); c->n = 0; }
    }
}
static void sha_final(sha_ctx *c, uint8_t out[32])
{
    uint64_t bits = c->len * 8;
    uint8_t pad = 0x80;
    sha_update(c, &pad, 1);
    pad = 0;
    while (c->n != 56) sha_update(c, &pad, 1);
    uint8_t lb[8];
    for (int i = 0; i < 8; i++) lb[i] = (uint8_t)(bits >> (56 - 8 * i));
    sha_update(c, lb, 8);
    for (int i = 0; i < 8; i++) { out[4 * i] = (uint8_t)(c->h[i] >> 24); out[4 * i + 1] = (uint8_t)(c->h[i] >> 16); out[4 * i + 2] = (uint8_t)(c->h[i] >> 8); out[4 * i + 3] = (uint8_t)c->h[i]; }
}
void fx_sha256(uint8_t out[32], const void *data, size_t len)
{
    sha_ctx c;
    sha_init(&c); sha_update(&c, data, len); sha_final(&c, out);
}
void fx_hmac_sha256(uint8_t out[32], const void *key, size_t klen, const void *msg, size_t mlen)
{
    uint8_t k[64] = { 0 }, ipad[64], opad[64], inner[32];
    if (klen > 64) fx_sha256(k, key, klen); else if (klen) memcpy(k, key, klen);
    for (int i = 0; i < 64; i++) { ipad[i] = k[i] ^ 0x36; opad[i] = k[i] ^ 0x5c; }
    sha_ctx c;
    sha_init(&c); sha_update(&c, ipad, 64); sha_update(&c, msg, mlen); sha_final(&c, inner);
    sha_init(&c); sha_update(&c, opad, 64); sha_update(&c, inner, 32); sha_final(&c, out);
}

int fx_random(void *buf, size_t n)
{
    uint8_t *p = buf;
    while (n) {
        ssize_t r = getrandom(p, n, 0);
        if (r < 0) {
            if (errno == EINTR) continue;
            int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
            if (fd < 0) return -1;
            while (n) { ssize_t g = read(fd, p, n); if (g <= 0) { close(fd); return -1; } p += g; n -= (size_t)g; }
            close(fd);
            return 0;
        }
        p += r; n -= (size_t)r;
    }
    return 0;
}

int fx_ip_allowed(const char *csv, const char *ip)
{
    size_t il = strlen(ip);
    if (!il || !csv) return 0;
    for (const char *p = csv; *p;) {
        const char *e = strchr(p, ',');
        size_t n = e ? (size_t)(e - p) : strlen(p);
        if (n == il && memcmp(p, ip, n) == 0) return 1;
        if (!e) break;
        p = e + 1;
    }
    return 0;
}

int fx_accept_allowed(int lfd, const char *allow_csv, int timeout_ms, char *peer, size_t peerlen)
{
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (;;) {
        struct timespec t1;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        long left = timeout_ms - ((long)(t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000);
        if (left <= 0) return -1;
        struct pollfd p = { lfd, POLLIN, 0 };
        int r = poll(&p, 1, (int)left);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return -1;
        struct sockaddr_in sa;
        socklen_t sl = sizeof sa;
        int fd = accept(lfd, (struct sockaddr *)&sa, &sl);
        if (fd < 0) continue;
        char ip[64] = "";
        if (sa.sin_family == AF_INET) inet_ntop(AF_INET, &sa.sin_addr, ip, sizeof ip);
        if (fx_ip_allowed(allow_csv, ip)) { snprintf(peer, peerlen, "%s", ip); return fd; }
        close(fd);                                                   /* a stranger is refused before it can try a token */
    }
}

/* ------------------------------------------------------------------------------------------------ chunk table */
int fx_table_init(fx_table *t, const uint64_t *sizes, int nfiles, size_t chunk)
{
    memset(t, 0, sizeof *t);
    if (chunk == 0 || chunk % FX_ALIGN) return -1;
    t->chunk = chunk; t->nfiles = nfiles;
    t->size = calloc(nfiles ? nfiles : 1, 8);
    t->first = calloc(nfiles ? nfiles : 1, 4);
    t->count = calloc(nfiles ? nfiles : 1, 4);
    uint64_t total = 0, nch = 0;
    for (int i = 0; i < nfiles; i++) { t->size[i] = sizes[i]; total += sizes[i]; nch += (sizes[i] + chunk - 1) / chunk; }
    if (nch > FX_MAX_CHUNKS) { fx_table_free(t); return -1; }
    t->nchunks = (uint32_t)nch; t->total = total;
    t->cfile = calloc(nch ? nch : 1, 4);
    uint32_t s = 0;
    for (int i = 0; i < nfiles; i++) {
        t->first[i] = s;
        t->count[i] = (uint32_t)((sizes[i] + chunk - 1) / chunk);
        for (uint32_t k = 0; k < t->count[i]; k++) t->cfile[s++] = (uint32_t)i;
    }
    return 0;
}

void fx_table_free(fx_table *t)
{
    free(t->size); free(t->first); free(t->count); free(t->cfile);
    memset(t, 0, sizeof *t);
}

/* chunk 0 of every file, then chunk 1 of every file, ...: files routed to different receivers keep all their disks busy at once
   (in file order the sender fed one receiver at a time) */
void fx_table_order(const fx_table *t, uint32_t *out)
{
    uint32_t n = 0, maxc = 0;
    for (int f = 0; f < t->nfiles; f++) if (t->count[f] > maxc) maxc = t->count[f];
    for (uint32_t k = 0; k < maxc; k++)
        for (int f = 0; f < t->nfiles; f++)
            if (k < t->count[f]) out[n++] = t->first[f] + k;
}

void fx_table_chunk(const fx_table *t, uint32_t seq, int *fi, uint64_t *off, size_t *n)
{
    int f = (int)t->cfile[seq];
    uint64_t o = (uint64_t)(seq - t->first[f]) * t->chunk;
    uint64_t left = t->size[f] - o;
    *fi = f; *off = o; *n = left < t->chunk ? (size_t)left : t->chunk;
}

uint32_t fx_file_crc(const uint32_t *crcs, const fx_table *t, int fi)
{
    uint32_t crc = 0;
    for (uint32_t k = 0; k < t->count[fi]; k++) {
        int f; uint64_t off; size_t n;
        fx_table_chunk(t, t->first[fi] + k, &f, &off, &n);
        crc = fx_crc32c_combine(crc, crcs[k], n);
    }
    return crc;
}

/* ------------------------------------------------------------------------------------------------ slot pool */
int fx_slots_init(fx_slots *s, int n)
{
    s->n = n; s->top = n;
    s->stack = malloc(sizeof(int) * (n ? n : 1));
    s->out = calloc(n ? n : 1, 1);
    if (!s->stack || !s->out) return -1;
    for (int i = 0; i < n; i++) s->stack[i] = n - 1 - i;     /* acquire hands out 0 first */
    return 0;
}
void fx_slots_free(fx_slots *s) { free(s->stack); free(s->out); memset(s, 0, sizeof *s); }
int fx_slots_acquire(fx_slots *s)
{
    if (s->top == 0) return -1;
    int i = s->stack[--s->top];
    s->out[i] = 1;
    return i;
}
int fx_slots_release(fx_slots *s, int i)
{
    if (i < 0 || i >= s->n || !s->out[i]) return -1;
    s->out[i] = 0;
    s->stack[s->top++] = i;
    return 0;
}

/* ------------------------------------------------------------------------------------------------ rails */
static int slurp(const char *path, char *buf, size_t len)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    size_t n = fread(buf, 1, len - 1, f);
    fclose(f);
    buf[n] = 0;
    while (n && (buf[n - 1] == '\n' || buf[n - 1] == ' ' || buf[n - 1] == '\r')) buf[--n] = 0;
    return 0;
}

static int cmp_str(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }
static int cmp_num(const void *a, const void *b) { return atoi(*(char *const *)a) - atoi(*(char *const *)b); }

static int list_dir(const char *dir, char ***names, int numeric)
{
    DIR *d = opendir(dir);
    if (!d) return -1;
    char **v = NULL; int n = 0;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        v = realloc(v, sizeof(char *) * (n + 1));
        v[n++] = strdup(e->d_name);
    }
    closedir(d);
    if (n) qsort(v, n, sizeof(char *), numeric ? cmp_num : cmp_str);
    *names = v;
    return n;
}
static void free_list(char **v, int n) { for (int i = 0; i < n; i++) free(v[i]); free(v); }

int fx_find_gid(const char *ip, const char *root, fx_rail *out)
{
    unsigned a, b, c, d;
    if (sscanf(ip, "%u.%u.%u.%u", &a, &b, &c, &d) != 4) return -1;
    char want[64];
    snprintf(want, sizeof want, "0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x", a, b, c, d);
    char **devs; int nd = list_dir(root, &devs, 0);
    for (int i = 0; i < nd; i++) {
        char pd[512]; snprintf(pd, sizeof pd, "%s/%s/ports", root, devs[i]);
        char **ports; int np = list_dir(pd, &ports, 1);
        for (int j = 0; j < np; j++) {
            char gd[640]; snprintf(gd, sizeof gd, "%s/%s/gids", pd, ports[j]);
            char **idx; int ni = list_dir(gd, &idx, 1);
            for (int k = 0; k < ni; k++) {
                char pth[800], gid[128], typ[64];
                snprintf(pth, sizeof pth, "%s/%s", gd, idx[k]);
                if (slurp(pth, gid, sizeof gid)) continue;
                for (char *q = gid; *q; q++) *q = (char)tolower((unsigned char)*q);
                snprintf(pth, sizeof pth, "%s/%s/gid_attrs/types/%s", pd, ports[j], idx[k]);
                if (slurp(pth, typ, sizeof typ)) continue;
                if (strcmp(gid, want) || strcmp(typ, "RoCE v2")) continue;
                memset(out, 0, sizeof *out);
                snprintf(out->ibdev, sizeof out->ibdev, "%s", devs[i]);
                out->port = atoi(ports[j]); out->gid_index = atoi(idx[k]);
                snprintf(out->ip, sizeof out->ip, "%s", ip);
                snprintf(pth, sizeof pth, "%s/%s/gid_attrs/ndevs/%s", pd, ports[j], idx[k]);
                slurp(pth, out->netdev, sizeof out->netdev);
                free_list(idx, ni); free_list(ports, np); free_list(devs, nd);
                return 0;
            }
            free_list(idx, ni);
        }
        free_list(ports, np);
    }
    free_list(devs, nd);
    return -1;
}

/* ------------------------------------------------------------------------------------------------ buffers and control */
static void buf_room(fx_buf *b, size_t n)
{
    if (b->len + n <= b->cap) return;
    size_t cap = b->cap ? b->cap : 256;
    while (cap < b->len + n) cap *= 2;
    b->p = realloc(b->p, cap);
    if (!b->p) abort();
    b->cap = cap;
}
void fx_buf_free(fx_buf *b) { free(b->p); memset(b, 0, sizeof *b); }
void fx_put_u8(fx_buf *b, uint8_t v) { buf_room(b, 1); b->p[b->len++] = v; }
void fx_put_u32(fx_buf *b, uint32_t v) { buf_room(b, 4); for (int i = 3; i >= 0; i--) b->p[b->len++] = (uint8_t)(v >> (8 * i)); }
void fx_put_u64(fx_buf *b, uint64_t v) { buf_room(b, 8); for (int i = 7; i >= 0; i--) b->p[b->len++] = (uint8_t)(v >> (8 * i)); }
void fx_put_bytes(fx_buf *b, const void *p, size_t n) { fx_put_u32(b, (uint32_t)n); buf_room(b, n); if (n) memcpy(b->p + b->len, p, n); b->len += n; }
void fx_put_str(fx_buf *b, const char *s) { fx_put_bytes(b, s, strlen(s)); }

uint8_t fx_get_u8(fx_rd *r) { if (r->pos + 1 > r->len) { r->bad = 1; return 0; } return r->p[r->pos++]; }
uint32_t fx_get_u32(fx_rd *r)
{
    if (r->pos + 4 > r->len) { r->bad = 1; return 0; }
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) v = (v << 8) | r->p[r->pos++];
    return v;
}
uint64_t fx_get_u64(fx_rd *r)
{
    if (r->pos + 8 > r->len) { r->bad = 1; return 0; }
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v = (v << 8) | r->p[r->pos++];
    return v;
}
const uint8_t *fx_get_bytes(fx_rd *r, uint32_t *n)
{
    *n = fx_get_u32(r);
    if (r->bad || r->pos + *n > r->len) { r->bad = 1; *n = 0; return (const uint8_t *)""; }
    const uint8_t *p = r->p + r->pos;
    r->pos += *n;
    return p;
}
char *fx_get_str(fx_rd *r)
{
    uint32_t n; const uint8_t *p = fx_get_bytes(r, &n);
    char *s = malloc(n + 1);
    memcpy(s, p, n); s[n] = 0;
    return s;
}

void fx_ctrl_init(fx_ctrl *c, int fd) { memset(c, 0, sizeof *c); c->fd = fd; }
void fx_ctrl_free(fx_ctrl *c) { free(c->buf); memset(c, 0, sizeof *c); c->fd = -1; }

int fx_ctrl_send(fx_ctrl *c, int type, const fx_buf *pl)
{
    size_t plen = pl ? pl->len : 0;
    uint8_t *frame = malloc(5 + plen);
    uint32_t fl = (uint32_t)(1 + plen);
    frame[0] = fl >> 24; frame[1] = fl >> 16; frame[2] = fl >> 8; frame[3] = (uint8_t)fl; frame[4] = (uint8_t)type;
    if (plen) memcpy(frame + 5, pl->p, plen);
    size_t off = 0; int rc = 0;
    while (off < 5 + plen) {
        ssize_t w = send(c->fd, frame + off, 5 + plen - off, MSG_NOSIGNAL);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) { rc = -1; break; }
        off += (size_t)w;
    }
    free(frame);
    return rc;
}

static int ctrl_extract(fx_ctrl *c, fx_msg *m)
{
    if (c->len < 4) return 0;
    uint32_t fl = ((uint32_t)c->buf[0] << 24) | (c->buf[1] << 16) | (c->buf[2] << 8) | c->buf[3];
    if (fl < 1 || fl > (1u << 28)) return -1;
    if (c->len < 4 + (size_t)fl) return 0;
    m->type = c->buf[4];
    m->len = fl - 1;
    m->data = malloc(m->len + 1);
    memcpy(m->data, c->buf + 5, m->len);
    memmove(c->buf, c->buf + 4 + fl, c->len - 4 - fl);
    c->len -= 4 + fl;
    return 1;
}

static int ctrl_fill(fx_ctrl *c, int flags)
{
    if (c->len + 65536 > c->cap) {
        c->cap = (c->len + 65536) * 2;
        c->buf = realloc(c->buf, c->cap);
    }
    ssize_t r = recv(c->fd, c->buf + c->len, c->cap - c->len, flags);
    if (r > 0) { c->len += (size_t)r; return 1; }
    if (r < 0 && (errno == EAGAIN || errno == EINTR)) return 0;
    return -1;
}

int fx_ctrl_poll(fx_ctrl *c, fx_msg *m)
{
    for (;;) {
        int e = ctrl_extract(c, m);
        if (e) return e;
        struct pollfd p = { c->fd, POLLIN, 0 };
        if (poll(&p, 1, 0) <= 0) return 0;
        int f = ctrl_fill(c, MSG_DONTWAIT);
        if (f < 0) return -1;
        if (f == 0) return 0;
    }
}

int fx_ctrl_recv(fx_ctrl *c, int timeout_ms, fx_msg *m)
{
    int waited = 0;
    for (;;) {
        int e = ctrl_extract(c, m);
        if (e < 0) return -1;
        if (e) return 0;
        struct pollfd p = { c->fd, POLLIN, 0 };
        int r = poll(&p, 1, 200);
        if (r > 0) { if (ctrl_fill(c, 0) < 0) return -1; }
        else if (r == 0) { waited += 200; if (waited >= timeout_ms) return -2; }
        else if (errno != EINTR) return -1;
    }
}

void fx_msg_free(fx_msg *m) { free(m->data); m->data = NULL; }

/* ------------------------------------------------------------------------------------------------ queue, pool */
void fx_queue_init(fx_queue *q) { pthread_mutex_init(&q->mu, NULL); q->head = q->tail = NULL; }
void fx_queue_push(fx_queue *q, fx_node *n)
{
    n->next = NULL;
    pthread_mutex_lock(&q->mu);
    if (q->tail) q->tail->next = n; else q->head = n;
    q->tail = n;
    pthread_mutex_unlock(&q->mu);
}
fx_node *fx_queue_pop(fx_queue *q)
{
    pthread_mutex_lock(&q->mu);
    fx_node *n = q->head;
    if (n) { q->head = n->next; if (!q->head) q->tail = NULL; }
    pthread_mutex_unlock(&q->mu);
    return n;
}

struct tp_job { struct tp_job *next; void (*fn)(void *); void *arg; };
struct fx_tp {
    pthread_mutex_t mu; pthread_cond_t cv; struct tp_job *head, *tail; int stop, n; pthread_t *th;
};
static void *tp_main(void *a)
{
    fx_tp *tp = a;
    for (;;) {
        pthread_mutex_lock(&tp->mu);
        while (!tp->head && !tp->stop) pthread_cond_wait(&tp->cv, &tp->mu);
        struct tp_job *j = tp->head;
        if (!j) { pthread_mutex_unlock(&tp->mu); return NULL; }
        tp->head = j->next;
        if (!tp->head) tp->tail = NULL;
        pthread_mutex_unlock(&tp->mu);
        j->fn(j->arg);
        free(j);
    }
}
fx_tp *fx_tp_new(int n)
{
    fx_tp *tp = calloc(1, sizeof *tp);
    pthread_mutex_init(&tp->mu, NULL); pthread_cond_init(&tp->cv, NULL);
    tp->n = n; tp->th = calloc(n, sizeof(pthread_t));
    for (int i = 0; i < n; i++) pthread_create(&tp->th[i], NULL, tp_main, tp);
    return tp;
}
void fx_tp_submit(fx_tp *tp, void (*fn)(void *), void *arg)
{
    struct tp_job *j = malloc(sizeof *j);
    j->next = NULL; j->fn = fn; j->arg = arg;
    pthread_mutex_lock(&tp->mu);
    if (tp->tail) tp->tail->next = j; else tp->head = j;
    tp->tail = j;
    pthread_cond_signal(&tp->cv);
    pthread_mutex_unlock(&tp->mu);
}
void fx_tp_free(fx_tp *tp)
{
    pthread_mutex_lock(&tp->mu);
    tp->stop = 1;
    pthread_cond_broadcast(&tp->cv);
    pthread_mutex_unlock(&tp->mu);
    for (int i = 0; i < tp->n; i++) pthread_join(tp->th[i], NULL);
    free(tp->th); free(tp);
}

/* ------------------------------------------------------------------------------------------------ disk */
int (*fx_write_hook)(int fd, const void *buf, size_t len, uint64_t off);

int fx_open(const char *path, int flags, int mode, int direct, int *got_direct, char *note, size_t notelen)
{
    if (direct) {
        int fd = open(path, flags | O_DIRECT, mode);
        if (fd >= 0) { *got_direct = 1; return fd; }
        if (errno != EINVAL) return -1;
        if (note) snprintf(note, notelen, "O_DIRECT is not supported for %s (EINVAL): buffered I/O for it", path);
    }
    *got_direct = 0;
    return open(path, flags, mode);
}

#ifndef SYS_openat2
#define SYS_openat2 437
#endif
struct fx_how { uint64_t flags, mode, resolve; };
#define FX_RESOLVE_NO_SYMLINKS 0x04
#define FX_RESOLVE_BENEATH 0x08
int fx_openat2_disabled;

/* returns the fd, or -1 with errno; *used = 0 if openat2 is not available here (the caller walks instead) */
static int try_openat2(int dirfd, const char *rel, int flags, int mode, int *used)
{
    static int unavailable;
    if (fx_openat2_disabled || unavailable) { *used = 0; return -1; }
    struct fx_how h = { (uint64_t)(unsigned)flags, (flags & O_CREAT) ? (uint64_t)mode : 0, FX_RESOLVE_BENEATH | FX_RESOLVE_NO_SYMLINKS };
    long r = syscall(SYS_openat2, dirfd, rel, &h, sizeof h);
    if (r >= 0) { *used = 1; return (int)r; }
    if (errno == ENOSYS || errno == EPERM) { unavailable = 1; *used = 0; return -1; }   /* old kernel, or a seccomp filter */
    *used = 1;
    return -1;
}

int fx_open_parent(int dirfd, const char *rel, int make_dirs, char *leaf, size_t leaflen)
{
    if (!fx_path_ok(rel, strlen(rel))) { errno = EINVAL; return -1; }
    int cur = openat(dirfd, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (cur < 0) return -1;
    const char *p = rel;
    for (;;) {
        const char *sl = strchr(p, '/');
        if (!sl) { snprintf(leaf, leaflen, "%s", p); return cur; }
        char comp[256];
        size_t n = (size_t)(sl - p);
        if (n >= sizeof comp) { close(cur); errno = ENAMETOOLONG; return -1; }
        memcpy(comp, p, n); comp[n] = 0;
        if (make_dirs && mkdirat(cur, comp, 0755) && errno != EEXIST) { int e = errno; close(cur); errno = e; return -1; }
        int nx = openat(cur, comp, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        int e = errno;
        close(cur);
        if (nx < 0) { errno = e; return -1; }
        cur = nx;
        p = sl + 1;
    }
}

static int open_once(int dirfd, const char *rel, int flags, int mode)
{
    int used;
    int fd = try_openat2(dirfd, rel, flags | O_CLOEXEC, mode, &used);
    if (used) return fd;
    char leaf[256];
    int parent = fx_open_parent(dirfd, rel, 0, leaf, sizeof leaf);
    if (parent < 0) return -1;
    fd = openat(parent, leaf, flags | O_NOFOLLOW | O_CLOEXEC, mode);
    int e = errno;
    close(parent);
    errno = e;
    return fd;
}

int fx_open_beneath(int dirfd, const char *rel, int flags, int mode, int direct, int *got_direct, int make_dirs)
{
    if (!fx_path_ok(rel, strlen(rel))) { errno = EINVAL; return -1; }
    if (make_dirs) {
        char leaf[256];
        int p = fx_open_parent(dirfd, rel, 1, leaf, sizeof leaf);
        if (p < 0) return -1;
        close(p);
    }
    if (direct) {
        int fd = open_once(dirfd, rel, flags | O_DIRECT, mode);
        if (fd >= 0) { *got_direct = 1; return fd; }
        if (errno != EINVAL) return -1;
    }
    *got_direct = 0;
    return open_once(dirfd, rel, flags, mode);
}

int fx_pread_chunk(int fd, void *buf, uint64_t off, size_t n, int direct, char *err, size_t errlen)
{
    size_t want = direct ? (size_t)fx_align_up(n) : n, got = 0;
    while (got < n) {
        ssize_t r = pread(fd, (char *)buf + got, want - got, (off_t)(off + got));
        if (r < 0) { if (errno == EINTR) continue; snprintf(err, errlen, "read at %llu: %s", (unsigned long long)off, strerror(errno)); return -1; }
        if (r == 0) break;
        got += (size_t)r;
    }
    if (got < n) {
        snprintf(err, errlen, "short read at offset %llu: got %zu of %zu bytes (the file changed while it was sent?)", (unsigned long long)off, got, n);
        return -1;
    }
    return 0;
}

int fx_pwrite_real(int fd, const void *buf, size_t len, uint64_t off)
{
    size_t done = 0;
    while (done < len) {
        ssize_t w = pwrite(fd, (const char *)buf + done, len - done, (off_t)(off + done));
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return -1;
        done += (size_t)w;
    }
    return 0;
}

int fx_pwrite_all(int fd, const void *buf, size_t len, uint64_t off)
{
    return fx_write_hook ? fx_write_hook(fd, buf, len, off) : fx_pwrite_real(fd, buf, len, off);
}

static __thread void *tl_buf;
static __thread size_t tl_size;
static void *tl_get(size_t n)
{
    if (tl_size < n) {
        free(tl_buf);
        if (posix_memalign(&tl_buf, FX_ALIGN, n)) { tl_buf = NULL; tl_size = 0; return NULL; }
        tl_size = n;
    }
    return tl_buf;
}

typedef struct {
    pthread_mutex_t mu; pthread_cond_t cv; int remaining, failed; char err[FX_ERRLEN];
} crc_group;
typedef struct {
    crc_group *g; int dirfd, secure; const char *path; uint64_t size; size_t chunk; int direct; uint32_t k; uint32_t *out;
} crc_job;

static void crc_job_run(void *a)
{
    crc_job *j = a;
    char err[FX_ERRLEN] = "";
    int bad = 0;
    uint64_t off = (uint64_t)j->k * j->chunk;
    size_t n = j->size - off < j->chunk ? (size_t)(j->size - off) : j->chunk;
    void *buf = tl_get(j->chunk);
    int d;
    int fd = !buf ? -1 : j->secure ? fx_open_beneath(j->dirfd, j->path, O_RDONLY, 0, j->direct, &d, 0)
                                    : fx_open(j->path, O_RDONLY, 0, j->direct, &d, NULL, 0);
    if (fd < 0) { snprintf(err, sizeof err, "%s: %s", j->path, strerror(errno)); bad = 1; }
    else {
        if (fx_pread_chunk(fd, buf, off, n, d, err, sizeof err)) bad = 1;
        else j->out[j->k] = fx_crc32c(0, buf, n);
        close(fd);
    }
    pthread_mutex_lock(&j->g->mu);
    if (bad && !j->g->failed) { j->g->failed = 1; snprintf(j->g->err, sizeof j->g->err, "%s", err); }
    j->g->remaining--;
    pthread_cond_signal(&j->g->cv);
    pthread_mutex_unlock(&j->g->mu);
    free(j);
}

int fx_file_chunk_crcs(const char *path, uint64_t size, size_t chunk, int direct, fx_tp *pool, uint32_t *out, char *err, size_t errlen)
{
    return fx_file_chunk_crcs_at(AT_FDCWD, 0, path, size, chunk, direct, pool, out, err, errlen);
}

int fx_file_chunk_crcs_at(int dirfd, int secure, const char *path, uint64_t size, size_t chunk, int direct, fx_tp *pool, uint32_t *out,
                          char *err, size_t errlen)
{
    uint32_t n = (uint32_t)((size + chunk - 1) / chunk);
    crc_group g;
    pthread_mutex_init(&g.mu, NULL); pthread_cond_init(&g.cv, NULL);
    g.remaining = (int)n; g.failed = 0; g.err[0] = 0;
    for (uint32_t k = 0; k < n; k++) {
        crc_job *j = malloc(sizeof *j);
        *j = (crc_job){ &g, dirfd, secure, path, size, chunk, direct, k, out };
        if (pool) fx_tp_submit(pool, crc_job_run, j); else crc_job_run(j);
    }
    pthread_mutex_lock(&g.mu);
    while (g.remaining > 0) pthread_cond_wait(&g.cv, &g.mu);
    pthread_mutex_unlock(&g.mu);
    if (g.failed) { snprintf(err, errlen, "%s", g.err); return -1; }
    return 0;
}
