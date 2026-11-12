/* fabric-xfer: shared declarations. C11, POSIX, pthreads. The wire (ibverbs) sits behind fx_transport. */
#ifndef FX_H
#define FX_H
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>

#define FX_PROTO 1
#define FX_ALIGN 4096u
#define FX_MAX_SLOTS 4096u          /* the slot number is 12 bits of the immediate value */
#define FX_MAX_CHUNKS (1u << 20)    /* the chunk number is the other 20 bits */
#define FX_MIN_SLOTS 2u
#define FX_DEFAULT_CHUNK (16u << 20)
#define FX_DEFAULT_RING_GIB 8u         /* the default ring's ceiling: a few seconds of a worker's disk writes */
#define FX_MAX_PEERS 63
#define FX_ERRLEN 512

/* ---- checksums: CRC-32C (Castagnoli), the ARMv8 CRC32C instructions when present, slice-by-8 otherwise */
uint32_t fx_crc32c(uint32_t crc, const void *buf, size_t len);       /* crc of "" = 0; chainable */
uint32_t fx_crc32c_sw(uint32_t crc, const void *buf, size_t len);
int fx_crc_hw_active(void);                                           /* 1 if fx_crc32c uses the instructions */
void fx_crc_force_sw(int on);
uint32_t fx_crc32c_combine(uint32_t crc1, uint32_t crc2, uint64_t len2);

/* ---- small pure helpers */
static inline uint64_t fx_align_up(uint64_t n) { return (n + FX_ALIGN - 1) / FX_ALIGN * FX_ALIGN; }
uint32_t fx_pack_imm(unsigned slot, unsigned seq);                    /* caller keeps slot<4096, seq<2^20 */
void fx_unpack_imm(uint32_t imm, unsigned *slot, unsigned *seq);
size_t fx_choose_chunk(const uint64_t *sizes, int n, size_t chunk);   /* doubles until the plan fits MAX_CHUNKS */
int fx_fit_chunk(size_t chunk, uint64_t budget, size_t *out);        /* shrink so the ring holds MIN_SLOTS; -1 if hopeless */
int64_t fx_parse_meminfo(const char *text);                           /* MemAvailable in bytes, -1 if absent */
uint64_t fx_memory_policy(uint64_t avail, int engine_running, double ring_gib, char *why, size_t whylen);
int fx_detect_engines(const char *const *patterns, int npat, char *names, size_t len); /* count, -1 if docker unusable */
int fx_expand_dest(const char *dest, const char *home, char *out, size_t outlen);      /* 0 ok, -1 bad */
int fx_check_ipv4(const char *s);
#define FX_PATH_MAX 3000
int fx_path_ok(const char *p, size_t len);                   /* 1 if a safe relative path: no NUL/control chars, no empty, '.' or '..' segment, not absolute, not long */
int fx_ct_equal(const void *a, size_t alen, const void *b, size_t blen);
void fx_sha256(uint8_t out[32], const void *data, size_t len);                                  /* self-contained: no libssl on a stock Spark */
void fx_hmac_sha256(uint8_t out[32], const void *key, size_t klen, const void *msg, size_t mlen);   /* RFC 2104 / RFC 4231 */
int fx_random(void *buf, size_t n);                                                             /* getrandom(2), else /dev/urandom; 0 ok */
int fx_ip_allowed(const char *csv, const char *ip);                                             /* ip is exactly one entry of the comma list */
int fx_accept_allowed(int lfd, const char *allow_csv, int timeout_ms, char *peer, size_t peerlen);   /* accept; close strangers and keep listening; -1 on timeout */   /* constant time in the contents; 1 if equal */

/* ---- chunk table: the numbering both ends derive from the same file list */
typedef struct {
    size_t chunk; int nfiles; uint32_t nchunks; uint64_t total;
    uint64_t *size; uint32_t *first, *count, *cfile;
} fx_table;
int fx_table_init(fx_table *t, const uint64_t *sizes, int nfiles, size_t chunk);   /* -1 if > FX_MAX_CHUNKS */
void fx_table_free(fx_table *t);
void fx_table_chunk(const fx_table *t, uint32_t seq, int *fi, uint64_t *off, size_t *n);
void fx_table_order(const fx_table *t, uint32_t *out);   /* the read order: chunk k of every file in turn (nchunks entries) */
uint32_t fx_file_crc(const uint32_t *chunk_crcs, const fx_table *t, int fi);        /* combined whole-file CRC-32C */

/* ---- slot pool: a slot is out between acquire and release; releasing one that is not out is a bug (-1) */
typedef struct { int n, top; int *stack; uint8_t *out; } fx_slots;
int fx_slots_init(fx_slots *s, int n);
void fx_slots_free(fx_slots *s);
int fx_slots_acquire(fx_slots *s);            /* -1 if none free */
int fx_slots_release(fx_slots *s, int i);
static inline int fx_slots_free_count(const fx_slots *s) { return s->top; }

/* ---- rails: the RoCE v2 GID that holds an IPv4 address, from sysfs */
typedef struct { char ibdev[64]; int port; int gid_index; char netdev[64]; char ip[32]; } fx_rail;
int fx_find_gid(const char *ip, const char *sysfs_root, fx_rail *out);                /* 0 ok, -1 not found */

/* ---- control channel: length-prefixed binary messages over TCP */
typedef struct { uint8_t *p; size_t len, cap; } fx_buf;
void fx_buf_free(fx_buf *b);
void fx_put_u8(fx_buf *b, uint8_t v);
void fx_put_u32(fx_buf *b, uint32_t v);
void fx_put_u64(fx_buf *b, uint64_t v);
void fx_put_bytes(fx_buf *b, const void *p, size_t n);   /* u32 length + bytes */
void fx_put_str(fx_buf *b, const char *s);
typedef struct { const uint8_t *p; size_t len, pos; int bad; } fx_rd;
uint8_t fx_get_u8(fx_rd *r);
uint32_t fx_get_u32(fx_rd *r);
uint64_t fx_get_u64(fx_rd *r);
const uint8_t *fx_get_bytes(fx_rd *r, uint32_t *n);       /* points into the message */
char *fx_get_str(fx_rd *r);                               /* malloc'd, NUL-terminated */

enum { FX_M_HELLO = 1, FX_M_LIST, FX_M_INV, FX_M_PLAN, FX_M_RING, FX_M_NOTHING, FX_M_QP, FX_M_READY, FX_M_FILE,
       FX_M_VERDICT, FX_M_DONE, FX_M_BYE, FX_M_ERROR, FX_M_LOG, FX_M_CHALLENGE };
typedef struct { int fd; uint8_t *buf; size_t len, cap; } fx_ctrl;
typedef struct { int type; uint8_t *data; size_t len; } fx_msg;
void fx_ctrl_init(fx_ctrl *c, int fd);
void fx_ctrl_free(fx_ctrl *c);
int fx_ctrl_send(fx_ctrl *c, int type, const fx_buf *payload);   /* 0 ok, -1 error */
int fx_ctrl_recv(fx_ctrl *c, int timeout_ms, fx_msg *m);          /* 0 ok, -1 closed/error, -2 timeout */
int fx_ctrl_poll(fx_ctrl *c, fx_msg *m);                          /* 1 message, 0 none, -1 closed/error */
void fx_msg_free(fx_msg *m);

/* ---- queue and thread pool */
typedef struct fx_node { struct fx_node *next; } fx_node;
typedef struct { pthread_mutex_t mu; fx_node *head, *tail; } fx_queue;
void fx_queue_init(fx_queue *q);
void fx_queue_push(fx_queue *q, fx_node *n);
fx_node *fx_queue_pop(fx_queue *q);                               /* NULL if empty */
typedef struct fx_tp fx_tp;
fx_tp *fx_tp_new(int nthreads);
void fx_tp_submit(fx_tp *tp, void (*fn)(void *), void *arg);
void fx_tp_free(fx_tp *tp);                                       /* runs what is queued, joins */

/* ---- disk */
extern int (*fx_write_hook)(int fd, const void *buf, size_t len, uint64_t off);  /* tests wrap the only data write */
int fx_open(const char *path, int flags, int mode, int direct, int *got_direct, char *note, size_t notelen);
extern int fx_openat2_disabled;                                 /* tests force the per-component fallback */
/* open REL under DIRFD without ever leaving it: openat2(RESOLVE_BENEATH|RESOLVE_NO_SYMLINKS) where the kernel has it, else a
 * per-component openat(O_NOFOLLOW) walk. make_dirs creates missing parents (mkdirat). -1 with errno on refusal. */
int fx_open_beneath(int dirfd, const char *rel, int flags, int mode, int direct, int *got_direct, int make_dirs);
int fx_open_parent(int dirfd, const char *rel, int make_dirs, char *leaf, size_t leaflen);   /* fd of REL's directory (walked with O_NOFOLLOW), leaf name out */
int fx_pread_chunk(int fd, void *buf, uint64_t off, size_t n, int direct, char *err, size_t errlen);
int fx_pwrite_real(int fd, const void *buf, size_t len, uint64_t off);
int fx_pwrite_all(int fd, const void *buf, size_t len, uint64_t off);
int fx_file_chunk_crcs_at(int dirfd, int secure, const char *path, uint64_t size, size_t chunk, int direct, fx_tp *pool, uint32_t *out,
                          char *err, size_t errlen);   /* secure: PATH is relative to DIRFD and opened with fx_open_beneath */
int fx_file_chunk_crcs(const char *path, uint64_t size, size_t chunk, int direct, fx_tp *pool, uint32_t *out,
                       char *err, size_t errlen);

/* ---- transport: what the engines need of the wire */
enum { FX_E_WDONE = 1, FX_E_IMM, FX_E_CREDIT, FX_E_CDONE, FX_E_ERR };
typedef struct { int kind, peer, rail; uint64_t a, b; } fx_event;   /* wdone: a=token b=nbytes; imm: a=imm b=nbytes; credit: a=slot */
#define FX_INFO_MAX 512
typedef struct fx_transport fx_transport;
struct fx_transport {
    void *ctx; int rails; int max_slots; char err[FX_ERRLEN];
    int (*alloc_ring)(fx_transport *, size_t nbytes, uint8_t **base);
    int (*create_peer)(fx_transport *, int peer, int depth, uint8_t *info, size_t *infolen);
    int (*connect_peer)(fx_transport *, int peer, const uint8_t *info, size_t infolen);
    int (*post_write)(fx_transport *, int peer, int rail, size_t local_off, size_t nbytes, size_t remote_off,
                      uint32_t imm, uint64_t token);
    int (*post_credit)(fx_transport *, int peer, int rail, uint32_t slot);
    int (*poll)(fx_transport *, fx_event *ev, int max);                /* count */
    void (*destroy)(fx_transport *);
};

/* loopback wire for tests (fx_loop.c) */
typedef struct fx_loopnet fx_loopnet;
fx_loopnet *fx_loopnet_new(int rails);
fx_transport *fx_loopnet_transport(fx_loopnet *n);
void fx_loopnet_free(fx_loopnet *n);
uint64_t fx_loopnet_rail_bytes(fx_loopnet *n, int rail);
int fx_loopnet_violations(fx_loopnet *n);
int fx_loopnet_max_busy(fx_loopnet *n);
void fx_loopnet_set_corrupt(fx_loopnet *n, void (*fn)(void *arg, unsigned seq, uint8_t *data, size_t len), void *arg);

/* ibverbs wire (fx_verbs.c) */
fx_transport *fx_verbs_new(const fx_rail *rails, int nrails, int max_peers, char *err, size_t errlen);
int fx_verbs_probe(const fx_rail *rail, char *out, size_t outlen);   /* for selftest: device/port numbers */

/* ---- engines */
typedef struct {
    size_t chunk; uint64_t ring_bytes; int io_threads, direct, readback, verify_existing, dry;
    const char *manifest; double progress_s;
    const char *token;             /* one-time shared secret; the sender presents it in HELLO, the receiver compares it; required on both */
    const uint64_t *path_masks;    /* sender: per path, the receivers it goes to (bit i = peer i); NULL = every receiver */
    void (*log)(void *ctx, const char *msg); void *logctx;
} fx_cfg;
typedef struct { char name[64]; fx_ctrl *ctrl; } fx_peer;
typedef struct { char name[64]; int files_ok, files_failed; uint64_t bytes, written; } fx_peer_res;   /* written: on the receiver's disk (credited) */
typedef struct {
    int ok, npeers; fx_peer_res peers[FX_MAX_PEERS];
    uint64_t total_bytes, would_send, chunks_read; double seconds;
    double t_plan, t_rings, t_alloc, t_ready, t_first, t_last, t_done;   /* seconds since the sender started: plan sent, ring registered,
                                                                   receivers ready, first write posted, last write done, all verdicts in */
    int nfail; char fail[16][FX_ERRLEN]; char err[FX_ERRLEN];
} fx_send_res;
typedef struct { int files_ok, files_failed; uint64_t bytes; int nfail; char fail[16][FX_ERRLEN]; char err[FX_ERRLEN]; } fx_recv_res;
typedef fx_transport *(*fx_transport_factory)(void *ctx);

/* 0 ok (res->ok says whether every file verified), -1 aborted (res->err) */
int fx_run_sender(const char *src, char **paths, int npaths, fx_peer *peers, int npeers, fx_transport_factory mk,
                  void *mkctx, const fx_cfg *cfg, fx_send_res *res);
int fx_run_receiver(fx_ctrl *ctrl, fx_transport_factory mk, void *mkctx, const char *dest, const fx_cfg *cfg,
                    uint64_t ring_bytes, fx_recv_res *res);

#endif
