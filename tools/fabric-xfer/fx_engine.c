/* fabric-xfer engines: the sender (reads each chunk once, fans it out to every receiver) and the receiver (disk writes,
 * checksums, credits). Both are single-threaded state machines; only disk I/O and CRC run on the thread pool. */
#include "fx.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define CTRL_TIMEOUT_MS 600000
#define ABSENT UINT64_MAX

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static void nap(void) { struct timespec ts = { 0, 100000 }; nanosleep(&ts, NULL); }

static void logf_(const fx_cfg *c, const char *fmt, ...)
{
    if (!c->log) return;
    char b[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    c->log(c->logctx, b);
}

static void add_fail(int *nfail, char (*fail)[FX_ERRLEN], const char *fmt, ...)
{
    if (*nfail >= 16) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(fail[(*nfail)++], FX_ERRLEN, fmt, ap);
    va_end(ap);
}

static int set_err(char *dst, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(dst, FX_ERRLEN, fmt, ap);
    va_end(ap);
    return -1;
}

static void send_error(fx_ctrl *c, const char *msg)
{
    fx_buf b = { 0 };
    fx_put_str(&b, msg);
    fx_ctrl_send(c, FX_M_ERROR, &b);
    fx_buf_free(&b);
}

/* ================================================================================================== sender */

typedef struct {
    int idx, active, done, slots, rr;
    fx_peer *peer; fx_ctrl *ctrl; fx_peer_res *res;
    uint8_t *need; uint64_t need_bytes;
    uint64_t *there; uint32_t **there_crc; uint32_t *there_n;
    fx_slots rfree;
    int *pend_ls; uint32_t *pend_seq; size_t ph, pn, pcap;
    size_t *slot_len;                /* the chunk length last written into each of the receiver's slots */
} sp;

typedef struct S S;
struct S {
    const char *src; char **paths; int np; const fx_cfg *cfg; fx_send_res *res;
    sp P[FX_MAX_PEERS]; int nP;
    uint64_t *sizes, *mtime; uint32_t *modes;
    size_t chunk;
    int *gp; int ngp; uint64_t *gmask; fx_table table;
    fx_transport *tr; uint8_t *ring; int Sn;
    uint32_t *crcs; uint8_t *crc_set;
    fx_tp *pool; fx_queue ready;
    uint32_t **mycrc;
    double t0;
};

typedef struct { fx_node n; S *s; uint32_t seq; int ls, bad; uint32_t crc; char err[FX_ERRLEN]; } rjob;

static void read_run(void *a)
{
    rjob *j = a;
    S *s = j->s;
    int fi; uint64_t off; size_t n;
    fx_table_chunk(&s->table, j->seq, &fi, &off, &n);
    char path[4352];
    snprintf(path, sizeof path, "%s/%s", s->src, s->paths[s->gp[fi]]);
    uint8_t *buf = s->ring + (size_t)j->ls * s->chunk;
    int d, fd = fx_open(path, O_RDONLY, 0, s->cfg->direct, &d, NULL, 0);
    if (fd < 0) { j->bad = 1; snprintf(j->err, sizeof j->err, "%.300s: %s", path, strerror(errno)); }
    else {
        if (fx_pread_chunk(fd, buf, off, n, d, j->err, sizeof j->err)) { j->bad = 1; }
        else j->crc = fx_crc32c(0, buf, n);
        close(fd);
    }
    fx_queue_push(&s->ready, &j->n);
}

static int recv_each(S *s, const int *kinds, int nk, fx_msg *m, int only_active)
{
    for (int i = 0; i < s->nP; i++) {
        sp *p = &s->P[i];
        m[i].data = NULL;
        if (only_active && !p->active) continue;
        int r = fx_ctrl_recv(p->ctrl, CTRL_TIMEOUT_MS, &m[i]);
        if (r) return set_err(s->res->err, "%s: control connection %s", p->peer->name, r == -2 ? "timed out" : "lost");
        if (m[i].type == FX_M_ERROR) {
            fx_rd rd = { m[i].data, m[i].len, 0, 0 };
            char *t = fx_get_str(&rd);
            set_err(s->res->err, "%s: %s", p->peer->name, t);
            free(t);
            return -1;
        }
        int okk = 0;
        for (int k = 0; k < nk; k++) okk |= m[i].type == kinds[k];
        if (!okk) return set_err(s->res->err, "%s: unexpected message type %d", p->peer->name, m[i].type);
    }
    return 0;
}
static void free_msgs(fx_msg *m, int n) { for (int i = 0; i < n; i++) if (m[i].data) fx_msg_free(&m[i]); }

static void send_all(S *s, int type, const fx_buf *b)
{
    for (int i = 0; i < s->nP; i++) fx_ctrl_send(s->P[i].ctrl, type, b);
}

static void json_str(FILE *f, const char *s)
{
    fputc('"', f);
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') { fputc('\\', f); fputc(*s, f); }
        else if ((unsigned char)*s < 0x20) fprintf(f, "\\u%04x", *s);
        else fputc(*s, f);
    }
    fputc('"', f);
}

static void write_manifest(S *s)
{
    FILE *f = fopen(s->cfg->manifest, "w");
    if (!f) { logf_(s->cfg, "cannot write manifest %s: %s", s->cfg->manifest, strerror(errno)); return; }
    int *gi = malloc(sizeof(int) * (s->np ? s->np : 1));
    for (int i = 0; i < s->np; i++) gi[i] = -1;
    for (int k = 0; k < s->ngp; k++) gi[s->gp[k]] = k;
    fprintf(f, "{\"version\":%d,\"chunk\":%zu,\"checksum\":\"CRC-32C per chunk, compared; file value combined\",\"files\":[\n", FX_PROTO, s->chunk);
    for (int i = 0; i < s->np; i++) {
        int have = 0; uint32_t crc = 0;
        if (gi[i] >= 0 && s->crcs) {
            uint32_t first = s->table.first[gi[i]], cnt = s->table.count[gi[i]];
            have = 1;
            for (uint32_t k = 0; k < cnt; k++) have &= s->crc_set[first + k];
            if (have) crc = fx_file_crc(s->crcs + first, &s->table, gi[i]);
        } else if (s->mycrc && s->mycrc[i]) {
            fx_table t1;
            if (fx_table_init(&t1, &s->sizes[i], 1, s->chunk) == 0) { crc = fx_file_crc(s->mycrc[i], &t1, 0); have = 1; fx_table_free(&t1); }
        }
        fprintf(f, "%s{\"path\":", i ? ",\n" : "");
        json_str(f, s->paths[i]);
        fprintf(f, ",\"size\":%llu,\"mode\":%u,\"mtime_ns\":%llu,\"crc32c\":", (unsigned long long)s->sizes[i], s->modes[i], (unsigned long long)s->mtime[i]);
        if (have) fprintf(f, "\"%08x\"", crc); else fputs("null", f);
        fputs(",\"sent_to\":[", f);
        int first = 1;
        for (int p = 0; p < s->nP; p++)
            if (s->P[p].need[i]) { fprintf(f, "%s", first ? "" : ","); json_str(f, s->P[p].peer->name); first = 0; }
        fputs("]}", f);
    }
    fputs("\n]}\n", f);
    fclose(f);
    free(gi);
}

static int on_msg(S *s, sp *p, fx_msg *m)
{
    fx_rd rd = { m->data, m->len, 0, 0 };
    if (m->type == FX_M_FILE) {
        uint32_t fi = fx_get_u32(&rd);
        char *err = fx_get_str(&rd);
        uint32_t n = fx_get_u32(&rd);
        if (rd.bad || fi >= (uint32_t)s->ngp || n > (rd.len - rd.pos) / 4) { free(err); return set_err(s->res->err, "%s: malformed file report", p->peer->name); }
        uint32_t first = s->table.first[fi], cnt = s->table.count[fi];
        char why[FX_ERRLEN] = "";
        if (err[0]) snprintf(why, sizeof why, "%s", err);
        else if (n != cnt) snprintf(why, sizeof why, "%u chunk checksums reported, %u sent", n, cnt);
        else {
            int bad = 0, firstbad = -1;
            for (uint32_t k = 0; k < cnt; k++) {
                uint32_t got = fx_get_u32(&rd);
                if (got != s->crcs[first + k]) { bad++; if (firstbad < 0) firstbad = (int)k; }
            }
            if (bad) snprintf(why, sizeof why, "checksum mismatch in chunk %d (offset %llu) and %d more", firstbad, (unsigned long long)firstbad * s->chunk, bad - 1);
        }
        free(err);
        int ok = why[0] == 0;
        fx_buf b = { 0 };
        fx_put_u32(&b, fi); fx_put_u8(&b, (uint8_t)ok); fx_put_str(&b, why);
        fx_ctrl_send(p->ctrl, FX_M_VERDICT, &b);
        fx_buf_free(&b);
        const char *path = s->paths[s->gp[fi]];
        if (ok) p->res->files_ok++;
        else {
            p->res->files_failed++;
            add_fail(&s->res->nfail, s->res->fail, "%s on %s: %s", path, p->peer->name, why);
            logf_(s->cfg, "ERROR: %s on %s: %s; not renamed into place", path, p->peer->name, why);
        }
    } else if (m->type == FX_M_DONE) p->done = 1;
    else if (m->type == FX_M_ERROR) {
        char *t = fx_get_str(&rd);
        set_err(s->res->err, "%s: %s", p->peer->name, t);
        free(t);
        return -1;
    } else if (m->type == FX_M_LOG) {
        char *t = fx_get_str(&rd);
        logf_(s->cfg, "[%s] %s", p->peer->name, t);
        free(t);
    }
    return 0;
}

static int data_loop(S *s)
{
    const fx_cfg *cfg = s->cfg;
    fx_transport *tr = s->tr;
    uint32_t nch = s->table.nchunks, next_seq = 0;
    int inflight = 0, max_inflight = cfg->io_threads * 2 > 2 ? cfg->io_threads * 2 : 2;
    fx_slots local;
    fx_slots_init(&local, s->Sn);
    int *refs = calloc(s->Sn, sizeof(int));
    uint32_t *order = malloc(sizeof(uint32_t) * (nch ? nch : 1));
    fx_table_order(&s->table, order);
    s->crcs = calloc(nch ? nch : 1, 4);
    s->crc_set = calloc(nch ? nch : 1, 1);
    uint64_t rail_bytes[8] = { 0 }, last_r[8] = { 0 }, last_w[FX_MAX_PEERS] = { 0 }, total = 0, last_wall = 0;
    double rate_avg = 0;
    for (int i = 0; i < s->nP; i++) if (s->P[i].active) total += s->P[i].need_bytes;
    double last_t = now_s();
    int rc = 0;
    fx_event ev[64];
    while (rc == 0) {
        int busy = 0;
        while (next_seq < nch && inflight < max_inflight && fx_slots_free_count(&local)) {
            rjob *j = calloc(1, sizeof *j);
            j->s = s; j->seq = order[next_seq++]; j->ls = fx_slots_acquire(&local);
            inflight++; busy = 1;
            fx_tp_submit(s->pool, read_run, j);
        }
        fx_node *nd;
        while ((nd = fx_queue_pop(&s->ready))) {
            rjob *j = (rjob *)nd;
            inflight--; busy = 1;
            if (j->bad) { rc = set_err(s->res->err, "%s", j->err); free(j); break; }
            s->crcs[j->seq] = j->crc; s->crc_set[j->seq] = 1; s->res->chunks_read++;
            int fi; uint64_t off; size_t n;
            fx_table_chunk(&s->table, j->seq, &fi, &off, &n);
            int targets = 0;
            for (int i = 0; i < s->nP; i++) {
                sp *p = &s->P[i];
                if (p->active && (s->gmask[fi] >> i & 1)) {
                    size_t at = (p->ph + p->pn) % p->pcap;
                    p->pend_ls[at] = j->ls; p->pend_seq[at] = j->seq; p->pn++;
                    targets++;
                }
            }
            refs[j->ls] = targets;
            if (!targets) fx_slots_release(&local, j->ls);
            free(j);
        }
        if (rc) break;
        for (int i = 0; i < s->nP && !rc; i++) {
            sp *p = &s->P[i];
            while (p->active && p->pn && fx_slots_free_count(&p->rfree)) {
                int ls = p->pend_ls[p->ph]; uint32_t seq = p->pend_seq[p->ph];
                p->ph = (p->ph + 1) % p->pcap; p->pn--;
                int rs = fx_slots_acquire(&p->rfree), rail = p->rr++ % tr->rails;
                int fi; uint64_t off; size_t n;
                fx_table_chunk(&s->table, seq, &fi, &off, &n);
                p->slot_len[rs] = n;
                if (tr->post_write(tr, i, rail, (size_t)ls * s->chunk, n, (size_t)rs * s->chunk, fx_pack_imm((unsigned)rs, seq), (uint64_t)ls * 64 + (uint64_t)i)) {
                    rc = set_err(s->res->err, "posting a write to %s failed: %s", p->peer->name, tr->err);
                    break;
                }
                busy = 1;
                if (s->res->t_first <= 0) s->res->t_first = now_s() - s->t0;
            }
        }
        if (rc) break;
        int ne;
        while ((ne = tr->poll(tr, ev, 64)) > 0) {
            busy = 1;
            for (int k = 0; k < ne && !rc; k++) {
                fx_event *e = &ev[k];
                if (e->kind == FX_E_WDONE) {
                    int ls = (int)(e->a / 64), pi = (int)(e->a % 64);
                    s->P[pi].res->bytes += e->b; rail_bytes[e->rail] += e->b;
                    s->res->t_last = now_s() - s->t0;
                    if (--refs[ls] == 0) fx_slots_release(&local, ls);
                } else if (e->kind == FX_E_CREDIT) {
                    sp *cp = &s->P[e->peer];
                    if (e->a < (uint64_t)cp->slots) cp->res->written += cp->slot_len[e->a];   /* a credit comes back once the slot is on disk */
                    if (fx_slots_release(&cp->rfree, (int)e->a)) rc = set_err(s->res->err, "%s returned slot %llu that was not out", s->P[e->peer].peer->name, (unsigned long long)e->a);
                } else if (e->kind == FX_E_ERR) {
                    rc = set_err(s->res->err, "queue pair error (%s) on rail %d: %s", e->peer >= 0 && e->peer < s->nP ? s->P[e->peer].peer->name : "?", e->rail, tr->err);
                }
            }
        }
        if (rc) break;
        int alldone = 1;
        for (int i = 0; i < s->nP && !rc; i++) {
            sp *p = &s->P[i];
            if (!p->active || p->done) continue;
            alldone = 0;
            fx_msg m;
            int r;
            while ((r = fx_ctrl_poll(p->ctrl, &m)) > 0) {
                busy = 1;
                int e = on_msg(s, p, &m);
                fx_msg_free(&m);
                if (e) { rc = -1; break; }
            }
            if (r < 0 && !rc) rc = set_err(s->res->err, "%s: control connection lost", p->peer->name);
        }
        if (rc) break;
        if (alldone) break;
        double t = now_s();
        if (cfg->progress_s > 0 && t - last_t >= cfg->progress_s) {
            /* what the receivers have WRITTEN (their credits: a slot comes back once it is on disk), not what was sent: the sender
               finishes long before the disks do, and a line stuck at 100 % while they drain says nothing */
            double dt = t - last_t; uint64_t wr = 0, sent = 0;
            char per[512] = "", rails[128] = "", tmp[128], eta[32] = "?";
            for (int i = 0; i < s->nP; i++) {
                sp *p = &s->P[i];
                if (!p->active) continue;
                wr += p->res->written; sent += p->res->bytes;
                snprintf(tmp, sizeof tmp, "%s%s %llu %% %.2f", per[0] ? ", " : "", p->peer->name,
                         (unsigned long long)(p->need_bytes ? p->res->written * 100 / p->need_bytes : 100), (double)(p->res->written - last_w[i]) / dt / 1e9);
                { size_t u = strlen(per); snprintf(per + u, sizeof per - u, "%s", tmp); }
                last_w[i] = p->res->written;
            }
            for (int r = 0; r < tr->rails; r++) {
                snprintf(tmp, sizeof tmp, "%s%c %.2f", r ? " " : "", 'A' + r, (double)(rail_bytes[r] - last_r[r]) / dt / 1e9);
                { size_t u = strlen(rails); snprintf(rails + u, sizeof rails - u, "%s", tmp); }
                last_r[r] = rail_bytes[r];
            }
            double rate = (double)(wr - last_wall) / dt;
            rate_avg = rate_avg > 0 ? 0.8 * rate_avg + 0.2 * rate : rate;
            last_wall = wr;
            if (rate_avg > 1e6 && total > wr) {
                unsigned long sec = (unsigned long)((double)(total - wr) / rate_avg);
                if (sec >= 3600) snprintf(eta, sizeof eta, "%lu:%02lu:%02lu", sec / 3600, sec / 60 % 60, sec % 60);
                else snprintf(eta, sizeof eta, "%lu:%02lu", sec / 60, sec % 60);
            } else if (total <= wr) snprintf(eta, sizeof eta, "0:00");
            logf_(cfg, "written %.1f of %.1f GB (%llu %%), %.2f GB/s, ETA %s | each receiver (%% written, GB/s): %s | sent %llu %%, GB/s a rail: %s",
                  (double)wr / 1e9, (double)total / 1e9, (unsigned long long)(total ? wr * 100 / total : 100), rate / 1e9, eta, per,
                  (unsigned long long)(total ? sent * 100 / total : 100), rails);
            last_t = t;
        }
        if (!busy) nap();
    }
    fx_slots_free(&local);
    free(refs);
    free(order);
    return rc;
}

int fx_run_sender(const char *src, char **paths, int npaths, fx_peer *peers, int npeers, fx_transport_factory mk, void *mkctx,
                  const fx_cfg *cfg, fx_send_res *res)
{
    S s;
    memset(&s, 0, sizeof s);
    memset(res, 0, sizeof *res);
    double t0 = now_s();
    int rc = -1;
    fx_msg msgs[FX_MAX_PEERS];
    memset(msgs, 0, sizeof msgs);
    s.t0 = t0;
    s.src = src; s.paths = paths; s.np = npaths; s.cfg = cfg; s.res = res; s.nP = npeers;
    fx_queue_init(&s.ready);
    if (npeers < 1 || npeers > FX_MAX_PEERS) return set_err(res->err, "1 to %d receivers", FX_MAX_PEERS);
    res->npeers = npeers;
    int np = npaths;
    s.sizes = calloc(np ? np : 1, 8); s.mtime = calloc(np ? np : 1, 8); s.modes = calloc(np ? np : 1, 4);
    for (int i = 0; i < npeers; i++) {
        sp *p = &s.P[i];
        p->idx = i; p->peer = &peers[i]; p->ctrl = peers[i].ctrl; p->res = &res->peers[i];
        snprintf(p->res->name, sizeof p->res->name, "%s", peers[i].name);
        p->need = calloc(np ? np : 1, 1);
        p->there = calloc(np ? np : 1, 8);
        p->there_crc = calloc(np ? np : 1, sizeof(uint32_t *));
        p->there_n = calloc(np ? np : 1, 4);
    }
    if (!cfg->token || !cfg->token[0]) { set_err(res->err, "no authentication token"); goto out; }
    for (int i = 0; i < np; i++) {
        char full[4352];
        struct stat st;
        if (!fx_path_ok(paths[i], strlen(paths[i]))) { set_err(res->err, "unsafe path (absolute, '.' or '..' or empty segment, control character, or too long): %.200s", paths[i]); goto out; }
        snprintf(full, sizeof full, "%s/%s", src, paths[i]);
        if (stat(full, &st) || !S_ISREG(st.st_mode)) { set_err(res->err, "not a file here (under %s): %s", src, paths[i]); goto out; }
        s.sizes[i] = (uint64_t)st.st_size; s.modes[i] = st.st_mode & 07777;
        s.mtime[i] = (uint64_t)st.st_mtim.tv_sec * 1000000000ull + (uint64_t)st.st_mtim.tv_nsec;
    }
    if (cfg->path_masks) {
        uint64_t all = (1ull << npeers) - 1;   /* npeers <= FX_MAX_PEERS = 63 */
        for (int i = 0; i < np; i++)
            if (!cfg->path_masks[i] || (cfg->path_masks[i] & ~all)) {
                set_err(res->err, "%.300s is routed to a receiver this run does not have (it has %d)", paths[i], npeers); goto out; }
    }
    s.pool = fx_tp_new(cfg->io_threads > 0 ? cfg->io_threads : 1);

    /* 1. hello: protocol and each receiver's memory budget */
    {
        /* challenge-response: the receiver's random nonce comes first; HELLO carries HMAC-SHA256(token, nonce), never the token */
        const int kc[] = { FX_M_CHALLENGE };
        if (recv_each(&s, kc, 1, msgs, 0)) goto out;
        for (int i = 0; i < npeers; i++) {
            fx_rd cr = { msgs[i].data, msgs[i].len, 0, 0 };
            uint32_t nl;
            const uint8_t *nonce = fx_get_bytes(&cr, &nl);
            if (cr.bad || nl < 16) { set_err(res->err, "%s: bad challenge", peers[i].name); goto out; }
            uint8_t mac[32];
            fx_hmac_sha256(mac, cfg->token, strlen(cfg->token), nonce, nl);
            fx_buf hb = { 0 };
            fx_put_u32(&hb, FX_PROTO);
            fx_put_bytes(&hb, mac, sizeof mac);
            fx_ctrl_send(peers[i].ctrl, FX_M_HELLO, &hb);
            fx_buf_free(&hb);
        }
        free_msgs(msgs, npeers);
        const int k[] = { FX_M_HELLO };
        if (recv_each(&s, k, 1, msgs, 0)) goto out;
        uint64_t budget = cfg->ring_bytes ? cfg->ring_bytes : UINT64_MAX;
        for (int i = 0; i < npeers; i++) {
            fx_rd rd = { msgs[i].data, msgs[i].len, 0, 0 };
            uint32_t v = fx_get_u32(&rd); uint64_t bg = fx_get_u64(&rd);
            if (rd.bad || v != FX_PROTO) { set_err(res->err, "%s: protocol %u, this side %d", peers[i].name, v, FX_PROTO); goto out; }
            if (bg < budget) budget = bg;
        }
        free_msgs(msgs, npeers);
        size_t chunk;
        if (fx_fit_chunk(cfg->chunk, budget, &chunk)) { set_err(res->err, "a ring of %llu bytes cannot hold %u slots", (unsigned long long)budget, FX_MIN_SLOTS); goto out; }
        s.chunk = fx_choose_chunk(s.sizes, np, chunk);
    }

    /* 2. inventory: what each worker holds */
    {
        fx_buf b = { 0 };
        fx_put_u64(&b, s.chunk); fx_put_u8(&b, (uint8_t)cfg->verify_existing); fx_put_u32(&b, (uint32_t)np);
        for (int i = 0; i < np; i++) { fx_put_str(&b, paths[i]); fx_put_u64(&b, s.sizes[i]); }
        send_all(&s, FX_M_LIST, &b);
        fx_buf_free(&b);
        const int k[] = { FX_M_INV };
        if (recv_each(&s, k, 1, msgs, 0)) goto out;
        for (int pi = 0; pi < npeers; pi++) {
            fx_rd rd = { msgs[pi].data, msgs[pi].len, 0, 0 };
            for (int i = 0; i < np; i++) {
                s.P[pi].there[i] = fx_get_u64(&rd);
                if (cfg->verify_existing && fx_get_u8(&rd)) {
                    uint32_t n = fx_get_u32(&rd);
                    if (rd.bad || n > (rd.len - rd.pos) / 4) { set_err(res->err, "%s: malformed inventory", peers[pi].name); goto out; }
                    s.P[pi].there_crc[i] = malloc(4 * (n ? n : 1)); s.P[pi].there_n[i] = n;
                    for (uint32_t q = 0; q < n; q++) s.P[pi].there_crc[i][q] = fx_get_u32(&rd);
                }
            }
            if (rd.bad) { set_err(res->err, "%s: malformed inventory", peers[pi].name); goto out; }
        }
        free_msgs(msgs, npeers);
    }
    if (cfg->verify_existing) {
        s.mycrc = calloc(np ? np : 1, sizeof(uint32_t *));
        for (int i = 0; i < np; i++) {
            int any = 0;
            for (int p = 0; p < npeers; p++) any |= s.P[p].there[i] == s.sizes[i];
            if (!any) continue;
            uint32_t n = (uint32_t)((s.sizes[i] + s.chunk - 1) / s.chunk);
            s.mycrc[i] = calloc(n ? n : 1, 4);
            char full[4352], err[FX_ERRLEN];
            snprintf(full, sizeof full, "%s/%s", src, paths[i]);
            if (fx_file_chunk_crcs(full, s.sizes[i], s.chunk, cfg->direct, s.pool, s.mycrc[i], err, sizeof err)) { set_err(res->err, "%s", err); goto out; }
        }
    }

    /* 3. who needs what; the global file table */
    s.gp = calloc(np ? np : 1, sizeof(int));
    for (int i = 0; i < np; i++) {
        int any = 0;
        for (int p = 0; p < npeers; p++) {
            sp *q = &s.P[p];
            int routed = !cfg->path_masks || (cfg->path_masks[i] >> p & 1);
            int need = routed && q->there[i] != s.sizes[i];
            if (!need && routed && cfg->verify_existing) {
                uint32_t n = (uint32_t)((s.sizes[i] + s.chunk - 1) / s.chunk);
                need = !q->there_crc[i] || q->there_n[i] != n || !s.mycrc[i] || memcmp(q->there_crc[i], s.mycrc[i], 4 * (size_t)n) != 0;
            }
            if (need) { q->need[i] = 1; q->need_bytes += s.sizes[i]; res->would_send += s.sizes[i]; any = 1; }
        }
        if (any) s.gp[s.ngp++] = i;
    }
    {
        uint64_t *gs = calloc(s.ngp ? s.ngp : 1, 8);
        s.gmask = calloc(s.ngp ? s.ngp : 1, 8);
        for (int k = 0; k < s.ngp; k++) {
            gs[k] = s.sizes[s.gp[k]];
            for (int p = 0; p < npeers; p++) if (s.P[p].need[s.gp[k]]) s.gmask[k] |= 1ull << p;
        }
        int e = fx_table_init(&s.table, gs, s.ngp, s.chunk);
        free(gs);
        if (e) { set_err(res->err, "the plan does not fit the chunk numbering"); goto out; }
    }
    for (int p = 0; p < npeers; p++) {
        fx_buf b = { 0 };
        fx_put_u64(&b, s.chunk); fx_put_u8(&b, (uint8_t)cfg->dry); fx_put_u32(&b, (uint32_t)s.ngp);
        for (int k = 0; k < s.ngp; k++) {
            int i = s.gp[k];
            fx_put_str(&b, paths[i]); fx_put_u64(&b, s.sizes[i]); fx_put_u32(&b, s.modes[i]); fx_put_u64(&b, s.mtime[i]);
            fx_put_u8(&b, s.P[p].need[i]);
        }
        fx_ctrl_send(s.P[p].ctrl, FX_M_PLAN, &b);
        fx_buf_free(&b);
    }
    res->t_plan = now_s() - t0;
    if (cfg->dry) {
        const int k[] = { FX_M_DONE };
        if (recv_each(&s, k, 1, msgs, 0)) goto out;
        free_msgs(msgs, npeers);
        send_all(&s, FX_M_BYE, NULL);
        rc = 0;
        goto out;
    }
    {
        const int k[] = { FX_M_RING, FX_M_NOTHING };
        if (recv_each(&s, k, 2, msgs, 0)) goto out;
        res->t_rings = now_s() - t0;
    }
    int nact = 0;
    for (int p = 0; p < npeers; p++) if (msgs[p].type == FX_M_RING) { s.P[p].active = 1; nact++; }
    if (nact) {
        /* 4. wire: ring, queue pairs */
        s.tr = mk(mkctx);
        if (!s.tr) { set_err(res->err, "cannot open the RDMA devices"); goto out; }
        s.Sn = (int)(cfg->ring_bytes / s.chunk);
        if (s.Sn > (int)FX_MAX_SLOTS) s.Sn = FX_MAX_SLOTS;
        if ((uint32_t)s.Sn > s.table.nchunks) s.Sn = (int)s.table.nchunks;
        if (s.Sn < 1) s.Sn = 1;
        if (s.tr->alloc_ring(s.tr, (size_t)s.Sn * s.chunk, &s.ring)) { set_err(res->err, "cannot allocate or register the ring: %s", s.tr->err); goto out; }
        res->t_alloc = now_s() - t0;
        for (int p = 0; p < npeers; p++) {
            sp *q = &s.P[p];
            if (!q->active) continue;
            fx_rd rd = { msgs[p].data, msgs[p].len, 0, 0 };
            q->slots = (int)fx_get_u32(&rd);
            uint32_t il; const uint8_t *info = fx_get_bytes(&rd, &il);
            if (rd.bad || q->slots < 1 || q->slots > (int)FX_MAX_SLOTS) { set_err(res->err, "%s: bad ring report", peers[p].name); goto out; }
            fx_slots_init(&q->rfree, q->slots);
            q->slot_len = calloc((size_t)q->slots, sizeof(size_t));
            q->pcap = (size_t)s.Sn + 1;
            q->pend_ls = calloc(q->pcap, sizeof(int)); q->pend_seq = calloc(q->pcap, 4);
            uint8_t mine[FX_INFO_MAX]; size_t ml = sizeof mine;
            if (s.tr->create_peer(s.tr, p, q->slots, mine, &ml)) { set_err(res->err, "queue pair for %s: %s", peers[p].name, s.tr->err); goto out; }
            if (s.tr->connect_peer(s.tr, p, info, il)) { set_err(res->err, "connecting to %s: %s", peers[p].name, s.tr->err); goto out; }
            fx_buf b = { 0 };
            fx_put_bytes(&b, mine, ml);
            fx_ctrl_send(q->ctrl, FX_M_QP, &b);
            fx_buf_free(&b);
        }
        free_msgs(msgs, npeers);
        const int k[] = { FX_M_READY };
        if (recv_each(&s, k, 1, msgs, 1)) goto out;
        free_msgs(msgs, npeers);
        res->t_ready = now_s() - t0;
        if (data_loop(&s)) goto out;
        res->t_done = now_s() - t0;
    }
    send_all(&s, FX_M_BYE, NULL);
    if (cfg->manifest) write_manifest(&s);
    rc = 0;

out:
    free_msgs(msgs, npeers);
    if (rc) for (int i = 0; i < npeers; i++) send_error(peers[i].ctrl, res->err[0] ? res->err : "sender stopped");
    if (s.pool) fx_tp_free(s.pool);
    fx_node *nd;
    while ((nd = fx_queue_pop(&s.ready))) free(nd);
    for (int i = 0; i < npeers; i++) {
        sp *p = &s.P[i];
        free(p->need); free(p->there); free(p->there_n);
        if (p->there_crc) for (int k = 0; k < np; k++) free(p->there_crc[k]);
        free(p->there_crc); free(p->pend_ls); free(p->pend_seq); free(p->slot_len);
        if (p->rfree.stack) fx_slots_free(&p->rfree);
    }
    if (s.mycrc) for (int k = 0; k < np; k++) free(s.mycrc[k]);
    free(s.mycrc); free(s.sizes); free(s.mtime); free(s.modes); free(s.gp); free(s.gmask); free(s.crcs); free(s.crc_set);
    fx_table_free(&s.table);
    if (s.tr) s.tr->destroy(s.tr);
    res->seconds = now_s() - t0;
    for (int i = 0; i < npeers; i++) res->total_bytes += res->peers[i].bytes;
    res->ok = rc == 0 && res->nfail == 0;
    for (int i = 0; i < npeers; i++) if (res->peers[i].files_failed) res->ok = 0;
    return rc;
}

/* ================================================================================================== receiver */

typedef struct {
    int i, fd, direct, awaiting, committed;
    char *path, *partial;            /* both relative to the DEST directory fd; PARTIAL is PATH + ".partial" */
    uint64_t size, mtime_ns; uint32_t mode, count, remaining;
    uint32_t *crcs;
} rf;

typedef struct R R;
struct R {
    fx_ctrl *ctrl; fx_transport *tr; const fx_cfg *cfg; fx_recv_res *res;
    fx_table table; size_t chunk; uint8_t *ring; fx_tp *pool; fx_queue doneq; rf *files; int nfiles;
    int dirfd; unsigned D; uint8_t *slot_busy, *seen;
};
typedef struct { fx_node n; R *r; rf *f; unsigned slot, seq; uint64_t off; size_t len; int rail, fin, bad; uint32_t crc; char err[FX_ERRLEN]; } wjob;

static void write_run(void *a)
{
    wjob *j = a;
    R *r = j->r;
    uint8_t *view = r->ring + (size_t)j->slot * r->chunk;
    j->crc = fx_crc32c(0, view, j->len);
    size_t wn = j->f->direct ? (size_t)fx_align_up(j->len) : j->len;
    if (fx_pwrite_all(j->f->fd, view, wn, j->off)) { j->bad = 1; snprintf(j->err, sizeof j->err, "write %.300s: %s", j->f->partial, strerror(errno)); }
    fx_queue_push(&r->doneq, &j->n);
}

static void fin_run(void *a)
{
    wjob *j = a;
    R *r = j->r;
    rf *f = j->f;
    if (ftruncate(f->fd, (off_t)f->size) || fsync(f->fd)) { j->bad = 1; snprintf(j->err, sizeof j->err, "finishing %.300s: %s", f->partial, strerror(errno)); }
    close(f->fd); f->fd = -1;
    if (!j->bad && r->cfg->readback && f->count) {
        uint32_t *got = calloc(f->count, 4);
        char err[FX_ERRLEN];
        if (fx_file_chunk_crcs_at(r->dirfd, 1, f->partial, f->size, r->chunk, r->cfg->direct, NULL, got, err, sizeof err)) { j->bad = 1; snprintf(j->err, sizeof j->err, "read-back: %.400s", err); }
        else for (uint32_t k = 0; k < f->count; k++)
            if (got[k] != f->crcs[k]) { j->bad = 1; snprintf(j->err, sizeof j->err, "read-back from disk differs from what arrived, first at chunk %u", k); break; }
        free(got);
    }
    fx_queue_push(&r->doneq, &j->n);
}

static int mkdir_p(const char *path)
{
    char tmp[4352];
    snprintf(tmp, sizeof tmp, "%s", path);
    for (char *p = tmp + 1; *p; p++)
        if (*p == '/') { *p = 0; if (mkdir(tmp, 0755) && errno != EEXIST) return -1; *p = '/'; }
    return 0;
}

static int rf_open(R *r, rf *f)
{
    f->fd = fx_open_beneath(r->dirfd, f->partial, O_WRONLY | O_CREAT | O_TRUNC, 0600, r->cfg->direct, &f->direct, 1);
    return f->fd < 0 ? -1 : 0;
}

/* rename .partial into place, then mode (never setuid/setgid/sticky: masked at parse) and mtime; all through the directory fd */
static int rf_commit(R *r, rf *f)
{
    char leaf[256], lp[300];
    int parent = fx_open_parent(r->dirfd, f->path, 0, leaf, sizeof leaf);
    if (parent < 0) return -1;
    snprintf(lp, sizeof lp, "%s.partial", leaf);
    if (renameat(parent, lp, parent, leaf)) { close(parent); return -1; }
    f->committed = 1;
    if (fchmodat(parent, leaf, f->mode & 0777, 0)) { /* the file is in place; a mode that cannot be set is reported by the final listing */ }
    struct timespec ts[2];
    ts[0].tv_sec = ts[1].tv_sec = (time_t)(f->mtime_ns / 1000000000ull);
    ts[0].tv_nsec = ts[1].tv_nsec = (long)(f->mtime_ns % 1000000000ull);
    utimensat(parent, leaf, ts, AT_SYMLINK_NOFOLLOW);
    fsync(parent);
    close(parent);
    return 0;
}

static void rf_discard(R *r, rf *f)
{
    if (f->fd >= 0) { close(f->fd); f->fd = -1; }
    if (f->committed || !f->partial || r->dirfd < 0) return;
    char leaf[256], lp[300];
    int parent = fx_open_parent(r->dirfd, f->path, 0, leaf, sizeof leaf);
    if (parent < 0) return;
    snprintf(lp, sizeof lp, "%s.partial", leaf);
    unlinkat(parent, lp, 0);
    close(parent);
}

static int recv_msg_t(R *r, int kind, fx_msg *m, int timeout_ms)
{
    int e = fx_ctrl_recv(r->ctrl, timeout_ms, m);
    if (e) return set_err(r->res->err, "control connection to the sender %s", e == -2 ? "timed out" : "lost");
    if (m->type == FX_M_ERROR) {
        fx_rd rd = { m->data, m->len, 0, 0 };
        char *t = fx_get_str(&rd);
        set_err(r->res->err, "sender: %s", t);
        free(t);
        fx_msg_free(m);
        return -1;
    }
    if (m->type != kind) { set_err(r->res->err, "expected message %d from the sender, got %d", kind, m->type); fx_msg_free(m); return -1; }
    return 0;
}

static int recv_msg(R *r, int kind, fx_msg *m) { return recv_msg_t(r, kind, m, CTRL_TIMEOUT_MS); }

static char *get_path(fx_rd *rd)
{
    uint32_t n;
    const uint8_t *p = fx_get_bytes(rd, &n);
    if (rd->bad || !fx_path_ok((const char *)p, n)) return NULL;
    char *s = malloc(n + 1);
    memcpy(s, p, n);
    s[n] = 0;
    return s;
}

static void wait_bye(R *r)
{
    fx_msg m;
    if (fx_ctrl_recv(r->ctrl, 30000, &m) == 0) fx_msg_free(&m);
}

static int rdata_loop(R *r, int nfiles)
{
    const fx_cfg *cfg = r->cfg;
    fx_transport *tr = r->tr;
    int finished = 0, rc = 0;
    fx_event ev[64];
    uint64_t last_b = 0;
    double last_t = now_s();
    for (int i = 0; i < r->nfiles; i++) {
        rf *f = &r->files[i];
        if (!f->path || f->count) continue;
        if (rf_open(r, f)) return set_err(r->res->err, "cannot create %s: %s", f->partial, strerror(errno));
        wjob *j = calloc(1, sizeof *j);
        j->r = r; j->f = f; j->fin = 1;
        fx_tp_submit(r->pool, fin_run, j);
    }
    while (finished < nfiles && rc == 0) {
        int busy = 0, ne;
        while ((ne = tr->poll(tr, ev, 64)) > 0) {
            busy = 1;
            for (int k = 0; k < ne && !rc; k++) {
                fx_event *e = &ev[k];
                if (e->kind == FX_E_IMM) {
                    unsigned slot, seq;
                    fx_unpack_imm((uint32_t)e->a, &slot, &seq);
                    if (slot >= r->D) { rc = set_err(r->res->err, "the sender wrote slot %u of a ring of %u slots", slot, r->D); break; }
                    if (seq >= r->table.nchunks) { rc = set_err(r->res->err, "the sender wrote chunk %u, beyond the plan", seq); break; }
                    if (r->slot_busy[slot]) { rc = set_err(r->res->err, "the sender wrote slot %u again before it was handed back", slot); break; }
                    if (r->seen[seq]) { rc = set_err(r->res->err, "the sender wrote chunk %u twice", seq); break; }
                    int fi; uint64_t off; size_t n;
                    fx_table_chunk(&r->table, seq, &fi, &off, &n);
                    if (e->b != n) { rc = set_err(r->res->err, "the sender wrote %llu bytes for chunk %u, which holds %zu", (unsigned long long)e->b, seq, n); break; }
                    rf *f = &r->files[fi];
                    if (!f->path) { rc = set_err(r->res->err, "the sender wrote chunk %u of a file this receiver did not ask for", seq); break; }
                    if (f->fd < 0 && rf_open(r, f)) { rc = set_err(r->res->err, "cannot create %s: %s", f->partial, strerror(errno)); break; }
                    r->slot_busy[slot] = 1; r->seen[seq] = 1;
                    wjob *j = calloc(1, sizeof *j);
                    j->r = r; j->f = f; j->slot = slot; j->seq = seq; j->off = off; j->len = n; j->rail = e->rail;
                    fx_tp_submit(r->pool, write_run, j);
                } else if (e->kind == FX_E_ERR) rc = set_err(r->res->err, "queue pair error on rail %d: %s", e->rail, tr->err);
            }
        }
        if (rc) break;
        fx_node *nd;
        while ((nd = fx_queue_pop(&r->doneq))) {
            wjob *j = (wjob *)nd;
            rf *f = j->f;
            busy = 1;
            if (!j->fin) {
                if (j->bad) { rc = set_err(r->res->err, "%s", j->err); free(j); break; }
                f->crcs[j->seq - r->table.first[f->i]] = j->crc;
                r->slot_busy[j->slot] = 0;
                if (tr->post_credit(tr, 0, j->rail, j->slot)) { rc = set_err(r->res->err, "posting a credit failed: %s", tr->err); free(j); break; }
                r->res->bytes += j->len;
                if (--f->remaining == 0) {
                    wjob *fj = calloc(1, sizeof *fj);
                    fj->r = r; fj->f = f; fj->fin = 1;
                    fx_tp_submit(r->pool, fin_run, fj);
                }
            } else {
                f->awaiting = 1;
                fx_buf b = { 0 };
                fx_put_u32(&b, (uint32_t)f->i); fx_put_str(&b, j->bad ? j->err : "");
                fx_put_u32(&b, f->count);
                for (uint32_t k = 0; k < f->count; k++) fx_put_u32(&b, f->crcs[k]);
                fx_ctrl_send(r->ctrl, FX_M_FILE, &b);
                fx_buf_free(&b);
            }
            free(j);
        }
        if (rc) break;
        fx_msg m;
        int pr;
        while ((pr = fx_ctrl_poll(r->ctrl, &m)) > 0) {
            busy = 1;
            fx_rd rd = { m.data, m.len, 0, 0 };
            if (m.type == FX_M_VERDICT) {
                uint32_t fi = fx_get_u32(&rd); uint8_t ok = fx_get_u8(&rd); char *why = fx_get_str(&rd);
                if (rd.bad || fi >= (uint32_t)r->nfiles || !r->files[fi].path) { rc = set_err(r->res->err, "malformed verdict"); free(why); fx_msg_free(&m); break; }
                rf *f = &r->files[fi];
                if (ok && rf_commit(r, f) == 0) r->res->files_ok++;
                else {
                    r->res->files_failed++;
                    add_fail(&r->res->nfail, r->res->fail, "%s: %s", f->path, ok ? "rename failed" : why);
                    logf_(cfg, "ERROR: %s: %s; the partial file was removed", f->path, ok ? "rename failed" : why);
                    rf_discard(r, f);
                }
                finished++;
                free(why);
            } else if (m.type == FX_M_ERROR) {
                char *t = fx_get_str(&rd);
                rc = set_err(r->res->err, "sender: %s", t);
                free(t);
            }
            fx_msg_free(&m);
            if (rc) break;
        }
        if (pr < 0 && !rc) rc = set_err(r->res->err, "control connection to the sender lost");
        double t = now_s();
        if (cfg->progress_s > 0 && t - last_t >= 5 * cfg->progress_s) {
            logf_(cfg, "written %.1f GB, %.2f GB/s", (double)r->res->bytes / 1e9, (double)(r->res->bytes - last_b) / (t - last_t) / 1e9);
            last_t = t; last_b = r->res->bytes;
        }
        if (!busy) nap();
    }
    return rc;
}

int fx_run_receiver(fx_ctrl *ctrl, fx_transport_factory mk, void *mkctx, const char *dest, const fx_cfg *cfg, uint64_t ring_bytes, fx_recv_res *res)
{
    R r;
    memset(&r, 0, sizeof r);
    memset(res, 0, sizeof *res);
    r.ctrl = ctrl; r.cfg = cfg; r.res = res;
    fx_queue_init(&r.doneq);
    fx_msg m = { 0 };
    int rc = -1;
    char **lpaths = NULL; uint32_t np = 0;
    int auth_failed = 0;
    r.dirfd = -1;
    if (!cfg->token || !cfg->token[0]) { auth_failed = 1; set_err(res->err, "no authentication token configured"); goto out; }
    uint8_t nonce[32];
    if (fx_random(nonce, sizeof nonce)) { auth_failed = 1; set_err(res->err, "no random source for the challenge"); goto out; }
    {
        fx_buf cb = { 0 };
        fx_put_bytes(&cb, nonce, sizeof nonce);
        int e = fx_ctrl_send(ctrl, FX_M_CHALLENGE, &cb);
        fx_buf_free(&cb);
        if (e) { auth_failed = 1; set_err(res->err, "control connection to the sender lost"); goto out; }
    }
    if (recv_msg_t(&r, FX_M_HELLO, &m, 30000)) goto out;
    {
        /* authentication: HMAC-SHA256(token, nonce), compared in constant time; a peer that fails gets no reply at all */
        fx_rd h = { m.data, m.len, 0, 0 };
        uint32_t v = fx_get_u32(&h), ml;
        const uint8_t *mac = fx_get_bytes(&h, &ml);
        uint8_t want[32];
        fx_hmac_sha256(want, cfg->token, strlen(cfg->token), nonce, sizeof nonce);
        if (h.bad || v != FX_PROTO || !fx_ct_equal(mac, ml, want, sizeof want)) {
            auth_failed = 1;
            set_err(res->err, "authentication failed");
            goto out;
        }
    }
    fx_msg_free(&m);
    {
        fx_buf b = { 0 };
        fx_put_u32(&b, FX_PROTO); fx_put_u64(&b, ring_bytes);
        fx_ctrl_send(ctrl, FX_M_HELLO, &b);
        fx_buf_free(&b);
    }
    if (recv_msg(&r, FX_M_LIST, &m)) goto out;
    r.pool = fx_tp_new(cfg->io_threads > 0 ? cfg->io_threads : 1);
    {
        fx_rd rd = { m.data, m.len, 0, 0 };
        r.chunk = (size_t)fx_get_u64(&rd);
        int verify = fx_get_u8(&rd);
        np = fx_get_u32(&rd);
        if (rd.bad || r.chunk == 0 || r.chunk % FX_ALIGN || np > (rd.len - rd.pos) / 12) { set_err(res->err, "malformed plan"); goto out; }
        lpaths = calloc(np ? np : 1, sizeof(char *));
        uint64_t *lsize = calloc(np ? np : 1, 8);
        for (uint32_t i = 0; i < np; i++) {
            lpaths[i] = get_path(&rd);
            lsize[i] = fx_get_u64(&rd);
            if (!lpaths[i]) { free(lsize); set_err(res->err, "malformed plan"); goto out; }
        }
        if (rd.bad) { free(lsize); set_err(res->err, "malformed plan"); goto out; }
        r.dirfd = open(dest, O_RDONLY | O_DIRECTORY | O_CLOEXEC);          /* -1 if DEST does not exist yet: nothing is there */
        fx_buf b = { 0 };
        for (uint32_t i = 0; i < np; i++) {
            struct stat st;
            int dd, ifd = r.dirfd < 0 ? -1 : fx_open_beneath(r.dirfd, lpaths[i], O_RDONLY, 0, 0, &dd, 0);
            int have = ifd >= 0 && fstat(ifd, &st) == 0 && S_ISREG(st.st_mode);
            if (ifd >= 0) close(ifd);
            if (!have) { fx_put_u64(&b, ABSENT); if (verify) fx_put_u8(&b, 0); continue; }
            fx_put_u64(&b, (uint64_t)st.st_size);
            if (!verify) continue;
            if ((uint64_t)st.st_size != lsize[i]) { fx_put_u8(&b, 0); continue; }
            uint32_t n = (uint32_t)(((uint64_t)st.st_size + r.chunk - 1) / r.chunk);
            uint32_t *c = calloc(n ? n : 1, 4);
            char err[FX_ERRLEN];
            if (fx_file_chunk_crcs_at(r.dirfd, 1, lpaths[i], (uint64_t)st.st_size, r.chunk, cfg->direct, r.pool, c, err, sizeof err)) { free(c); free(lsize); fx_buf_free(&b); set_err(res->err, "%s", err); goto out; }
            fx_put_u8(&b, 1); fx_put_u32(&b, n);
            for (uint32_t q = 0; q < n; q++) fx_put_u32(&b, c[q]);
            free(c);
        }
        free(lsize);
        fx_ctrl_send(ctrl, FX_M_INV, &b);
        fx_buf_free(&b);
    }
    fx_msg_free(&m);
    if (recv_msg(&r, FX_M_PLAN, &m)) goto out;
    {
        fx_rd rd = { m.data, m.len, 0, 0 };
        uint64_t chunk = fx_get_u64(&rd);
        int dry = fx_get_u8(&rd);
        uint32_t nf = fx_get_u32(&rd);
        if (rd.bad || chunk != r.chunk || nf > (rd.len - rd.pos) / 25) { set_err(res->err, "malformed plan"); goto out; }
        r.nfiles = (int)nf;
        r.files = calloc(nf ? nf : 1, sizeof(rf));
        uint64_t *gs = calloc(nf ? nf : 1, 8);
        int nneed = 0;
        uint64_t mychunks = 0;
        for (uint32_t i = 0; i < nf; i++) {
            rf *f = &r.files[i];
            char *p = get_path(&rd);
            f->i = (int)i; f->fd = -1; f->size = fx_get_u64(&rd); f->mode = fx_get_u32(&rd) & 0777; f->mtime_ns = fx_get_u64(&rd);
            int need = fx_get_u8(&rd);
            gs[i] = f->size;
            if (!p) { free(gs); set_err(res->err, "malformed plan"); goto out; }
            if (need && (f->size + r.chunk - 1) / r.chunk > FX_MAX_CHUNKS) { free(p); free(gs); set_err(res->err, "malformed plan"); goto out; }
            if (need) {
                char full[4352];
                f->path = p;
                snprintf(full, sizeof full, "%s.partial", p);
                f->partial = strdup(full);
                f->count = f->remaining = (uint32_t)((f->size + r.chunk - 1) / r.chunk);
                f->crcs = calloc(f->count ? f->count : 1, 4);
                nneed++; mychunks += f->count;
            } else free(p);
        }
        if (rd.bad) { free(gs); set_err(res->err, "malformed plan"); goto out; }
        if (fx_table_init(&r.table, gs, (int)nf, r.chunk)) { free(gs); set_err(res->err, "the plan does not fit the chunk numbering"); goto out; }
        free(gs);
        fx_msg_free(&m);
        if (dry) {
            fx_buf b = { 0 };
            fx_put_u32(&b, 0); fx_put_u32(&b, 0);
            fx_ctrl_send(ctrl, FX_M_DONE, &b);
            fx_buf_free(&b);
            wait_bye(&r);
            rc = 0;
            goto out;
        }
        if (!nneed) { fx_ctrl_send(ctrl, FX_M_NOTHING, NULL); wait_bye(&r); rc = 0; goto out; }
        if (r.dirfd < 0) {
            char dd[4352];
            snprintf(dd, sizeof dd, "%s", dest);
            if (mkdir_p(dd) || (mkdir(dd, 0755) && errno != EEXIST) || (r.dirfd = open(dest, O_RDONLY | O_DIRECTORY | O_CLOEXEC)) < 0) {
                set_err(res->err, "cannot create %s: %s", dest, strerror(errno));
                goto out;
            }
        }
        r.tr = mk(mkctx);
        if (!r.tr) { set_err(res->err, "cannot open the RDMA devices"); goto out; }
        uint64_t D = ring_bytes / r.chunk;
        if (D > FX_MAX_SLOTS) D = FX_MAX_SLOTS;
        if (D > (uint64_t)r.tr->max_slots) D = (uint64_t)r.tr->max_slots;
        if (D > mychunks) D = mychunks;
        if (D < 1) D = 1;
        r.D = (unsigned)D;
        r.slot_busy = calloc(D, 1);
        r.seen = calloc(r.table.nchunks ? r.table.nchunks : 1, 1);
        double ta = now_s();
        if (r.tr->alloc_ring(r.tr, (size_t)D * r.chunk, &r.ring)) { set_err(res->err, "cannot allocate or register the ring: %s", r.tr->err); goto out; }
        logf_(cfg, "ring %.2f GiB (%llu slots of %zu MiB) allocated and registered in %.2f s", (double)D * (double)r.chunk / (double)(1ull << 30), (unsigned long long)D, r.chunk >> 20, now_s() - ta);
        uint8_t info[FX_INFO_MAX]; size_t il = sizeof info;
        if (r.tr->create_peer(r.tr, 0, (int)D, info, &il)) { set_err(res->err, "queue pair: %s", r.tr->err); goto out; }
        fx_buf b = { 0 };
        fx_put_u32(&b, (uint32_t)D); fx_put_bytes(&b, info, il);
        fx_ctrl_send(ctrl, FX_M_RING, &b);
        fx_buf_free(&b);
        if (recv_msg(&r, FX_M_QP, &m)) goto out;
        fx_rd q = { m.data, m.len, 0, 0 };
        uint32_t ql; const uint8_t *qi = fx_get_bytes(&q, &ql);
        if (q.bad || r.tr->connect_peer(r.tr, 0, qi, ql)) { set_err(res->err, "connecting to the sender: %s", r.tr->err); goto out; }
        fx_msg_free(&m);
        fx_ctrl_send(ctrl, FX_M_READY, NULL);
        if (rdata_loop(&r, nneed)) goto out;
        fx_buf d = { 0 };
        fx_put_u32(&d, (uint32_t)res->files_ok); fx_put_u32(&d, (uint32_t)res->files_failed);
        fx_ctrl_send(ctrl, FX_M_DONE, &d);
        fx_buf_free(&d);
        wait_bye(&r);
        rc = 0;
    }
out:
    if (m.data) fx_msg_free(&m);
    if (rc && !auth_failed) send_error(ctrl, res->err[0] ? res->err : "receiver stopped");
    if (r.pool) fx_tp_free(r.pool);
    fx_node *nd;
    while ((nd = fx_queue_pop(&r.doneq))) free(nd);
    for (int i = 0; i < r.nfiles; i++) {
        rf *f = &r.files[i];
        rf_discard(&r, f);
        free(f->path); free(f->partial); free(f->crcs);
    }
    free(r.files); free(r.slot_busy); free(r.seen);
    if (r.dirfd >= 0) close(r.dirfd);
    if (lpaths) { for (uint32_t i = 0; i < np; i++) free(lpaths[i]); free(lpaths); }
    fx_table_free(&r.table);
    if (r.tr) r.tr->destroy(r.tr);
    return rc;
}
