/* ibverbs wire: one RC QP per rail per peer, one CQ per rail, the ring registered once per rail's protection domain.
 * Chunks go out as RDMA WRITE_WITH_IMM; credits are one-byte SEND_WITH_IMM. Linux only (needs libibverbs). */
#include "fx.h"

#include <arpa/inet.h>
#include <errno.h>
#include <infiniband/verbs.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define MAXR 8
#define MAXQ (FX_MAX_PEERS + 1)
#define TOKENS ((size_t)FX_MAX_SLOTS * 64)

typedef struct {
    struct ibv_context *ctx; struct ibv_pd *pd; struct ibv_cq *cq; struct ibv_mr *ctrl_mr, *ring_mr;
    int port, gid_index; enum ibv_mtu mtu; union ibv_gid gid;
} vrail;
typedef struct { struct ibv_qp *qp[MAXR]; uint32_t psn[MAXR]; uint32_t rkey[MAXR]; uint64_t raddr[MAXR]; } vpeer;
typedef struct { uint32_t qpn, psn, mtu, rkey; uint64_t addr; uint8_t gid[16]; } vinfo;   /* same-endian peers (both aarch64) */

typedef struct {
    fx_transport t; int nr, max_peers; vrail r[MAXR]; vpeer peer[MAXQ];
    uint8_t *ctrl; uint8_t *ring; size_t ring_len; uint32_t *nbytes;
    struct { uint32_t qpn; int peer, rail; } map[MAXQ * MAXR]; int nmap;
} V;

static int verr(V *v, const char *fmt, const char *what)
{
    snprintf(v->t.err, sizeof v->t.err, fmt, what, strerror(errno));
    return -1;
}

static int post_recv(V *v, int peer, int rail)
{
    struct ibv_sge sge = { .addr = (uintptr_t)v->ctrl, .length = 1, .lkey = v->r[rail].ctrl_mr->lkey };
    struct ibv_recv_wr wr = { .wr_id = 0, .sg_list = &sge, .num_sge = 1 }, *bad;
    int e = ibv_post_recv(v->peer[peer].qp[rail], &wr, &bad);
    if (e) { errno = e; return verr(v, "%s: %s", "ibv_post_recv"); }
    return 0;
}

static int v_alloc_ring(fx_transport *t, size_t n, uint8_t **base)
{
    V *v = t->ctx;
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return verr(v, "%s: %s", "mmap of the ring");
    madvise(p, n, MADV_HUGEPAGE);
    for (int r = 0; r < v->nr; r++) {
        v->r[r].ring_mr = ibv_reg_mr(v->r[r].pd, p, n, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!v->r[r].ring_mr) return verr(v, "%s: %s (RLIMIT_MEMLOCK too small for this ring?)", "ibv_reg_mr of the ring");
    }
    v->ring = p; v->ring_len = n; *base = p;
    return 0;
}

static int v_create_peer(fx_transport *t, int peer, int depth, uint8_t *info, size_t *infolen)
{
    V *v = t->ctx;
    uint32_t nr = (uint32_t)v->nr;
    memcpy(info, &nr, 4);
    for (int r = 0; r < v->nr; r++) {
        vrail *vr = &v->r[r];
        struct ibv_qp_init_attr ia = { 0 };
        ia.send_cq = ia.recv_cq = vr->cq; ia.qp_type = IBV_QPT_RC; ia.sq_sig_all = 1;
        ia.cap.max_send_wr = (uint32_t)depth + 16; ia.cap.max_recv_wr = (uint32_t)depth + 16;
        ia.cap.max_send_sge = 1; ia.cap.max_recv_sge = 1;
        struct ibv_qp *qp = ibv_create_qp(vr->pd, &ia);
        if (!qp) return verr(v, "%s: %s", "ibv_create_qp");
        v->peer[peer].qp[r] = qp;
        struct ibv_qp_attr a = { 0 };
        a.qp_state = IBV_QPS_INIT; a.pkey_index = 0; a.port_num = (uint8_t)vr->port;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
        if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS))
            return verr(v, "%s: %s", "ibv_modify_qp to INIT");
        v->map[v->nmap].qpn = qp->qp_num; v->map[v->nmap].peer = peer; v->map[v->nmap].rail = r; v->nmap++;
        for (int i = 0; i < depth + 8; i++) if (post_recv(v, peer, r)) return -1;
        v->peer[peer].psn[r] = (uint32_t)(random() & 0xFFFFFF);
        vinfo vi = { .qpn = qp->qp_num, .psn = v->peer[peer].psn[r], .mtu = (uint32_t)vr->mtu,
                     .rkey = vr->ring_mr ? vr->ring_mr->rkey : 0, .addr = (uint64_t)(uintptr_t)v->ring };
        memcpy(vi.gid, vr->gid.raw, 16);
        memcpy(info + 4 + (size_t)r * sizeof vi, &vi, sizeof vi);
    }
    *infolen = 4 + (size_t)v->nr * sizeof(vinfo);
    return 0;
}

static int v_connect_peer(fx_transport *t, int peer, const uint8_t *info, size_t len)
{
    V *v = t->ctx;
    uint32_t nr;
    if (len < 4) return -1;
    memcpy(&nr, info, 4);
    if ((int)nr != v->nr || len != 4 + (size_t)nr * sizeof(vinfo)) {
        snprintf(v->t.err, sizeof v->t.err, "the peer has %u rails, this side %d", nr, v->nr);
        return -1;
    }
    for (int r = 0; r < v->nr; r++) {
        vrail *vr = &v->r[r];
        vinfo vi;
        memcpy(&vi, info + 4 + (size_t)r * sizeof vi, sizeof vi);
        struct ibv_qp_attr a = { 0 };
        a.qp_state = IBV_QPS_RTR;
        a.path_mtu = (enum ibv_mtu)(vi.mtu < (uint32_t)vr->mtu ? vi.mtu : (uint32_t)vr->mtu);
        a.dest_qp_num = vi.qpn; a.rq_psn = vi.psn; a.max_dest_rd_atomic = 1; a.min_rnr_timer = 12;
        a.ah_attr.is_global = 1; a.ah_attr.port_num = (uint8_t)vr->port; a.ah_attr.dlid = 0; a.ah_attr.sl = 0;
        memcpy(a.ah_attr.grh.dgid.raw, vi.gid, 16);
        a.ah_attr.grh.sgid_index = (uint8_t)vr->gid_index; a.ah_attr.grh.hop_limit = 1; a.ah_attr.grh.traffic_class = 0;
        struct ibv_qp *qp = v->peer[peer].qp[r];
        if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                      IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER))
            return verr(v, "%s: %s", "ibv_modify_qp to RTR");
        struct ibv_qp_attr b = { 0 };
        b.qp_state = IBV_QPS_RTS; b.timeout = 14; b.retry_cnt = 7; b.rnr_retry = 7; b.sq_psn = v->peer[peer].psn[r]; b.max_rd_atomic = 1;
        if (ibv_modify_qp(qp, &b, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC))
            return verr(v, "%s: %s", "ibv_modify_qp to RTS");
        v->peer[peer].rkey[r] = vi.rkey; v->peer[peer].raddr[r] = vi.addr;
    }
    return 0;
}

static int v_post_write(fx_transport *t, int peer, int rail, size_t lo, size_t n, size_t ro, uint32_t imm, uint64_t token)
{
    V *v = t->ctx;
    struct ibv_sge sge = { .addr = (uintptr_t)(v->ring + lo), .length = (uint32_t)n, .lkey = v->r[rail].ring_mr->lkey };
    struct ibv_send_wr wr, *bad;
    memset(&wr, 0, sizeof wr);
    wr.wr_id = token; wr.sg_list = &sge; wr.num_sge = 1; wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.send_flags = IBV_SEND_SIGNALED; wr.imm_data = htonl(imm);
    wr.wr.rdma.remote_addr = v->peer[peer].raddr[rail] + ro; wr.wr.rdma.rkey = v->peer[peer].rkey[rail];
    v->nbytes[token] = (uint32_t)n;
    int e = ibv_post_send(v->peer[peer].qp[rail], &wr, &bad);
    if (e) { errno = e; return verr(v, "%s: %s", "ibv_post_send (write)"); }
    return 0;
}

static int v_post_credit(fx_transport *t, int peer, int rail, uint32_t slot)
{
    V *v = t->ctx;
    struct ibv_sge sge = { .addr = (uintptr_t)v->ctrl, .length = 1, .lkey = v->r[rail].ctrl_mr->lkey };
    struct ibv_send_wr wr, *bad;
    memset(&wr, 0, sizeof wr);
    wr.wr_id = 1ull << 62; wr.sg_list = &sge; wr.num_sge = 1; wr.opcode = IBV_WR_SEND_WITH_IMM;
    wr.send_flags = IBV_SEND_SIGNALED; wr.imm_data = htonl(slot);
    int e = ibv_post_send(v->peer[peer].qp[rail], &wr, &bad);
    if (e) { errno = e; return verr(v, "%s: %s", "ibv_post_send (credit)"); }
    return 0;
}

static int v_poll(fx_transport *t, fx_event *out, int max)
{
    V *v = t->ctx;
    int per = max / v->nr > 0 ? max / v->nr : 1, n = 0;
    struct ibv_wc wc[64];
    for (int r = 0; r < v->nr && n < max; r++) {
        int k = ibv_poll_cq(v->r[r].cq, per > 64 ? 64 : per, wc);
        if (k < 0) { snprintf(v->t.err, sizeof v->t.err, "ibv_poll_cq failed on rail %d", r); out[n++] = (fx_event){ FX_E_ERR, -1, r, 0, 0 }; continue; }
        for (int i = 0; i < k; i++) {
            int peer = -1;
            for (int q = 0; q < v->nmap; q++) if (v->map[q].qpn == wc[i].qp_num) { peer = v->map[q].peer; break; }
            if (wc[i].status != IBV_WC_SUCCESS) {
                snprintf(v->t.err, sizeof v->t.err, "work completion %s (vendor_err 0x%x, opcode %d)", ibv_wc_status_str(wc[i].status), wc[i].vendor_err, (int)wc[i].opcode);
                out[n++] = (fx_event){ FX_E_ERR, peer, r, 0, 0 };
                continue;
            }
            switch (wc[i].opcode) {
            case IBV_WC_RDMA_WRITE: out[n++] = (fx_event){ FX_E_WDONE, peer, r, wc[i].wr_id, v->nbytes[wc[i].wr_id] }; break;
            case IBV_WC_SEND: out[n++] = (fx_event){ FX_E_CDONE, peer, r, 0, 0 }; break;
            case IBV_WC_RECV_RDMA_WITH_IMM:
                out[n++] = (fx_event){ FX_E_IMM, peer, r, ntohl(wc[i].imm_data), wc[i].byte_len };
                if (peer >= 0) post_recv(v, peer, r);
                break;
            case IBV_WC_RECV:
                out[n++] = (fx_event){ FX_E_CREDIT, peer, r, ntohl(wc[i].imm_data), 0 };
                if (peer >= 0) post_recv(v, peer, r);
                break;
            default: break;
            }
        }
    }
    return n;
}

static void v_destroy(fx_transport *t)
{
    V *v = t->ctx;
    for (int p = 0; p < MAXQ; p++) for (int r = 0; r < v->nr; r++) if (v->peer[p].qp[r]) ibv_destroy_qp(v->peer[p].qp[r]);
    for (int r = 0; r < v->nr; r++) {
        if (v->r[r].ring_mr) ibv_dereg_mr(v->r[r].ring_mr);
        if (v->r[r].ctrl_mr) ibv_dereg_mr(v->r[r].ctrl_mr);
        if (v->r[r].cq) ibv_destroy_cq(v->r[r].cq);
        if (v->r[r].pd) ibv_dealloc_pd(v->r[r].pd);
        if (v->r[r].ctx) ibv_close_device(v->r[r].ctx);
    }
    if (v->ring) munmap(v->ring, v->ring_len);
    if (v->ctrl) munmap(v->ctrl, FX_ALIGN);
    free(v->nbytes);
    free(v);
}

static struct ibv_context *open_dev(const char *name)
{
    int n = 0;
    struct ibv_device **l = ibv_get_device_list(&n);
    struct ibv_context *c = NULL;
    for (int i = 0; l && i < n; i++)
        if (!strcmp(ibv_get_device_name(l[i]), name)) { c = ibv_open_device(l[i]); break; }
    if (l) ibv_free_device_list(l);
    return c;
}

fx_transport *fx_verbs_new(const fx_rail *rails, int nrails, int max_peers, char *err, size_t errlen)
{
    if (nrails < 1 || nrails > MAXR || max_peers < 1 || max_peers > FX_MAX_PEERS) { snprintf(err, errlen, "bad rail or peer count"); return NULL; }
    V *v = calloc(1, sizeof *v);
    v->nr = nrails; v->max_peers = max_peers;
    v->nbytes = calloc(TOKENS, 4);
    v->ctrl = mmap(NULL, FX_ALIGN, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    v->t = (fx_transport){ .ctx = v, .rails = nrails, .max_slots = FX_MAX_SLOTS, .alloc_ring = v_alloc_ring, .create_peer = v_create_peer,
                           .connect_peer = v_connect_peer, .post_write = v_post_write, .post_credit = v_post_credit, .poll = v_poll,
                           .destroy = v_destroy };
    for (int r = 0; r < nrails; r++) {
        vrail *vr = &v->r[r];
        vr->port = rails[r].port; vr->gid_index = rails[r].gid_index;
        vr->ctx = open_dev(rails[r].ibdev);
        if (!vr->ctx) { snprintf(err, errlen, "cannot open RDMA device %s", rails[r].ibdev); v_destroy(&v->t); return NULL; }
        struct ibv_device_attr da;
        struct ibv_port_attr pa;
        if (ibv_query_device(vr->ctx, &da) || ibv_query_port(vr->ctx, vr->port, &pa) || ibv_query_gid(vr->ctx, vr->port, vr->gid_index, &vr->gid)) {
            snprintf(err, errlen, "cannot query %s port %d gid %d", rails[r].ibdev, vr->port, vr->gid_index);
            v_destroy(&v->t); return NULL;
        }
        if (da.max_qp_wr - 32 < v->t.max_slots) v->t.max_slots = da.max_qp_wr - 32;
        vr->mtu = pa.active_mtu;
        vr->pd = ibv_alloc_pd(vr->ctx);
        int cqe = 2 * (int)(FX_MAX_SLOTS + 64) * max_peers;
        if (cqe > da.max_cqe) cqe = da.max_cqe;
        vr->cq = vr->pd ? ibv_create_cq(vr->ctx, cqe, NULL, NULL, 0) : NULL;
        vr->ctrl_mr = vr->cq ? ibv_reg_mr(vr->pd, v->ctrl, FX_ALIGN, IBV_ACCESS_LOCAL_WRITE) : NULL;
        if (!vr->ctrl_mr) { snprintf(err, errlen, "cannot create pd/cq/mr on %s: %s", rails[r].ibdev, strerror(errno)); v_destroy(&v->t); return NULL; }
    }
    return &v->t;
}

int fx_verbs_probe(const fx_rail *rail, char *out, size_t outlen)
{
    struct ibv_context *c = open_dev(rail->ibdev);
    if (!c) { snprintf(out, outlen, "cannot open %s", rail->ibdev); return -1; }
    struct ibv_device_attr da;
    struct ibv_port_attr pa;
    int bad = ibv_query_device(c, &da) || ibv_query_port(c, rail->port, &pa);
    if (!bad)
        snprintf(out, outlen, "%s port %d: active_mtu enum %d, max_msg_sz %u, max_qp_wr %d, max_cqe %d, max_mr_size 0x%llx", rail->ibdev, rail->port,
                 (int)pa.active_mtu, pa.max_msg_sz, da.max_qp_wr, da.max_cqe, (unsigned long long)da.max_mr_size);
    else snprintf(out, outlen, "cannot query %s", rail->ibdev);
    ibv_close_device(c);
    return bad ? -1 : 0;
}
