/* In-process wire for tests: a write is a memcpy into the other transport's ring. It also checks the slot protocol:
 * a receiver slot is never written before it was handed back, and a credit is only for a slot that is out. */
#include "fx.h"

#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

typedef struct lt lt;
struct fx_loopnet {
    pthread_mutex_t mu; int rails; uint64_t rail_bytes[8]; int violations, max_busy;
    void (*corrupt)(void *, unsigned, uint8_t *, size_t); void *corrupt_arg;
};
struct lt {
    fx_loopnet *net; fx_transport t; uint8_t *ring; size_t ring_len;
    pthread_mutex_t mu; fx_event *ev; size_t evn, evcap;
    struct { lt *remote; int rpeer; } peer[64];
    uint8_t busy[FX_MAX_SLOTS]; int nbusy;
};

static void ev_push(lt *l, int kind, int peer, int rail, uint64_t a, uint64_t b)
{
    pthread_mutex_lock(&l->mu);
    if (l->evn == l->evcap) { l->evcap = l->evcap ? l->evcap * 2 : 256; l->ev = realloc(l->ev, l->evcap * sizeof *l->ev); }
    l->ev[l->evn++] = (fx_event){ kind, peer, rail, a, b };
    pthread_mutex_unlock(&l->mu);
}

static int lt_alloc_ring(fx_transport *t, size_t n, uint8_t **base)
{
    lt *l = t->ctx;
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return -1;
    l->ring = p; l->ring_len = n; *base = p;
    return 0;
}
static int lt_create_peer(fx_transport *t, int peer, int depth, uint8_t *info, size_t *infolen)
{
    (void)depth;
    lt *l = t->ctx;
    uint64_t me = (uint64_t)(uintptr_t)l;
    memcpy(info, &me, 8); memcpy(info + 8, &peer, 4);
    *infolen = 12;
    return 0;
}
static int lt_connect_peer(fx_transport *t, int peer, const uint8_t *info, size_t len)
{
    lt *l = t->ctx;
    uint64_t them; int rp;
    if (len != 12) return -1;
    memcpy(&them, info, 8); memcpy(&rp, info + 8, 4);
    l->peer[peer].remote = (lt *)(uintptr_t)them; l->peer[peer].rpeer = rp;
    return 0;
}
static int lt_post_write(fx_transport *t, int peer, int rail, size_t lo, size_t n, size_t ro, uint32_t imm, uint64_t token)
{
    lt *l = t->ctx, *r = l->peer[peer].remote;
    fx_loopnet *net = l->net;
    unsigned slot, seq;
    fx_unpack_imm(imm, &slot, &seq);
    pthread_mutex_lock(&net->mu);
    if (r->busy[slot]) net->violations++;
    else { r->busy[slot] = 1; r->nbusy++; }
    if (r->nbusy > net->max_busy) net->max_busy = r->nbusy;
    net->rail_bytes[rail] += n;
    pthread_mutex_unlock(&net->mu);
    memcpy(r->ring + ro, l->ring + lo, n);
    if (net->corrupt) net->corrupt(net->corrupt_arg, seq, r->ring + ro, n);
    ev_push(r, FX_E_IMM, l->peer[peer].rpeer, rail, imm, n);
    ev_push(l, FX_E_WDONE, peer, rail, token, n);
    return 0;
}
static int lt_post_credit(fx_transport *t, int peer, int rail, uint32_t slot)
{
    lt *l = t->ctx, *r = l->peer[peer].remote;
    pthread_mutex_lock(&l->net->mu);
    if (slot >= FX_MAX_SLOTS || !l->busy[slot]) l->net->violations++;
    else { l->busy[slot] = 0; l->nbusy--; }
    pthread_mutex_unlock(&l->net->mu);
    ev_push(r, FX_E_CREDIT, l->peer[peer].rpeer, rail, slot, 0);
    ev_push(l, FX_E_CDONE, peer, rail, 0, 0);
    return 0;
}
static int lt_poll(fx_transport *t, fx_event *out, int max)
{
    lt *l = t->ctx;
    pthread_mutex_lock(&l->mu);
    int n = (int)(l->evn < (size_t)max ? l->evn : (size_t)max);
    if (n) {
        memcpy(out, l->ev, n * sizeof *out);
        memmove(l->ev, l->ev + n, (l->evn - n) * sizeof *out);
        l->evn -= n;
    }
    pthread_mutex_unlock(&l->mu);
    return n;
}
static void lt_destroy(fx_transport *t)
{
    lt *l = t->ctx;
    if (l->ring) munmap(l->ring, l->ring_len);
    free(l->ev);
    free(l);
}

fx_loopnet *fx_loopnet_new(int rails)
{
    fx_loopnet *n = calloc(1, sizeof *n);
    pthread_mutex_init(&n->mu, NULL);
    n->rails = rails;
    return n;
}
fx_transport *fx_loopnet_transport(fx_loopnet *n)
{
    lt *l = calloc(1, sizeof *l);
    l->net = n;
    pthread_mutex_init(&l->mu, NULL);
    l->t = (fx_transport){ .ctx = l, .rails = n->rails, .max_slots = FX_MAX_SLOTS, .alloc_ring = lt_alloc_ring,
                           .create_peer = lt_create_peer, .connect_peer = lt_connect_peer, .post_write = lt_post_write,
                           .post_credit = lt_post_credit, .poll = lt_poll, .destroy = lt_destroy };
    return &l->t;
}
void fx_loopnet_free(fx_loopnet *n) { free(n); }
uint64_t fx_loopnet_rail_bytes(fx_loopnet *n, int rail) { return n->rail_bytes[rail]; }
int fx_loopnet_violations(fx_loopnet *n) { return n->violations; }
int fx_loopnet_max_busy(fx_loopnet *n) { return n->max_busy; }
void fx_loopnet_set_corrupt(fx_loopnet *n, void (*fn)(void *, unsigned, uint8_t *, size_t), void *arg)
{
    n->corrupt = fn; n->corrupt_arg = arg;
}
