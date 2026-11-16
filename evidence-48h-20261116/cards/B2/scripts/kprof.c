// LD_PRELOAD CUDA launch profiler (no CUPTI/nsys in image). One cudaEvent after each launch; per-stream delta = kernel time (+ idle gap if GPU starved).
// Phase label read from file $KPROF_PHASE (polled 20 ms); label "idle"/"off"/empty = not recording. Output $KPROF_OUT.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>
typedef int err_t; typedef struct { unsigned x, y, z; } d3;
typedef struct { void *ev, *prev, *stream; const void *func; unsigned gx, bx; int kind, label; long bytes; double th; } rec_t;
static err_t (*r_launch)(const void*, d3, d3, void**, size_t, void*);
static err_t (*r_launchEx)(const void*, const void*, void**);
static err_t (*r_launchCoop)(const void*, d3, d3, void**, size_t, void*);
static err_t (*r_memcpyA)(void*, const void*, size_t, int, void*);
static err_t (*r_memcpy)(void*, const void*, size_t, int);
static err_t (*e_create)(void**, unsigned), (*e_record)(void*, void*), (*e_elapsed)(float*, void*, void*), (*e_query)(void*), (*dev_sync)(void);
static err_t (*f_name)(const char**, const void*);
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
#define CAP 120000
static rec_t *R; static int nR; static volatile int cur_label; static int nlabels;
static FILE *fo; static void *pool[400000]; static int npool_all, nfree; static void *freel[400000];
static void *last_ev[32], *last_st[32]; static int need_base[32], nst;
static const void *seen[4096]; static int nseen; static long segid;
static void *H;
static void *sym(const char *n) { if (!H) H = dlopen("libcudart.so.12", RTLD_NOW); return H ? dlsym(H, n) : NULL; }
static void resolve(void) {
  if (e_create) return;
  r_launch = sym("cudaLaunchKernel"); r_launchEx = sym("cudaLaunchKernelExC"); r_launchCoop = sym("cudaLaunchCooperativeKernel");
  r_memcpyA = sym("cudaMemcpyAsync"); r_memcpy = sym("cudaMemcpy");
  e_record = sym("cudaEventRecord"); e_elapsed = sym("cudaEventElapsedTime"); e_query = sym("cudaEventQuery"); dev_sync = sym("cudaDeviceSynchronize"); f_name = sym("cudaFuncGetName");
  e_create = sym("cudaEventCreateWithFlags");
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void *getev(void) {
  if (nfree) return freel[--nfree];
  void *e; if (e_create(&e, 0)) return NULL; pool[npool_all++] = e; return e;
}
static int sidx(void *s) { for (int i = 0; i < nst; i++) if (last_st[i] == s) return i; if (nst >= 32) return 31; last_st[nst] = s; need_base[nst] = 1; return nst++; }
static void addrec(int kind, const void *func, unsigned gx, unsigned bx, void *stream, long bytes) {
  if (!cur_label) return;
  resolve(); if (!e_create || !e_record) return;
  pthread_mutex_lock(&mu);
  if (nR >= CAP - 2) { pthread_mutex_unlock(&mu); return; }
  int si = sidx(stream);
  if (need_base[si] && kind != 2) { rec_t *b = &R[nR++]; memset(b, 0, sizeof *b); b->kind = 1; b->ev = getev(); e_record(b->ev, stream); last_ev[si] = b->ev; need_base[si] = 0; b->label = cur_label; b->stream = stream; }
  rec_t *r = &R[nR++]; memset(r, 0, sizeof *r); r->kind = kind; r->func = func; r->gx = gx; r->bx = bx; r->stream = stream; r->bytes = bytes; r->th = now(); r->label = cur_label;
  r->ev = getev(); r->prev = last_ev[si]; e_record(r->ev, stream); last_ev[si] = r->ev;
  if (kind == 2) for (int i = 0; i < nst; i++) need_base[i] = 1;
  pthread_mutex_unlock(&mu);
}
typedef struct { const void *f; int n; double ms; unsigned gx; } kacc_t;
static void flush_locked(int all) {
  if (!nR) return; resolve(); int L = -1;
  for (int i = nR - 1; i >= 0; i--) if (R[i].kind == 2) { L = i; break; }
  if (all && L < 0) L = nR - 1;
  if (L < 0) return;
  dev_sync();
  static kacc_t acc[1024]; int na = 0, nl = 0; double gms = 0, th0 = 0; long minh = 0;
  for (int i = 0; i <= L; i++) {
    rec_t *r = &R[i]; float ms = 0;
    if (r->kind != 1 && r->prev) e_elapsed(&ms, r->prev, r->ev);
    if (r->kind == 0) {
      if (!nl) th0 = r->th; nl++; gms += ms; int k; for (k = 0; k < na; k++) if (acc[k].f == r->func) break;
      if (k == na && na < 1024) { acc[na].f = r->func; acc[na].n = 0; acc[na].ms = 0; acc[na].gx = 0; na++; }
      if (k < na) { acc[k].n++; acc[k].ms += ms; if (r->gx > acc[k].gx) acc[k].gx = r->gx; }
    } else if (r->kind == 3) { if (r->bytes >= 4 && r->bytes <= 256 && r->bytes % 4 == 0 && (!minh || r->bytes < minh)) minh = r->bytes; }
    else if (r->kind == 2) {
      fprintf(fo, "S\t%ld\t%d\t%ld\t%d\t%.4f\t%.6f\t%.6f\t%ld\t%.4f\n", segid, r->label, minh / 4, nl, gms, th0, r->th, r->bytes, ms);
      for (int k = 0; k < na; k++) {
        int s; for (s = 0; s < nseen; s++) if (seen[s] == acc[k].f) break;
        if (s == nseen && nseen < 4096) { seen[nseen++] = acc[k].f; const char *nm = NULL; if (f_name) f_name(&nm, acc[k].f); fprintf(fo, "N\t%p\t%s\n", acc[k].f, nm ? nm : "?"); }
        fprintf(fo, "K\t%ld\t%p\t%d\t%.4f\t%u\n", segid, acc[k].f, acc[k].n, acc[k].ms, acc[k].gx);
      }
      segid++; na = 0; nl = 0; gms = 0; th0 = 0; minh = 0;
    }
  }
  for (int i = 0; i <= L; i++) if (R[i].ev) freel[nfree++] = R[i].ev;
  memmove(R, R + L + 1, (nR - L - 1) * sizeof(rec_t)); nR -= L + 1; fflush(fo);
}
static void *mon(void *a) {
  char *pf = getenv("KPROF_PHASE"); char buf[128], lastbuf[128] = "";
  for (;;) {
    usleep(20000);
    FILE *f = fopen(pf, "r"); if (f) { buf[0] = 0; if (!fgets(buf, sizeof buf, f)) buf[0] = 0; fclose(f); buf[strcspn(buf, "\r\n")] = 0;
      if (strcmp(buf, lastbuf)) { strcpy(lastbuf, buf);
        pthread_mutex_lock(&mu);
        int id = 0; if (buf[0] && strcmp(buf, "idle") && strcmp(buf, "off")) { id = ++nlabels; fprintf(fo, "L\t%d\t%s\n", id, buf); }
        cur_label = id; pthread_mutex_unlock(&mu); } }
    pthread_mutex_lock(&mu);
    if (nR && R[nR - 1].ev && e_query(R[nR - 1].ev) == 0) flush_locked(!cur_label);
    pthread_mutex_unlock(&mu);
  }
  return NULL;
}
__attribute__((constructor)) static void init(void) {
  const char *o = getenv("KPROF_OUT"); if (!o) return; char pp[600]; snprintf(pp, sizeof pp, "%s.%d", o, (int)getpid()); fo = fopen(pp, "w"); if (!fo) return;
  R = calloc(CAP, sizeof(rec_t));
  fprintf(fo, "I\tinit\n"); fflush(fo);
  pthread_t t; pthread_create(&t, NULL, mon, NULL); pthread_detach(t);
}
err_t cudaLaunchKernel(const void *f, d3 g, d3 b, void **a, size_t s, void *st) {
  resolve();
  err_t e = r_launch(f, g, b, a, s, st); if (fo && e == 0) addrec(0, f, g.x, b.x, st, 0); return e; }
err_t cudaLaunchCooperativeKernel(const void *f, d3 g, d3 b, void **a, size_t s, void *st) {
  resolve();
  err_t e = r_launchCoop(f, g, b, a, s, st); if (fo && e == 0) addrec(0, f, g.x, b.x, st, 0); return e; }
err_t cudaLaunchKernelExC(const void *cfg, const void *f, void **a) {
  resolve();
  err_t e = r_launchEx(cfg, f, a); if (fo && e == 0) { const d3 *g = cfg; const d3 *b = (const d3*)((const char*)cfg + 12); void *st = *(void**)((const char*)cfg + 32); addrec(0, f, g->x, b->x, st, 0); } return e; }
err_t cudaMemcpyAsync(void *d, const void *s, size_t n, int k, void *st) {
  resolve();
  err_t e = r_memcpyA(d, s, n, k, st); if (fo && e == 0 && (k == 1 || k == 2)) addrec(k == 2 ? 2 : 3, 0, 0, 0, st, n); return e; }
err_t cudaMemcpy(void *d, const void *s, size_t n, int k) {
  resolve();
  err_t e = r_memcpy(d, s, n, k); if (fo && e == 0 && (k == 1 || k == 2)) addrec(k == 2 ? 2 : 3, 0, 0, 0, 0, n); return e; }
