// Standalone large-batch correctness test for coli_metal_gemm — reproduces the
// long-context OpenCode prefill corruption WITHOUT the model or a 33-minute run.
//
// Isolation established (see INVESTIGATION_LOG_metal_moe_race.md): corruption <=> dense
// GEMM on GPU (`coli_metal_gemm`), independent of MoE. The engine's existing kernel tests
// only exercise S<=64, which is why this fault was never caught. The trigger is the DISPATCH
// GRID (NT=S*O), not S alone: only the largest-O shape here (kv_b, O=28672) crosses the device
// limit, and only at S=7478 (~5.4e7 tg) -- it is still clean at S=4376 (~3.1e7 tg), and the
// smaller-O shapes (gate/up, down) stay clean at every S in the sweep.
//
// This sweeps S across the clean/corrupt bracket for a few real GLM matmul shapes,
// compares GPU vs CPU reference, prints the first diverging (row,col), AND runs the GPU
// twice to report determinism (race vs fixed compute error).
//
// Build/run:  make gemm-test
#include "../backend_metal.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

enum { I8=1, I4=2 };

// CPU reference: dequant->f32 MAC * per-row scale. OMP-parallel over output cols so the
// big shapes stay fast.
static void cpu_ref(int fmt, const void *W, const float *s, const float *x,
                    float *y, int S, int I, int O) {
  const int8_t *q8 = (const int8_t*)W; const uint8_t *q4 = (const uint8_t*)W;
  int rb4=(I+1)/2;
  #pragma omp parallel for schedule(static)
  for (int o=0;o<O;o++) for (int si=0;si<S;si++){
    const float *xr = x + (size_t)si*I; float acc=0;
    for (int i=0;i<I;i++){
      float w;
      if (fmt==I8) w=(float)q8[(size_t)o*I+i];
      else { uint8_t b=q4[(size_t)o*rb4+(i>>1)]; int v=(i&1)?(b>>4):(b&0xF); w=(float)(v-8); }
      acc += w*xr[i];
    }
    y[(size_t)si*O+o]=acc*s[o];
  }
}

static size_t roundpg(size_t n){ size_t p=16384; return ((n+p-1)/p)*p; }

// One (fmt,O,I,S) case: GPU vs CPU + GPU-vs-GPU determinism. Returns 0 = clean.
static int run_gemm(int fmt, int O, int I, int S, const char *name) {
  int rb4=(I+1)/2;
  size_t wn = (fmt==I8)?(size_t)O*I : (size_t)O*rb4;
  size_t wb = roundpg(wn), sb = roundpg((size_t)O*4);
  uint8_t *W=nullptr; float *Sc=nullptr;
  posix_memalign((void**)&W,16384,wb); posix_memalign((void**)&Sc,16384,sb);
  srand(1234 + S*7 + fmt);
  for(size_t i=0;i<wn;i++) W[i]=(uint8_t)((fmt==I8)?((rand()%255)-127):(rand()&0xFF));
  for(int i=0;i<O;i++) Sc[i]=0.01f+(rand()%50)/50000.f;
  coli_metal_register(W,wb); coli_metal_register(Sc,sb);

  std::vector<float> x((size_t)S*I), yr((size_t)S*O), yg((size_t)S*O), yg2((size_t)S*O);
  for(auto&v:x) v=((rand()%2000)-1000)/1000.f;

  cpu_ref(fmt, W, Sc, x.data(), yr.data(), S, I, O);
  int ok1 = coli_metal_gemm(yg.data(),  x.data(), W, Sc, fmt, S, I, O);
  int ok2 = coli_metal_gemm(yg2.data(), x.data(), W, Sc, fmt, S, I, O);

  // GPU vs CPU: max normalized error + first diverging element.
  double maxabs=0, ymax=0; long badidx=-1;
  for(size_t i=0;i<(size_t)S*O;i++){
    double d=fabs((double)yg[i]-yr[i]);
    if(d>maxabs) maxabs=d;
    if(fabs(yr[i])>ymax) ymax=fabs(yr[i]);
    if(badidx<0 && d > 1e-2*(fabs(yr[i])+1e-6)) badidx=(long)i;
  }
  double nerr = maxabs/(ymax+1e-9);
  // GPU vs GPU: determinism.
  long detdiff=0; for(size_t i=0;i<(size_t)S*O;i++) if(yg[i]!=yg2[i]) detdiff++;

  int clean = ok1 && ok2 && nerr < 1e-3;
  printf("  %-26s S=%-5d O=%-6d I=%-5d  nerr=%.2e  det:%s  %s",
         name, S, O, I, nerr,
         detdiff? "RACE(diff)" : "same",
         clean? "ok" : "*** CORRUPT");
  if(badidx>=0) printf("  first-bad row=%ld col=%ld (gpu=%.4g cpu=%.4g)",
                       badidx/O, badidx%O, (double)yg[badidx], (double)yr[badidx]);
  if(detdiff) printf("  [%ld/%zu gpu elems differ run-to-run]", detdiff, (size_t)S*O);
  printf("\n");

  coli_metal_unregister(W); coli_metal_unregister(Sc); free(W); free(Sc);
  return clean?0:1;
}

int main(void){
  if(!coli_metal_init()){ printf("Metal unavailable (skipping)\n"); return 0; }
  printf("coli_metal_gemm large-batch sweep (GPU vs CPU, + determinism):\n");
  int fail=0;
  // Real GLM-5.2 matmul shapes. kv_b (O=28672,I=512) has the largest S*O.
  struct { int O,I; const char*n; } shapes[] = {
    { 2048, 6144, "gate/up (O2048,I6144)" },
    { 6144, 2048, "down    (O6144,I2048)" },
    { 28672, 512, "kv_b    (O28672,I512)" },
  };
  int Ss[] = { 512, 2153, 4376, 7478 };   // control / clean / clean, just under the kv_b cliff / CORRUPT for kv_b
  for(auto &sh : shapes){
    for(int S : Ss) fail |= run_gemm(I4, sh.O, sh.I, S, sh.n);
    printf("\n");
  }
  printf(fail? "GEMM large-batch: CORRUPTION DETECTED\n" : "GEMM large-batch: all clean\n");
  coli_metal_shutdown();
  return fail;
}
