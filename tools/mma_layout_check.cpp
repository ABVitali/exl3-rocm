// CPU emulation: does "per-lane FMA + 4-lane xor reduce" reproduce row 0 of
// mma.sync.m16n8k16 for the exact fragment layout run_gemv_tile uses?
// Fragment layouts per PTX ISA "Matrix Fragments for mma.m16n8k16 with .f16":
//   A (row): lane l holds A[l/4][(l%4)*2+{0,1}] (a0), A[l/4+8][...] (a1),
//            A[l/4][(l%4)*2+8+{0,1}] (a2), A[l/4+8][+8..] (a3)
//   B (col): lane l holds B[(l%4)*2+{0,1}][l/4] (b0), B[(l%4)*2+8+{0,1}][l/4] (b1)
//   C:       lane l holds C[l/4][(l%4)*2+{0,1}] (c0,c1), C[l/4+8][...] (c2,c3)
// run_gemv_tile sets a1=a3=0 and only lanes 0-3 carry A (row 0), so M=1.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <array>
int main() {
    srand(7);
    int bad = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        float A[16]; float B[16][8];
        for (int k = 0; k < 16; ++k) A[k] = (rand() % 17) - 8;
        for (int k = 0; k < 16; ++k) for (int n = 0; n < 8; ++n) B[k][n] = (rand() % 17) - 8;
        // reference: row 0 of A(16x16, rows 1..15 zero) x B(16x8)
        float Cref[8];
        for (int n = 0; n < 8; ++n) { float s = 0; for (int k = 0; k < 16; ++k) s += A[k] * B[k][n]; Cref[n] = s; }
        // per-lane B fragment exactly as dq8 emits it (b0 = frag[0], b1 = frag[1])
        float b0[32][2], b1[32][2], partial[32];
        for (int l = 0; l < 32; ++l) {
            int n = l / 4, k0 = (l % 4) * 2;
            b0[l][0] = B[k0][n];     b0[l][1] = B[k0 + 1][n];
            b1[l][0] = B[k0 + 8][n]; b1[l][1] = B[k0 + 9][n];
            // FMA form: EVERY lane loads A at its own (l&3) slot: A2[slice*8 + (lane&3)] and +4
            float a0x = A[k0], a0y = A[k0 + 1], a2x = A[k0 + 8], a2y = A[k0 + 9];
            partial[l] = a0x * b0[l][0] + a0y * b0[l][1] + a2x * b1[l][0] + a2y * b1[l][1];
        }
        // xor-reduce within aligned groups of 4 lanes (xor 1, then xor 2)
        float r1[32], r2[32];
        for (int l = 0; l < 32; ++l) r1[l] = partial[l] + partial[l ^ 1];
        for (int l = 0; l < 32; ++l) r2[l] = r1[l] + r1[l ^ 2];
        // lane 4n (any lane of the group) now holds C[0][n]
        for (int n = 0; n < 8; ++n) {
            float got = r2[4 * n];
            if (std::fabs(got - Cref[n]) > 1e-3f) { ++bad; if (bad < 5) printf("trial %d n %d got %g want %g\n", trial, n, got, Cref[n]); }
            // and every lane in the group agrees
            for (int j = 1; j < 4; ++j) if (r2[4*n+j] != got) { ++bad; }
        }
        // cross-check the C->sh_red mapping used by the ORIGINAL code: lanes 0-3 hold c0,c1 = C[0][(l%4)*2 +{0,1}]
        // (so the FMA version must write col n from lane 4n instead). Nothing to compute; documented.
    }
    printf("%s: %d mismatches over 2000 trials\n", bad ? "FAIL" : "OK", bad);
    return bad != 0;
}
