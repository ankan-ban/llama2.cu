#pragma once

// copied from AWQ's implementation
__device__ uint4 dequantize_s4_to_fp16x2(uint32_t const& source)
{
    uint4 result;

    uint32_t* h = reinterpret_cast<uint32_t*>(&result);
    uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
    static constexpr uint32_t TOP_MASK = 0x00f000f0;
    static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

    // Note that the entire sequence only requires 1 shift instruction. This is thanks to the register packing
    // format and the fact that we force our integers to be unsigned, and account for this in the fp16 subtractions.
    // In addition, I exploit the fact that sub and fma have the same throughput in order to convert elt_23 and
    // elt_67 to fp16 without having to shift them to the bottom bits before hand.

    // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we issue
    // immediately before required.
    const uint32_t top_i4s = i4s >> 8;
    // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[0])
        : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[1])
        : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[2])
        : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(h[3])
        : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    // I use inline PTX below because I am not sure if the compiler will emit float2half instructions if I use the
    // half2 ctor. In this case, I chose performance reliability over code readability.

    // This is the half2 {1032, 1032} represented as an integer.
    // static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
    // Haotian: subtract {1024, 1024} instead, we do not need to map to [-8, 7]
    static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
    // This is the half2 {1 / 16, 1 / 16} represented as an integer.
    static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
    // This is the half2 {-72, -72} represented as an integer.
    // static constexpr uint32_t NEG_72 = 0xd480d480;
    // Haotian: Let's use {-64, -64}.
    static constexpr uint32_t NEG_64 = 0xd400d400;

    // Finally, we construct the output numbers.
    // Convert elt_01
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
    // Convert elt_23
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
    // Convert elt_45
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
    // Convert elt_67
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));

    return result;
}


// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
    unsigned v0 = *((unsigned short*)&x);
    unsigned v1 = *((unsigned short*)&y);
    return (v1 << 16) | v0;
}

__global__ void __launch_bounds__(64) gemm_forward_4bit_cuda_m16n128k32(int split_k_iters, half* __restrict__ A, int* __restrict__ B, half* __restrict__ scaling_factors, int* __restrict__ zeros, int M, int IC, int OC, half* __restrict__ C)
{
    static constexpr uint32_t ZERO = 0x0;
    float C_warp[32];
    __shared__ half A_shared[16 * (32 + 8)];
    __shared__ half B_shared[32 * (128 + 8)];

    __shared__ half scaling_factors_shared[128];
    __shared__ half zeros_shared[128];

    int j_factors1 = ((OC + 128 - 1) / 128);

    int blockIdx_x = 0;
    int blockIdx_y = blockIdx.x % ((M + 16 - 1) / 16 * j_factors1);
    int blockIdx_z = blockIdx.x / ((M + 16 - 1) / 16 * j_factors1);

    half A_shared_warp[8];
    half B_shared_warp[32];
    for (int j_0_4_init = 0; j_0_4_init < 4; ++j_0_4_init) {
        for (int i = 0; i < 8; ++i) {
            C_warp[(j_0_4_init * 8) + i] = 0.0;
        }
    }

    static constexpr int row_stride_warp = 32 * 8 / 32;
    static constexpr int row_stride = 2 * 32 * 8 / 128;
    bool ld_zero_flag = (threadIdx.y * 32 + threadIdx.x) * 8 < 128;
    // TODO: Haotian: blockIdx_y / j_factors1 in A loading to support bsz > 16
    bool ld_A_flag = (blockIdx_y / j_factors1 * 16 + threadIdx.y * row_stride_warp + threadIdx.x * 8 / 32) < M;     // threadIdx.y is warp_id
    // bool wb_C_flag = (threadIdx.x / 4) < M;

    half* A_ptr = A
        + (((int)blockIdx_y) / j_factors1 * 16 + (((int)threadIdx.y) * row_stride_warp) + ((int)threadIdx.x) / (32 / 8)) * IC
        + (((int)threadIdx.x) % (32 / 8)) * 8;

    int* B_ptr = B
        + ((int)threadIdx.y) * (OC / 8) * 2
        + (((int)threadIdx.x) / (128 / 8)) * (OC / 8)
        + (((int)blockIdx_y) % j_factors1) * (128 / 8)
        + (((int)threadIdx.x) % (128 / 8)) * 1;

    half* A_shared_ptr = A_shared
        + ((int)threadIdx.y) * row_stride_warp * (32 + 8)
        + (((int)threadIdx.x) / (32 / 8)) * (32 + 8)
        + (((int)threadIdx.x) % (32 / 8)) * 8;

    half* B_shared_ptr = B_shared
        + ((int)threadIdx.y) * (row_stride / 2) * (128 + 8)
        + (((int)threadIdx.x) / (128 / 8)) * (128 + 8)
        + (((int)threadIdx.x) % (128 / 8)) * 8;

    int* zeros_ptr = zeros
        + (((int)blockIdx_y) % j_factors1) * (128 / 8)
        + ((int)threadIdx.x) % (128 / 8);

    half* scaling_factors_ptr = scaling_factors
        + (((int)blockIdx_y) % j_factors1) * (128)
        + (((int)threadIdx.x) % (128 / 8)) * 8;

    half* C_ptr = C
        + blockIdx_z * M * OC        // blockIdz.x -> split_k dim
        + (((int)blockIdx_y) % j_factors1) * 128
        + ((int)threadIdx.y) * 64
        + (((int)threadIdx.x) % 4) * 2;

    // preload s.f. and zeros
    int k_bound = (IC / 32 + split_k_iters - 1) / split_k_iters;
    if ((k_bound - 1) * 32 + blockIdx_z >= IC) k_bound -= 1;
    for (int _k_0_0 = 0; _k_0_0 < k_bound; ++_k_0_0) {
        int k_0_0 = _k_0_0 * split_k_iters + blockIdx_z;
        __syncthreads();
        // TODO: Haotian: blockIdx_y / j_factors1 in A loading to support bsz > 16
        if (ld_A_flag)
        {
            *(uint4*)(A_shared_ptr) = *(uint4*)(A_ptr + (k_0_0 * 32));
        }
        else
        {
            *(uint4*)(A_shared_ptr) = make_uint4(0, 0, 0, 0);
        }

        // for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 2; ++ax0_ax1_fused_0) {
        uint32_t zeros_loaded = *(uint32_t*)(zeros_ptr + k_0_0 * 32 / 128 * (OC / 8));
        uint4 B_loaded_zero = dequantize_s4_to_fp16x2(zeros_loaded);
        uint4 B_loaded_scale = *(uint4*)(scaling_factors_ptr + k_0_0 * 32 / 128 * (OC));
        /*
        if (blockIdx_z == 0 && blockIdx_y == 0 && k_0_0 == 0 && threadIdx.x == 0 && threadIdx.y == 0){
          printf("%x %x %x %x %x %x %x %x\n", B_loaded_scale.x, B_loaded_scale.y, B_loaded_scale.z, B_loaded_scale.w, B_loaded_zero.x, B_loaded_zero.y, B_loaded_zero.z, B_loaded_zero.w);
        }
        */
        // uint4 B_loaded_scale = make_uint4(0, 0, 0, 0);
        int* B_ptr_local = B_ptr + k_0_0 * 32 * (OC / 8);

        for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 8; ++ax0_ax1_fused_0) {

            // B: 32 x 136 (128+8) float16
            // each warp: 32 x 4
            // each thr: read 32 bit -> convert to 8xFP16 (a UINT4) -> scale and minus zero -> WB UINT4
            // *(uint4*)(B_shared + ((((ax0_ax1_fused_0 * 544) + (((int)threadIdx.y) * 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8))) = *(uint4*)(B + ((((((k_0_0 * 163840) + (ax0_ax1_fused_0 * 20480)) + (((int)threadIdx.y) * 10240)) + ((((int)threadIdx.x) >> 4) * 5120)) + (((int)blockIdx_y) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
            // row stride in shared memory: (NWARPS * 32 * 8 / cta_N) 
            uint32_t B_loaded = *(uint32_t*)(B_ptr_local + ax0_ax1_fused_0 * row_stride * (OC / 8));
            uint4 B_loaded_fp16 = dequantize_s4_to_fp16x2(B_loaded);
            //uint4 B_loaded_zero = *(uint4*)(zeros_shared + (threadIdx.x % (cta_N / 8)) * 8);

            // uint4 B_loaded_scale = *(uint4*)(scaling_factors_shared + (threadIdx.x % (cta_N / 8)) * 8);
            // - zero and * scale
            // TODO (Haotian): can save 4 assembly instructions if sormulate as deq = q * scale - zero * scale.
            asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_zero.x));
            asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.x) : "r"(B_loaded_fp16.x), "r"(B_loaded_scale.x), "r"(ZERO));
            asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_zero.y));
            asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.y) : "r"(B_loaded_fp16.y), "r"(B_loaded_scale.y), "r"(ZERO));
            asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_zero.z));
            asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.z) : "r"(B_loaded_fp16.z), "r"(B_loaded_scale.z), "r"(ZERO));
            asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_zero.w));
            asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(B_loaded_fp16.w) : "r"(B_loaded_fp16.w), "r"(B_loaded_scale.w), "r"(ZERO));
            /*
            if (ax0_ax1_fused_0 == 0 && blockIdx_z == 0 && blockIdx_y == 0 && k_0_0 == 0 && threadIdx.x == 17 && threadIdx.y == 0){
              printf("[x] %X %X %X %X\n", B_loaded_fp16.x, B_loaded_fp16.y, B_loaded_fp16.z, B_loaded_fp16.w);
            }
            */

            // write back
            *(uint4*)(B_shared_ptr + ax0_ax1_fused_0 * row_stride * (128 + 8)) = B_loaded_fp16;
        }
        __syncthreads();

        for (int k_0_1 = 0; k_0_1 < 2; ++k_0_1) {
            {
                unsigned int addr;
                asm volatile (
                    "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                    : "=r"(addr)
                    : "l"((void*)((&(A_shared[(k_0_1 * 16)])) + (((((int)threadIdx.x) & 15) * 40) + ((((int)threadIdx.x) >> 4) * 8))))
                    );


                asm volatile (
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
                    "{%0, %1, %2, %3}, [%4];\n"
                    : "=r"(((unsigned*)(A_shared_warp + 0))[0]), "=r"(((unsigned*)(A_shared_warp + 0))[1]), "=r"(((unsigned*)(A_shared_warp + 0))[2]), "=r"(((unsigned*)(A_shared_warp + 0))[3])
                    : "r"(addr)
                    );
            }

            for (int ax1_0 = 0; ax1_0 < 4; ++ax1_0) {
                {
                    unsigned int addr;
                    asm volatile (
                        "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
                        : "=r"(addr)
                        : "l"((void*)((&(B_shared[(((k_0_1 * 2176) + (((int)threadIdx.y) * 64)) + (ax1_0 * 16))])) + (((((int)threadIdx.x) & 15) * 136) + ((((int)threadIdx.x) >> 4) * 8))))
                        );
                    asm volatile (
                        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
                        "{%0, %1, %2, %3}, [%4];\n"
                        : "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[0]), "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[1]), "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[2]), "=r"(((unsigned*)(B_shared_warp + (ax1_0 * 8)))[3])
                        : "r"(addr)
                        );
                }
            }
            for (int j_0_4 = 0; j_0_4 < 4; ++j_0_4) {
                {
                    asm volatile (
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        :  "=f"(((float*)(C_warp + (j_0_4 * 8)))[0]), "=f"(((float*)(C_warp + (j_0_4 * 8)))[1]), "=f"(((float*)(C_warp + (j_0_4 * 8)))[2]), "=f"(((float*)(C_warp + (j_0_4 * 8)))[3])
                        : "r"(((unsigned*)(A_shared_warp + 0))[0]), "r"(((unsigned*)(A_shared_warp + 0))[1]), "r"(((unsigned*)(A_shared_warp + 0))[2]), "r"(((unsigned*)(A_shared_warp + 0))[3]), "r"(((unsigned*)(B_shared_warp + (j_0_4 * 8)))[0]), "r"(((unsigned*)(B_shared_warp + (j_0_4 * 8)))[1]), "f"(((float*)(C_warp + (j_0_4 * 8)))[0]), "f"(((float*)(C_warp + (j_0_4 * 8)))[1]), "f"(((float*)(C_warp + (j_0_4 * 8)))[2]), "f"(((float*)(C_warp + (j_0_4 * 8)))[3]));
                }

                {
                    asm volatile (
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        :  "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]), "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]), "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]), "=f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3])
                        : "r"(((unsigned*)(A_shared_warp + 0))[0]), "r"(((unsigned*)(A_shared_warp + 0))[1]), "r"(((unsigned*)(A_shared_warp + 0))[2]), "r"(((unsigned*)(A_shared_warp + 0))[3]), "r"(((unsigned*)(B_shared_warp + ((j_0_4 * 8) + 4)))[0]), "r"(((unsigned*)(B_shared_warp + ((j_0_4 * 8) + 4)))[1]), "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[0]), "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[1]), "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[2]), "f"(((float*)(C_warp + ((j_0_4 * 8) + 4)))[3]));
                }
            }
        }
    }

    // TODO: Shang: Hoist loop invariance.
    for (int ax1_0_1 = 0; ax1_0_1 < 4; ++ax1_0_1) {
        for (int local_id = 0; local_id < 8; ++local_id) {
            int row_offset = (((int)blockIdx_y) / j_factors1) * 16 + ((int)threadIdx.x) / 4 + (local_id % 4) / 2 * 8;
            if (row_offset < M)
            {
                *(C_ptr + ax1_0_1 * 16 + row_offset * OC + (local_id / 4) * 8 + local_id % 2) = __float2half(C_warp[(ax1_0_1 * 8) + local_id]);
            }
        }
    }
}
