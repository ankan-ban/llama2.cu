/*
Inference for Llama-2 Transformer model in pure Cuda.

### INT4 - AWQ quantization version ###

1. First generate AWQ int-4 quantized weights following steps in https://github.com/mit-han-lab/llm-awq
 E.g:
  python -m awq.entry --model_path /path-to-model/Llama-2-7b-chat-hf --w_bit 4 --q_group_size 128 --run_awq --dump_awq awq_cache/llama2-7b-chat-metadata.pt
  python -m awq.entry --model_path /path-to-model/Llama-2-7b-chat-hf --w_bit 4 --q_group_size 128 --load_awq awq_cache/llama2-7b-chat-metadata.pt --q_backend real --dump_quant awq_weights/llama2-7b-awq.pt
 Note - AWQ scripts doesn't run on Windows. Use Linux or WSL.

2. Convert AWQ weights into individual weight binary files using convert_awq_to_bin.py

3. Run this program pointing to the directory containing the binary weight files.

*/

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

// ----------------------------------------------------------------------------
// GPU kernels

__global__ void element_wise_add_kernel(half* dest, half* src, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        dest[i] = (half)((float)dest[i] + (float)src[i]);
}

__global__ void convert_fp32_to_fp16(half* out, float* in, int elements) {
    int index = blockIdx.x * 256 + threadIdx.x;
    if (index < elements)
        out[index] = (half)in[index];
}

__global__ void convert_fp16_to_fp32(float* out, half* in, int elements) {
    int index = blockIdx.x * 256 + threadIdx.x;
    if (index < elements)
        out[index] = (float)in[index];
}

// Single block - not enough parallelism for the GPU, but it's just 1% of total time
__global__ void rmsnorm_kernel(half* o, half* x, half* weight, int size, int elementsPerThread) {
    float ss = 0.0f;
    for (int i = 0; i < elementsPerThread; i++) {
        int index = threadIdx.x + i * 1024;
        if (index < size) {
            float val = (float)x[index];
            ss += val * val;
        }
    }

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    ss = BlockReduce(temp).Sum(ss);

    __shared__ float shared_ss;
    if (threadIdx.x == 0) {
        ss /= size;
        ss += 1e-5f;      // Ankan - test!
        ss = 1.0f / sqrtf(ss);
        shared_ss = ss;
    }
    __syncthreads();
    ss = shared_ss;

    // normalize
    for (int i = 0; i < elementsPerThread; i++) {
        int index = threadIdx.x + i * 1024;
        if (index < size) {
            float val = (float)x[index];
            val *= ss * (float)weight[index];
            o[index] = (half)val;
        }
    }
}


// Note that ~95% of total time is spent here, so optimizing this is important
// 1. One output generated per warp so that we can parallelize the dot product across the warp
// 2. We load 8 elements at a time for efficiency (assume dimensions to be multiple of 8)
__global__ void mat_vec_kernel(half* op, const half* ip, const half* wt, int n, int d, int numSerialLoads, 
    int ip_stride, int w_stride, int op_stride, int w_row_stride, float alpha) {
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= d)
        return;
    const half* __restrict__ input = ip + blockIdx.y * ip_stride;
    const half* __restrict__ weight = wt + blockIdx.y * w_stride;
    half* output = op + blockIdx.y * op_stride;

    float sum = 0;

    for (int i = 0; i < numSerialLoads; i++) {
        int j = (i * 32 + threadIdx.x) * 8;
        if (j < n) {
            half w[8];
            half ip[8];
            *((uint4 *)(&w)) = *((uint4 *)(&weight[index * w_row_stride + j]));
            *((uint4 *)(&ip)) = *((uint4 *)(&input[j]));
            for (int el = 0; el < 8; el++)
                sum += float(w[el]) * float(ip[el]);
        }
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);
    sum *= alpha;

    if (threadIdx.x == 0)
        output[index] = (half)sum;
}

// Simpler version of the above - to handle non multiple of 8 dimensions too (needed for MHA block)
__global__ void mat_vec_kernel_simple(half* op, half* ip, half* wt, int n, int d, int numSerialElements,
    int ip_stride, int w_stride, int op_stride, int w_row_stride, float alpha) {

    const half* __restrict__ input = ip + blockIdx.y * ip_stride;
    const half* __restrict__ weight = wt + blockIdx.y * w_stride;
    half* output = op + blockIdx.y * op_stride;

    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= d)
        return;

    float sum = 0;
    for (int i = 0; i < numSerialElements; i++) {
        int j = i * 32 + threadIdx.x;
        if (j < n)
            sum += ((float)weight[index * w_row_stride + j]) * ((float)input[j]);
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);
    sum *= alpha;

    if (threadIdx.x == 0)
        output[index] = (half)sum;
}

#include "awq_kernel.h"

__device__ __host__ float sint4tofloat(uint32_t x)
{
    //uint8_t last4 = x & 0x0F;
    //int8_t s4 = (last4 ^ 0x08) - (last4 & 0x08);
    //int32_t s32 = s4;
    return float(x & 0x0F);
}


__global__ void mat_vec_kernel_int4_simple(half* __restrict__ output, const half* __restrict__ input,
    const uint32_t* __restrict__ q_weight, const uint32_t* __restrict__ q_zeros, const half* __restrict__ scales,
    int n, int d, int group_size, half *deq_weights)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= d)
        return;

    int groupIndex = index / group_size;

    float sum = 0;
    for (int k = 0; k < n; k += 8)
    {
        half sc[8];
        half ip[8];
        *((uint4*)(&ip)) = *((uint4*)(&input[k]));
        *((uint4*)(&sc)) = *((uint4*)(&scales[groupIndex * n + k]));

        uint32_t packed_q_wt = q_weight[(index * n + k) / 8];
        uint32_t packed_q_z = q_zeros[(groupIndex * n + k) / 8];

        for (int i = 0; i < 8; i++) {
            float q_wt = sint4tofloat(packed_q_wt); packed_q_wt >>= 4;
            float q_z = sint4tofloat(packed_q_z); packed_q_z >>= 4;
            float scale = (float)sc[i];
            float w = (q_wt - q_z) * scale;
            //if (index == 0)
            //{
            //    printf("%10.6f", w);
            //}

            sum += w * float(ip[i]);
            //w = deq_weights[index * n + k+i];

            sum += w * float(input[k+i]);
        }

        //if ((index == 0) && (k % 16 == 8))
        //    printf("\n");

    }

    output[index] = sum;

}

__global__ void mat_vec_kernel_int4(half* __restrict__ output, const half* __restrict__ input, 
    const uint32_t* __restrict__ q_weight, const uint32_t* __restrict__ q_zeros, const half* __restrict__ scales,
    int n, int d, int numSerialLoads, int group_size)
{
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= d)
        return;

    int groupIndex = index / group_size;        // can replace with index & (num_groups - 1) for power of 2 num_groups

    float sum = 0;

    for (int i = 0; i < numSerialLoads; i++) {
        int j = (i * 32 + threadIdx.x) * 8;
        if (j < n) {
            half sc[8];
            half ip[8];
            *((uint4*)(&ip)) = *((uint4*)(&input[j]));
            *((uint4*)(&sc)) = *((uint4*)(&scales[groupIndex * n + j]));

            uint32_t packed_q_wt = q_weight[(index * n + j) / 8];
            uint32_t packed_q_z = q_zeros[(groupIndex * n + j) / 8];

            for (int i = 0; i < 8; i++) {
                float q_wt = sint4tofloat(packed_q_wt); packed_q_wt >>= 4;
                float q_z = sint4tofloat(packed_q_z); packed_q_z >>= 4;
                float scale = (float)sc[i];
                float w = (q_wt - q_z) * scale;
                sum += w * float(ip[i]);
            }
        }
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);

    if (threadIdx.x == 0)
        output[index] = (half)sum;
}

void deq_weights_cpu(half* op, const uint32_t* q_weight, const uint32_t* q_zeros, const half* scales, int height, int width)
{
    for (int y = 0; y < height; y++) {
        int group_y = y / 128;      // assume group size of 128
        for (int x = 0; x < width; x += 8) {
            uint32_t packed_q_wt = q_weight[(y * width + x) / 8];
            uint32_t packed_q_z = q_zeros[(group_y * width + x) / 8];
             
            float q[8];
            float z[8];
            int order_map[] = { 0, 2, 4, 6, 1, 3, 5, 7 };
            for (int i = 0; i < 8; i++) {
                float q_wt = (float)(packed_q_wt & 0xF);
                float q_z = (float)(packed_q_z & 0xF);
                q[order_map[i]] = q_wt;
                z[order_map[i]] = q_z;
                packed_q_wt = (packed_q_wt >> 4);   // go to next 4-bits
                packed_q_z = (packed_q_z >> 4);
            }


            for (int i = 0; i < 8; i++) {
                float scale = (float)scales[(group_y * width + x + i)];
                float w = (q[i] - z[i]) * scale;
                //op[y * width + x + i] = (half)w;        // Ankan - TODO: transpose here for testing
                op[(x + i) * height + y] = (half)w;
            }
        }
    }
}

__global__ void deq_weights(half* op, const uint32_t* q_weight, const uint32_t* q_zeros, const half* scales, int width, int height)
{
    // each thread de-quantizes 8 weights
    int x = (blockIdx.x * 16 + threadIdx.x) * 8;
    int y = blockIdx.y * 16 + threadIdx.y;

    int y_g = y / 128;

    if (x >= width || y >= height)
        return;

    int index = y * width + x;
    int index_g = y_g * width + x;

    uint32_t packed_q_wt = q_weight[index / 8];
    uint32_t packed_q_z = q_zeros[index_g / 8];

    half sc[8];
    *((uint4*)(sc)) = *((uint4*)(&scales[index_g]));
    half out[8];
    for (int i = 0; i < 8; i++) {
        float q_wt = sint4tofloat(packed_q_wt); packed_q_wt >>= 4;
        float q_z = sint4tofloat(packed_q_z); packed_q_z >>= 4;
        float scale = (float)sc[i];
        float w = (q_wt - q_z) * scale;
        out[i] = (half)w;
    }

    *((uint4*)(&op[index])) = *((uint4*)out);
}


// Here we make use of shared memory to achieve better memory access pattern, and transpose a 32x32 chunk of the matrix on the fly
__global__ void vec_mat_kernel(half* op, const half* __restrict__ ip, const half* __restrict__ wt, int N, int K, int elementsPerThread,
    int ip_stride, int w_stride, int op_stride, int w_row_stride) {

    const half* __restrict__ input = ip + blockIdx.y * ip_stride;
    const half* __restrict__ weight = wt + blockIdx.y * w_stride;
    half* output = op + blockIdx.y * op_stride;

    int start_n = blockIdx.x * 32;
    int i = start_n + threadIdx.y;

    // 2x for double buffering
    // +2 to avoid shared memory bank conflicts
    __shared__ half loaded_fragment[2][32][32 + 2];

    // OOB check
    if (i >= N)
        return;

    // load the first 32x32 fragment
    int n = start_n + threadIdx.x;
    int k = threadIdx.y;
    int offset = k * w_row_stride + n;
    loaded_fragment[0][threadIdx.y][threadIdx.x] = ((n < N) && (k < K)) ? weight[offset] : 0;

    float sum = 0;
    // Loop over the matrix row and vector elements
    for (int e = 0; e < elementsPerThread;)
    {
        __syncthreads();    // wait for the load

        int start_k = e * 32;
        k = start_k + threadIdx.x;
        int buf_i = e & 1;
        sum += float(loaded_fragment[buf_i][threadIdx.x][threadIdx.y]) * ((k < K) ? (float) input[k] : 0);

        // load for the next iteration
        e++;
        start_k = e * 32;
        buf_i = e & 1;
        n = start_n + threadIdx.x;
        k = start_k + threadIdx.y;
        int offset = k * w_row_stride + n;
        loaded_fragment[buf_i][threadIdx.y][threadIdx.x] = ((n < N) && (k < K)) ? weight[offset] : 0;
    }

    using WarpReduce = cub::WarpReduce<float>;
    __shared__ typename WarpReduce::TempStorage temp;
    sum = WarpReduce(temp).Sum(sum);

    if (threadIdx.x == 0)
        output[i] = (half)sum;
}

// Each block processes a single head
__global__ void RoPERotation_kernel(half* sq, half* sk, float* f_real, float* f_imag, int num_heads, int head_size) {
    int h = blockIdx.x;
    half* q = sq + h * head_size;
    half* k = sk + h * head_size;

    int i = threadIdx.x;
    float q0 = q[i];
    float q1 = q[i + head_size/2];
    float k0 = k[i];
    float k1 = k[i + head_size / 2];
    float fcr = f_real[i];
    float fci = f_imag[i];
    q[i] = q0 * fcr - q1 * fci;
    q[i + head_size / 2] = q0 * fci + q1 * fcr;
    k[i] = k0 * fcr - k1 * fci;
    k[i + head_size / 2] = k0 * fci + k1 * fcr;
}

#define MAX_SEQ_LEN 8192
__global__ void softmax_kernel(half* __restrict__ arr, int num_heads, int size) {
    __shared__ float att[MAX_SEQ_LEN];
    int h = blockIdx.x;
    int tid = threadIdx.x;
    int step = blockDim.x;

    // load input to shared memory
    for (int t = tid; t < size; t += step)
        att[t] = (float) arr[h * size + t];
    __syncthreads();

    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    // find max value (for numerical stability)
    float max_val = tid < size ? att[tid] : 0;
    for (int i = tid + step; i < size; i += step)
        if (att[i] > max_val)
            max_val = att[i];

    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = max_val;
    __syncthreads();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        att[i] = expf(att[i] - max_val);
        sum += att[i];
    }

    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0)
        shared_val = sum;
    __syncthreads();
    sum = shared_val;

    // normalize and write the result
    for (int t = tid; t < size; t += step)
        arr[h * size + t] = (half) (att[t] / sum);
}

__global__ void silu_element_wise_mul_kernel(half* dest, half* src, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = (float)dest[i];
        val *= 1.0f / (1.0f + expf(-val));
        val *= (float)src[i];
        dest[i] = (half)val;
    }
}

// ----------------------------------------------------------------------------
// Transformer and RunState structs, and related memory management

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

struct QWeight {
    uint32_t* weight;
    uint32_t* zeros;
    half* scales;
};

struct PerLayerWeight {
    half* rms_att_weight; // (layer, dim) rmsnorm weights
    half* rms_ffn_weight; // (layer, dim)
    /*
    half* wq; // (layer, dim, dim)
    half* wk; // (layer, dim, dim)
    half* wv; // (layer, dim, dim)
    half* wo; // (layer, dim, dim)
    */
    QWeight wq_q;
    QWeight wq_k;
    QWeight wq_v;
    QWeight wq_o;

    /*
    half* gate_proj; // (layer, hidden_dim, dim)
    half* up_proj;   // (layer, hidden_dim, dim)
    half* down_proj; // (layer, dim, hidden_dim)
    */

    QWeight wq_gate;
    QWeight wq_up;
    QWeight wq_down;
};

typedef struct {
    // token embedding table
    half* token_embedding_table;    // (vocab_size, dim)
    /*
    // weights for rmsnorms
    half* rms_att_weight; // (layer, dim) rmsnorm weights
    half* rms_ffn_weight; // (layer, dim)
    // weights for matmuls
    half* wq; // (layer, dim, dim)
    half* wk; // (layer, dim, dim)
    half* wv; // (layer, dim, dim)
    half* wo; // (layer, dim, dim)
    // weights for ffn
    half* w1; // (layer, hidden_dim, dim)
    half* w2; // (layer, dim, hidden_dim)
    half* w3; // (layer, hidden_dim, dim)
    */

    // final rmsnorm
    half* rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    float* freq_cis_real; // (seq_len, dim/2)
    float* freq_cis_imag; // (seq_len, dim/2)
    // (optional) classifier weights for the logits, on the last layer
    half* wcls;

    PerLayerWeight* layers;
    int num_layers;
} TransformerWeights;

typedef struct {
    // current wave of activations
    half* x; // activation at current time stamp (dim,)
    half* xb; // same, but inside a residual branch (dim,)
    half* xb2; // an additional buffer just for convenience (dim,)
    half* hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    half* hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    half* q; // query (dim,)
    half* att; // buffer for scores/attention values (n_heads, seq_len)
    half* logits_gpu; // output logits
    float* logits_temp; // logits in GPU memory converted to float
    float* logits; // logits copied CPU side
    // kv cache
    half* key_cache;   // (layer, seq_len, dim)
    half* value_cache; // (layer, seq_len, dim)
} RunState;

void malloc_run_state(RunState* s, Config* p) {
    cudaMalloc((void**)&s->x, p->dim * sizeof(half));
    cudaMalloc((void**)&s->xb, p->dim * sizeof(half));
    cudaMalloc((void**)&s->xb2, p->dim * sizeof(half));
    cudaMalloc((void**)&s->hb, p->hidden_dim * sizeof(half));
    cudaMalloc((void**)&s->hb2, p->hidden_dim * sizeof(half));
    cudaMalloc((void**)&s->q, p->dim * sizeof(half));
    cudaMalloc((void**)&s->att, p->n_heads * p->dim * sizeof(half));
    cudaMalloc((void**)&s->logits_gpu, p->vocab_size * sizeof(half));
    cudaMalloc((void**)&s->key_cache, p->n_layers * p->seq_len * p->dim * sizeof(half));    // potentially huge allocs
    cudaMalloc((void**)&s->value_cache, p->n_layers * p->seq_len * p->dim * sizeof(half));
    cudaMalloc((void**)&s->logits_temp, p->vocab_size * sizeof(float));
    s->logits = (float*)malloc(p->vocab_size * sizeof(float));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
        || !s->att || !s->logits || !s->key_cache
        || !s->value_cache || !s->logits_gpu || !s->logits_temp) {
        printf("malloc failed!\n");
        exit(1);
    }
}

void free_run_state(RunState* s) {
    cudaFree(s->x);
    cudaFree(s->xb);
    cudaFree(s->xb2);
    cudaFree(s->hb);
    cudaFree(s->hb2);
    cudaFree(s->q);
    cudaFree(s->att);
    cudaFree(s->logits_gpu);
    cudaFree(s->logits_temp);
    free(s->logits);
    cudaFree(s->key_cache);
    cudaFree(s->value_cache);
}

void allocQWeight(QWeight* pWeight, size_t size)
{
    cudaMalloc((void**)&pWeight->weight, size / 2);
    cudaMalloc((void**)&pWeight->zeros, size / (2 * 128));
    cudaMalloc((void**)&pWeight->scales, sizeof(half) * size / 128);
}

void freeQWeight(QWeight* pWeight)
{
    cudaFree(pWeight->weight);
    cudaFree(pWeight->zeros);
    cudaFree(pWeight->scales);
}

void malloc_weights(TransformerWeights* w, Config* p, int shared_weights) {
    cudaMalloc((void**)&w->token_embedding_table, p->vocab_size * p->dim * sizeof(half));

    /*
    cudaMalloc((void**)&w->rms_att_weight, p->n_layers * p->dim * sizeof(half));
    cudaMalloc((void**)&w->rms_ffn_weight, p->n_layers * p->dim * sizeof(half));
    cudaMalloc((void**)&w->wq, p->n_layers * p->dim * p->dim * sizeof(half));
    cudaMalloc((void**)&w->wk, p->n_layers * p->dim * p->dim * sizeof(half));
    cudaMalloc((void**)&w->wv, p->n_layers * p->dim * p->dim * sizeof(half));
    cudaMalloc((void**)&w->wo, p->n_layers * p->dim * p->dim * sizeof(half));
    cudaMalloc((void**)&w->w1, p->n_layers * p->hidden_dim * p->dim * sizeof(half));
    cudaMalloc((void**)&w->w2, p->n_layers * p->dim * p->hidden_dim * sizeof(half));
    cudaMalloc((void**)&w->w3, p->n_layers * p->hidden_dim * p->dim * sizeof(half));
    */
    w->layers = (PerLayerWeight*)malloc(p->n_layers * sizeof(PerLayerWeight));
    w->num_layers = p->n_layers;
    for (int l = 0; l < p->n_layers; l++)
    {
        PerLayerWeight* layer = &(w->layers[l]);
        cudaMalloc((void**)&layer->rms_att_weight,  p->dim * sizeof(half));
        cudaMalloc((void**)&layer->rms_ffn_weight,  p->dim * sizeof(half));

#if 0
        cudaMalloc((void**)&layer->wq,  p->dim * p->dim * sizeof(half));
        cudaMalloc((void**)&layer->wk,  p->dim * p->dim * sizeof(half));
        cudaMalloc((void**)&layer->wv,  p->dim * p->dim * sizeof(half));
        cudaMalloc((void**)&layer->wo,  p->dim * p->dim * sizeof(half));
        cudaMalloc((void**)&layer->gate_proj,  p->hidden_dim * p->dim * sizeof(half));
        cudaMalloc((void**)&layer->up_proj,  p->hidden_dim * p->dim * sizeof(half));
        cudaMalloc((void**)&layer->down_proj,  p->dim * p->hidden_dim * sizeof(half));
#endif
        allocQWeight(&layer->wq_q, p->dim * p->dim);
        allocQWeight(&layer->wq_k, p->dim * p->dim);
        allocQWeight(&layer->wq_v, p->dim * p->dim);
        allocQWeight(&layer->wq_o, p->dim * p->dim);
        allocQWeight(&layer->wq_gate, p->hidden_dim * p->dim);
        allocQWeight(&layer->wq_up, p->hidden_dim * p->dim);
        allocQWeight(&layer->wq_down, p->dim * p->hidden_dim);
    }

    cudaMalloc((void**)&w->rms_final_weight, p->dim * sizeof(half));
    int head_size = p->dim / p->n_heads;
    cudaMalloc((void**)&w->freq_cis_real, p->seq_len * head_size / 2 * sizeof(float));
    cudaMalloc((void**)&w->freq_cis_imag, p->seq_len * head_size / 2 * sizeof(float));

    if (shared_weights)
        w->wcls = w->token_embedding_table;
    else
        cudaMalloc((void**)&w->wcls, p->vocab_size * p->dim * sizeof(half));

    // ensure all mallocs went fine
    if (!w->token_embedding_table || !w->layers ||
        !w->rms_final_weight || !w->freq_cis_real || !w->freq_cis_imag || !w->wcls) {
        printf("malloc failed!\n");
        exit(1);
    }
}

void free_weights(TransformerWeights* w, int shared_weights) {
    cudaFree(w->token_embedding_table);
    cudaFree(w->rms_final_weight);
    cudaFree(w->freq_cis_real);
    cudaFree(w->freq_cis_imag);
    if (!shared_weights)
        cudaFree(w->wcls);

    for (int l = 0; l < w->num_layers; l++)
    {
        PerLayerWeight* layer = &(w->layers[l]);
        cudaFree(layer->rms_att_weight);
        cudaFree(layer->rms_ffn_weight);
/*
        cudaFree(layer->wq);
        cudaFree(layer->wk);
        cudaFree(layer->wv);
        cudaFree(layer->wo);
        cudaFree(layer->gate_proj);
        cudaFree(layer->up_proj);
        cudaFree(layer->down_proj);
        */
        freeQWeight(&layer->wq_q);
        freeQWeight(&layer->wq_k);
        freeQWeight(&layer->wq_v);
        freeQWeight(&layer->wq_o);
        freeQWeight(&layer->wq_gate);
        freeQWeight(&layer->wq_up);
        freeQWeight(&layer->wq_down);
    }
    free(w->layers);
}

int divUp(int a, int b) {
    return (a - 1) / b + 1;
}

int uploadWeight(void *w, int elements, FILE* f, void *scratchCpu, void *scratchGpu, bool real_upload = false, bool fp32 = false) {
    int count = fread(scratchCpu, sizeof(float), elements, f);
    if (count != elements) return 1;

    if (real_upload)    // otherwise just skip through the file
    {
        if (fp32)
        {
            cudaMemcpyAsync(w, scratchCpu, sizeof(float) * elements, cudaMemcpyHostToDevice);
        }
        else
        {
            // copy and convert fp32->fp16
            cudaMemcpyAsync(scratchGpu, scratchCpu, sizeof(float) * elements, cudaMemcpyHostToDevice);
            convert_fp32_to_fp16 << <divUp(elements, 256), 256 >> > ((half*)w, (float*)scratchGpu, elements);
        }
    }
    return 0;
}

void dumpTensor(char* message, half* tensor, int elements) {
    half* arr = (half*)malloc(elements * sizeof(half));
    cudaMemcpy(arr, tensor, elements * sizeof(half), cudaMemcpyDeviceToHost);

    printf("\n%s :- \n", message);
    for (int i = 0; i < elements; i++)
    {
        printf("%10.6f", (float)arr[i]);
        if (i % 16 == 15) printf("\n");
    }

    free(arr);
}


void getFileContents(void *buf, char* filename, size_t bytes)
{
    FILE* fp = fopen(filename, "rb");
    if (!fp) { printf("\nUnable to open %s\n", filename); exit(1); }
    if (fread(buf, 1, bytes, fp) != bytes) { printf("error reading weights from %s", filename);  exit(1); }
    fclose(fp);
}

// get a matrix by name and dequantize it
void getDeqWeightByName(void *scratchCpu, char *fileNameBase, char *weightName, size_t height, size_t width, bool save = false)
{
    uint32_t* qweight;
    uint32_t* qzeros;
    half* scales;

    qweight = (uint32_t*)malloc(height * width / 2);        // 4 bit per element
    qzeros = (uint32_t*)malloc(height * width / (2 * 128));
    scales = (half*)malloc(sizeof(half) * height * width / 128);

    char filename[512];
    sprintf(filename, "%s.%s.qweight.bin", fileNameBase, weightName);
    getFileContents(qweight, filename, height * width / 2);
    sprintf(filename, "%s.%s.qzeros.bin", fileNameBase, weightName);
    getFileContents(qzeros, filename, height * width / (2 * 128));
    sprintf(filename, "%s.%s.scales.bin", fileNameBase, weightName);
    getFileContents(scales, filename, sizeof(half) * height * width / 128);

    deq_weights_cpu((half*)scratchCpu, qweight, qzeros, scales, height, width);

    free(qweight);
    free(qzeros);
    free(scales);
}

#include <math.h>
// A function to precompute the frequencies and complex numbers for a given dimension, end, and theta
void precompute_freqs_cis(int dim, int end, float theta, float* freqs_re, float* freqs_im) {
    // Allocate memory for the freqs array
    float* freqs = (float*) malloc(dim / 2 * sizeof(float));
    // Compute the freqs values as 1 / (theta ^ (i / dim)) for even i
    for (int i = 0; i < dim / 2; i++) {
        freqs[i] = 1.0 / pow(theta, (2 * i) / (float)dim);
    }
    // Loop over the range from 0 to end
    for (int t = 0; t < end; t++) {
        // Loop over the freqs array
        for (int i = 0; i < dim / 2; i++) {
            // Compute the product of t and freqs[i]
            float prod = t * freqs[i];
            // Compute the real and imaginary parts of the complex number as cos(prod) and sin(prod)
            freqs_re[t * dim / 2 + i] = cos(prod);
            freqs_im[t * dim / 2 + i] = sin(prod);
        }
    }
    // Free the memory for the freqs array
    free(freqs);
}

// get a matrix by name and dequantize it
void uploadQWeightByName(QWeight &weight, char* fileNameBase, char* weightName, size_t height, size_t width, bool save = false)
{
    uint32_t* qweight;
    uint32_t* qzeros;
    half* scales;

    qweight = (uint32_t*)malloc(height * width / 2);        // 4 bit per element
    qzeros = (uint32_t*)malloc(height * width / (2 * 128));
    scales = (half*)malloc(sizeof(half) * height * width / 128);

    char filename[512];
    sprintf(filename, "%s.%s.qweight.bin", fileNameBase, weightName);
    getFileContents(qweight, filename, height * width / 2);
    sprintf(filename, "%s.%s.qzeros.bin", fileNameBase, weightName);
    getFileContents(qzeros, filename, height * width / (2 * 128));
    sprintf(filename, "%s.%s.scales.bin", fileNameBase, weightName);
    getFileContents(scales, filename, sizeof(half) * height * width / 128);


    cudaMemcpy(weight.weight, qweight, height * width / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(weight.zeros, qzeros, height * width / (2 * 128), cudaMemcpyHostToDevice);
    cudaMemcpy(weight.scales, scales, sizeof(half) * height * width / 128, cudaMemcpyHostToDevice);

    free(qweight);
    free(qzeros);
    free(scales);
}



// ----------------------------------------------------------------------------
// initialization: read from checkpoint

int checkpoint_init_weights(TransformerWeights* w, Config* p, FILE* f, int shared_weights) {
    size_t scratch_size = p->n_layers * std::max(p->dim, p->hidden_dim) * p->dim;
    scratch_size = std::max((size_t)p->vocab_size * p->dim, scratch_size);
    scratch_size *= sizeof(float);
    void* scratchCpu = malloc(scratch_size);
    void* scratchGpu = nullptr;
    cudaMalloc(&scratchGpu, scratch_size);

    // Ankan - try different set of weights
#if 1
    // read weights from different files and upload them at appriopriate offsets in the allocations we created.
    // OVERWRITES THE ABOVE WEIGHTS
    getFileContents(scratchCpu, "E:\\Work\\INT4\\AWQ\\new\\llm-awq\\awq\\awq_weights\\bin_wt\\model.embed_tokens.weight.bin", 32000 * 4096 * sizeof(half));
    cudaMemcpy(w->token_embedding_table, scratchCpu, 32000 * 4096 * sizeof(half), cudaMemcpyHostToDevice);

    getFileContents(scratchCpu, "E:\\Work\\INT4\\AWQ\\new\\llm-awq\\awq\\awq_weights\\bin_wt\\lm_head.weight.bin", 32000 * 4096 * sizeof(half));
    cudaMemcpy(w->wcls, scratchCpu, 32000 * 4096 * sizeof(half), cudaMemcpyHostToDevice);

    getFileContents(scratchCpu, "E:\\Work\\INT4\\AWQ\\new\\llm-awq\\awq\\awq_weights\\bin_wt\\model.norm.weight.bin", 4096 * sizeof(half));
    cudaMemcpy(w->rms_final_weight, scratchCpu, 4096 * sizeof(half), cudaMemcpyHostToDevice);

    // upload decoder block weight for each layer
#pragma omp parallel for
    for (int i = 0; i < 32; i++)
    {
        void* scratchCpu = malloc(11008 * 4096 * sizeof(float));
        void* scratchGpu = nullptr;
        cudaMalloc(&scratchGpu, 11008 * 4096 * sizeof(float));

        printf("\nParsing weights for layer: %d\n", i);

        char fileNameBase[512];
        char filename[512];
        sprintf(fileNameBase, "E:\\Work\\INT4\\AWQ\\new\\llm-awq\\awq\\awq_weights\\bin_wt\\model.layers.%d", i);

#if 0
        getDeqWeightByName(scratchCpu, fileNameBase, "self_attn.q_proj", 4096, 4096);
        cudaMemcpy(w->layers[i].wq, scratchCpu, 4096 * 4096 * sizeof(half), cudaMemcpyHostToDevice);

        getDeqWeightByName(scratchCpu, fileNameBase, "self_attn.k_proj", 4096, 4096);
        cudaMemcpy(w->layers[i].wk, scratchCpu, 4096 * 4096 * sizeof(half), cudaMemcpyHostToDevice);

        getDeqWeightByName(scratchCpu, fileNameBase, "self_attn.v_proj", 4096, 4096);
        cudaMemcpy(w->layers[i].wv, scratchCpu, 4096 * 4096 * sizeof(half), cudaMemcpyHostToDevice);

        getDeqWeightByName(scratchCpu, fileNameBase, "self_attn.o_proj", 4096, 4096);
        cudaMemcpy(w->layers[i].wo, scratchCpu, 4096 * 4096 * sizeof(half), cudaMemcpyHostToDevice);

        getDeqWeightByName(scratchCpu, fileNameBase, "mlp.up_proj", 4096, 11008);
        cudaMemcpy(w->layers[i].up_proj, scratchCpu, 11008 * 4096 * sizeof(half), cudaMemcpyHostToDevice);

        getDeqWeightByName(scratchCpu, fileNameBase, "mlp.gate_proj", 4096, 11008);
        cudaMemcpy(w->layers[i].gate_proj, scratchCpu, 11008 * 4096 * sizeof(half), cudaMemcpyHostToDevice);

        getDeqWeightByName(scratchCpu, fileNameBase, "mlp.down_proj", 11008, 4096);
        cudaMemcpy(w->layers[i].down_proj, scratchCpu, 11008 * 4096 * sizeof(half), cudaMemcpyHostToDevice);
#endif

        uploadQWeightByName(w->layers[i].wq_q, fileNameBase, "self_attn.q_proj", 4096, 4096);
        uploadQWeightByName(w->layers[i].wq_k, fileNameBase, "self_attn.k_proj", 4096, 4096);
        uploadQWeightByName(w->layers[i].wq_v, fileNameBase, "self_attn.v_proj", 4096, 4096);
        uploadQWeightByName(w->layers[i].wq_o, fileNameBase, "self_attn.o_proj", 4096, 4096);

        uploadQWeightByName(w->layers[i].wq_up, fileNameBase, "mlp.up_proj", 4096, 11008);
        uploadQWeightByName(w->layers[i].wq_gate, fileNameBase, "mlp.gate_proj", 4096, 11008);
        uploadQWeightByName(w->layers[i].wq_down, fileNameBase, "mlp.down_proj", 11008, 4096);

        sprintf(filename, "%s.input_layernorm.weight.bin", fileNameBase);
        getFileContents(scratchCpu, filename, 4096 * sizeof(half));
        cudaMemcpy(w->layers[i].rms_att_weight, scratchCpu, 4096 * sizeof(half), cudaMemcpyHostToDevice);

        sprintf(filename, "%s.post_attention_layernorm.weight.bin", fileNameBase);
        getFileContents(scratchCpu, filename, 4096 * sizeof(half));
        cudaMemcpy(w->layers[i].rms_ffn_weight, scratchCpu, 4096 * sizeof(half), cudaMemcpyHostToDevice);

        /*
        sprintf(filename, "%s.self_attn.rotary_emb.inv_freq.bin", fileNameBase);
        float test[64];
        getFileContents(test, filename, 64 * sizeof(float));
        printf("rotary_emb.inv_freq: ");
        for (int k = 0; k < 64; k++)
            printf("%g ", test[k]);
        printf("\n");
        */

        cudaFree(scratchGpu);
        free(scratchCpu);
    }
#endif

    int head_size = p->dim / p->n_heads;
#if 0
    if (uploadWeight(w->token_embedding_table, p->vocab_size * p->dim, f, scratchCpu, scratchGpu, true)) return 1;

    if (uploadWeight(nullptr, p->n_layers * p->dim, f, scratchCpu, scratchGpu)) return 1;
    if (uploadWeight(nullptr, p->n_layers * p->dim * p->dim, f, scratchCpu, scratchGpu)) return 1;
    if (uploadWeight(nullptr, p->n_layers * p->dim * p->dim, f, scratchCpu, scratchGpu)) return 1;
    if (uploadWeight(nullptr, p->n_layers * p->dim * p->dim, f, scratchCpu, scratchGpu)) return 1;
    if (uploadWeight(nullptr, p->n_layers * p->dim * p->dim, f, scratchCpu, scratchGpu)) return 1;

    if (uploadWeight(nullptr, p->n_layers * p->dim, f, scratchCpu, scratchGpu)) return 1;
    if (uploadWeight(nullptr, p->n_layers * p->dim * p->hidden_dim, f, scratchCpu, scratchGpu)) return 1;
    if (uploadWeight(nullptr, p->n_layers * p->hidden_dim * p->dim, f, scratchCpu, scratchGpu)) return 1;
    if (uploadWeight(nullptr, p->n_layers * p->dim * p->hidden_dim, f, scratchCpu, scratchGpu)) return 1;
    if (uploadWeight(nullptr, p->dim, f, scratchCpu, scratchGpu)) return 1;

    int head_size = p->dim / p->n_heads;
    if (uploadWeight(w->freq_cis_real, p->seq_len * head_size / 2, f, scratchCpu, scratchGpu, true)) return 1;
    if (uploadWeight(w->freq_cis_imag, p->seq_len * head_size / 2, f, scratchCpu, scratchGpu, true)) return 1;

    if (!shared_weights)
        if (uploadWeight(nullptr, p->vocab_size * p->dim, f, scratchCpu, scratchGpu)) return 1;
#endif


    // Ankan - test compute it at our end
    float* freq_real, * freq_imag;
    freq_real = (float*)malloc(sizeof(float) * p->seq_len * head_size / 2);
    freq_imag = (float*)malloc(sizeof(float) * p->seq_len * head_size / 2);
    precompute_freqs_cis(head_size, p->seq_len, 10000.0f, freq_real, freq_imag);
    cudaMemcpy(w->freq_cis_real, freq_real, sizeof(float) * p->seq_len * head_size / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(w->freq_cis_imag, freq_imag, sizeof(float) * p->seq_len * head_size / 2, cudaMemcpyHostToDevice);
    free(freq_real);
    free(freq_imag);

    //dumpTensor("baseline freq_cis_computed_with_half", w->freq_cis_real, 256 * head_size / 2);
    //dumpTensor("freq_cis_imag", w->freq_cis_imag, 256 * head_size / 2);
    //exit(0);

    cudaFree(scratchGpu);
    free(scratchCpu);
    return 0;
}



half* deq_wts;

void dumpTensor_cpu(char* message, half* tensor, int elements) {
    half* arr = tensor;

    printf("\n%s :- \n", message);
    for (int i = 0; i < elements; i++)
    {
        printf("%10.6f", (float)arr[i]);
        if (i % 16 == 15) printf("\n");
    }
}


// ----------------------------------------------------------------------------
// neural net blocks

void accum(half* a, half* b, int size) {
    int blocks = divUp(size, 256);
    element_wise_add_kernel <<< blocks, 256 >>> (a, b, size);
}


void rmsnorm(half* o, half* x, half* weight, int size) {
    int elementsPerThread = divUp(size, 1024);
    rmsnorm_kernel <<< 1, 1024 >>> (o, x, weight, size, elementsPerThread);
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(half* xout, half* x, half* w, int n, int d, int batch = 1, int x_stride = 0, int w_stride = 0, int op_stride = 0, int w_row_stride = -1, float alpha = 1.0f) {


    //dumpTensor("MatMul XX input", x, n);

    int serialElements = divUp(n, 32);
    int serialLoads = divUp(serialElements, 8);     // we load 8 elements in parallel
    dim3 block_dim(32, 4);
    dim3 grid_dim(divUp(d, 4), batch);
    if (w_row_stride == -1) w_row_stride = n;
    if ((n & 7) || (d & 7))
        mat_vec_kernel_simple <<<grid_dim, block_dim >>> (xout, x, w, n, d, serialElements, x_stride, w_stride, op_stride, w_row_stride, alpha);
    else
        mat_vec_kernel <<<grid_dim, block_dim >>> (xout, x, w, n, d, serialLoads, x_stride, w_stride, op_stride, w_row_stride, alpha);

    //dumpTensor("MatMul XX output", xout, d);
    //printf("\n\n");
}


__global__ void reduce_kernel(half *output, half *input, int size, int iters)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= size) return;

    float sum = 0;
    for (int i = 0; i < iters; i++)
    {
        sum += (float)input[index + size * i];
    }

    output[index] = (half)sum;
}

half* split_k_scratch = nullptr;
void matmul(half* xout, half* x, QWeight &w, int inpSize, int opSize) {
    // try existing awq kernel
    int j_factors1 = opSize / 128 / 1;

    int split_k_iters = 8;     // Ankan - tune this! (only upto 8 works!)

    dim3 num_blocks(j_factors1 * split_k_iters);
    dim3 threads_per_block(32, 2);
 
    if (split_k_iters == 1)
    {
        gemm_forward_4bit_cuda_m16n128k32 << < num_blocks, threads_per_block >> > (
            1, x, (int*)w.weight, w.scales, (int*)w.zeros, 1, inpSize, opSize, xout);
    }
    else
    {
        if (!split_k_scratch)
            cudaMalloc((void**)&split_k_scratch, 1024 * 1024);

        gemm_forward_4bit_cuda_m16n128k32 << < num_blocks, threads_per_block >> > (
            split_k_iters, x, (int*)w.weight, w.scales, (int*)w.zeros, 1, inpSize, opSize, split_k_scratch);

        reduce_kernel <<< divUp(opSize, 64), 64 >>> (xout, split_k_scratch, opSize, split_k_iters);
    }
}

void RoPERotation(half *q, half *k, float *f_real, float *f_imag, int num_heads, int head_size) {
    RoPERotation_kernel <<<num_heads, head_size / 2 >>> (q, k, f_real, f_imag, num_heads, head_size);
}


__device__ void softmax_gpu(float* __restrict__ x, int size) {
    using BlockReduce = cub::BlockReduce<float, 1024>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    int tid = threadIdx.x;
    int step = blockDim.x;

    // find max value (for numerical stability)
    float max_val = tid < size ? x[tid] : -1000000000.0f;
    for (int i = tid + step; i < size; i += step)
        if (x[i] > max_val)
            max_val = x[i];

    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0)
        shared_val = max_val;
    __syncthreads();
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0)
        shared_val = sum;
    __syncthreads();
    sum = shared_val;

    // normalize
    for (int i = tid; i < size; i += step)
        x[i] /= sum;
}

// Each block processes a single head
// Poor parallelism and even poorer memory access pattern.
// Ankan - TODO: optimize this.
#define MAX_SEQ_LEN 8192
__global__ void MultiHeadAttention_kernel(half* __restrict__ output, const half* __restrict__ sq,
    const half* __restrict__ key_cache, const half* __restrict__ value_cache,
    int num_heads, int head_size, int loff, int seq_len, int dim) {
    int h = blockIdx.x;

    // get the query vector for this head
    const half* q = sq + h * head_size;
    // attention scores for this head
    __shared__ float att[MAX_SEQ_LEN];

    // iterate over all timesteps, including the current one
    for (int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        // get the key vector for this head and at this timestep
        const half* k = key_cache + loff + t * dim + h * head_size;
        // calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for (int i = 0; i < head_size; i++)
            score += (float)q[i] * (float)k[i];
        score /= sqrtf(head_size);
        // save the score to the attention buffer
        att[t] = score;
    }
    __syncthreads();

    // softmax the scores to get attention weights
    softmax_gpu(att, seq_len);
    __syncthreads();

    // weighted sum of the values, store back into xb
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float val = 0.0f;
        for (int t = 0; t < seq_len; t++)
            val += att[t] * (float)value_cache[loff + t * dim + h * head_size + i];
        output[h * head_size + i] = (half)val;
    }
}

void MultiHeadAttention(half *output, half *q, half *key_cache, half * value_cache, half *att, int num_heads, int head_size, int seq_len) {
    int dim = head_size * num_heads;
    //MultiHeadAttention_kernel << <num_heads, 1024 >> > (output, q, key_cache, value_cache, num_heads, head_size, 0, seq_len, dim);
#if 1
    // 1. Get attention scores
    int serialElements = divUp(head_size, 32);
    dim3 block_dim(32, 32);
    dim3 grid_dim1(divUp(seq_len, 32), num_heads);
    mat_vec_kernel_simple <<< grid_dim1, block_dim >>> (att, q, key_cache, head_size, seq_len, serialElements, head_size, head_size, seq_len, dim, 1.0 / sqrt(head_size));

    // 2. Run softmax kernel
    softmax_kernel <<< num_heads, 1024 >>> (att, num_heads, seq_len);

    // 3. weighted sum of the values to get the final result
    serialElements = divUp(seq_len, 32);    
    dim3 grid_dim2(divUp(head_size, 32), num_heads);
    vec_mat_kernel <<< grid_dim2, block_dim >>> (output, att, value_cache, head_size, seq_len, serialElements, seq_len, head_size, head_size, dim);

#endif
}

void siluElementwiseMul(half *hb, half *hb2, int size) {
   silu_element_wise_mul_kernel <<< divUp(size, 256), 256 >>> (hb, hb2, size);
}

void transformer(int token, int pos, Config* p, RunState* s, TransformerWeights* w) {

    // a few convenience variables
    half* x = s->x;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    /*
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    */

    // ankan - test!
    //token = 13;
    // printf("\nToken: %d\n", token);

    // copy the token embedding into x
    half* content_row = &(w->token_embedding_table[token * dim]);
    cudaMemcpyAsync(x, content_row, dim * sizeof(half), cudaMemcpyDeviceToDevice);


    // Ankan - test! dump after embedding
    if (pos > 20000)
        dumpTensor("after initial embedding", x, 3);
    //exit(0);


    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    float* freq_cis_real_row = w->freq_cis_real + pos * head_size / 2;
    float* freq_cis_imag_row = w->freq_cis_imag + pos * head_size / 2;

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->layers[l].rms_att_weight, dim);

        if (pos > 20000 && l == 0)
            dumpTensor("after rms-norm", s->xb, 3);

        // we directly store (key, value) at this time step (pos) to our kv cache
        int loff = l * p->seq_len * dim; // kv cache layer offset for convenience
        half* key_cache_row = s->key_cache + loff + pos * dim;
        half* value_cache_row = s->value_cache + loff + pos * dim;

        // print some weights for example

        //dumpTensor("first few elements of scale tensor", layer0_q_proj_scales, 16);
        //dumpTensor("first few elements of original fp16 weights: ", w->wq + l * dim * dim, 16);
        //exit(0);

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->layers[l].wq_q, dim, dim);

        if (pos > 20000 && l == 0)
            dumpTensor("after q matmul", s->q, dim);

        matmul(key_cache_row, s->xb, w->layers[l].wq_k, dim, dim);

        if (pos > 20000 && l == 0)
            dumpTensor("after k matmul", key_cache_row, dim);

        matmul(value_cache_row, s->xb, w->layers[l].wq_v, dim, dim);

        if (pos > 20000 && l == 0)
            dumpTensor("after v matmul", value_cache_row, dim);

        // apply RoPE rotation to the q and k vectors for each head
        // also save the output (key, value) at this time step (pos) to our kv cache
        RoPERotation(s->q, key_cache_row, freq_cis_real_row, freq_cis_imag_row, p->n_heads, head_size);

        if (pos > 20000 && l == 0) {
            dumpTensor("q after RoPE rotation", s->q, 3);
            dumpTensor("k after RoPE rotation", key_cache_row, 3);
        }

        // apply MHA using the query and the key-value cache
        MultiHeadAttention(s->xb, s->q, s->key_cache + loff, s->value_cache + loff, s->att, p->n_heads, head_size, pos+1);

        if (pos > 20000 && l == 0)
            dumpTensor("after MHA", s->xb, dim);

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->layers[l].wq_o, dim, dim);

        if (pos > 20000 && l == 0)
        {
            dumpTensor("after final attention block matmul", s->xb2, dim);
        }

        // residual connection back into x
        accum(x, s->xb2, dim);

        if (pos > 20000 && l == 0)
            dumpTensor("after skip connection add", s->x, 3);

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->layers[l].rms_ffn_weight, dim);

        if (pos > 20000 && l == 0)
            dumpTensor("after ffn layernorm", s->xb, 3);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->layers[l].wq_gate, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->layers[l].wq_up, dim, hidden_dim);

        if (pos > 20000 && l == 0)
        {
            dumpTensor("after ffn hb", s->hb, 3);
            dumpTensor("after ffn hb2", s->hb2, 3);
        }

        // apply F.silu activation on hb and multiply it with hb2
        siluElementwiseMul(s->hb, s->hb2, hidden_dim);

        matmul(s->xb, s->hb, w->layers[l].wq_down, hidden_dim, dim);

        if (pos > 20000 && l == 0)
            dumpTensor("after final matmul", s->xb, 3);

        // residual connection
        accum(x, s->xb, dim);

        // Ankan - test! dump output after first decoder block and exit
        // 
        if (pos > 20000 && l == 0)
            dumpTensor("after resi add - final at end of block", x, 3);

        //exit(0);
    }

    //printf("\n------------------------------------------------------------------------------\n");

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits_gpu, x, w->wcls, p->dim, p->vocab_size);


    //dumpTensor("final scores", s->logits_gpu, 32000);
    //exit(0);


    // copy logits from GPU->CPU
    convert_fp16_to_fp32 <<<divUp(p->vocab_size, 256), 256 >>> (s->logits_temp, s->logits_gpu, p->vocab_size);

    /*
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    printf(" t: %g ", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    */

    cudaMemcpy(s->logits, s->logits_temp, p->vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
}

// ----------------------------------------------------------------------------
// byte pair encoding (BPE) tokenizer, encodes strings into tokens so we can prompt

int str_lookup(char *str, char **vocab, int vocab_size) {
    // find the first perfect match for str in vocab, return its index or -1 if not found
    for (int i = 0; i < vocab_size; i++) {
        if (strcmp(str, vocab[i]) == 0) {
            return i;
        }
    }
    return -1;
}

void bpe_encode(char *text, char **vocab, float *vocab_scores, int vocab_size, unsigned int max_token_length, int *tokens, int *n_tokens) {
    
    // a temporary buffer to merge two consecutive tokens
    char* str_buffer = (char*) malloc((max_token_length*2+1) * sizeof(char)); // *2 for concat, +1 for null terminator

    // first encode every individual byte in the input string
    *n_tokens = 0; // the number of tokens
    for (char *c = text; *c != '\0'; c++) {
        sprintf(str_buffer, "%c", *c);
        int id = str_lookup(str_buffer, vocab, vocab_size);
        if (id == -1) { printf("not good\n"); exit(1);}
        tokens[*n_tokens] = id;
        (*n_tokens)++;
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", vocab[tokens[i]], vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, vocab, vocab_size);
            if (id != -1 && vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// utilities

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    timespec_get(&time, TIME_UTC);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

unsigned long long rng_seed;
unsigned int random_u32() {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    rng_seed ^= rng_seed >> 12;
    rng_seed ^= rng_seed << 25;
    rng_seed ^= rng_seed >> 27;
    return (rng_seed * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32() { // random float32 in [0,1)
    return (random_u32() >> 8) / 16777216.0f;
}

int sample(float* probabilities, int n) {
    // sample index from probabilities, they must sum to 1
    float r = random_f32();
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (r < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int argmax(float* v, int n) {
    // return argmax of v in elements 0..n
    int max_i = 0;
    float max_p = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > max_p) {
            max_i = i;
            max_p = v[i];
        }
    }
    return max_i;
}
// ----------------------------------------------------------------------------

int main(int argc, char *argv[]) {

    // Ankan - test!
    //loadQWeights();

    // poor man's C argparse
    char *checkpoint = NULL;  // e.g. out/model.bin
    float temperature = 0.9f; // e.g. 1.0, or 0.0
    int steps = 256;          // max number of steps to run for, 0: use seq_len
    char *prompt = NULL;      // prompt string

    // 'checkpoint' is necessary arg
    if (argc < 2) {
        printf("Usage: %s <checkpoint_file> [temperature] [steps] [prompt]\n", argv[0]);
        return 1;
    }
    if (argc >= 2) {
        checkpoint = argv[1];
    }
    if (argc >= 3) {
        // optional temperature. 0.0 = (deterministic) argmax sampling. 1.0 = baseline
        temperature = atof(argv[2]);
    }
    if (argc >= 4) {
        steps = atoi(argv[3]);
    }
    if (argc >= 5) {
        prompt = argv[4];
    }

    // seed rng with time. if you want deterministic behavior use temperature 0.0
    rng_seed = (unsigned int)time(NULL);

    // read in the model.bin file
    Config config;
    TransformerWeights weights;
    int shared_weights;
    {
        FILE *file = fopen(checkpoint, "rb");
        if (!file) { printf("Couldn't open file %s\n", checkpoint); return 1; }
        // read in the config header
        if (fread(&config, sizeof(Config), 1, file) != 1) { return 1; }

        // Dump model config
        printf("\nModel params:- \ndim: %d \nhidden_dim: %d\nn_heads: %d\nn_kv_heads: %d\nn_layers: %d\nseq_len: %d\nvocab_size: %d\n\n",
            config.dim, config.hidden_dim, config.n_heads, config.n_kv_heads, config.n_layers, config.seq_len, config.vocab_size);

        // negative vocab size is hacky way of signaling unshared weights. bit yikes.
        shared_weights = config.vocab_size > 0 ? 1 : 0;
        config.vocab_size = abs(config.vocab_size);
        // read in the Transformer weights
        malloc_weights(&weights, &config, shared_weights);
        if (checkpoint_init_weights(&weights, &config, file, shared_weights)) { return 1; }
    }
    // right now we cannot run for more than config.seq_len steps
    if (steps <= 0 || steps > config.seq_len) { steps = config.seq_len; }

    // read in the tokenizer.bin file
    char** vocab = (char**)malloc(config.vocab_size * sizeof(char*));
    float* vocab_scores = (float*)malloc(config.vocab_size * sizeof(float));
    unsigned int max_token_length;
    {
        FILE *file = fopen("tokenizer.bin", "rb");
        if (!file) { printf("couldn't load tokenizer.bin\n"); return 1; }
        if (fread(&max_token_length, sizeof(int), 1, file) != 1) { printf("failed read\n"); return 1; }
        int len;
        for (int i = 0; i < config.vocab_size; i++) {
            if (fread(vocab_scores + i, sizeof(float), 1, file) != 1) { printf("failed read\n"); return 1;}
            if (fread(&len, sizeof(int), 1, file) != 1) { printf("failed read\n"); return 1; }
            vocab[i] = (char *)malloc(len + 1);
            if (fread(vocab[i], len, 1, file) != 1) { printf("failed read\n"); return 1; }
            vocab[i][len] = '\0'; // add the string terminating token
        }
        fclose(file);
    }

#if 0
    // dump all of vocab into a file
    FILE* fp = fopen("vocab.txt", "w+");
    for (int i = 0; i < config.vocab_size; i++) {
        fprintf(fp, "%d: %s\n", i, vocab[i]);
    }
    fclose(fp);
#endif

    // create and init the application RunState
    RunState state;
    malloc_run_state(&state, &config);

    // process the prompt, if any
    int *prompt_tokens = NULL;
    int num_prompt_tokens = 0;
    prompt_tokens = (int*)malloc(config.seq_len * sizeof(int));

    char input_message[2048];
    strcpy(input_message, prompt);

    while (1)
    {
        if (input_message != NULL) {
            bpe_encode(input_message, vocab, vocab_scores, config.vocab_size, max_token_length, prompt_tokens, &num_prompt_tokens);
        }

        printf("\nprompt tokens: ");
        for (int i = 0; i < num_prompt_tokens; i++)
            printf("%d ", prompt_tokens[i]);
        printf("\n");

        // start the main loop
        long start = 0;  // used to time our code, only initialized after first iteration
        int next;        // will store the next token in the sequence
        int token = 1;   // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
        int pos = 0;     // position in the sequence
        printf("<s>\n"); // explicit print the initial BOS token for stylistic symmetry reasons
        while (pos < steps) {

            // forward the transformer to get logits for the next token
            transformer(token, pos, &config, &state, &weights);

            if (pos < num_prompt_tokens) {
                // if we are still processing the input prompt, force the next prompt token
                next = prompt_tokens[pos];
            }
            else {
                // sample the next token
                if (temperature == 0.0f) {
                    // greedy argmax sampling: take the token with the highest probability
                    next = argmax(state.logits, config.vocab_size);
                }
                else {
                    // apply the temperature to the logits
                    for (int q = 0; q < config.vocab_size; q++) { state.logits[q] /= temperature; }
                    // apply softmax to the logits to get the probabilities for next token
                    softmax(state.logits, config.vocab_size);
                    // we sample from this distribution to get the next token
                    next = sample(state.logits, config.vocab_size);
                }
            }

            // following BOS token (1), sentencepiece decoder strips any leading whitespace (see PR #89)
            char* token_str = (token == 1 && vocab[next][0] == ' ') ? vocab[next] + 1 : vocab[next];
            printf("%s", token_str);
            //printf(" [%d - %s] ", next, token_str);
            fflush(stdout);

            if (next == 2) break; // break if EOS token is reached

            // advance forward
            token = next;
            pos++;
            // init our timer here because the first iteration could be slow
            if (start == 0) { start = time_in_ms(); }
        }

        // report achieved tok/s
        long end = time_in_ms();
        double time = (end - start) / 1000.0;
        int timed_tokens = pos - 1;
        printf("\nachieved tok/s: %f. Tokens: %d, seconds: %g\n", timed_tokens / time, timed_tokens, time);

        printf("enter next prompt: ");
        gets_s(input_message);
    }



    // memory cleanup
    free_run_state(&state);
    free_weights(&weights, shared_weights);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    free(vocab_scores);
    if (prompt_tokens != NULL) free(prompt_tokens);
    return 0;
}