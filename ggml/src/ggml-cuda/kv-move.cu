#include "kv-move.cuh"

// [TAG_KV_MOVE] Block (t, c) walks every move in order over tensor t's columns [c0, c1) of each row. A byte only moves to the
// same column of another row, so the blocks never touch each other's bytes and need no grid barrier. Within a block each
// chunk of a move is read into registers, the block syncs, then writes it: the chunks run row-major, so a move whose
// destination overlaps its source from below writes rows d + r <= s + r - 1, which the block has already read, and a later
// move only overwrites rows an earlier one read before the sync that ended it. Copied as raw words: a quantized row moves exactly.

#define KV_MOVE_THREADS 256
#define KV_MOVE_UNROLL  4

template <typename T>
static __global__ void kv_move_rows_kernel(const kv_move_tensor * __restrict__ tensors, const uint32_t * __restrict__ moves,
        const int n_moves, const int n_split) {
    const kv_move_tensor tt = tensors[blockIdx.x];
    const uint32_t row_el = (uint32_t) (tt.row_bytes / sizeof(T));
    const uint32_t c0     = (uint32_t) ((uint64_t) row_el*blockIdx.y/n_split);
    const uint32_t c1     = (uint32_t) ((uint64_t) row_el*(blockIdx.y + 1)/n_split);
    const uint32_t w      = c1 - c0;
    if (w == 0) {
        return; // the whole block: w depends on blockIdx only
    }
    T * base = (T *) tt.data;

    for (int i = 0; i < n_moves; ++i) {
        const uint32_t s     = moves[3*i + 0];
        const uint32_t d     = moves[3*i + 1];
        const uint32_t total = moves[3*i + 2]*w;
        for (uint32_t e0 = 0; e0 < total; e0 += KV_MOVE_THREADS*KV_MOVE_UNROLL) {
            T v[KV_MOVE_UNROLL];
#pragma unroll
            for (int u = 0; u < KV_MOVE_UNROLL; ++u) {
                const uint32_t e = e0 + u*KV_MOVE_THREADS + threadIdx.x;
                if (e < total) {
                    v[u] = base[(size_t) (s + e/w)*row_el + c0 + e%w];
                }
            }
            __syncthreads();
#pragma unroll
            for (int u = 0; u < KV_MOVE_UNROLL; ++u) {
                const uint32_t e = e0 + u*KV_MOVE_THREADS + threadIdx.x;
                if (e < total) {
                    base[(size_t) (d + e/w)*row_el + c0 + e%w] = v[u];
                }
            }
            __syncthreads();
        }
    }
}

void ggml_cuda_kv_move_rows(ggml_backend_cuda_context & ctx, const kv_move_tensor * tensors, int n_tensors,
        const uint32_t * moves, int n_moves) {
    if (n_tensors <= 0 || n_moves <= 0) {
        return;
    }
    // the widest word every row and every tensor's start is a multiple of
    size_t align = 16;
    for (int t = 0; t < n_tensors; ++t) {
        while (align > 1 && (tensors[t].row_bytes % align != 0 || (uintptr_t) tensors[t].data % align != 0)) {
            align /= 2;
        }
    }
    uint64_t min_row = UINT64_MAX;
    for (int t = 0; t < n_tensors; ++t) {
        min_row = std::min<uint64_t>(min_row, tensors[t].row_bytes/align);
    }

    // the descriptors and moves go up in one stream-ordered pool buffer, as the KQ mask's rows do (kq-mask.cu)
    cudaStream_t stream = ctx.stream();
    const size_t bytes_t = (size_t) n_tensors*sizeof(kv_move_tensor);
    const size_t bytes_m = (size_t) n_moves*3*sizeof(uint32_t);
    ggml_cuda_pool_alloc<char> buf(ctx.pool(), bytes_t + bytes_m);
    CUDA_CHECK(cudaMemcpyAsync(buf.get(),           tensors, bytes_t, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(buf.get() + bytes_t, moves,   bytes_m, cudaMemcpyHostToDevice, stream));
    const kv_move_tensor * d_tensors = (const kv_move_tensor *) buf.get();
    const uint32_t *       d_moves   = (const uint32_t *) (buf.get() + bytes_t);

    // about two blocks a multiprocessor, each over at least 8 words of a row
    const int nsm     = ggml_cuda_info().devices[ctx.device].nsm;
    const int n_split = (int) std::max<int64_t>(1, std::min<int64_t>((2*nsm + n_tensors - 1)/n_tensors, (int64_t) min_row/8));
    const dim3 grid(n_tensors, n_split);
    switch (align) {
        case 16: kv_move_rows_kernel<int4>    <<<grid, KV_MOVE_THREADS, 0, stream>>>(d_tensors, d_moves, n_moves, n_split); break;
        case 8:  kv_move_rows_kernel<int2>    <<<grid, KV_MOVE_THREADS, 0, stream>>>(d_tensors, d_moves, n_moves, n_split); break;
        case 4:  kv_move_rows_kernel<int>     <<<grid, KV_MOVE_THREADS, 0, stream>>>(d_tensors, d_moves, n_moves, n_split); break;
        case 2:  kv_move_rows_kernel<short>   <<<grid, KV_MOVE_THREADS, 0, stream>>>(d_tensors, d_moves, n_moves, n_split); break;
        default: kv_move_rows_kernel<char>    <<<grid, KV_MOVE_THREADS, 0, stream>>>(d_tensors, d_moves, n_moves, n_split); break;
    }
    CUDA_CHECK(cudaGetLastError());
}
