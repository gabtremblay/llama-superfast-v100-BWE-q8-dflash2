#pragma once

#include "common.cuh"

// [TAG_KV_MOVE] the KV cache's row moves (a pack, a compaction) in one launch on the backend's stream, in the order of the
// moves (llama_kv_cache::move_rows_device has the callers' rules)

struct kv_move_tensor {
    char *   data;      // the tensor's first row
    uint64_t row_bytes; // nb[1]: a cell's row
};

// moves: [n_moves][3] = first source row, first destination row, rows. Each move's destination is below its source or does
// not overlap it, and no move reads rows an earlier move wrote (the caller checks both)
void ggml_cuda_kv_move_rows(ggml_backend_cuda_context & ctx, const kv_move_tensor * tensors, int n_tensors,
        const uint32_t * moves, int n_moves);

// [TAG_CACHE_BG_SAVE] byte ranges gathered into one device buffer in one launch: ranges [n][3] = source address, offset in dst,
// bytes. The ranges must not overlap dst
void ggml_cuda_gather_bytes(ggml_backend_cuda_context & ctx, const uint64_t * ranges, int n, char * dst);

// [TAG_CACHE_BG_RESTORE] the inverse: a device buffer's byte ranges scattered to their places in one launch: ranges [n][3] =
// destination address, offset in src, bytes. The ranges must not overlap src
void ggml_cuda_scatter_bytes(ggml_backend_cuda_context & ctx, const uint64_t * ranges, int n, const char * src);
