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
