# What this fork changed, and why

The design notes of every change on top of [bitexploder/llama-superfast-v100](https://github.com/bitexploder/llama-superfast-v100),
in the order they landed, each with what it replaced, what was measured and the switches that turn it off or tune it.
All of it is in `main`; where a later section replaced an earlier one as the default, the section says so and the earlier
path stays behind its switch. The [README](README.md) lists these one line each and holds the build and run guide.

Measurements are on one V100 PCIe 32 GB behind a PCIe Gen3 x4 eGPU link, Qwen3.8-27B `UD-Q4_K_XL` with the DFlash2
draft, unless a section says otherwise.

## 1. Q8_0 KV cache on the Volta attention paths

Commits `bcf1877`, `5c15ec5`.

Upstream's fast paths, the CUTLASS 6-head prefill attention (`fattn-gqa6`) and the streaming decode/verify kernel
(`fattn-verify-stream`), took only an f16 KV cache. With `-ctk q8_0 -ctv q8_0`, every step fell to the generic kernels
over a temporary f16 copy of the cache.

- Both paths now read `q8_0` directly. The prefill kernel widens key blocks to f16 in shared memory; the verify kernel
  has a `KVQ8` variant with 8-byte row loads and aligned windows.
- Allocations no longer reserve the f16 copy on those routes.
- `test-backend-ops` has `q8_0` cases for both kernels, including the 27B's shapes.
- Switches: `LLAMA_FA_GQA6_Q8=0` and `LLAMA_FATTN_VERIFY_STREAM_Q8=0` fall back to the generic kernels;
  `LLAMA_FATTN_VERIFY_STREAM_Q8_KP` and `LLAMA_FATTN_VERIFY_STREAM_Q8_W` tune the verify kernel.

With sections 3 and 5, this took the server from upstream's 2 × 65,536 tokens (f16 cache, DFlash2) to one
245,760-token pool.

## 2. Faster prefill on repacked weights

Commits `044ef3b`.

- Long prompts expand each weight to fp16 for cuBLAS. The expansion now decodes straight from the repacked layout with
  16-byte stores, instead of one warp rebuilding the GGUF blocks while the rest of the block waits. It matches ggml's
  dequantizers bit for bit.
- Products of 129 to 191 tokens go to MMQ on a device-side unrepack instead of fp16 + cuBLAS. The Volta MMQ crossover is
  192 tokens (64 tokens: 0.5 vs 1.25 ms per 27B FFN product; 256 tokens: 1.72 vs 1.50 ms).
- The tensor-core matvec takes up to 128 tokens by default.
- Switches: `LLAMA_QPN_MAX_TOKENS` (default 128, at most 256); `GGML_CUDA_MMQ_MAX_BATCH` overrides the crossover;
  `LLAMA_QPN_MMQ=0` and `LLAMA_QPN_FAST_EXPAND=0` restore the earlier routes; `LLAMA_QPN_EXPAND_CHECK=1` runs the
  earlier kernels alongside and reports any difference.

## 3. Prompt work kept on the GPU

Commits `64107c5`.

- **DFlash2 features stay on the device.** The draft reads the target's layer inputs. For any decode that is one
  ubatch they now stay on the GPU, and the draft copies only the rows its 2,048-token window can see, device to
  device. Before, every prefill ubatch sent all of them to the host and back (about 2.3 GB for a 24k prompt). Verify
  steps with two slots take this path too. Switches: `LLAMA_DFLASH2_FEAT_DEV=0`, `LLAMA_DFLASH2_FEAT_DEV_ROWS`.
- **Checkpoints copy in the background.** A prompt checkpoint (the 27B's recurrent state, ~150 MiB) goes into a GPU
  staging buffer, then to the host on a second stream, so the next compute does not wait for the link. The draft's
  small state is read first. Switch: `LLAMA_CKPT_STAGE=0`.
- **No duplicate checkpoints.** A checkpoint at the position of an existing one is kept, not copied again; before, every
  new conversation re-saved the shared system prompt's checkpoint. Switch: `LLAMA_CKPT_KEEP_SAME=0`.
- **Cheaper prompt-cache saves.** A save reuses evicted entries' buffers, and moves checkpoints instead of copying them
  when the next prompt won't need them.
- **Two-user scheduler.** A lone prompt is no longer cut into small slices when no other slot is waiting
  (`LLAMA_TU_IDLE_JOB_MIN` sets the minimum slice).
- **Smaller draft compute buffer** with `--spec-draft-ubatch 512`: 1.7 GB freed at no measured speed cost.

## 4. Unified KV at full speed

Commits `146f3c4`, `9a20120`.

- **A per-batch window.** With `--kv-unified`, every batch attended over the other slot's cells too, masked out but
  still paid for. A batch of one sequence now reads only the cells from its first to its last (aligned to 256): the
  K/V views and the mask start there, and graph reuse keys on the window. Batches holding both slots skip it (their
  window would span everything, and the scan cost ~4% of two-slot decode). The window is on for the qwen35 line only,
  whose attention uses nothing else. Switches: `LLAMA_KV_WINDOW=0`, `LLAMA_KV_WINDOW_MIN_TOKENS`.
- **Both slots' masks on the GPU.** The device mask builder handled only sequence 0 of a unified cache, so slot 1's
  masks were built on the host and sent over the link every batch. It now keeps one cell mirror per sequence and
  builds mixed batches run by run.
- With the other slot holding a long conversation, prefill and decode in the unified cache run at the split cache's
  speed.
- Verified: device masks match the host's bit for bit (`LLAMA_KQ_MASK_DEVICE_CHECK=1`, 711 masks under two-slot load),
  and temperature-0 outputs are identical with the window on and off.

## 5. A smaller CUDA graph cache

Commits `0b1a768`.

- Under real traffic VRAM grew ~2 GB over the first ~25 minutes, which no fill test showed. `GGML_CUDA_MEM_TRACE`
  traced it to the CUDA graph cache: each instantiated graph of the 27B holds 8 MiB, decode graphs are keyed by cache
  length in steps of 256, and the cache kept up to 512 graphs for 600 s.
- The defaults are now 48 graphs for 120 s (`GGML_CUDA_GRAPH_MAX`, `GGML_CUDA_GRAPH_TTL_S`). A conversation moves to the
  next step every ~256 tokens, so old steps are rarely met again. Decode and TTFT are unchanged.
- Growth over a long run of mixed agentic traffic is now +784 MiB (temporary-buffer pools plus the capped cache),
  against ~2 GB with upstream's defaults, which made room for 32k more tokens of context.
- Diagnostics: `GGML_CUDA_MEM_TRACE=1` logs every drop of free device memory of 8 MiB or more and where it happened
  (`=2` also per node); `GGML_CUDA_POOL_LOG=1` logs each pool growth; `GGML_CUDA_GRAPH_STATS=1` logs the cached graph
  count at the server's debug level.

## 6. Cell placement in a unified cache

Commits `263398e`, `dd1665f`.

The window of section 4 is only as tight as a sequence's cells are contiguous. llama.cpp's head search fills free
cells from a moving head, token by token, so in a shared pool a conversation's cells scattered into holes left by the
other slot or by its own earlier prompts, and every later window spanned them.

- A ubatch of one sequence in a unified cache now goes right after that sequence's last cell when those cells are free.
  Otherwise it goes to the start of the largest free run that holds it, so a conversation stays one run of cells.
  The head search is the fallback.
- `263398e` does this for prompts and `dd1665f` for decode, which had restarted the search at cell 0 and decoded into
  small holes beside the other slot's conversation.
- Known trade-off: a new conversation placed right after the other slot's cells can block that slot's next prompt
  chunk, which then jumps past it and slows its long-prompt prefill. Placing new conversations in the middle of the
  free run avoids that but slows two-slot decode, so the current build keeps the tight placement; section 14 moves or
  clears what blocks a prompt's start.
- Switches: `LLAMA_KV_PLACE_MIN_TOKENS` (default 1) sets the smallest ubatch placed this way; `LLAMA_KV_WINDOW_LOG=1`
  logs placements and windows.

## 7. Build time

Commits `716cd08`.

`mmvq-qpn.cu`, upstream's tensor-core matvec, instantiates ~1,000 kernels in one file, which took 14 minutes on one core
and set the length of every build. Its kernels now live in `mmvq-qpn-impl.cuh`, with each launch entry point
instantiated in its own file under `ggml/src/ggml-cuda/template-instances/` (`generate_qpn_cu_files.py`), like upstream
llama.cpp's MMQ. The longest of those files takes ~90 s. Every kernel's PTX is unchanged.

## 8. The two-user scheduler with DFlash2 on a unified cache

The two-user scheduler reads a long prompt as a job of layer groups interleaved with the other slot's decode. It refused
every DFlash2 job (the draft needs the target's layer inputs) and did not run on a unified cache, so with DFlash2 a side
request's prompt waited until the other slot finished generating, or (`LLAMA_TU_FIX=0`) the generating slot fell to
~3 tok/s while the prompt was read.

- A job's groups copy each draft layer's input into a persistent f32 buffer (~210 MiB at `-b 2048`), which becomes the
  device feature path's source when the job publishes. Greedy outputs and draft acceptance are identical with the
  scheduler on and off.
- On a unified cache jobs are one ubatch (`-b <= -ub`), and the job's cell hash covers only its own sequence.
- With a slot generating, a read of at most 256 rows joins the decode round (`LLAMA_TU_GEN_JOB_MIN`), a longer one runs
  as jobs of at most 1,024 rows (`LLAMA_TU_GEN_JOB_ROWS`), and a shorter job's groups take more layers. A group's time
  grows with the depth of the prompt read so far (1,024 rows over 4 layers: 82 ms at 8k tokens, 138 ms at 125k), so
  the generating slot gets one round per 80 ms of a group's measured time (`LLAMA_TU_GEN_ROUND_MS`, at most
  `LLAMA_TU_GEN_ROUNDS_MAX` 4): one at 8k as before, two from ~60k. Beside a 123k-token read, at 60-90k of depth the
  generating slot 5.5 -> 8.1 rounds/s (30 -> 46 tok/s) and the read 375 -> 318 tok/s, at 90-120k 4.8 -> 7.6 (26 -> 43)
  and 327 -> 287; the whole read 308 -> 322 s. A 29k-token read is unchanged (529 tok/s, the generating slot 38).
  Shrinking the job's rows instead (`LLAMA_TU_GEN_GROUP_MS`, a target group time, default off) costs the read more:
  a group's cost has a part per group (the context's KV), and at 60-90k of depth a 55 ms target took the read from
  375 to 182 tok/s for the generating slot's 5.5 -> 9.3 rounds/s, an 80 ms target to 241 tok/s for 7.6. Idle, a job's
  slices and groups are longer (`LLAMA_TU_IDLE_SLICE_MS`, `LLAMA_TU_IDLE_GROUP`), so a lone long prompt reads within
  ~1% of the scheduler being off.
- A 24k-token side prompt beside a generating slot: it starts at once, and the generating slot keeps ~34 tok/s with
  gaps of at most 0.22 s.

## 9. Short prompts (32-token passes, no graph capture for prompts)

- The tensor-core matvec's single products take passes of 32 tokens (four 8-token tiles over one read and decode of the
  weights; each column summed in the same order as before, so outputs are bit-identical), and products up to 224
  tokens stay on it: prompts of 68 / 113 / 171 tokens 202 / 310 / 421 ms -> 168 / 240 / 361. `LLAMA_QPN_W32=0` restores
  passes of 16.
- A graph whose products with a weight take more than 64 tokens is no longer captured as a CUDA graph
  (`GGML_CUDA_GRAPH_MAX_TOKENS`): each prompt length was captured on its second occurrence and seldom replayed.

## 10. Checkpoints

- The checkpoint 4 tokens before a prompt's end, which the next turn of the conversation restores, is written from the
  recurrent state's rollback snapshot after the first token is out (`llama_memory_rs_view_back`), instead of reading the
  last 4 tokens as a decode of their own (~50 ms of every prompt's first token). `LLAMA_CKPT_TAIL_SNAP=0` restores it.
- Two checkpoints are kept on the device as well (`LLAMA_CKPT_DEV`, default 2): a slot's latest tail checkpoint and the
  last prefix restored from the host, such as a shared system prompt. A restore of a held prefix takes 1.5 ms instead of
  50 ms over the host link.

## 11. Compaction of a boxed-in prefix

A new conversation keeps the cells of the prefix it restored. When the other slot's conversation sits right after them,
its prompt went past that conversation and every window spanned it. Before a ubatch of a sequence that cannot grow in
place, up to 1,024 of its own cells move next to the free run its ubatch will use (`LLAMA_KV_COMPACT_MAX`): a 15.6k-token
prompt beside a 37k-token resident conversation, 708 -> 933 tok/s.

## 12. Four slots: the verify rows budgeted, the draft one width at a time

Commits `100cd2c`.

A verify step of the 27B costs ~40 ms whatever its width up to 8 rows, +15% to 16 rows and +35% to 32 (the tensor-core
matvec's 32-token kernel runs one block per SM). Three slots at the draft's 7 tokens made 24 rows a step, and a third
slot added nothing (93 vs 93 tok/s combined).

- The generating slots share a budget of verify rows, 8 up to two slots and 16 from three (`LLAMA_SPEC_WIDTH_SLOTS=8,16`):
  each drafts `budget / generating − 1` tokens, so two slots keep 7, three draft 4 and four draft 3.
- A slot alone always drafts the full 7. The older adaptive policy, which narrowed the draft to 3 tokens when its cost
  table said a wide verify was too dear, is off by default (`LLAMA_SPEC_ADAPT_WIDTH=1` turns it back on): its table
  predates the streaming verify kernel, which reads the cache once a step whatever the width. On a 148k-token
  conversation it drafted 3 tokens most rounds for 52 tok/s; the full width at the same depth gives 80–113 tok/s.
- The DFlash2 draft graph needs equal block widths in one ubatch, so the draft now runs one width at a time when slots
  draft different widths. Before, a narrowed slot beside full ones mis-blocked silently at two slots and asserted at
  three.

## 13. A shared pool of rollback snapshots

Commits `36bd290`, `f8aaffa`.

Each slot kept its own rollback snapshots of the recurrent state, as many rows as the draft's tokens: 150 MiB a row,
1.2 GB a slot at 7, so four slots at full depth would have cost 2.4 GB more (~70k tokens of context), and four slots at
a depth of 3 made a lone slot draft 3 as well (decode 74–82 tok/s instead of 90–116).

- `LLAMA_RS_POOL=12` makes the rollback rows one pool the slots share: a slot alone takes 7, two take 6 each, three 4
  and four 3. The runs are planned when the set of active slots changes, and a snapshot still to be read is applied to
  the live row before its run moves. Four live rows and 12 shared ones are 2.4 GB, the same as two slots before, so
  four slots fit the 245,760-token pool.
- A fifth live cell (`LLAMA_RS_SPARE=0` drops it, 150 MiB) keeps a prompt job's state where it is: llama.cpp gathers
  the cells of the sequences decoding together into one run by swapping cells, and with a cell a slot, three slots
  decoding while the fourth held a prompt job swapped the job's cell into their run, and the next decode copied another
  slot's state over it (the job's row check aborted the server about one run of the 8-in-flight probe in three). A prompt
  job's sequence now moves to the fifth cell, at the top, when the job begins (one copy of its state row), so the runs of
  the other sequences, all below it, never straddle it.
- The graph writes each sequence's snapshots as a copy of its own (the fused GDN cache write takes per-sequence bases),
  and the snapshot rows belong to the sequence, not to the storage cell: `f8aaffa` fixed a lookup through the cell,
  which read another sequence's snapshot once the cache had moved a sequence's state (three pinned slots generating
  together: one answer ended after 4 tokens; a released slot's cell tripped an assert in BetterBench's concurrency
  sweep).
- Greedy outputs of a lone slot are identical with the pool on and off (600 tokens, the same draft counts).
- Switches: `LLAMA_RS_POOL=N` (default 0: the per-slot layout), `LLAMA_RS_POOL_LOG=1`, `LLAMA_GDN_CACHE_FUSE=0` (the
  unfused writes, ~10% slower with several slots).

## 14. A prompt's cells stay one run, low in the pool

Commits `39dfacc`, `5e4a216`.

With four slots, a prompt's cached prefix (a shared system prompt, or the longest common prefix of an earlier prompt in
that slot) sat in cells far from the free run its body took, and every ubatch of the prompt read the window spanning
both: a 6k-token prompt read 50k cells, and cold 47k-token prompts fell from 716 to 491 tok/s. The per-ubatch compaction
of section 11 skipped those prefixes (its 1,024-cell cap).

- At a prompt's start the server asks the cache how many tokens the sequence can add while staying one run: the free
  cells after its last cell, or the largest free run when it has none.
- When the prompt does not fit, its whole cached prefix moves to the start of the largest free run that holds it and
  the prompt (a copy of a few ms, once per prompt; `LLAMA_KV_COMPACT_PROMPT_MAX` caps the cells moved).
- If it still does not fit, the idle slots' conversations are cleared, the least recently used first and only those
  shorter than this prompt, so a long resident conversation is never dropped for a shorter prompt
  (`LLAMA_KV_EVICT_IDLE=0` keeps them all; with `--cache-ram` they are saved to the prompt cache first).
- A new conversation takes the lowest free run that holds its whole prompt with room to grow
  (`LLAMA_KV_PLACE_MARGIN` cells, default 16,384), the largest run only when none does. The largest run alone sent new
  conversations to the top of an emptying pool, and the slots' decodes then read the whole used range: after four
  47k-token prompts, four short conversations decoded at 95 tok/s combined instead of 131.
- BetterBench's prefill sweep on four slots: 12k / 24k / 47k tokens 961 / 915 / 830 tok/s, against 920 / 836 / 716 on
  two slots before; its concurrency sweep 154 tok/s with four requests in flight.
- Before an idle conversation is cleared, the idle slots' cells and the prompt's own move down over the free cells
  below them (`llama_kv_pack_seqs`: the idle ones first, the prompt's on top, a device copy of the rows in ranges), so
  the free space is one run right after the slot that asked, and a conversation is cleared only when the pool is
  short, not fragmented. Four agents on the 262,144-token pool had cleared 37-52k-token conversations for follow-ups
  of 29-254 tokens, re-read minutes later beside three generating slots (135-427 s each). On a 65,536-token pool with
  four conversations, two replaced by shorter ones and one growing past the hole after it: without the packing two
  17.8k-token conversations were cleared and 41.6k tokens re-read (47 s); with it 23,739 cells moved in 24.4 ms and
  nothing was cleared, the others' follow-ups fully cached, their greedy answers identical to a fresh read.
  `LLAMA_KV_PACK=0` turns it off; the `kv_pack` log line gives the cells moved, the time and the room gained.
  The copies go to the device backend's stream and synchronize once: a move by a short distance is cut into chunks no
  longer than that distance, and the synchronous copy (one synchronize each) took 1.5 s for 23.7k cells moved by 2,048.

## 16. The prompt cache gives back the conversations that left the slots

With `--cache-ram` and `--no-cache-idle-slots`, the host prompt cache held what `kv_evict_idle` saved before clearing a
conversation and never read it back: slot selection refused every cache use with idle-slot caching off, and with it
on (the server's default) every idle slot's conversation was saved and cleared at every new task, then restored on its
next request, ~0.5 s each way over the eGPU link for 40k tokens, a pause of all slots. Now, with idle-slot caching off,
a slot chosen by LRU, or whose conversation the prompt would mostly discard, saves its own conversation and loads a
better entry when the cache holds one (`server_prompt_cache::load` takes only a strictly better entry and consumes
it), whether or not a peer is busy; nothing is saved or cleared otherwise. The LRU choice prefers an empty slot, so a
returning conversation does not displace a live one. On a 65,536-token pool: a 23.8k-token conversation cleared for a
34k-token prompt came back in 0.97 s with 23,819 tokens restored (26.4 s re-read without the cache), its 48 greedy
tokens identical. Saves cost ~0.2 s per 6k tokens and ~0.35 s per 24k. `LLAMA_CACHE_RETURN=0` restores upstream's
rule; the `prompt_cache` log line gives what was saved and loaded.

The cache must not take the host down. Its limit counts its state vectors and checkpoints, but its entries take
checkpoint blocks from the pinned reserve, the slots' next checkpoints then take fresh anonymous memory, and a recycle
pool keeps more: with `--cache-ram 15360` beside the 10.5 GB reserve the server held 17.5 GB of anonymous memory for
a 13.9 GB cache, and under BetterBench the host ran out (OOM-killed in production; on the dev build a watchdog stopped
it at 1.5 GB available, 16 requests into the concurrency sweep). A new entry now also needs the host's `MemAvailable`
to stay above `LLAMA_CACHE_RESERVE_MIB` (default 4096) after it: the oldest entries go first, and the save is skipped
when none are left. A conversation shorter than `LLAMA_CACHE_MIN_TOKENS` (default 8192) is not saved: every
single-stream request of the bench paid ~150 ms to first token saving the previous one-off conversation (its
recurrent state and tail checkpoint, ~300 MiB over the link; 248 ms for a 612-token one), and it never came back.
The production launcher runs `--cache-ram 15360 --no-cache-idle-slots` and 16 checkpoints per slot (the default 32
reserved 19.6 GB of pinned host memory).

## 15. Rollback by replay: one state a slot instead of a snapshot a token

A verify step runs the draft's 8 tokens through the recurrent layers at once, and the rollback to the accepted count
needed the state after each of them: the snapshots of section 13, 12 shared rows of 150 MiB plus the four live rows,
2.4 GB. Now (the default; `LLAMA_RS_REPLAY=0` for the pool) every write keeps, in one row a slot, the state as it was
8 tokens before its end (the GDN kernel's slot 1, written from its registers as it passes that token, the input state
when the ubatch is shorter), and the GDN inputs of those last 8 tokens in two rings (the pre-conv projections and the
raw alpha/beta product: one `set_rows` a tensor a layer, 13 MB a slot). A rollback of n tokens replays the first 8-n
of them from the kept state into the live row before the slot's next step: a graph of the recurrent layers only, the
conv, the gate and the delta rule as the forward ran them, so the state is the one the forward would have reached.
The replay of a whole step reproduces the live state exactly (`LLAMA_RS_REPLAY_SELFTEST=1` checks it before every
rollback; `=2` also checks a replay of several slots against each slot's own).

- All slots with a pending rollback replay in one graph, of a length rounded up to a multiple of 4: the shorter
  replays are padded with a pad token the delta rule leaves the state alone for (the x ring's zero row, the
  alpha/beta ring's -inf row: gate 0, beta 0), and each slot's conv state is gathered from the three input columns
  ending at its last real token. A whole rollback replays 0 tokens. One graph a (length, slots) shape, its rows all
  inputs, so after its first run a replay is inputs and a CUDA graph launch (0.3 ms of host time); the kernel reads
  each slot's kept row in place and writes its live row itself (the `set_rows` after it is fused into the launch, the
  rows read from a device table), so a replay moves each state row once.
- The replay runs on its own backend instance (its own CUDA stream), launched from `seq_rm`, so its GPU time (the
  GDN kernel's serial pass over the replayed tokens, about 4 us a token a layer, and a dozen small kernels a layer)
  overlaps the draft's work instead of running ahead of the next verify graph. Events order it: the side stream first
  waits for the main stream's enqueued graphs (a pipelined decode may still read the rows), the main stream waits for
  the replay before its next graph, and host-side readers of the rows (row copies, the checkpoint view) wait on the
  host. `LLAMA_RS_REPLAY_SIDE=0` keeps the replay on the main stream, batched before the decode (then the cost is
  about 1.5 ms a round, 4% of a lone stream). BetterBench: 105.5 tok/s single stream against 106.1 with the pool's
  snapshots, 158.1 / 153.9 tok/s aggregate with four / eight requests in flight against the pool's 145.6 / 144.3;
  the rows are 10 instead of 17, 1.0 GB back, which is what takes the pool from 245,760 tokens to the model's
  262,144.
- The checkpoint 4 tokens before a prompt's end (section 10) replays into a scratch row instead of reading a snapshot.
- The pool's plan, locks and re-packing are off under the replay; the spare cell of section 13 stays.

## 17. A full pool: the slot kept, the prompt that cannot fit refused, room made for a restore

Four agents on the 262,144-token pool held about 105k, 93k, 58k and 22k tokens: two prompt reads ran for minutes
beside the generating slots and failed when the pool ran out ("Context size has been exceeded"), the 58k request's
retry took by prompt similarity the slot of a 94k-token conversation idle between two of its turns while another slot
was empty, and that conversation, back a second later, could not be restored (94k free cells were not there) and was
re-read beside three other reads.

- **Slot choice.** While a slot is empty, a request no longer takes by similarity a slot whose conversation it would
  mostly discard: it must keep `LLAMA_SLOT_SIM_KEEP` of it (default 0.5; 0: upstream's rule), else the choice falls to
  the least recently used slot, which prefers an empty one.
- **Admission.** A prompt is read only if its new tokens, a margin for its answer (`LLAMA_KV_ADMIT_MARGIN`, default
  2048) and what the other slots still have to read fit the pool's free cells after the packing and the allowed
  evictions (`llama_kv_n_free`); else it is refused at once with the context-size error, so the client can compact or
  retry instead of waiting for a read that cannot finish. It runs before each chunk of a long read, so of two reads
  that cannot both fit, one goes on. The generating slots' growth is not reserved: a prompt that fits now is read.
  `LLAMA_KV_ADMIT=0` turns it off; the `kv_admit_refused` log line gives what was needed and what was free.
- **Restore.** Before a cached conversation is restored into a slot, the idle conversations shorter than it are cleared
  (saved first, the least recently used first) until the pool holds it, and the idle conversations pack down; with no
  room the restore is skipped and the entry stays, and the request goes through the admission check like any prompt.
  `LLAMA_CACHE_ROOM=0` restores as before; the `cache_restore_room` log line gives the entry, the room and the outcome.

Measured on a 65,536-token pool with four slots and the prompt cache on, each case with its fix off then on:

| Case | Fix off | Fix on |
|---|---|---|
| A (24k) idle; B shares A's first 5k and adds 17k | B takes A's slot; A goes to the cache (0.5 s) and back (0.8 s), pushing B out | B takes an empty slot; A's follow-up finds A in place |
| Two slots generate 2,500 tokens while D (41k) arrives with 32k cells free | D is read for 75 s; the two generating requests fail with an HTTP 500 when the pool runs out | D is refused in 0.08 s (HTTP 400, the context-size error); both answers complete |
| A (24k) cleared to the cache, then back with 21k free | the restore fails, C is cleared anyway and A is re-read: 26.4 s | C is saved and cleared first, A is restored: 4.4 s |

With the pool full, the old code did not fail the late prompt but the slots already generating: their verify step found
no cell, the batch was halved down to 2 rows, and the speculative batch bookkeeping threw. The first build of the
admission check crashed the server on its first refusal (the two-user scheduler read the refused slot's task after
its release); `tu_remaining()` now returns 0 for a slot without a task and the refusal clears the round's reader pick.

