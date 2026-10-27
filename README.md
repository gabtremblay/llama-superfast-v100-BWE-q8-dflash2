# llama-superfast-v100-q8-dflash2

A fork of [bitexploder/llama-superfast-v100](https://github.com/bitexploder/llama-superfast-v100), the llama.cpp fork
tuned for the Tesla V100 (Volta, `sm_70`), made for **one V100 card**, a **single user**, **up to four concurrent slots**
on a **unified KV cache**, and the **DFlash2 speculative draft**: agentic work made palatable at the model's whole
262,144-token context on one 32 GB card.

The model is Qwen3.8-27B (`UD-Q4_K_XL`) on a V100 PCIe 32 GB, decoded with [DFlash2](https://github.com/z-lab/dflash),
z-lab's speculative draft for it: a small draft model that reads the target's own layer inputs and proposes a block of
up to 7 tokens, which the 27B verifies in one step; about half are accepted, and that is what takes decode from the
~40 tok/s a 27B manages alone on this card to ~105. Upstream ships DFlash2 beside Qwen's MTP head; this fork runs
DFlash2 by default, and much of what it adds (features kept on the GPU, the rollback by replay, the scheduler's jobs)
is built around it. The slots serve one user: the main thinking thread, and the side requests an agent harness sends
in parallel with it. Our card sits behind a PCIe Gen3 x4 eGPU link (~3.3 GB/s), so anything that crosses the host link
is on the critical path; the gains from keeping work on the GPU are largest on a link like that.

The rest of upstream (the tensor-core matvec on repacked K-quants, MTP, the two-user scheduler's design, two-card
tensor parallel) is documented further down, from [Upstream: llama-superfast-v100](#upstream-llama-superfast-v100).

Part of this work was sponsored by [Inscora](https://inscora.com). If you are an MSP or MSSP and want to offer cyber
insurance services to your clients, have a look at [inscora.com](https://inscora.com).

## What this fork adds

One line each, against upstream at `403f06f`. The design, the measurements and the switches of every item are in
[CHANGES.md](CHANGES.md).

1. **`q8_0` KV cache on the Volta attention kernels**: the CUTLASS prefill and the streaming verify kernels read 8-bit K/V directly, ~3% of decode for 2.7× the context. ([§1](CHANGES.md#1-q8_0-kv-cache-on-the-volta-attention-paths))
2. **Faster prefill on repacked weights**: the fp16 expansion decodes straight from the repacked layout, products of 129–191 tokens go to MMQ. ([§2](CHANGES.md#2-faster-prefill-on-repacked-weights))
3. **Prompt work kept on the GPU**: DFlash2 features, checkpoints and prompt-cache saves no longer cross the host link every ubatch. ([§3](CHANGES.md#3-prompt-work-kept-on-the-gpu))
4. **Unified KV cache at the split cache's speed**: each batch attends over its own sequence's cells only, with device-built masks for every slot. ([§4](CHANGES.md#4-unified-kv-at-full-speed))
5. **A capped CUDA graph cache**: 48 graphs for 120 s instead of 512 for 600 s, ~1.2 GB of VRAM that only grew under real traffic. ([§5](CHANGES.md#5-a-smaller-cuda-graph-cache))
6. **Cell placement**: a conversation's cells stay one run of the shared pool, so its attention window stays tight. ([§6](CHANGES.md#6-cell-placement-in-a-unified-cache))
7. **Build time**: the tensor-core matvec's ~1,000 kernels split into per-instance files, the 14-minute serial compile gone. ([§7](CHANGES.md#7-build-time))
8. **The two-user scheduler with DFlash2 on a unified cache**: a side prompt is read beside the generating slot, which gets more rounds per group as the read deepens. ([§8](CHANGES.md#8-the-two-user-scheduler-with-dflash2-on-a-unified-cache))
9. **Short prompts**: 32-token matvec passes and no CUDA graph capture for prompt shapes, the first token 15–20% sooner. ([§9](CHANGES.md#9-short-prompts-32-token-passes-no-graph-capture-for-prompts))
10. **Checkpoints from the rollback state**: the tail checkpoint without a 50 ms decode of its own, two checkpoints kept on the device. ([§10](CHANGES.md#10-checkpoints))
11. **Compaction of a boxed-in prefix**: up to 1,024 of a sequence's cells move before a ubatch that cannot grow in place. ([§11](CHANGES.md#11-compaction-of-a-boxed-in-prefix))
12. **Four slots**: the verify rows budgeted across the generating slots, the draft run one width at a time. ([§12](CHANGES.md#12-four-slots-the-verify-rows-budgeted-the-draft-one-width-at-a-time))
13. **A shared pool of rollback snapshots**: four slots in the memory of two; now behind `LLAMA_RS_REPLAY=0`. ([§13](CHANGES.md#13-a-shared-pool-of-rollback-snapshots))
14. **A prompt's cells one run, low in the pool**: cached prefixes move, idle slots are evicted for a longer prompt, new conversations take the lowest run. ([§14](CHANGES.md#14-a-prompts-cells-stay-one-run-low-in-the-pool))
15. **Rollback by replay on its own stream**: one kept state a slot and the accepted tokens replayed under the draft; 1.0 GB back, the pool at 262,144. ([§15](CHANGES.md#15-rollback-by-replay-one-state-a-slot-instead-of-a-snapshot-a-token))
16. **Vision with the scheduler on**: a loaded projector no longer turns the two-user scheduler off; only a prompt holding media pauses the other slots. ([Vision](#vision))
17. **Two or more cards by code reading**: every addition adapted to upstream's tensor-parallel path, untested here. ([below](#two-or-four-cards-tensor-parallel-untested))

## Results

One V100 PCIe 32 GB, Qwen3.8-27B `UD-Q4_K_XL`, DFlash2 draft, `--cache-ram 0`, the same probes for both.
Upstream is [bitexploder/llama-superfast-v100](https://github.com/bitexploder/llama-superfast-v100) at `403f06f` with its
f16 KV cache and its recipe (`-b 4096 -ub 2048`), at the largest two-slot context it fits with the DFlash2 draft. This
fork runs the [Run](#run) line below.

| | Upstream, f16 KV | This fork, `q8_0` KV |
|---|---|---|
| Cold prefill, ~8k-token prompt | 880 tok/s | 981 tok/s |
| Cold prefill, ~48k-token prompt | 797 tok/s | 831 tok/s |
| Time to first token, new ~125-token chat turn (median) | 657 ms | 262 ms |
| Decode, 400 tokens, three prompts | 95 / 93 / 99 tok/s | 86 / 85 / 109 tok/s |
| Context | 2 concurrent slots of 65,536 each | 4 concurrent slots sharing one 262,144-token pool (`--kv-unified`) |

Decode on the first two prompts is ~9% slower (the 8-bit cache, four slots instead of two, and the rollback's replay of
[section 15](CHANGES.md#15-rollback-by-replay-one-state-a-slot-instead-of-a-snapshot-a-token)) in exchange for twice the context and four slots; the third prompt's rate moves with how many draft tokens
it accepts (the probe decodes greedily). BetterBench's weighted decode below puts the gap at 3%.

Upstream's two-user scheduler refuses DFlash2 prompt jobs, so there a side request's prompt waits until the other slot
finishes generating. Here it starts at once and the generating slot keeps ~34 tok/s while it is read ([section 8](CHANGES.md#8-the-two-user-scheduler-with-dflash2-on-a-unified-cache)).

The same comparison under [BetterBench](https://github.com/bitexploder/betterbench) (20 passes per category, a 47k-token
conversation resident in one slot, the same corpus and sampling for both). Upstream runs with its scheduler off
(`LLAMA_TU=0`: on, its prompt jobs refuse the DFlash2 draft and long prompts fail) at 2 × 49,152 tokens, the largest
two-slot context that survived the run: at its 2 × 65,536 the card ran out of memory four minutes in, as its graph
cache grew under traffic ([section 5](CHANGES.md#5-a-smaller-cuda-graph-cache)). This fork runs the [Run](#run) line: four slots, one 262,144-token pool.

| BetterBench | Upstream, f16 KV, 2 × 49,152 | This fork, `q8_0` KV, 4 slots on 262,144 |
|---|---|---|
| Single-stream decode (weighted median) | 108.9 tok/s | 105.5 tok/s |
| Time to first token, single stream (p50) | 610 ms | 191 ms |
| Cold prefill, 12k / 24k / 47k-token prompts | 877 / 858 / 795 tok/s | 963 / 915 / 830 tok/s |
| Aggregate decode, 1 / 2 / 4 / 8 requests in flight | 90 / 101 / 89 / 88 tok/s | 88 / 91 / 158 / 154 tok/s |
| Time to first token with 2 / 4 / 8 in flight (p50) | 2.3 s / 12.1 s / 32.7 s | 0.36 s / 0.37 s / 9.2 s |
| Peak VRAM over the run | 31,044 MiB at 98,304 tokens | 32,028 MiB at 262,144 tokens |

A lone stream decodes 3% slower, the cost of the 8-bit cache (the rollback's replay of [section 15](CHANGES.md#15-rollback-by-replay-one-state-a-slot-instead-of-a-snapshot-a-token) runs under the draft
and costs it under 1%). Everything else is ahead: the first token three times sooner, prompts read 5–10% faster, 2.7× the context, and with
four or eight requests in flight 1.75× the throughput, while the fourth request starts in 0.4 s instead of 12 s and the
eighth in 9 s instead of 33 s (upstream's two slots serve eight requests four at a time). The 262,144-token pool peaks
at ~32.0 GB of the card's 32.5 over the run.

## Build

Requirements:

- A Tesla V100 (`sm_70`). The changes here are Volta-specific; other GPUs are untested.
- CUDA 12.x (12.8 tested; CUDA 13 dropped Volta), gcc/g++ 14, CMake, Ninja. ccache is optional and recommended.
- The NVIDIA container toolkit, for the Docker route.

**Native:**

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70-real -DGGML_NATIVE=OFF -DGGML_CUDA_CUB_3DOT2=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=OFF -DGGML_CCACHE=ON \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build build --target llama-server -j 16
```

- `GGML_CUDA_FA_ALL_QUANTS=OFF` still builds the `f16` and `q8_0` KV pairs this setup reads.
- `GGML_CUDA_CUB_3DOT2=ON` fetches CCCL 3.2 at configure time.
- A cold build takes about 15 minutes on 16 threads. The CUTLASS attention units need several GB of RAM per `nvcc`;
  lower `-j` on a machine with less than ~32 GB.
- Add `--target test-backend-ops` (with `-DLLAMA_BUILD_TESTS=ON`) for the kernel tests.

**Docker:**

```bash
DOCKER_BUILDKIT=1 docker build -f .devops/cuda-v100.Dockerfile -t llama-superfast-v100 .devops
```

The image clones this repository at `--build-arg LLAMA_REF=` (default `main`; pin a commit for a reproducible image).
Other build args: `BUILD_JOBS` (default 8), `CUDA_VERSION` (12.8.1). Its compiler cache persists across builds, so
rebuilding after a small change takes a few minutes.

## Models

```bash
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q4_K_XL.gguf MTP/mtp-Qwen3.8-27B-Q4_0.gguf --local-dir models
hf download z-lab/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-Q4_K_M.gguf --local-dir models
hf download unsloth/Qwen3.8-27B-GGUF mmproj-F16.gguf --local-dir models   # for images (the Vision section)
```

Put the files flat in `./models` (move `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` up). DFlash2 needs the MTP file too: its output
head reads that file's rows. The draft vocabulary ranking, `models/draft-vocab-qwen3.8-27b.txt`, ships in this
repository.

## Run

One card, four slots, DFlash2, one 262,144-token pool:

```bash
LLAMA_DFLASH2_HEAD_FILE=models/mtp-Qwen3.8-27B-Q4_0.gguf LLAMA_SPEC_DRAFT_VOCAB_FILE=models/draft-vocab-qwen3.8-27b.txt build/bin/llama-server -m models/Qwen3.8-27B-UD-Q4_K_XL.gguf -md models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-ubatch 512 -c 262144 --parallel 4 --kv-unified -fa on -ngl 99 -ctk q8_0 -ctv q8_0 -ctkd f16 -ctvd f16 -t 12 -b 2048 -ub 2048 --cache-ram 0 --jinja --metrics --temp 1.0 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8080
```

The same with the Docker image (models in `$PWD/models`):

```bash
docker run -d --name llama-superfast --init --gpus all -p 8080:8080 -v "$PWD/models:/models:ro" -e LLAMA_DFLASH2_HEAD_FILE=/models/mtp-Qwen3.8-27B-Q4_0.gguf llama-superfast-v100 --host 0.0.0.0 --port 8080 -m /models/Qwen3.8-27B-UD-Q4_K_XL.gguf -md /models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-ubatch 512 -c 262144 --parallel 4 --kv-unified -fa on -ngl 99 -ctk q8_0 -ctv q8_0 -ctkd f16 -ctvd f16 -t 12 -b 2048 -ub 2048 --cache-ram 0 --jinja --metrics --temp 1.0 --top-p 0.95 --top-k 20
```

What the settings do:

- The two-user scheduler is on ([section 8](CHANGES.md#8-the-two-user-scheduler-with-dflash2-on-a-unified-cache)): a long prompt in one slot no longer stalls the other slots' decode.
  `LLAMA_TU=0` turns it off.
- `--parallel 4` ([sections 12](CHANGES.md#12-four-slots-the-verify-rows-budgeted-the-draft-one-width-at-a-time) and [15](CHANGES.md#15-rollback-by-replay-one-state-a-slot-instead-of-a-snapshot-a-token)): four slots, each with one kept recurrent state; the draft's rollbacks replay.
  `LLAMA_RS_REPLAY=0` goes back to the snapshot pool of [section 13](CHANGES.md#13-a-shared-pool-of-rollback-snapshots) (then add `LLAMA_RS_POOL=12` and `-c 245760`).
- `LLAMA_DFLASH2_HEAD_FILE` points DFlash2's output head at the MTP file's rows.
- `LLAMA_SPEC_DRAFT_VOCAB_FILE` (set by the Docker image) restricts the draft to the top 98,304 ids of the file's
  131,072-id ranking. Without it, decode is ~5–10% slower.
- `--kv-unified` gives the slots one 262,144-token pool: any slot can take most of it (one took a 157k-token prompt
  while another answered side requests). `--kv-unified-per-slot N` caps each slot inside the pool;
  `--no-kv-unified` gives each slot a fixed share (with two slots 2 × 122,880) at the same speed.
- `-b 2048` equal to `-ub` keeps the DFlash2 features on the GPU during prefill ([section 3](CHANGES.md#3-prompt-work-kept-on-the-gpu)).
- `--spec-draft-ubatch 512` sizes the draft's compute buffer for 512-token batches, which frees 1.7 GB.
- `--cache-ram 0` turns off the host prompt cache. Saving a slot's state to host memory costs ~220 ms over a slow link
  each time a slot switches conversations; with one user and pinned slots (`id_slot` in the request) the cache brings
  little. On a fast link, try it on.
- 262,144 tokens with four slots peaks at ~32.0 GB under BetterBench. VRAM keeps growing for the first minutes of real
  traffic, so size a different context from a long mixed run, not a fill test ([section 5](CHANGES.md#5-a-smaller-cuda-graph-cache)).

The host side of this setup — powering an OcuLink eGPU dock from a USB relay, training the PCIe link through an S3
suspend, loading the driver with the 200 W cap, and cooling the passive card with a PWM fan driven from its
temperature, as systemd units an inference service chains after — is published separately as
[v100-egpu-host](https://github.com/gabtremblay/v100-egpu-host), with the bill of materials.

Check the startup log for `two-user scheduling (LLAMA_TU): on`, the draft model loading, and
`n_slots = 4, n_ctx_slot = 262144`. Draft acceptance on this target is ~0.5; near zero means the draft did not load.

For MTP instead of DFlash2: `-md models/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp`, without
`LLAMA_DFLASH2_HEAD_FILE` or `--spec-draft-ubatch`. It decodes slower here.

### Two or four cards (tensor parallel, untested)

This fork was built and measured on one card. Upstream runs the 27B on two cards in tensor parallel (`--split-mode
tensor`, the ggml meta backend, `GGML_CUDA_P2P=1`, `iommu=pt`; see its `README-27B.md`), and the fork's additions were
adapted to that path, for any number of cards, so far only by reading the code: there is no multi-card machine here to
run it on. What was done, and what to expect:

- **The unified pool, cell placement, compaction, idle-slot eviction, the q8_0 cache and the per-batch window** work per
  device: a split cache keeps every cell on every card (each holds its heads of it), so the cell bookkeeping is the same
  and the row moves of a compaction run on each card's own slice. The device mask builder was already split-aware.
- **DFlash2's device-fed features** (the draft reading the target's layer inputs on the GPU) copy card to card: with the
  draft split like the target, each card feeds its own copy; with the draft pinned to one card (`--device-draft`), that
  card's copy is read. A split target with a draft split differently refuses the device path and says so; set
  `LLAMA_DFLASH2_FEAT_DEV=0` then.
- **The two-user scheduler** runs its prompt jobs on the plain layer loop under the split (upstream's interleaved
  two-half prefill is off while the draft reads layer inputs, as upstream decided), with MTP or DFlash2.
- **The shared snapshot pool** restores and backs up its rows per card; the fused write is a per-card kernel already.
- **Checkpoints** on a split cache go through upstream's path; the fork's staged copy (`LLAMA_CKPT_STAGE`) steps aside
  for them.

The line is upstream's two-card line with this fork's switches, for example on two cards:

```bash
CUDA_VISIBLE_DEVICES=0,1 GGML_CUDA_P2P=1 LLAMA_RS_POOL=12 LLAMA_DFLASH2_HEAD_FILE=models/mtp-Qwen3.8-27B-Q4_0.gguf LLAMA_SPEC_DRAFT_VOCAB_FILE=models/draft-vocab-qwen3.8-27b.txt build/bin/llama-server -m models/Qwen3.8-27B-UD-Q4_K_XL.gguf -md models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-ubatch 512 -c 491520 --parallel 4 --kv-unified --split-mode tensor -fa on -ngl 99 -ctk q8_0 -ctv q8_0 -ctkd f16 -ctvd f16 -t 12 -b 2048 -ub 2048 --cache-ram 0 --jinja --metrics --temp 1.0 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8080
```

Two 32 GB cards hold about twice the pool (the weights are split too), so `-c` can double; start lower and raise it
while the server fits. Expect upstream's gains (1.35× decode, 1.7× prefill on two cards) on top of this fork's, not
more: the draft's acceptance and the verify budget are the same. If something fails, the one-card line in `## Run`
with `CUDA_VISIBLE_DEVICES=0` is the fallback, and a report with the server log is welcome.

## Vision

The server takes images when Qwen3.8-27B's vision projector is loaded: `--mmproj models/mmproj-F16.gguf`
(from the same Unsloth repository, 884 MiB; a BF16 one is there too, and any Qwen3.8-27B projector fits, a Q8_0 one
saving ~440 MiB). Chat requests then carry `image_url` parts (data: URLs or http). The projector and its encode buffers
need room the 245,760-token pool does not have, so the pool gives up 49,152 tokens:

```bash
LLAMA_RS_POOL=12 LLAMA_DFLASH2_HEAD_FILE=models/mtp-Qwen3.8-27B-Q4_0.gguf LLAMA_SPEC_DRAFT_VOCAB_FILE=models/draft-vocab-qwen3.8-27b.txt build/bin/llama-server -m models/Qwen3.8-27B-UD-Q4_K_XL.gguf --mmproj models/mmproj-F16.gguf -md models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-ubatch 512 -c 196608 --parallel 4 --kv-unified -fa on -ngl 99 -ctk q8_0 -ctv q8_0 -ctkd f16 -ctvd f16 -t 12 -b 2048 -ub 2048 --cache-ram 0 --jinja --metrics --temp 1.0 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8080
```

(the Docker line takes the same two additions: `--mmproj /models/mmproj-F16.gguf` and `-c 196608`).

What was measured (four slots, `LLAMA_RS_POOL=12`, `--cache-ram 0`):

- 196,608 tokens is the largest pool that keeps a margin: 29.6 GB after load, 31.6 GB peak over 50 minutes of concurrent
  image and text requests, 47k-token prompts and four-slot decode, against the card's 32.5 GB; the text-only pool of
  245,760 already peaks at 32.3 GB, and 212,992 with the projector would leave ~0.6 GB, less than the first minutes'
  growth ([section 5](CHANGES.md#5-a-smaller-cuda-graph-cache)).
- A 2048 × 1536 image is ~3,100 prompt tokens, read at ~520 tok/s; an 800 × 600 one ~500 tokens. Its answer decodes at
  85–108 tok/s with the draft (a red square, a blue circle and a white band in a test image are all named).
- Text requests are unchanged: four in flight 130–134 tok/s combined, 47k-token prompts at 843 tok/s. The two-user
  scheduler stays on with a projector loaded (`TAG_TU_MTMD`): a text prompt is read as a job beside the other slots'
  decodes as before, and only a prompt that holds media is read in the batch, so a decoding slot pauses for the ~6 s a
  large image takes. Without that change the scheduler went off as soon as `--mmproj` was given, and four text
  requests in flight fell to 71 tok/s combined.
- With a projector loaded, llama.cpp's prompt cache reuse and context shift are off for every slot (upstream's rule), and
  so is the fork's tail-snapshot checkpoint for prompts that hold media.

## Known limits

- Tested only on V100 PCIe 32 GB with Qwen3.8-27B. The window and placement changes are on for the qwen35 model line
  only.
- Combined decode tops out near 130–160 tok/s with four slots: a verify step costs about the same up to 8 rows, so the
  slots share 16 verify rows a step ([section 12](CHANGES.md#12-four-slots-the-verify-rows-budgeted-the-draft-one-width-at-a-time)), and each slot drafts less when more generate.
- Four slots at 262,144 tokens (the model's training context) use ~32.0 GB of the card's 32.5 under BetterBench. The
  recurrent rows are 150 MiB each ([sections 13](CHANGES.md#13-a-shared-pool-of-rollback-snapshots) and [15](CHANGES.md#15-rollback-by-replay-one-state-a-slot-instead-of-a-snapshot-a-token)), 10 of them with the replay.
- The first two prompts after a start pay ~0.4 s once each to allocate a device checkpoint copy.
- Two or more cards (`--split-mode tensor`) are supported by code reading only (the section after `## Run`): nothing
  here has run on more than one card.

## Credits and license

MIT, like [llama.cpp](https://github.com/ggml-org/llama.cpp) and [ggml](https://github.com/ggml-org/ggml). Built on
[bitexploder/llama-superfast-v100](https://github.com/bitexploder/llama-superfast-v100); its credits for the models,
the draft vocabulary and ported kernels are in [`README-27B.md`](README-27B.md). DFlash2 is
[z-lab/dflash](https://github.com/z-lab/dflash). Part of this work was sponsored by [Inscora](https://inscora.com).

---

# Upstream: llama-superfast-v100

The rest of this page is upstream's README: its headline figures, hardware notes and serving modes for one and two
cards.

## Llama Superfast

There are many llama forks, but this one is mine :)

We have made llama ultra superfast for v100. We specialize in two models: Qwen 27B 3.8 and Qwen Flash Next 3.8.

The operational models are pretty simple: 27B can run one user on two cards or one user on one card. It's fast in either mode. Flash Next can run with one user on two cards. Two users on two cards for flash next is still experimental. I don't recommend it yet.

### Headline figures: Qwen3.8-27B

Tesla V100 32 GB PCIe cards, MTP speculative decoding, temperature 1.0. Decode speeds are the server's steady rate on real coding-agent work. Prefill means a cold prompt with nothing cached.

- **One user, one card (131,072-token context).**
  - **Decode:** about 100–113 tok/s on long agentic coding sessions, and about 125 tok/s on short coding tasks.
  - **Prefill:** 951 tok/s for a 16K prompt and 770 tok/s for a 64K prompt.
- **One user, two cards (tensor parallel).**
  - **Decode:** 152–156 tok/s on the same agentic work, 1.35× one card.
  - **Prefill:** averages 1,383 tok/s reading a prompt up to 64K, 969 tok/s up to 128K and 686 tok/s up to 240K. On agents' real requests it reads 1.7× as fast as one card.
- **Two users, two cards (131,072 tokens each).**
  - **Decode:** about 200 tok/s combined while both users are decoding (about 100 each), and 150–160 tok/s for one user while the other is idle.
  - **Prefill:** each user's prompts read at 1,020–1,080 tok/s of its own GPU time, or 800–830 tok/s by the wall clock with the other user active.
  - **Throughput:** over a whole agentic session, the pair does about 1.2× the work of one user on the same two cards.

How these were measured, figures by context depth and the 27B's build and run details are in [`README-27B.md`](README-27B.md).

Build and run instructions are at the bottom of this page, under [llama-superfast-v100](#llama-superfast-v100).

## llama-superfast-v100

A llama.cpp fork tuned for the Tesla V100 (Volta, `sm_70`): Qwen3.8-27B on one or two cards and Qwen3.8-Flash-Next on two, with MTP speculative decoding. Build steps, model files, measurements and credits are in [`README-27B.md`](README-27B.md).

### Hardware and software

- One or two Tesla V100 32 GB PCIe (no NVLink needed). Other GPUs are untested.
- NVIDIA driver 580, CUDA 12.8, gcc/g++ 14, Ubuntu 26.04.
- **Two cards in tensor parallel (`GGML_CUDA_P2P=1`) need the IOMMU in passthrough mode.** The cards write to each other's memory directly over PCIe. On our AMD board with the IOMMU translating, the first peer copy raised hundreds of `AMD-Vi IO_PAGE_FAULT`s and stopped the second card (Xid 62), and the server's output was garbage from the first token. Boot with `iommu=pt`: add it to `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`, run `sudo update-grub`, reboot, and check that `grep -o iommu=pt /proc/cmdline` prints it. Intel boards with VT-d on are untested and may need the same. Layer split (the Flash-Next line below) does not set `GGML_CUDA_P2P`.

### Build

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14 -DGGML_NATIVE=OFF -DGGML_CUDA_CUB_3DOT2=ON
cmake --build build --target llama-server -j 12
```

### Models

Put them flat in `./models`.

- **Qwen3.8-27B:** `Qwen3.8-27B-UD-Q4_K_XL.gguf` and the MTP head `mtp-Qwen3.8-27B-Q4_0.gguf` from `unsloth/Qwen3.8-27B-GGUF` (download lines in `README-27B.md`). The draft vocabulary ships in `models/`.
- **Qwen3.8-Flash-Next:** ISTA-DASLab's `Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (`IQ3_XXS`, two shards) and the shared MTP head `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`.

### Run options

Our default serving modes on 32 GB V100s. All commands run from the repository root, with the models in `./models` (`$MODELS`). Sampling: temperature 1.0, top-p 0.95, top-k 20, thinking on.

**1. Qwen3.8-27B, one card** (131,072 tokens)

```
scripts/serve-27b.sh mtp
```

That is one user. For two users on the card, add `--parallel 2` to the script's line in `README-27B.md`; they share the 131,072 tokens (65,536 each). Two users on one card is untested.

**2. Qwen3.8-27B, two cards, two users** (tensor parallel, 131,072 tokens each)

```
CUDA_VISIBLE_DEVICES=0,1 GGML_CUDA_P2P=1 LLAMA_SPEC_DRAFT_VOCAB_FILE=models/draft-vocab-qwen3.8-27b.txt build/bin/llama-server -m $MODELS/Qwen3.8-27B-UD-Q4_K_XL.gguf -md $MODELS/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 7 -c 262144 -fa on -ngl 99 -ctk f16 -ctv f16 -ctkd f16 -ctvd f16 -t 12 -b 4096 -ub 2048 --jinja --parallel 2 --split-mode tensor --temp 1.0 --top-p 0.95 --top-k 20
```

For one user, use `--parallel 1 -c 131072`. Two-user scheduling is on by default; `LLAMA_TU=0` turns it off.

**3. Qwen3.8-Flash-Next, two cards, two users** (layer split, 131,072 tokens each)

```
CUDA_VISIBLE_DEVICES=0,1 GGML_SCHED_COPIES=2 build/bin/llama-server -m $MODELS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf -md $MODELS/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 -c 262144 -fa on -ngl 99 -ncmoe 0 --tensor-split 55,45 -ctk bf16 -ctv bf16 -ctkd f16 -ctvd f16 -t 12 -b 4096 -ub 1024 --spec-draft-ubatch 512 -lm mmap --lazy-mode on --jinja --parallel 2 --no-cache-idle-slots --temp 1.0 --top-p 0.95 --top-k 20
```

This leaves only about 240 MB free on the second card; `-b 6144 -ub 1536` does not fit with two users. For one user with the whole 262,144 tokens, use `--parallel 1 -b 6144 -ub 1536`, which reads prompts faster.

**Switches** (environment, all optional)

- `LLAMA_SPEC_REJECTION=0`: exact-match draft acceptance instead of rejection sampling.
- `LLAMA_SPEC_DRAFT_VOCAB=-1`: draft over the full vocabulary.
- `LLAMA_TU=0`: no two-user scheduling.
- `LLAMA_QSA_ATTN_MMA=1`: Flash-Next's alternative tensor-core prompt attention (slower here, off by default).
- `LLAMA_KQ_MASK_DEVICE=0`: build the attention mask on the host.

Add `--host`, `--port` and your sampling settings as usual. Ours: temperature 1.0, top-p 0.95, top-k 20, thinking on.
