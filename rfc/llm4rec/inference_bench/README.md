# LLM4Rec inference benchmark: sid-gr-inference vs SGLang beam search

Reproduction of the recsys-examples `examples/sid-gr-inference` benchmark
(NVIDIA PR #414: Qwen3-1.7B bf16, beam 256, 3 SID tokens, contexts 1000/5000)
on tzrec-relevant hardware, complementing the README's H100 reference numbers.
Feeds RFC section 10 (`rfc/llm4rec/index.html`).

## Runs

| Target                                                                       | Env                                                                           | Status                                                                                        |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Local 1x A10 23GB (SM86)                                                     | docker `sidgr-bench:cu13` (lmsysorg/sglang:dev-cu13 + cuda-compat-13 on r535) | done - see `results/` and `experiments/llm4rec/rfc_research/serving-sidgr-bench-repro-a10.md` |
| `tianyi_l20n_llm4rec_serve` 72GB SM120 (RTX PRO 5000 Blackwell, driver r580) | native host env (the machine is the dev-cu13 image; no compat)                | this directory                                                                                |

## Code topology (all code flows local -> ty_git -> remote pull; never edit remote in place)

| Repo                            | Bench branch                  | Content                                                                                                                                          |
| ------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| tiankongdeguiji/sglang          | `llm4rec/beam-search-bench`   | pin of cswuyg/sglang `feature/beam_search` @ 2aac32adcf (upstream PR #15645)                                                                     |
| tiankongdeguiji/recsys-examples | `llm4rec/sidgr-bench`         | base 5dc46a2 + env-gated SGLang knobs (`SGLANG_MEM_FRACTION_STATIC` / `SGLANG_MAX_RUNNING_REQUESTS` / `SGLANG_CONTEXT_LENGTH`), inert by default |
| tiankongdeguiji/TorchEasyRec    | `rfc_llm4rec_inference_bench` | this directory                                                                                                                                   |

Remote checkouts live under `/mnt/data/hongsheng.jhs/{sglang,recsys-examples,TorchEasyRec}`.

## Reproduce on the remote host

```bash
# on tianyi_l20n_llm4rec_serve
cd /mnt/data/hongsheng.jhs/TorchEasyRec/rfc/llm4rec/inference_bench
bash setup_remote.sh            # branches, pip install -e (no-deps), import gates, model via hf-mirror
nohup bash run_l20n_bench.sh > /mnt/data/hongsheng.jhs/bench_results/bench.log 2>&1 &
```

`run_l20n_bench.sh` runs, in order: env manifest; gates (kernel backend selection,
`gr_decode_atten` smoke + correctness pytest - the first SM120 validation - and the
beam-fork Engine sanity); the faithful offline perf grid (`REPEAT=3`, prefill CUDA
graphs ON, no deviation envs); the offline accuracy grid; online serving for both
sides (GR HTTP :8000, SGLang fork :30000) driven by `sglang.bench_serving`.

## Operational notes (learned on the A10 run; apply here too)

- The SGLang fork's `/health_generate` reports a FALSE 503 under `--enable-beam-search`
  (detokenizer heartbeat never starts) while `/generate` serves correctly - the runner
  probes `/generate`.
- `bench_serving` warmup slots: warmup 16 x beam 256 = 4096 request slots; the fork
  treats slot exhaustion as a FATAL scheduler error. At 72GB the default clamp is 4096
  (exact fit, same as H100); on smaller GPUs set `WARMUP_REQUESTS<=8` and the env-gated
  knobs from the recsys branch.
- The harness reports wall times even for cells whose requests all FAILED - the runner
  prints per-cell `succeeded_requests` and results are only valid where it equals the
  request count.

## Results

`results/comparison.md` carries the three-way comparison (A10 / SM120-72GB / H100
README): GR beats the SGLang beam fork in all 8 offline cells on every arch
(SM120: 1.451-1.607x, online 1.50x at 64/64 completion, top1 parity 28/30), and the
run doubles as the first SM120/Blackwell validation of `gr_decode_atten`.
Raw artifacts under `results/sm120/`.
