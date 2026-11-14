# benchmarker — benchmark and tuning tool for local GGUF models

One tool for all GGUF models on the GPU PC. It reads each model's architecture, layers, experts and native context from the GGUF file. It finds the image encoder (`mmproj`) of vision models and tests text and images. It writes a ready llama-swap config from the results.

Requirements: Python 3.11 or later, llama.cpp (`llama-server`, `llama-bench`), an AMD GPU. No extra Python packages.

## What is new in 2.3

Version 2.2 rejected a context after one wrong value at 90% fill. This hid contexts above 130K that work well in practice. Version 2.3 uses **profiles**: sets of rules that you can read, change and compare.

| Change                                          | Effect                                                                                                                                                        |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Profiles `strict`, `balanced` (default), `long` | `strict` = the rules of 2.2. `balanced` = the new default. `long` = largest context first                                                                     |
| Target context (131072)                         | The report says if each model reaches it. If not, it says which rule stopped it                                                                               |
| Recall test with 5 values                       | One wrong value out of 5 is allowed (`balanced`). 90% fill is no longer a pass/fail gate                                                                      |
| `keyed` test                                    | The model finds `Project kelmora: access code 41872` by name, like a symbol in code. The old test (`line`) is still available                                 |
| Speed rule                                      | Slow = more than 25% below the best speed, or below 15 t/s with a filled context. Speed is measured with the context 50% full for contexts from the target up |
| Decision table per model                        | Each tested context shows speed, recall and the reason for a rejection                                                                                        |
| Cache type choice                               | If more than one KV cache type reaches the target: the best quality that keeps 85% of the fastest speed                                                       |
| Fixes                                           | A context with only request errors can no longer win over a context that passed. Server logs go to `results/logs/` (not `/tmp`)                               |
| `report` and `swap`                             | Judge saved results with the current rules. Use `--profile` to compare profiles without a new sweep                                                           |
| Progress display                                | Every line has a tag with model, phase and step. `status` shows the run from a second terminal (see "Progress")                                               |
| `tests/`                                        | 45 pytest tests for the rules, progress display, and config — `python3 -m pytest tests/`                              |

History: 2.2 cache choice by speed at depth. 2.1 stepwise accuracy search. 2.0 automatic paths, vision tests, `doctor`, `swap --install`.

## Install

```bash
mkdir -p ~/llm-tests && cd ~/llm-tests
# clone or copy the benchmarker package
uv pip install -e .                    # installs benchmarker CLI
python -m benchmarker doctor           # all lines must be OK or WARN, no FAIL
python -m benchmarker init             # scans the models folder and writes models.toml
python -m benchmarker info             # check what it found
```

## Tests

```bash
uv pip install -e ".[dev]"             # or: uv add --dev pytest
uv run pytest tests/ -v                # 45 tests, all must pass
```

## Profiles

Set the profile in `models.toml` (`[defaults]`, `profile = "long"`), with `--profile`, or with `BENCHMARKER_PROFILE`.

| Rule                                                    | `strict` (2.2) | `balanced` (default) | `long`      |
| ------------------------------------------------------- | -------------- | -------------------- | ----------- |
| `target_context`                                        | 65536          | **131072**           | **131072**  |
| `speed_tolerance_pct` (slower than the best = rejected) | 10             | 25                   | 40          |
| `min_tg_tps` (with a filled context)                    | off            | 15                   | 10          |
| `needle_style`                                          | `line`         | `keyed`              | `keyed`     |
| `needles` (values in one test)                          | 3              | 5                    | 5           |
| `accuracy_pct` (fill levels that decide)                | 50, 90         | 50, 75               | 50          |
| `accuracy_min_pct` (share that must be right)           | 100            | 80 (4 of 5)          | 80 (4 of 5) |

Any value can be set in `[defaults]`; it then replaces the profile value.

Use `balanced` for daily work. Use `long` if `balanced` rejects a context you want and you accept slower generation. Use `strict` to compare with results from 2.2. Recall results of the `line` and `keyed` tests are not equal. Do not compare them.

**What the rules mean for coding agents.** OpenCode starts to compact (summarize) the conversation before the context is full. A test at 50% and 75% fill is close to real use. A test at 90% shows a limit that the agent rarely reaches.

## Progress

Each long command (`sweep`, `fill`, `vision`) shows where it is.

```
---- sweep model 2/3: my-model | accuracy 0/3 | run 41:10 | model 12:02 | later models ~28:00 ----
[2/3 accuracy 1/3 +41:12]   accuracy: ctx=196608 ngl=41 kv=f16 n-cpu-moe=25
[2/3 accuracy 1/3 +46:30]   fill   50%  ctx=196608 prompt=97245 ... 5/5 PASS
  ... still working: sweep model 2/3: my-model | accuracy 2/3 | long-prompt tests: ctx=131072 ... | run 52:00
```

| Part                        | Meaning                                                                                                                                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `[2/3 accuracy 1/3 +41:12]` | Model 2 of 3, phase `accuracy`, step 1 of 3, 41 min 12 s since the start. At the `fit` phase the step number only counts the tests (`fit #7`), because the number of fit tests is not known in advance |
| `----` line                 | Starts each model and each phase. It shows the current detail, the time for the run and for this model, and (after the first model) the time for the models that follow                                |
| `... still working`         | Printed when nothing else was printed for 60 seconds, for example during a model load or a long prompt. It shows what the tool is doing now                                                            |

Phases: `fit`, `speed`, `accuracy`, `tune`, `image` (for `fill`: `fill`).

In a second terminal (for example another SSH session):

```bash
python -m benchmarker status               # once
python -m benchmarker status --watch 10    # every 10 seconds, ends with the run
```

`status` reads `results/progress.json`. It shows `running`, `finished`, `stopped` (Ctrl+C or kill) or `failed`. If the state is `running` but the process is gone, it says so. Use `--results` if you started the run with this option.

## How the decision works

For each model and each KV cache type, benchmarker checks every tested context:

1. **Fit.** The model loads with the safe margin (`min_free_vram_mb`, `min_free_ram_mb`) and answers a text request (and an image request for vision models).
2. **Speed.** Generation speed (`tg128`) is not more than `speed_tolerance_pct` below the best speed of this cache type. With a filled context it is at least `min_tg_tps`. For contexts from the target up, benchmarker measures the speed with the context 50% full (`bench_deep_fill_pct`).
3. **Recall** (only with `--accuracy`). The context is filled to each level of `accuracy_pct`. The model must find `accuracy_min_pct` of the hidden values. The test starts at the largest context that passes the speed rule and goes down (`accuracy_max_steps`, default 4) until one passes. If a level fails, the next level is not tested.

**Optimum context** = the largest context that passes 1, 2 and 3. Without `--accuracy`, the result says `speed only`.

**Cache type.** If more than one cache type reaches the target, benchmarker takes the best quality (f16, then q8_0, then q4_0) whose speed with a filled context is at least 85% of the fastest. If none reaches the target, it takes the largest context.

**Target.** The result line and the report say `TARGET 131072: met` or `NOT met` with the reason:

| Reason in the report                             | Action                                                                                                                                                                 |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `no tested context >= 131072 fits in memory`     | Add `"q4_0"` to `cache_types` of the model. Vision models: `mmproj_offload = false`. Or lower `min_free_vram_mb`                                                       |
| `too slow: ... % below the best`                 | Use profile `long`, or accept the smaller context                                                                                                                      |
| `too slow: ... t/s with N tokens in the context` | Lower `min_tg_tps`, or use a smaller context                                                                                                                           |
| `long-prompt test failed (2/5@50%, needs 80%)`   | The model loses values at this context. Use a smaller context, `cache_types = ["f16", "q8_0"]` (a less compressed KV cache), or accept it with `accuracy_min_pct = 60` |
| `long-prompt test not run for this context`      | Run `sweep <model> --accuracy` again                                                                                                                                   |

Compression of the KV cache (`q4_0`) can lower recall at long contexts. The recall test shows this: run the test for `q8_0` and `q4_0` and compare.

## Commands

| Command                | Function                                                                                                      | llama-swap |
| ---------------------- | ------------------------------------------------------------------------------------------------------------- | ---------- |
| `doctor`               | Checks paths, GPU devices, flags of the build, groups, VRAM, API key, llama-swap, profile                     | Any        |
| `init`                 | Adds new models to `models.toml` and shows the `mmproj` it will use. Keeps your changes                       | —          |
| `info`                 | Model details, size and vision file                                                                           | —          |
| `sweep <models\|all>`  | Fit (text + image), speed, tuning, image test, optional recall. Writes the report                             | Stopped    |
| `fill <models\|all>`   | Long-prompt recall test through llama-swap                                                                    | Running    |
| `vision <models\|all>` | Image test through llama-swap                                                                                 | Running    |
| `status [--watch N]`   | Shows which model, phase and step is running now                                                              | Any        |
| `report`               | Writes one Markdown report with all results                                                                   | —          |
| `swap [--install]`     | Writes the llama-swap config from the results. `--install` makes a backup and replaces `/etc/llama-swap.yaml` | —          |

Options for all commands: `--profile`, `--backend`, `--llama-dir`, `--models-dir`, `--server`, `--bench`, `--config`, `--results`.

## Typical runs

```bash
cd ~/llm-tests

python -m benchmarker sweep all --stop-swap                         # all models, speed only (approx. 1–2 h)
python -m benchmarker sweep all --accuracy --stop-swap              # with recall (overnight)
python -m benchmarker sweep <model-name> --accuracy --stop-swap  # one model
python -m benchmarker sweep all --quick --skip-tune --stop-swap     # fast check after an update

python -m benchmarker report --profile long                         # judge the saved results with another profile
python -m benchmarker swap --profile long                           # config for that profile

python -m benchmarker fill <model-name> --pct 50,75              # through llama-swap
python -m benchmarker vision all                                    # images through llama-swap

python -m benchmarker swap                                          # check the file and the warnings
python -m benchmarker swap --install                                # backup + install
```

`--stop-swap` and `swap --install` use `sudo`.

**Time.** A recall test at 131072 contexts fills 65K and 98K tokens. At about 1000 tokens/s for the prompt, this is 2 to 3 minutes per fill. `accuracy_max_steps` limits the number of contexts to test for each cache type. The speed check with a half-full context adds a few minutes for each context from the target up.

## What `sweep` measures

| Phase                         | What it does                                                                                                                                                                                                                          | Output                         |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| 1. Fit                        | For each context, it finds the setting that loads with a safe margin. **MoE:** fewest expert layers in RAM (`--n-cpu-moe`). **Dense:** largest context per KV cache type; if nothing fits, the number of layers that must stay in RAM | `fit.csv`                      |
| 2. Speed                      | `llama-bench`: prompt (`pp512`) and generation (`tg128`) with an empty context and at 8K and 32K filled. Contexts from the target up: generation with the context 50% full                                                            | `speed.csv`, `bench-raw.jsonl` |
| 3. Recall (`--accuracy`)      | See "How the decision works"                                                                                                                                                                                                          | `accuracy.csv`                 |
| 4. Tune                       | On the final optimum: batch sizes 256–2048 and, when the CPU does part of the work, 3 thread counts. The best one must still load                                                                                                     | `tune.csv`                     |
| 5. Image test (vision models) | Starts the tuned optimum and sends 3 generated images with a 4-digit number in a coloured frame. The model must read the number and the colour                                                                                        | `vision.csv`                   |

`llama-bench` does not load the image encoder. The speed values are for text. The fit tests and the image test use `llama-server` with the image encoder.

## models.toml

`init` writes this file. Any value of `DEFAULTS` or of the profile in `benchmarker.config` can be set in `[defaults]`.

```toml
[defaults]
# models_dir = "/srv/models"        # found automatically
# llama_dir = "/opt/llama.cpp"      # found automatically
backend = "rocm"
min_free_vram_mb = 512              # 1024 if the screen is on this GPU
min_free_ram_mb = 4096
profile = "balanced"                # strict, balanced or long
# target_context = 196608           # replaces the value of the profile
# accuracy_max_steps = 4
# visible_devices = "auto"          # "0" = this device index, "" = do not set
# key_file = "/etc/llama-swap.env"
# swap_config = "/etc/llama-swap.yaml"
# swap_url = "http://127.0.0.1:9931"

[models."my-model"]
path = "my-model/MyModel-UD-Q4_K_M.gguf"
display_name = "My Model (text)"
cache_types = ["f16", "q8_0"]
run_flags = ["--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.01"]

[models."my-vision-model"]
path = "my-vision-model/MyVisionModel-UD-Q4_K_M.gguf"
display_name = "My Vision Model (text + image)"
cache_types = ["f16", "q8_0"]
mmproj_offload = false              # image encoder in RAM: more layers on the GPU
run_flags = ["--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0.0"]
```

| Key                       | Function                                                                                               |
| ------------------------- | ------------------------------------------------------------------------------------------------------ |
| `path`                    | Model file, relative to `models_dir` (absolute also works). For split models, the `00001` part         |
| `mmproj`                  | `"auto"` (default) = best `mmproj-*.gguf` in the model folder. A path = that file. `""` = no vision    |
| `mmproj_offload`          | `false` keeps the image encoder in RAM (less VRAM, slower images)                                      |
| `display_name`            | Name in llama-swap and in clients                                                                      |
| `run_flags`               | Sampling settings for llama-swap only, not used in the tests                                           |
| `extra_flags`             | Added to each test and to llama-swap. Included in the fit tests                                        |
| `contexts`, `cache_types` | Own lists. Contexts above the native context are removed. Include a value at or above `target_context` |
| `tools`                   | `false` if the model has no tool calling (written to `capabilities`)                                   |
| `skip`                    | `true` leaves the model out of `all`                                                                   |

## Add a new model

```bash
mkdir -p /srv/models/<model-name>
hf download <repo> --include "*<QUANT>*" "mmproj-F16.gguf" --local-dir /srv/models/<model-name>
python -m benchmarker init && python -m benchmarker info
python -m benchmarker sweep <model-name> --accuracy --stop-swap
python -m benchmarker swap --install
```

The folder name becomes the model ID in llama-swap. Leave out `mmproj-F16.gguf` for text-only models.

After `swap --install`, set `limit.context` of each model in the OpenCode config to the `-c` value in `/etc/llama-swap.yaml`.

## Results

All files are in `~/llm-tests/results/`. Each run has a `run_id`. Old results stay for comparison.

| File                               | Content                                                                                                                                                                                                       |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `report-<run_id>.md`               | **Send this file for review.** System, rules, models, recommendations with target status, decision table per context, speed, tuning, image test, recall, fit tests, current and recommended llama-swap config |
| `fit.csv`, `speed.csv`, `tune.csv` | Fit, speed and tuning values                                                                                                                                                                                  |
| `vision.csv`                       | Each image: prompt tokens, image prompt speed, time to first token, expected value, answer                                                                                                                    |
| `accuracy.csv`                     | Each recall test: context, cache, fill level, score, style, error                                                                                                                                             |
| `state-<model>@<backend>.json`     | Latest result per model and backend (used by `report` and `swap`)                                                                                                                                             |
| `logs/<model>-<phase>.log`         | Output of the last test server (first line = full command)                                                                                                                                                    |
| `llama-swap.recommended.yaml`      | Written by `swap`                                                                                                                                                                                             |

The report and the generated config contain no API key (`apiKeys` reads `${env.LLAMA_SWAP_API_KEY}`).

## Read the results

| Value                    | Meaning                                                                                     |
| ------------------------ | ------------------------------------------------------------------------------------------- |
| `pp512@0` / `tg128@0`    | Prompt and generation speed with an empty context                                           |
| `tg128@65536`            | Generation speed with 65K tokens in the context                                             |
| `recall 4/5@50% 5/5@75%` | Values found at each fill level                                                             |
| `LOW_MARGIN`             | Loaded, but less than the safe margin was left. Not counted as fit                          |
| `FAIL_REQUEST`           | Loaded, but the warm-up or image request failed (often out of memory for the image buffers) |
| Image `PARTIAL`          | Number or colour wrong. Check the `mmproj` file, or try `image_px = 1024`                   |
| Image `FAIL`             | Request failed. Read `results/logs/<model>-optimum.log`                                     |

## Troubleshooting

| Problem                                   | Action                                                                                    |
| ----------------------------------------- | ----------------------------------------------------------------------------------------- |
| `doctor`: server or bench `FAIL`          | Set `llama_dir` or `backend` in `[defaults]`, or use `--llama-dir`                        |
| `doctor`: GPU devices `WARN`              | More than one GPU is visible. Set `visible_devices = "0"` or disable the iGPU in the BIOS |
| `profile override` WARN                   | A value in `[defaults]` replaces the profile. Remove the line, or keep it on purpose      |
| `llama-swap is running`                   | Add `--stop-swap`                                                                         |
| Every fit test is `FAIL`                  | Read `results/logs/<model>-fit.log` (the first line is the full command)                  |
| Dense model: "layers must stay in RAM"    | Set `mmproj_offload = false` for that model, then run `sweep` again                       |
| `bench failed`                            | Run the printed command yourself. Old builds may not support `--n-cpu-moe` or `-d`        |
| `Cannot read /etc/llama-swap.env`         | `sudo chown $USER: /etc/llama-swap.env`, or `export LLAMA_SWAP_API_KEY=<key>`             |
| `swap --install`: models would be removed | The current config has models without results. Run `sweep` for them, or use `--force`     |
| Run stopped with Ctrl+C                   | All test servers are stopped. Finished models are saved; run `report`                     |
| After a change of the code                | `python3 -m pytest tests/`                                                              |
