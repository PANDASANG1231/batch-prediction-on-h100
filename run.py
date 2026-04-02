"""
Batch Inference: vLLM + Qwen on H100

Step 1: FP16 + Prefix Caching (baseline)
Step 2: FP8 Quantization
Step 3: Guided Decoding (constrained output)
"""

import time
import json
import pandas as pd
from vllm import LLM, SamplingParams

# ============ CONFIG ============
# Model
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # FILL IN
DTYPE = "float16"
MAX_MODEL_LEN = 512
GPU_MEM_UTIL = 0.9
MAX_NUM_SEQS = 256

# Data
INPUT_COL = "description"                  # FILL IN
IO_BATCH_SIZE = 1000

# Sampling
TEMPERATURE = 0
MAX_TOKENS = 20
MAX_TOKENS_GUIDED = 5

# [Step 3] Categories for guided decoding
CATEGORIES = ["groceries", "dining", "travel", "utilities", "entertainment", "transfer", "other"]  # FILL IN

SYSTEM_PROMPT = f"""You are a transaction classifier. Classify the transaction into one of these categories:
{json.dumps(CATEGORIES)}
Respond with ONLY the category name, nothing else."""
# ================================


def load_data(path, n=None):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if n:
        df = df.head(n)
    df = df.sort_values(by=INPUT_COL, key=lambda x: x.str.len()).reset_index(drop=True)
    return df


def init_model(use_fp8=False):
    """
    use_fp8=False → Step 1: FP16 baseline
    use_fp8=True  → Step 2: FP8 quantization
    """
    llm = LLM(
        model=MODEL_NAME,
        dtype=DTYPE,
        quantization="fp8" if use_fp8 else None,  # [Step 2] FP8: halves model weight size, ~1.3-1.5x throughput gain
        enable_prefix_caching=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_num_seqs=MAX_NUM_SEQS,
    )
    return llm


def make_sampling_params(guided=False):
    """
    guided=False → Step 1/2: standard greedy decode
    guided=True  → Step 3: constrain output to predefined categories only
    """
    kwargs = dict(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    if guided:
        # [Step 3] Guided Decoding: forces output to be one of CATEGORIES
        # reduces decode steps from ~10+ to 1-3, faster and no malformed outputs
        kwargs["guided_decoding"] = {
            "choice": CATEGORIES,  # vLLM automatically constrains output to these choices
        }
        kwargs["max_tokens"] = MAX_TOKENS_GUIDED  # [Step 3] constrained output needs far fewer tokens

    return SamplingParams(**kwargs)


def run_inference(df, llm, params, batch_size=IO_BATCH_SIZE):
    prompts = [
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": str(d)}]
        for d in df[INPUT_COL]
    ]

    results = []
    total_output_tokens = 0
    t0 = time.time()
    total = len(prompts)

    for i in range(0, total, batch_size):
        batch = prompts[i : i + batch_size]
        outputs = llm.chat(batch, params)
        for o in outputs:
            results.append(o.outputs[0].text.strip())
            total_output_tokens += len(o.outputs[0].token_ids)

        done = i + len(batch)
        speed = done / (time.time() - t0)
        print(f"[{done}/{total}] {speed:.0f} rec/s | ETA: {(total - done) / speed / 60:.1f} min")

    elapsed = time.time() - t0
    df["prediction"] = results
    stats = {
        "records": total,
        "elapsed_s": round(elapsed, 2),
        "rec_per_s": round(total / elapsed, 1),
        "tok_per_s": round(total_output_tokens / elapsed, 1),
    }
    print(f"\nDone. {total} records in {elapsed / 60:.1f} min | {stats['rec_per_s']} rec/s | {stats['tok_per_s']} tok/s")
    return df, stats


def benchmark(data_path, n=500):
    """
    Run all 3 steps on the same n records and print a comparison table.
    Each LLM is destroyed before loading the next to free GPU memory.
    """
    import gc
    import torch

    df = load_data(data_path, n=n)
    results = {}

    steps = [
        ("Step 1: FP16 + prefix cache",  dict(use_fp8=False), dict(guided=False)),
        ("Step 2: FP8 + prefix cache",   dict(use_fp8=True),  dict(guided=False)),
        ("Step 3: FP8 + guided decoding", dict(use_fp8=True), dict(guided=True)),
    ]

    for label, model_kwargs, param_kwargs in steps:
        print(f"\n{'='*50}\n{label}\n{'='*50}")
        llm = init_model(**model_kwargs)
        params = make_sampling_params(**param_kwargs)
        _, stats = run_inference(df.copy(), llm, params)
        results[label] = stats

        # release GPU memory before next step
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    # Print comparison table
    print("\n" + "=" * 78)
    print(f"{'Step':<35} {'rec/s':>8} {'tok/s':>10} {'min/1M rec':>12} {'speedup':>9}")
    print("-" * 78)
    baseline_rps = results[steps[0][0]]["rec_per_s"]
    for label, stats in results.items():
        min_per_1m = round(1_000_000 / stats["rec_per_s"] / 60, 1)
        speedup = f"{stats['rec_per_s'] / baseline_rps:.2f}x"
        print(f"{label:<35} {stats['rec_per_s']:>8.1f} {stats['tok_per_s']:>10.1f} {min_per_1m:>12.1f} {speedup:>9}")
    print("=" * 78)


# ============================================================
# Usage A: run one step at a time
# ============================================================

# --- Step 1: FP16 baseline ---
# llm = init_model(use_fp8=False)
# params = make_sampling_params(guided=False)
# df = load_data("data.parquet", n=100)
# df = run_inference(df, llm, params)

# --- Step 2: one change only → enable FP8 ---
# llm = init_model(use_fp8=True)          # <-- changed here
# params = make_sampling_params(guided=False)
# df = load_data("data.parquet", n=100)
# df = run_inference(df, llm, params)

# --- Step 3: one change only → enable Guided Decoding ---
# llm = init_model(use_fp8=True)
# params = make_sampling_params(guided=True)  # <-- changed here
# df = load_data("data.parquet", n=100)
# df, stats = run_inference(df, llm, params)

# ============================================================
# Usage B: benchmark all 3 steps back-to-back, prints comparison table
# ============================================================
# benchmark("data.parquet", n=500)
