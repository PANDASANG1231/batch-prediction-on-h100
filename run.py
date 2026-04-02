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
        quantization="fp8" if use_fp8 else None,  # [Step 2] FP8: 模型权重减半，吞吐提升~1.3-1.5x
        enable_prefix_caching=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_num_seqs=MAX_NUM_SEQS,
    )
    return llm


def make_sampling_params(guided=False):
    """
    guided=False → Step 1/2: 普通greedy decode
    guided=True  → Step 3: 限制输出只能是预定义类别
    """
    kwargs = dict(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)

    if guided:
        # [Step 3] Guided Decoding: 强制输出为CATEGORIES中的一个
        # 效果：decode步数从~10+降到1-3步，速度更快，且不会出现脏输出
        kwargs["guided_decoding"] = {
            "choice": CATEGORIES,  # vLLM会自动约束输出为这些选项之一
        }
        kwargs["max_tokens"] = MAX_TOKENS_GUIDED  # [Step 3] 输出被约束后不需要20 tokens了

    return SamplingParams(**kwargs)


def run_inference(df, llm, params, batch_size=IO_BATCH_SIZE):
    prompts = [
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": str(d)}]
        for d in df[INPUT_COL]
    ]

    results = []
    t0 = time.time()
    total = len(prompts)

    for i in range(0, total, batch_size):
        batch = prompts[i : i + batch_size]
        outputs = llm.chat(batch, params)
        results.extend([o.outputs[0].text.strip() for o in outputs])

        done = i + len(batch)
        speed = done / (time.time() - t0)
        print(f"[{done}/{total}] {speed:.0f} rec/s | ETA: {(total - done) / speed / 60:.1f} min")

    df["prediction"] = results
    print(f"\nDone. {total} records in {(time.time() - t0) / 60:.1f} min")
    return df


# ============================================================
# Usage: 逐步升级，每步对比throughput
# ============================================================

# --- Step 1: FP16 baseline ---
# llm = init_model(use_fp8=False)
# params = make_sampling_params(guided=False)
# df = load_data("data.parquet", n=100)
# df = run_inference(df, llm, params)

# --- Step 2: 只改一处 → 开FP8 ---
# llm = init_model(use_fp8=True)          # <-- 这里改了
# params = make_sampling_params(guided=False)
# df = load_data("data.parquet", n=100)
# df = run_inference(df, llm, params)

# --- Step 3: 再改一处 → 开Guided Decoding ---
# llm = init_model(use_fp8=True)
# params = make_sampling_params(guided=True)  # <-- 这里改了
# df = load_data("data.parquet", n=100)
# df = run_inference(df, llm, params)
