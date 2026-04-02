# Batch Inference Plan: Qwen on H100 with vLLM

## Step 1: vLLM + FP16 + Prefix Caching Baseline
- Install vLLM, load Qwen model with FP16
- Enable prefix caching: `--enable-prefix-caching`
- Write batch inference script: read input CSV/parquet → async feed to vLLM → write results
- Run 100 records first, log per-record latency and throughput (tokens/s)
- Then run full 1M records

## Step 2: FP8 Quantization
- Switch to FP8: `--quantization fp8` (H100 native support)
- Re-run 100 records, compare throughput vs FP16 baseline
- Verify output quality unchanged on sample

## Step 3: Guided Decoding
- Define allowed output categories as a JSON schema or regex
- Use vLLM `--guided-decoding-backend outlines` with constrained output
- Set `max_tokens` to minimum needed (e.g. 20)

## Step 4: Tuning
- `temperature=0` (greedy decode)
- Increase `max_num_seqs` to maximize GPU utilization (start 256, try 512)
- Sort input by token length before batching to reduce padding waste
- Async I/O: don't let data loading block GPU

## Notes
- Model: [FILL IN exact model name, e.g. Qwen/Qwen2.5-7B-Instruct]
- Input: [FILL IN path and format]
- Output: classification label per record
- Each step: log throughput, compare to previous step