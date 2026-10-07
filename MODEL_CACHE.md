# Local model cache

Inference is offline by design. Provide either:

1. A Hugging Face cache containing the snapshot for
   `depth-anything/Depth-Anything-V2-Small-hf`, then run with
   `--cache-dir /path/to/hf_cache`; or
2. An extracted local Transformers model directory and pass
   `--model /path/to/Depth-Anything-V2-Small-hf`.

The directory must contain `config.json`, `preprocessor_config.json`, and
`model.safetensors` (plus any tokenizer/image-processor files supplied by the
model). The converter sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`local_files_only=True`; if files are missing it fails with an actionable error
instead of attempting a network download. Model weights are intentionally not
checked into this repository.
