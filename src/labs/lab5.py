"""GPT-2 baseline submission for local Lab 4 smoke tests.

Replace this with your model's Hugging Face ID for a real submission.
"""

eval_model_id = "Dxnn1u/llama-3.2-1B"
training_run_url = "https://wandb.ai/<WANDB_ENTITY>/lab5-training-llama/runs/9tgz2zs6"
training_config = {
    "optimizer": {
        "name": "Muon (hidden 2D matrices) + AdamW (everything else)",
        "parameter_groups": {
            "muon": {
                "params": "all 2D weights except the embedding: q/k/v/o and gate/up/down projections of the 16 layers",
                "lr": 1e-3,
                "momentum": 0.95,
                "nesterov": True,
                "ns_steps": 5,
                "weight_decay": 0.1,
                "adjust_lr_fn": "match_rms_adamw",
            },
            "adamw_embedding": {
                "params": "tied input/output embedding (lm_head shares it)",
                "lr": 1e-3,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1,
            },
            "adamw_norms": {
                "params": "all RMSNorm weights (1D)",
                "lr": 1e-3,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
        },
        "schedule": {
            "type": "linear warmup then cosine decay to 0 (per optimizer step, all groups)",
            "peak_lr": 1e-3,
            "warmup_steps": 573,
            "total_steps": 11444,
            "min_lr": 0.0,
        },
        "grad_clip_norm": 1.0,
        "batch": {
            "tokens_per_step": 262_144,
            "sequences_per_step": 32,
            "per_gpu_micro_batch": 1,
            "gradient_accumulation": 4,
            "gpus": 8,
        },
        "weights_dtype": "float32",
        "compute_dtype": "bfloat16 autocast",
        "torch_compile": True,
        "seed": 0,
    },
    "context_length": 8192,
    "initialization": "random",
    "dataset": "allenai/dolma3_mix-150B-1025",
    "shuffle_seed": 42,
    "holdout_documents": 50_000,
    "holdout_excluded_from_training": True,
    "non_padding_tokens_seen": 2_999_975_936,
    "repeated_tokens_counted": True,
    "duration_seconds": 12_270,
    "gpu_type": "H200",
    "gpu_count": 8,
    "validation_documents": 10_000,
}

