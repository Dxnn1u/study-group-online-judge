"""Lab 5 submission: Llama 3.2 1B (random init) trained on Dolma 3 with Muon-VS + AdamW.

Run: 8x H200, 2.0B tokens, 7,629 optimizer steps of 262,144 tokens each, 16.97 H200-hours.
Code: https://github.com/Dxnn1u/ML-study-group-lab-5 (train.py, data.py, split.py).
"""

eval_model_id = "Dxnn1u/llama-3.2-1B-lab5-2b"
training_run_url = "https://wandb.ai/cerulean-labs/lab5-training-llama/runs/d77j8ri2"
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
                "weight_decay": 0.8,
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
            "type": "warmup, constant, then linear decay over the last 40% (per optimizer step, all groups)",
            "peak_lr": 1e-3,
            "warmup_steps": 200,
            "decay_steps": 3052,
            "total_steps": 7629,
            "min_lr": 1e-4,
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
        "attention": "sdpa with a block-diagonal mask derived from document starts",
        "torch_compile": False,
        "seed": 0,
    },
    "context_length": 8192,
    "initialization": "random",
    "dataset": "allenai/dolma3_mix-150B-1025",
    "shuffle_seed": 42,
    "holdout_documents": 50_000,
    "holdout_excluded_from_training": True,
    "holdout_note": (
        "the split runs before tokenizing and packing; 5,195 of the 50,000 held-out documents "
        "fall in the 611 shards this run's data was drawn from and were skipped there, and the "
        "rest are in shards that were never read"
    ),
    "packing": {
        "documents_truncated_to": 8192,
        "document_attention_mask": True,
        "position_ids_reset_per_document": True,
        "documents_interleaved_across_shards": True,
    },
    "data_selection": {
        "shards_used": 611,
        "shards_total": 6081,
        "method": "stratified by source-topic group, then sub-sampled within each shard",
        "tokens_available": 9_482_398_080,
        "source_mix_total_variation_vs_holdout": 0.0089,
    },
    "non_padding_tokens_seen": 1_999_896_576,
    "repeated_tokens_counted": True,
    "duration_seconds": 7_638,
    "h200_hours": 16.97,
    "gpu_type": "H200",
    "gpu_count": 8,
    "validation_documents": 10_000,
}
