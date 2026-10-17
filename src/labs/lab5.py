"""Lab 5 submission: Llama 3.2 1B (random init) trained on Dolma 3 with Muon-VS + AdamW.

Run: 8x H200, 6.0B non-padding tokens, 23,034 optimizer steps, 52.44 H200-hours.
Code: https://github.com/Dxnn1u/ML-study-group-lab-5 (train.py, data.py, split.py).
"""

eval_model_id = "Dxnn1u/llama-3.2-1B-lab5-6b-d40"
training_run_url = "https://wandb.ai/cerulean-labs/lab5-training-llama/runs/qbw8ooaq"
training_config = {
    "optimizer": {
        "name": "Muon-VS (hidden 2D matrices) + AdamW (everything else)",
        "muon_variant": (
            "variance-scaled Muon (arXiv 2601.14603): the momentum is divided by its "
            "running second moment before the Newton-Schulz orthogonalization; no extra "
            "hyperparameters, 1.7% slower per step"
        ),
        "parameter_groups": {
            "muon": {
                "params": "all 2D weights except the embedding: q/k/v/o and gate/up/down projections of the 16 layers",
                "lr": 2e-3,
                "momentum": 0.95,
                "nesterov": True,
                "ns_steps": 5,
                "weight_decay": 0.8,
                "adjust_lr_fn": "match_rms_adamw",
                "variance_scaling": True,
            },
            "adamw_embedding": {
                "params": "tied input/output embedding (lm_head shares it)",
                "lr": 2e-3,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1,
            },
            "adamw_norms": {
                "params": "all RMSNorm weights (1D)",
                "lr": 2e-3,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
        },
        "schedule": {
            "type": "warmup, constant, then linear decay over the last 40% (per optimizer step, all groups)",
            "peak_lr": 2e-3,
            "warmup_steps": 200,
            "decay_steps": 9214,
            "total_steps": 23_034,
            "min_lr": 2e-4,
        },
        "schedule_note": (
            "the first 13,818 steps are shared with a sibling run that decayed over its last "
            "20% instead; both are 23,034 steps over the same 6B tokens, and the branch point "
            "is inside the constant-learning-rate phase of both, so each model's trajectory is "
            "exactly the one it would have had on its own. 52.44 H200-hours of compute produced "
            "this trajectory, 20.86 of them after the branch"
        ),
        "grad_clip_norm": 1.0,
        "batch": {
            "non_padding_tokens_per_step": 260_478,
            "positions_per_step": 262_144,
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
        "no_document_crosses_a_block_boundary": True,
        "packing_note": (
            "documents are packed into 8192-token blocks by random probe until one fits the "
            "room left, and the remainder of a block is padded rather than filled with the "
            "start of a document whose rest would land in the next block. Writing documents "
            "end to end and cutting every 8192 tokens instead leaves 20.65% of the tokens in "
            "fragments that begin mid-text with no BOS at position 0, which is a context the "
            "judge never scores"
        ),
        "pad_token": "<|finetune_right_pad_id|>",
        "pad_token_id": 128_004,
        "padding_fraction_of_positions": 0.0064,
        "padding_masked_from_loss": True,
    },
    "data_selection": {
        "shards_used": 611,
        "shards_total": 6081,
        "method": "stratified by source-topic group, then sub-sampled within each shard",
        "tokens_available": 9_482_398_080,
        "blocks_available": 1_164_921,
        "source_mix_total_variation_vs_holdout": 0.0089,
    },
    "non_padding_tokens_seen": 5_999_850_252,
    "non_padding_tokens_note": (
        "steps x non-padding tokens per step, the latter from the measured 99.3646% of "
        "positions that carry text; padding is masked out of the loss and not counted"
    ),
    "tokens_repeated": False,
    "repeated_tokens_counted": True,
    "duration_seconds": 23_597,
    "h200_hours": 52.44,
    "gpu_type": "H200",
    "gpu_count": 8,
    "validation_documents": 10_000,
    "validation_note": (
        "val.bin packs the held-out documents into blocks the same way, so the logged "
        "eval/perplexity scores each document from its own BOS, as the judge does. The "
        "rebuild landed mid-run, so this run's logged curve switches metrics partway and "
        "should not be read; a direct replication of the judge's scorer gives 14.53 on "
        "10,000 held-out documents"
    ),
}
