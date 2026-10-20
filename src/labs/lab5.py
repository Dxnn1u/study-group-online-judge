"""Lab 5 submission: Llama 3.2 1B (random init) trained on Dolma 3 with Muon-VS + AdamW,
distilled from a precomputed Llama-3.2-1B teacher.

Run: 8x H200, 6.0B non-padding tokens, 23,034 optimizer steps, 59.39 H200-hours for the
student, plus 44.0 H200-hours spent once, offline, recording the teacher's distribution.
Code: https://github.com/Dxnn1u/ML-study-group-lab-5 (train.py, data.py, split.py,
prepare_teacher_logits.py).
"""

eval_model_id = "Dxnn1u/llama-3.2-1B-lab5-6b-kd"
training_run_url = "https://wandb.ai/cerulean-labs/lab5-training-llama/runs/t3zo8r86"
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
                "lr": 1e-3,
                "momentum": 0.95,
                "nesterov": True,
                "ns_steps": 5,
                "weight_decay": 0.8,
                "adjust_lr_fn": "match_rms_adamw",
                "variance_scaling": True,
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
            "type": "warmup, constant, then linear decay over the last 80% (per optimizer step, all groups)",
            "peak_lr": 1e-3,
            "warmup_steps": 200,
            "decay_steps": 18_427,
            "total_steps": 23_034,
            "min_lr": 0.0,
        },
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
    "distillation": {
        "used": True,
        "teacher": "meta-llama/Llama-3.2-1B",
        "teacher_is_pretrained": True,
        "teacher_disclosure": (
            "the teacher is the released Llama-3.2-1B, which Meta pretrained on roughly 9T "
            "tokens. The student is random init and reads only Dolma 3, but the teacher's "
            "distribution carries information learned outside Dolma 3 into the student's loss. "
            "Flagging this explicitly: if the lab means the student to see nothing but its own "
            "6B tokens, this run does not qualify and the non-distilled sibling "
            "Dxnn1u/llama-3.2-1B-lab5-6b-d40 (perplexity 14.5338) is the entry to score instead"
        ),
        "objective": "loss = 0.25 * cross_entropy + 0.75 * T^2 * KL(teacher || student), T = 1.5",
        "distill_weight": 0.75,
        "temperature": 1.5,
        "teacher_run_in_the_loop": False,
        "teacher_store": (
            "the teacher was run once over all 1,164,921 training blocks and its distribution "
            "stored per position: the exact top-16 probabilities, the remaining tail mass, and "
            "32 index/log-probability draws from the renormalised tail. The KL is then the exact "
            "head term plus an unbiased Monte-Carlo estimate of the tail term, so the student "
            "step gathers 48 logits per position instead of materialising a 128,256-wide "
            "distribution. 2.6 TB on disk, built from the same train.bin this run reads"
        ),
        "estimator_validation": (
            "three checks, all in the repo. (1) tests/check_sampled_distill.py: the sampled "
            "tail estimator's per-step error against the dense KL is 21.98% and unbiased, while "
            "truncating at the top-16 head instead is a 266% bias that averaging cannot remove. "
            "(2) the store's loss and gradients reproduce a dense log_softmax to 5.96e-08 and "
            "~2e-06, and the stored teacher to 0.50%. (3) end to end at 2B tokens, same recipe: "
            "store 2.681090 vs teacher-in-the-loop 2.680046, a gap of 0.001043 against a "
            "0.00156 seed-noise floor, so the 22% per-step noise averages out over training"
        ),
        "store_matches_data_check": (
            "train.py refuses to start unless the store has exactly as many blocks as the "
            "dataset and a probe of the stored teacher's top-1 agreement with the data's true "
            "next token clears 0.25; this run measured 0.4973 (misaligned data scores 0.0067). "
            "An earlier attempt trained for an hour against a store built from a different "
            "snapshot of the data, which is what the check now prevents"
        ),
        "teacher_pass_cost_h200_hours": 44.0,
        "teacher_pass_note": (
            "6 jobs x 8 H200 x 0.917 h, spent once before the student started. It is excluded "
            "from h200_hours below, which covers the student trajectory only; whether this lab "
            "counts an offline teacher pass against the budget is a call for the staff. Counted "
            "together the two come to 103.4 H200-hours"
        ),
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
    "duration_seconds": 26_726,
    "h200_hours": 59.39,
    "gpu_type": "H200",
    "gpu_count": 8,
    "validation_documents": 10_000,
    "validation_note": (
        "a direct replication of the judge's scorer on 10,000 held-out documents "
        "(12,637,446 tokens, bfloat16) gives corpus perplexity 12.3211, mean loss 2.5113. "
        "The same scorer gives 14.5338 for the non-distilled sibling run, which is the honest "
        "measure of what distillation bought here"
    ),
}
