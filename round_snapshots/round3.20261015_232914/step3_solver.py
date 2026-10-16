
import argparse
import os
import re
import json
from typing import Optional

from dataclasses import dataclass, field
from datasets import Dataset, load_dataset
import torch

from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
    get_quantization_config,
)


from rewards_solver import format_reward_func, accuracy_reward_func

@dataclass
class MyScriptArguments(ScriptArguments):
    generated_data_path: str = field(
        default="solver_train.json",
        metadata={"help": "Path to the generated data"},
    )


def main():

    parser = TrlParser((MyScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    print(f"Loading dataset from {script_args.generated_data_path}...")


    with open(script_args.generated_data_path, 'r') as f:
        data = json.load(f)

    for item in data:
        if "answer" in item:
            item["answer"] = str(item["answer"])

    train_dataset = Dataset.from_list(data)
    print(f"\nSuccessfully loaded {len(train_dataset)} examples...")
    print(train_dataset)
    print("\n")


    # Imported rather than restated: step3 carried its own copy of this prompt,
    # so the solver was trained under instructions that differed from the ones
    # step2_genverify verifies it with (and from what step1's difficulty reward
    # probes it with in the next iteration).
    from step2_genverify import SYSTEM_PROMPT_SOLVER

    
    def convert_answer_to_string(example):
        """Convert answer field from dict/list to JSON string without newlines"""
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT_SOLVER},
                {"role": "user", "content": example["question"]},
            ],
            "answer": example["answer"],
            "model_name_or_path": model_args.model_name_or_path,
        }
    
    train_dataset = train_dataset.map(convert_answer_to_string)

    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[format_reward_func, accuracy_reward_func],
        train_dataset=train_dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    merge_lora_checkpoint(model_args, training_args)


def _latest_checkpoint(output_dir: str) -> Optional[str]:
    """Return the highest-numbered checkpoint-* directory under output_dir."""
    best, best_step = None, -1
    if not os.path.isdir(output_dir):
        return None
    for name in os.listdir(output_dir):
        m = re.fullmatch(r"checkpoint-(\d+)", name)
        if m and int(m.group(1)) > best_step:
            best, best_step = os.path.join(output_dir, name), int(m.group(1))
    return best


def merge_lora_checkpoint(model_args, training_args) -> None:
    """Fold the LoRA adapter into the base weights and rewrite the checkpoint.

    TRL saves a PEFT run as an adapter-only directory, but step0/step2 load the
    solver checkpoint with vLLM as a plain model, so the adapter has to be merged
    back into the base weights first. No-op when LoRA is disabled.
    """
    if not model_args.use_peft:
        return

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt = _latest_checkpoint(training_args.output_dir)
    if ckpt is None:
        print(f"[merge] no checkpoint found under {training_args.output_dir}, skipping")
        return

    print(f"[merge] merging {ckpt} into {model_args.model_name_or_path}")
    base = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, ckpt).merge_and_unload()

    # Drop the adapter files so the directory is an unambiguous full model.
    for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
        path = os.path.join(ckpt, name)
        if os.path.exists(path):
            os.remove(path)

    merged.save_pretrained(ckpt)
    AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    ).save_pretrained(ckpt)
    print(f"[merge] wrote full model to {ckpt}")


if __name__ == "__main__":
    main()
    

