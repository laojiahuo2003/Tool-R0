
import argparse
import os
import re
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

from rewards_generator import (
    reward_format_accuracy,
    reward_validity_accuracy,
    reward_curriculum,
)

# Imported rather than restated: step1 used to carry its own copy of the
# generator prompt, which had drifted from the one step2_gen actually verifies
# with (four-space indentation on every line, and "Generate a new" where
# step2_gen says "Generate one new"). SFT data is now trained against the
# step2_gen wording, so both stages have to read the same string.
from step2_gen import SYSTEM_PROMPT_GENERATOR_TEMPLATE, USER_PROMPT_GENERATOR

@dataclass
class MyScriptArguments(ScriptArguments):
    number_of_generated_data: int = field(
        default=100,
        metadata={"help": "Number of generated data"},
    )


import random
import json
from datasets import Dataset
# 有各种领域
DOMAINS = [
    # --- Primary Functional Domains (Explicitly listed in docs) ---
    "finance",              # Stock trading, mortgage calculation, banking APIs
    "travel",               # Flight booking, hotel reservation, airport info
    "math",                 # Algebra, statistics, calculator functions
    "sports",               # Sports scores (e.g., Soccer/Football, NBA)
    "weather",              # Real-time weather data
    "system",               # File System (ls, cd, cat), OS commands
    "database",             # SQL queries (SELECT, INSERT, etc.)
    "vehicle_control",      # Car status, engine control, EV charging
    "communication",        # Messaging (Slack, Email, SMS)
    "entertainment",        # Movies (TMDB), Music (Spotify)
    "retail_ecommerce",     # Inventory management, order status, product search
    "scheduling",           # Calendar management, meeting booking
    "cloud_infrastructure", # VM management, AWS/Cloud resource handling
    "geolocation",          # Maps, routing, distance estimation
    
    # --- Agentic & Technical Domains (Live Updates) ---
    "web_search",           # Multi-hop reasoning, internet search (SerpAPI)
    "memory_management",    # Key-value store, vector database, recursive summarization
    "programming",          # Code execution, debugging helpers
    "iot",                  # Internet of Things (smart home device control)
    "social_media",         # Twitter/X, Reddit API interactions
    "logistics",            # Shipping tracking, supply chain
    "real_estate",          # Property listing, housing data
    
    # --- Specialized & "Live" API Domains ---
    "food_ordering",        # Restaurant delivery services
    "healthcare",           # Basic medical info (via public APIs)
    "education",            # Course searching, academic references
    "productivity",         # Note-taking (Notion), task management
    "insurance",            # Purchase insurance, policy checks
    "cybersecurity",        # Basic security checks, auth verification
    "legal",                # Regulatory data lookup
    "government",           # Public service APIs
    "news",                 # News aggregation
    "translation",          # Language translation services
    "utilities",            # Energy usage, utility billing
    "customer_support"      # FAQ retrieval, ticket creation
]
# 每个领域的权重
DOMAIN_WEIGHTS = {
    "finance": 0.03125,
    "healthcare": 0.03125,
    "productivity": 0.03125,
    "retail_ecommerce": 0.03125,
    "scheduling": 0.03125,
    "database": 0.03125,
    "cloud_infrastructure": 0.03125,
    "system": 0.03125,
    "programming": 0.03125,
    "geolocation": 0.03125,
    "logistics": 0.03125,
    "communication": 0.03125,
    "iot": 0.03125,
    "cybersecurity": 0.03125,
    "insurance": 0.03125,
    "legal": 0.03125,
    "news": 0.03125,
    "weather": 0.03125,
    "sports": 0.03125,
    "entertainment": 0.03125,
    "education": 0.03125,
    "real_estate": 0.03125,
    "food_ordering": 0.03125,
    "translation": 0.03125,
    "utilities": 0.03125,
    "government": 0.03125,
    "memory_management": 0.03125,
    "web_search": 0.03125,
    "social_media": 0.03125,
    "math": 0.03125,
    "vehicle_control": 0.03125,
    "travel": 0.03125,
    # "customer_support": 0.03125  # Uncomment if this is in your DOMAINS list
}

# DOMAIN_WEIGHTS = {
#     # higher weight on precision/routing friendly eval-like domains
#     "finance": 0.10,
#     "healthcare": 0.08,
#     "productivity": 0.08,
#     "retail_ecommerce": 0.08,
#     "scheduling": 0.07,
#     "database": 0.07,
#     "cloud_infrastructure": 0.07,
#     "system": 0.06,
#     "programming": 0.06,
#     "geolocation": 0.05,
#     "logistics": 0.05,
#     "communication": 0.05,
#     "iot": 0.04,
#     "cybersecurity": 0.04,
#     "insurance": 0.03,
#     "legal": 0.03,
#     "news": 0.03,
#     "weather": 0.03,
#     "sports": 0.02,
#     "entertainment": 0.02,
#     "education": 0.02,
#     "real_estate": 0.02,
#     "food_ordering": 0.02,
#     "translation": 0.02,
#     "utilities": 0.02,
#     "government": 0.01,
#     "memory_management": 0.01,
#     "web_search": 0.01,
#     "social_media": 0.01,
#     "math": 0.01,
#     "vehicle_control": 0.01,
#     "travel": 0.02,  # intentionally low
# }

# 采样任务领域
def weighted_choice(weight_dict):
    items = list(weight_dict.items())
    keys, weights = zip(*items)
    return random.choices(keys, weights=weights, k=1)[0]
# 设置任务难度和领域
def sample_spec():

    context_type = random.choices(["single_turn", "multi_turn"], weights=[0.9, 0.1], k=1)[0]# 90概率是单轮

    if context_type == "multi_turn":
        num_calls = 1
    else:
        num_calls = random.choices([1, 2], weights=[0.8, 0.2], k=1)[0]

    if num_calls > 1:
        tool_menu_size = random.choices([3, 4, 5], weights=[0.3, 0.4, 0.3], k=1)[0]
    else:
        bucket = random.choices(["SMALL5", "LARGE"], weights=[0.4, 0.6], k=1)[0]
        tool_menu_size = random.randint(2, 4) if bucket == "SMALL5" else random.randint(5, 8)

    domain = weighted_choice(DOMAIN_WEIGHTS)

    return {
        "domain": domain,
        "num_calls": num_calls,
        "tool_menu_size": tool_menu_size,
        "context_type": context_type,
    }




def main():
    parser = TrlParser((MyScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    N = script_args.number_of_generated_data
    specs = [sample_spec() for _ in range(N)]

    dummy_examples = [{"question": ""} for _ in range(N)]
    train_dataset = Dataset.from_list(dummy_examples)
    train_dataset = train_dataset.add_column("gen_spec", [json.dumps(s) for s in specs])


    # Imported rather than restated: step1 used to carry its own copy of this
    # prompt, so the difficulty reward during training probed the solver with a
    # different system prompt than step2_genverify actually verifies with.
    from step2_genverify import SYSTEM_PROMPT_SOLVER

    def make_conversation_generator(example):
        spec = json.loads(example["gen_spec"])
        system_prompt = SYSTEM_PROMPT_GENERATOR_TEMPLATE.format(**spec)
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT_GENERATOR},
            ],
            "gen_spec": example["gen_spec"],
        }

    train_dataset = train_dataset.map(make_conversation_generator)
    train_dataset = train_dataset.remove_columns(["question"])
    train_dataset = train_dataset.add_column("solver_prompt", [SYSTEM_PROMPT_SOLVER] * len(train_dataset))
    train_dataset = train_dataset.add_column("model_name_or_path", [model_args.model_name_or_path] * len(train_dataset))


    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[
            reward_format_accuracy,
            reward_validity_accuracy,
            reward_curriculum,
        ],
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

    TRL saves a PEFT run as an adapter-only directory, but step2 loads the
    generator checkpoint with vLLM as a plain model, so the adapter has to be
    merged back into the base weights first. No-op when LoRA is disabled.
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
    
