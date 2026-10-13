#!/usr/bin/env python3
"""Generator 的 SFT 热启动：用 gen_sft_data.py 产出的合格样例把 generator 先拽进正确格式。
模型规模由 --model_name_or_path 决定，这段热启动逻辑与规模无关。

为什么先做 SFT：RL(step1) 阶段没热启动过的 generator 几乎产不出规定的四段标签，
奖励常为 0，GRPO 拿不到梯度。先监督学一遍"输入 control-spec → 输出四段标签"的映射，
再进 RL，起点才站得住。

与现有流程的一致性：
  - 训练数据就是 gen_sft_data.py 的输出（已用训练时那套 reward 逻辑过滤成满分正例）。
  - 只对 assistant 段算 loss（completion_only_loss），prompt 段不计 —— 这正是
    我们想强化的"照着 spec 写答案"这一件事。
  - 结构对齐 step1_generator.py：TrlParser + LoRA + 训练后把 adapter 合并回完整模型，
    合并后的目录可直接当作 step1 GRPO 的 --model_name_or_path，也可直接被 step2 加载。

用法见 run_sft.sh。
"""

import json
import os
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)

# 复用 step1 里同一套"把 LoRA 折回完整权重"的逻辑，避免两处各写一份合并实现。
from step1_generator import merge_lora_checkpoint


@dataclass
class MyScriptArguments(ScriptArguments):
    sft_data_path: str = field(
        default="sft_generator_data.jsonl",
        metadata={"help": "gen_sft_data.py 产出的 jsonl 路径"},
    )


def build_prompt_completion(dataset, tokenizer):
    """把每条 messages 拆成 (prompt, completion)。

    - prompt：system+user 渲染到 add_generation_prompt（即推理时模型真正看到的那段）。
    - completion：整段渲染 - prompt 的前缀，等价于 assistant 内容 + 模板自带的结束符。
      用"整段渲染减去前缀"而不是手拼结束符，模板怎么改都不用动这里。

    这样交给 SFTTrainer 的就是标准 prompt/completion 格式，配 completion_only_loss
    即可只对 completion 算 loss —— 不需要给 chat 模板塞 generation 标记。
    """

    def _row(example):
        msgs = example["messages"]
        assistant = [m for m in msgs if m.get("role") == "assistant"]
        prompt_msgs = [m for m in msgs if m.get("role") != "assistant"]
        if not assistant or not prompt_msgs:
            return {"prompt": "", "completion": ""}
        prompt = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        full = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        completion = full[len(prompt):] if full.startswith(prompt) else assistant[-1]["content"]
        return {"prompt": prompt, "completion": completion}

    out = dataset.map(_row, remove_columns=dataset.column_names)
    return out.filter(lambda x: bool(x["prompt"]) and bool(x["completion"]))


def main():
    parser = TrlParser((MyScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    if not os.path.isfile(script_args.sft_data_path):
        raise SystemExit(f"找不到 SFT 数据：{script_args.sft_data_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    raw = load_dataset("json", data_files=script_args.sft_data_path, split="train")
    train_dataset = build_prompt_completion(raw, tokenizer)

    if len(train_dataset) == 0:
        raise SystemExit("过滤后训练集为空，检查 jsonl 是否为 messages 格式。")
    print(f"[sft] 训练样本 {len(train_dataset)} 条")

    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # 合并 LoRA -> 完整权重。step1 的 GRPO 与 step2 的 vLLM 都按普通模型目录加载，
    # 所以必须留一个不带 adapter 的完整 checkpoint。
    merge_lora_checkpoint(model_args, training_args)


if __name__ == "__main__":
    main()