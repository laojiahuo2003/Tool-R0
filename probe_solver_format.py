"""Decisive test: does the BASE Qwen2.5-0.5B-Instruct emit the solver's
<tool_call_answer> tag under the OFFICIAL bare prompt?

The released Tool-R0-Qwen2.5-0.5B trainer_state.json reports
rewards/format_reward_func/mean = 0.978 at step 1, i.e. the base model scores
0.98 on the official format reward before any training. Our own probe measured
11% tag emission. One of the two is wrong. This runs the official prompt and the
official format_reward_func against the base model.
"""
import sys, json, random
sys.path.insert(0, "/tmp")
import importlib.util

spec = importlib.util.spec_from_file_location("official_solver_rewards", "/tmp/official_rewards_solver.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT_SOLVER = (
    "A conversation between user and tool-calling assistant. The user asks a question, and the assistant uses tools to solve it. The "
    "assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think></think> and <tool_call_answer></tool_call_answer> tags, i.e., <think>\nThis is my "
    "reasoning.\n</think>\n<tool_call_answer>[{\"name\": \"<tool_name>\", \"arguments\": {\"arg1\": \"value\", \"arg2\": \"value2\", ...}}, ...]</tool_call_answer>."
)
USER_TMPL = """User request:
{question}

Available tools (JSON):
{tools_json}
"""

# Well-formed samples of the kind a trained generator produces: natural
# question, flat-primitive tool schema, gold call whose argument values all
# appear in the question.
SAMPLES = [
    {
        "question": "Cancel my dentist appointment on Friday June 14 at 3pm, the id is appt_3pm_friday.",
        "tools": [
            {"name": "list_appointments", "description": "List appointments for a date",
             "parameters": {"type": "object", "properties": {"date": {"type": "string", "description": "Date YYYY-MM-DD"}}, "required": ["date"]}},
            {"name": "cancel_appointment", "description": "Cancel an appointment by id",
             "parameters": {"type": "object", "properties": {"appointment_id": {"type": "string", "description": "Appointment id"}}, "required": ["appointment_id"]}},
        ],
    },
    {
        "question": "How many open tickets does the payments team have right now?",
        "tools": [
            {"name": "list_teams", "description": "List teams", "parameters": {"type": "object", "properties": {}, "required": []}},
            {"name": "count_tickets", "description": "Count tickets for a team",
             "parameters": {"type": "object", "properties": {"team": {"type": "string"}, "status": {"type": "string"}}, "required": ["team"]}},
        ],
    },
    {
        "question": "What is the weather in Chicago tomorrow?",
        "tools": [
            {"name": "get_weather", "description": "Get weather forecast for a city",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}},
        ],
    },
    {
        "question": "Convert 250 US dollars to euros using today's rate.",
        "tools": [
            {"name": "convert_currency", "description": "Convert an amount between currencies",
             "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from": {"type": "string"}, "to": {"type": "string"}}, "required": ["amount", "from", "to"]}},
        ],
    },
]

llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=4096,
          gpu_memory_utilization=0.85, enforce_eager=True)
tok = AutoTokenizer.from_pretrained(MODEL)

prompts, owners = [], []
for si, s in enumerate(SAMPLES):
    for _ in range(8):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT_SOLVER},
            {"role": "user", "content": USER_TMPL.format(question=s["question"], tools_json=json.dumps(s["tools"]))},
        ]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        owners.append(si)

for temp in (1.0, 0.7, 0.001):
    sp = SamplingParams(temperature=temp, max_tokens=1024, n=1)
    outs = llm.generate(prompts, sp)
    texts = [o.outputs[0].text for o in outs]

    comps = [[{"content": t}] for t in texts]
    fmt = R.format_reward_func(prompts, comps)

    by_sample = {}
    for si, f in zip(owners, fmt):
        by_sample.setdefault(si, []).append(f)

    print(f"\n=== temperature = {temp}  (n={len(texts)} draws) ===")
    print(f"  official format_reward mean = {sum(fmt)/len(fmt):.4f}")
    print(f"  得满分(1.0)的比例            = {sum(1 for f in fmt if f >= 0.999)/len(fmt)*100:.1f}%")
    print(f"  得 0.0 的比例               = {sum(1 for f in fmt if f == 0.0)/len(fmt)*100:.1f}%")
    print("  逐样本均值: " + "  ".join(f"s{si}={sum(v)/len(v):.2f}" for si, v in sorted(by_sample.items())))

    if temp == 1.0:
        print("\n  --- 头两条原始输出 ---")
        for t in texts[:2]:
            print("   ", repr(t[:300]))
