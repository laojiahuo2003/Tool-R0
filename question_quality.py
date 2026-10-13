#!/usr/bin/env python3
"""Shared question-quality gate for the generator.

Two places decide whether a generated sample is worth anything:
  - rewards_generator._question_gate_mask   (the GRPO reward gate, during training)
  - step2_gen.non_triviality_heuristic       (offline data generation, step2)

They have to agree. If training rewards a question that step2 then throws away,
the generator is optimised toward exactly the samples the pipeline discards.

Both checks missed the failure mode that dominated the first full run. Of the 18
samples step2_gen kept out of 300 generations:

  - 5 had the generator's own instructions sitting in the question field, e.g.
    'Write a natural user question (in domain "cloud_infrastructure") about
    creating a new VPC in AWS.'
  - 6 asked the model to name or suggest a tool rather than to use one, e.g.
    'I want to create a new tool that will help athletes swim more efficiently.
    What should it be called?' with gold calls for a tool called "SwimMaster".
  - 1 leaked a conversation transcript ('User: ... Agent: ...') into the field.
  - 1 was a plain arithmetic question with a single-argument call.

A meta-instruction is perfectly well-formed text, so reward_tags_parsed and
reward_gold_valid_json both score it highly. Nothing pushed back, so the
generator drifted toward it -- it is the cheapest way to earn format reward
without having to produce a coherent task. Worse, once verify could parse the
tag, the incoherent samples were the ones that passed it: a gold call named
"travel_itinerary_generator" is easy for a solver to reproduce from a question
that mentions travel itineraries, so garbage verified more reliably than sense.

This module is the single definition of "not a usable question". Import it from
both sides rather than restating the rules.
"""
import re
from typing import Optional

# The generator's own instructions leaking into the question field. A blocklist
# of exact phrases loses to this: a fresh 300-generation draw produced "Write a
# brief, realistic task now.", "Craft a new logistical tool that could
# revolutionize supply chain management...", "How should I design a 2D navigation
# tool for travel?", none of which any fixed phrase list anticipates.
#
# The discriminator is verb + object. A usable question asks the assistant to do
# something in the world; a meta-instruction asks it to *author a task*. So
# reject a task-authoring verb applied to a task-authoring noun, and leave
# "Create a calendar event for tomorrow at 3pm" (create + event) alone.
_META_VERB = (r"(?:write|craft|create|generate|design|compose|draft|devise|"
              r"formulate|come up with|produce|build|develop|make|suggest|"
              r"propose|recommend|give me)")
_META_OBJECT = (r"(?:task|question|questions|query|queries|tool|tools|scenario|"
                r"example|examples|prompt|spec|problem|exercise|use case|"
                r"benchmark|dataset|generator)")

_META_INSTRUCTION_RES = [
    rf"\b{_META_VERB}\b[^.?]{{0,60}}?\b{_META_OBJECT}\b",
    r"\bhow (?:can|should|would|could) (?:we|i|you)\b",
    r"\bit must be a question\b",
    r"\bno bullet points\b",
    r"\bno meta-?instructions\b",
    r"\bno placeholders\b",
    r"\bshould be specific about\b",
    r"\bsingle turn\b",
    r"\bhuman-?like statement\b",
    r"\brequires gold tool\b",
    r"\bcannot be solved without\b",
    r"\butilizes the agent'?s knowledge\b",
    r"\bfollow the control spec\b",
    r"\bcontrol spec\b",
    r"\bformat the output exactly\b",
    r"\bavailable tools \(json\)\b",
    r"\bin domain\b",
    r"\bthat involves (?:using|multiple|calling)\b",
    r"\bcould be used to call\b",
]

# Questions that ask *about* tools -- what to call one, whether one exists --
# rather than asking for something a tool would do.
_TOOL_ABOUT_RES = [
    r"\bwhat should (?:it|this|the tool|the model) be called\b",
    r"\bwhat should be the name\b",
    r"\bname of (?:this|the|that) tool\b",
    r"\bsuitable name for\b",
    r"\bname for a new\b",
]

# A conversation transcript that leaked into the field instead of a question.
_LEAKED_TRANSCRIPT_RES = [
    r"^\s*user\s*:",
    r"^\s*agent\s*:",
    r"^\s*assistant\s*:",
    r"^\s*system\s*:",
]

_META_RE = [re.compile(p, re.IGNORECASE) for p in _META_INSTRUCTION_RES]
_TOOL_RE = [re.compile(p, re.IGNORECASE) for p in _TOOL_ABOUT_RES]
_TRANSCRIPT_RE = [re.compile(p, re.MULTILINE) for p in _LEAKED_TRANSCRIPT_RES]


def question_quality_reason(question: str) -> Optional[str]:
    """None if the question is usable, else a short reason string.

    Only rejects. Anything that survives is still only *potentially* usable --
    non_triviality_heuristic and the rest of the validator do the rest.
    """
    if not question or not question.strip():
        return "empty_question"
    q = question.strip()
    if len(q) < 15:
        return "question_too_short"

    for r in _META_RE:
        if r.search(q):
            return "meta_instruction_in_question"
    for r in _TOOL_RE:
        if r.search(q):
            return "question_about_tools_not_task"
    for r in _TRANSCRIPT_RE:
        if r.search(q):
            return "leaked_transcript_in_question"
    return None


def is_unusable_question(question: str) -> bool:
    return question_quality_reason(question) is not None
