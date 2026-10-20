"""M5 — Skill system.

The right 1–3 skills load into a small context at the right time, and the library improves
itself from failures without rotting. Four sub-parts: format & loading (agentskills.io
SKILL.md, progressive disclosure), ranking & composition (SkillComposer / SkillRAG),
writing new skills from successful traces (MUSE-Autoskill), and maintenance (EvoSkill
validation-gated pruning + SkillOps typed contracts and a skill graph). Skill bodies are
capped to a token budget, since every loaded skill eats the 32K window.
"""

from __future__ import annotations
