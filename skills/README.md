# skills/ — Keel runtime skill library

The agent's own skills, in the **agentskills.io `SKILL.md`** format (trigger metadata +
instructions + optional scripts), loaded into the model's context at inference time via
progressive disclosure. The skill system (M5) ranks and composes them, writes new ones from
successful traces, and prunes ones that fail held-out validation.

This is populated starting in **Phase 4 (M5)** — see [docs/ROADMAP.md](../docs/ROADMAP.md).

> Not to be confused with `.claude/skills/`, which holds Claude Code *developer* skills for
> people working on Keel. These here are consumed by the Keel agent itself.

Constraint: each skill body is capped to a token budget — every loaded skill eats the 32K
active-context window.
