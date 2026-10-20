"""The plain single-agent baseline loop.

Linear append-only history; one bash action per step; stateless execution; no context
folding and no skills — that minimalism is the point, so later modules have a yardstick
to beat. Model access goes through the ``InferenceBackend`` interface's ``chat()`` method,
so an instruct model gets its real chat template (when the backend has one) and the loop
runs against any local backend, or a scripted fake in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from keel.baseline.actions import BashAction, ParseError, SubmitAction, parse_action
from keel.baseline.bash import BashTool
from keel.serving.backends.base import ChatMessage, InferenceBackend
from keel.trace.schema import ToolCall, TraceRecord

SYSTEM_PROMPT = """You are a coding agent working in a shell. Each turn, reply with a
short thought and then EXACTLY ONE fenced code block.

To run a command:
```bash
<a single shell command>
```

The environment is STATELESS: a `cd` does not persist between commands, so write
self-contained commands. When the task is complete, reply instead with:
```submit
<your final answer / summary of what you did>
```
"""

MAX_STEPS = 30
MAX_TOKENS_PER_STEP = 1024

# A model may ignore its stop token and keep going past its turn, hallucinating the next
# role's text; keep only what looks like the model's own turn.
_ROLE_DELIMITERS = ("\nOBSERVATION:", "\nUSER:", "\nSYSTEM:", "\nASSISTANT:")


def _first_turn(text: str) -> str:
    cut = len(text)
    for delim in _ROLE_DELIMITERS:
        index = text.find(delim)
        if index != -1:
            cut = min(cut, index)
    return text[:cut].strip()


@dataclass
class AgentResult:
    outcome: str  # "submitted" | "max_steps"
    final_message: str
    steps: int
    transcript: list[str] = field(default_factory=list)
    trace: TraceRecord | None = None


class BaselineAgent:
    def __init__(
        self,
        backend: InferenceBackend,
        *,
        workspace: Path | str = ".",
        max_steps: int = MAX_STEPS,
        bash: BashTool | None = None,
    ) -> None:
        self.backend = backend
        self.max_steps = max_steps
        self.bash = bash or BashTool(workspace=workspace)

    def run(self, task: str, *, task_id: str = "baseline", model_id: str = "") -> AgentResult:
        messages: list[ChatMessage] = [
            ChatMessage("system", SYSTEM_PROMPT),
            ChatMessage("user", task),
        ]
        transcript: list[str] = [f"SYSTEM:\n{SYSTEM_PROMPT}", f"USER:\n{task}"]
        tool_calls: list[ToolCall] = []
        peak_active_tokens = 0
        parse_errors = 0
        failed_commands = 0

        for step in range(1, self.max_steps + 1):
            gen = self.backend.chat(messages, max_tokens=MAX_TOKENS_PER_STEP, temperature=0.0)
            peak_active_tokens = max(peak_active_tokens, gen.prompt_tokens + gen.completion_tokens)
            reply = _first_turn(gen.text)
            messages.append(ChatMessage("assistant", reply))
            transcript.append(f"ASSISTANT:\n{reply}")

            try:
                action = parse_action(reply)
            except ParseError as exc:
                parse_errors += 1
                note = f"invalid action ({exc}). Reply with exactly one ```bash or ```submit block."
                messages.append(ChatMessage("user", note))
                transcript.append(f"OBSERVATION:\n{note}")
                continue

            if isinstance(action, SubmitAction):
                return AgentResult(
                    outcome="submitted",
                    final_message=action.message,
                    steps=step,
                    transcript=transcript,
                    trace=self._trace(
                        task_id,
                        model_id,
                        task,
                        "submitted",
                        tool_calls,
                        step,
                        peak_active_tokens,
                        parse_errors,
                        failed_commands,
                    ),
                )

            assert isinstance(action, BashAction)
            obs = self.bash.run(action.command)
            if obs.exit_code != 0:
                failed_commands += 1
            tool_calls.append(
                ToolCall(
                    name="bash",
                    arguments={"command": action.command},
                    findings=obs.render(),
                    ok=obs.exit_code == 0,
                )
            )
            observation = f"OBSERVATION:\n{obs.render()}"
            messages.append(ChatMessage("user", observation))
            transcript.append(observation)

        return AgentResult(
            outcome="max_steps",
            final_message="",
            steps=self.max_steps,
            transcript=transcript,
            trace=self._trace(
                task_id,
                model_id,
                task,
                "max_steps",
                tool_calls,
                self.max_steps,
                peak_active_tokens,
                parse_errors,
                failed_commands,
            ),
        )

    @staticmethod
    def _trace(
        task_id: str,
        model_id: str,
        task: str,
        outcome: str,
        tool_calls: list[ToolCall],
        steps: int,
        peak_active_tokens: int,
        parse_errors: int,
        failed_commands: int,
    ) -> TraceRecord:
        total_actions = len(tool_calls) + parse_errors
        error_rate = (parse_errors + failed_commands) / total_actions if total_actions else 0.0
        return TraceRecord(
            task_id=task_id,
            model_id=model_id,
            prompt=task,
            # The baseline does not verify; the eval harness (M8) overwrites this with a
            # verified pass/fail. Here it records only how the loop terminated.
            outcome=outcome,
            tool_calls=tool_calls,
            metrics={
                "steps": steps,
                "peak_active_tokens": peak_active_tokens,
                "tool_call_error_rate": round(error_rate, 4),
                "parse_errors": parse_errors,
                "failed_commands": failed_commands,
                "escalations": 0,
            },
        )
