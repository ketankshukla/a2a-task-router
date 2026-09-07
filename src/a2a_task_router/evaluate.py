"""Score a router on tasks it has never seen.

The evaluation set is written deliberately unlike the card examples — same
intent, different words — because a router scored on the strings it trained on
tells you nothing. It also contains out-of-scope tasks whose correct answer is
"decline", which is the measurement most routing demos leave out.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class EvalTask:
    text: str
    expected: str | None  # None means: nothing should claim this


def load_tasks(path: Path | None = None) -> list[EvalTask]:
    path = path or DATA / "eval_tasks.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [
            EvalTask(r["text"], r["expected"])
            for r in (json.loads(line) for line in fh if line.strip())
        ]


@dataclass
class Result:
    router: str
    in_scope_total: int = 0
    in_scope_correct: int = 0
    in_scope_wrong: int = 0
    in_scope_declined: int = 0
    out_scope_total: int = 0
    out_scope_declined: int = 0
    out_scope_claimed: int = 0
    ms_per_task: float = 0.0
    confusions: list[tuple[str, str, str]] | None = None

    @property
    def accuracy(self) -> float:
        """Correct routes over in-scope tasks. A decline counts as a miss."""
        return self.in_scope_correct / self.in_scope_total if self.in_scope_total else 0.0

    @property
    def precision(self) -> float:
        """Of the in-scope tasks it *did* claim, how many went to the right agent."""
        claimed = self.in_scope_correct + self.in_scope_wrong
        return self.in_scope_correct / claimed if claimed else 0.0

    @property
    def abstention(self) -> float:
        """Of the out-of-scope tasks, how many it correctly refused."""
        return (
            self.out_scope_declined / self.out_scope_total
            if self.out_scope_total
            else 0.0
        )

    @property
    def balanced(self) -> float:
        """One number covering both halves.

        A router that always answers scores 1.0 on abstention's opposite and
        0.0 on abstention; one that never answers does the reverse. Averaging
        the two stops either degenerate strategy looking good.
        """
        return (self.accuracy + self.abstention) / 2


def evaluate(router, tasks: list[EvalTask], keep_confusions: bool = True) -> Result:
    result = Result(router=router.name, confusions=[] if keep_confusions else None)

    started = time.perf_counter()
    for task in tasks:
        decision = router.route(task.text)

        if task.expected is None:
            result.out_scope_total += 1
            if decision.routed:
                result.out_scope_claimed += 1
                if result.confusions is not None:
                    result.confusions.append((task.text, "—", decision.agent))
            else:
                result.out_scope_declined += 1
            continue

        result.in_scope_total += 1
        if not decision.routed:
            result.in_scope_declined += 1
            if result.confusions is not None:
                result.confusions.append((task.text, task.expected, "declined"))
        elif decision.agent == task.expected:
            result.in_scope_correct += 1
        else:
            result.in_scope_wrong += 1
            if result.confusions is not None:
                result.confusions.append((task.text, task.expected, decision.agent))

    elapsed = time.perf_counter() - started
    result.ms_per_task = elapsed / len(tasks) * 1000 if tasks else 0.0
    return result


def format_table(results: list[Result]) -> str:
    header = (
        f"| {'router':<12} | accuracy | precision | abstention | balanced | ms/task |"
    )
    lines = [header, "|" + "-" * (len(header) - 2) + "|"]
    for r in results:
        lines.append(
            f"| {r.router:<12} |    {r.accuracy:.3f} |     {r.precision:.3f} "
            f"|      {r.abstention:.3f} |    {r.balanced:.3f} | {r.ms_per_task:7.2f} |"
        )
    return "\n".join(lines)
