"""Compare the routers head to head on held-out tasks.

    python scripts/run_eval.py             # all three, downloads a model
    python scripts/run_eval.py --no-embed  # keyword + learned only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from a2a_task_router.evaluate import evaluate, format_table, load_tasks  # noqa: E402
from a2a_task_router.registry import load_cards  # noqa: E402
from a2a_task_router.routers import (  # noqa: E402
    EmbeddingRouter,
    KeywordRouter,
    LearnedRouter,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--show-errors", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="sweep the abstention threshold")
    args = ap.parse_args()

    cards = load_cards()
    tasks = load_tasks()
    in_scope = sum(1 for t in tasks if t.expected)
    print(
        f"{len(cards)} agents · {sum(len(c.skills) for c in cards)} skills · "
        f"{len(tasks)} held-out tasks ({in_scope} in scope, "
        f"{len(tasks) - in_scope} out of scope)\n"
    )

    routers = [KeywordRouter(), LearnedRouter()]
    if not args.no_embed:
        routers.append(EmbeddingRouter())

    results = []
    for r in routers:
        r.fit(cards)
        results.append(evaluate(r, tasks))

    print(format_table(results))

    print("\nWhere each router loses:\n")
    for r in results:
        print(
            f"  {r.router:<12} wrong agent {r.in_scope_wrong:>2} · "
            f"declined a real task {r.in_scope_declined:>2} · "
            f"claimed an out-of-scope task {r.out_scope_claimed:>2}"
        )

    if args.sweep:
        print()
        print("=" * 74)
        print("Abstention threshold sweep - the actual design decision")
        head = (
            f"| {'threshold':>9} | accuracy | precision | abstention | balanced "
            f"| declined | wrong |"
        )
        sweeps = [(LearnedRouter, "learned")]
        if not args.no_embed:
            sweeps.append((EmbeddingRouter, "embedding"))
        for factory, label in sweeps:
            print()
            print(f"  {label}")
            print("  " + head)
            print("  |" + "-" * (len(head) - 2) + "|")
            for t in (0.0, 0.15, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60):
                res = evaluate(
                    factory(min_confidence=t).fit(cards), tasks, keep_confusions=False
                )
                print(
                    f"  |     {t:.2f} |    {res.accuracy:.3f} |     {res.precision:.3f} "
                    f"|      {res.abstention:.3f} |    {res.balanced:.3f} |       "
                    f"{res.in_scope_declined:>2} |    {res.in_scope_wrong:>2} |"
                )

    if args.show_errors:
        for r in results:
            print(f"\n--- {r.router} mistakes ---")
            for text, expected, got in (r.confusions or []):
                print(f"  {expected:<18} -> {got:<18} {text[:52]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
