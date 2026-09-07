# 🔀 A2A Task Router

**Which agent should handle this task — and when should none of them?**

Agent-to-agent systems need a router. Most demos write a keyword match, measure accuracy, and ship. This repo measures three routers properly and finds that **accuracy is the wrong metric**: the keyword baseline scores respectably on it and is unusable in production, because it never says no.

Routing is learned from the agents' own published capability cards. No separate labelled file, no annotation step — an agent that describes itself well becomes routable without anyone editing the router.

🧭 **The routers:** [`src/a2a_task_router/routers.py`](src/a2a_task_router/routers.py)
📋 **The protocol types:** [`src/a2a_task_router/protocol.py`](src/a2a_task_router/protocol.py)
📝 **Related write-up:** [One agent found 0 of 36. A crew found 36 of 36, for half the tokens.](https://ketanshukla.dev/blog/one-agent-found-zero-of-thirty-six)

---

## The headline

5 agents · 8 skills · 48 held-out tasks (40 in scope, 8 deliberately not)

| router | accuracy | precision | abstention | balanced | ms/task |
|---|---|---|---|---|---|
| keyword (baseline) | 0.625 | 0.625 | **0.000** | 0.312 | **0.04** |
| learned (scikit-learn) | 0.500 | 0.909 | 0.750 | 0.625 | 4.01 |
| **embedding (PyTorch)** | **0.850** | **0.919** | **0.875** | **0.863** | 41.55 |

- **accuracy** — correct routes over in-scope tasks; a decline counts as a miss
- **precision** — of the in-scope tasks it *claimed*, how many were right
- **abstention** — of the out-of-scope tasks, how many it correctly refused
- **balanced** — the mean of accuracy and abstention, so neither degenerate strategy wins

### The baseline looks fine until you look at what it refuses

```
keyword      wrong agent 15 · declined a real task  0 · claimed an out-of-scope task 8
learned      wrong agent  2 · declined a real task 18 · claimed an out-of-scope task 2
embedding    wrong agent  3 · declined a real task  3 · claimed an out-of-scope task 1
```

The keyword router got **8 out of 8** out-of-scope tasks wrong, because it claimed every one of them. Ask it to translate a sentence into Japanese and it confidently hands the job to an agent that reviews pull requests.

Its abstention score is **0.000**. Not low — zero. It has no mechanism for declining that ever fires, and its 0.625 accuracy hides that completely.

This is the same shape as [an approval gate that trains reflexive clicking](https://ketanshukla.dev/blog/danger-hint-is-not-a-permission-model): a component that always answers gives you the *feeling* of a decision without the decision.

### The learned router has the opposite problem

Precision 0.909 — when it commits, it is nearly always right. Accuracy 0.500, because it declined **18 of 40 real tasks**. Safe and useless: nearly half the work falls on the floor.

---

## The actual finding: the threshold matters more than the model

Both learned routers take a `min_confidence` below which they decline. That one constant moves the result more than the choice of model does.

```
python scripts/run_eval.py --sweep
```

**Embedding router, sweeping the abstention threshold:**

| threshold | accuracy | precision | abstention | balanced | declined | wrong |
|---|---|---|---|---|---|---|
| 0.00 | 0.900 | 0.900 | 0.000 | 0.450 | 0 | 4 |
| 0.15 | 0.900 | 0.900 | 0.250 | 0.575 | 0 | 4 |
| 0.25 | 0.875 | 0.897 | 0.750 | 0.812 | 1 | 4 |
| **0.30** | 0.850 | 0.919 | 0.875 | **0.863** | 3 | 3 |
| 0.35 | 0.675 | 0.931 | 0.875 | 0.775 | 11 | 2 |
| 0.40 | 0.600 | 0.960 | 0.875 | 0.738 | 15 | 1 |
| 0.50 | 0.275 | 0.917 | 1.000 | 0.637 | 28 | 1 |
| 0.60 | 0.100 | 1.000 | 1.000 | 0.550 | 36 | 0 |

Read the last two columns. **Every point of precision past 0.30 is bought by dropping real work on the floor.** At 0.60 the router is perfect and answers almost nothing.

My first guess for this constant was 0.35. That guess cost 8 points of accuracy and 9 of balanced score — a larger swing than switching from scikit-learn to PyTorch would have bought me at the wrong threshold.

<sub>**The honest caveat:** with 48 tasks there is no separate development set, so 0.30 was chosen on the same data it is reported on. That is overfitting, and the number to trust is the *shape* of the curve, not the specific optimum. On a real deployment this threshold belongs in config, tuned on held-out traffic and revisited when agents are added.</sub>

---

## Routing is learned from the agent cards

The [A2A protocol](https://a2a-protocol.org/) idea worth stealing even if you never adopt the spec: **an agent's capability declaration is data it publishes**, not configuration you write about it. So routing can be learned from it.

```json
{
  "name": "billing-agent",
  "skills": [{
    "skill_id": "explain_invoice",
    "name": "Explain an invoice",
    "description": "Break down what a customer was charged for...",
    "examples": [
      "why is my invoice higher this month",
      "I was charged twice for March",
      "my card was declined, what now"
    ]
  }]
}
```

Those `examples` are not documentation — they are the training set. `training_pairs()` flattens every card into `(text, agent-name)` rows and all three routers fit on exactly that. Adding a sixth agent means publishing a card, not editing a router.

The evaluation tasks are written deliberately *unlike* the examples — same intent, different words — because a router scored on the strings it trained on tells you nothing:

| card example | eval task |
|---|---|
| `I was charged twice for March` | `there are two identical charges on the same day` |
| `error rate spiked at 3am, what happened` | `something started throwing 502s about twenty minutes ago` |
| `find 30 minutes with the design team next week` | `get half an hour with the platform folks before Wednesday` |

---

## Design notes

### The baseline is not a straw man

`KeywordRouter` weights each agent's terms by inverse document frequency across agents, so a word every agent uses counts for almost nothing. Without that, the agent with the longest card wins every ambiguous task — and a comparison against a deliberately bad baseline proves nothing. There is a test asserting the weighting works.

### Character n-grams, on purpose

`LearnedRouter` unions word n-grams with `char_wb` 3–5 grams. They survive typos, plurals and compound technical words, where a pure word model sees `rerank` and `reranking` as unrelated. There is a test that routes `revew this pul reqest` correctly.

### `UNROUTABLE` is not `FAILED`

A task nothing claimed is a distinct terminal state. Nothing went wrong — the router declined to guess. Collapsing the two loses the distinction that matters when you are looking at why work stopped, which is [the same argument as marking a failed run explicitly](https://ketanshukla.dev/blog/a-long-agent-run-should-not-die-on-one-failure) rather than leaving it stuck.

### Every decision carries a reason

`RoutingDecision(agent, confidence, reason)`. A router that returns only a name gives you nothing to look at when it sends a refund question to the code reviewer.

---

## Running it

```bash
pip install -r requirements.txt

python scripts/run_eval.py                 # all three (downloads ~90 MB)
python scripts/run_eval.py --no-embed      # keyword + learned only, no downloads
python scripts/run_eval.py --sweep         # the threshold curve above
python scripts/run_eval.py --show-errors   # every misroute, with the task text
```

```bash
pytest -q     # 30 tests, no network, no model downloads
```

---

## Tests

30 tests covering the protocol round-trip, the orchestrator's three terminal states, and both routers that need no download. The parametrised `TestRouterContract` runs the same five assertions against `KeywordRouter` and `LearnedRouter`, so the harness can swap them without knowing which it holds.

The ones worth reading assert a *property*: that an unfitted router declines rather than crashing, that a raising agent fails the task rather than the process, that a router which always declines scores 0.5 balanced and not something flattering, and that setting `min_confidence=0.0` makes abstention disappear entirely.

---

## Stack

Python · PyTorch (CPU) · sentence-transformers · scikit-learn · NumPy · pytest

MIT licensed. Built by [Ketan Shukla](https://ketanshukla.dev).
