# SPEC — a2a-task-router

> A specification, not a tutorial. It says what this repo is for, what it
> deliberately does not do, and what must stay true for a change to be correct.

**Status:** complete for its purpose · **Version:** 0.1.0 · **Last verified:** 7 Sep 2026

---

## 1. Purpose

Answer two questions about multi-agent routing with measurements:

> **Which agent should handle this task — and when should none of them?**

The second half is the point. Most routing demos measure accuracy and ship. This
repo shows that a competent keyword baseline scores 0.625 accuracy while
claiming **every** out-of-scope task, and that accuracy cannot see the problem.

Routing is learned from the agents' own published capability cards, so onboarding
an agent is a card publication rather than a router edit.

## 2. Scope

### In scope
- Agent2Agent-shaped protocol types: `AgentCard`, `Skill`, `Task`, `Message`
- Agent discovery via a registry of cards
- Three routers behind one interface, including abstention
- A held-out evaluation set containing deliberately unroutable tasks
- Four metrics, and a threshold sweep

### Explicitly out of scope
- **Transport.** No HTTP, no JSON-RPC, no streaming, no push notifications. A
  handler is a Python callable. Transport would obscure the routing decision,
  which is the subject.
- **Authentication and authorisation.** Who may call an agent is a real problem
  and a different one.
- **Task lifecycle beyond one hop.** No multi-turn negotiation, no delegation
  chains, no task cancellation.
- **Full A2A conformance.** Three ideas are borrowed — cards, tasks, message
  envelopes. This is not a compliant implementation and does not claim to be.
- **Model training.** Both learned routers fit on published card text at
  startup. Nothing is fine-tuned.

## 3. Architecture

```
data/agents.json ──► load_cards() ──► Registry
                                         │
                                         ▼
                                   Router.fit(cards)      training_pairs()
                                         │                flattens every skill
                                         ▼                and example into
task text ──────────────────────► Router.route(text)      (text, agent) rows
                                         │
                                         ▼
                                  RoutingDecision(agent | None, confidence, reason)
                                         │
                                         ▼
                              Orchestrator.submit() ──► Task(state)
```

| module | responsibility |
|---|---|
| `protocol.py` | wire types; no logic beyond serialisation |
| `routers.py` | the three routing strategies |
| `registry.py` | card loading, agent registry, orchestration |
| `evaluate.py` | scoring against held-out tasks |

## 4. Interfaces

### Router protocol

```python
class Router(Protocol):
    name: str
    def fit(self, cards: list[AgentCard]): ...
    def route(self, text: str) -> RoutingDecision: ...
```

```python
@dataclass(frozen=True)
class RoutingDecision:
    agent: str | None      # None means declined
    confidence: float      # 0.0–1.0
    reason: str            # why, for diagnosis after the fact
```

### The three routers

| router | mechanism | fit cost | route cost |
|---|---|---|---|
| `KeywordRouter` | IDF-weighted term overlap | instant | 0.04 ms |
| `LearnedRouter` | TF-IDF (word + char n-grams) → logistic regression | ~1 s | 4 ms |
| `EmbeddingRouter` | nearest published example by cosine | model load | 42 ms |

### Data formats

`data/agents.json` — a list of agent cards. The `examples` on each skill are the
training data, not documentation.

`data/eval_tasks.jsonl`:
```json
{"text": "there are two identical charges on the same day", "expected": "billing-agent"}
{"text": "what is the capital of Portugal", "expected": null}
```

`expected: null` means **nothing should claim this**. These tasks are what make
abstention measurable.

### CLI

```
python scripts/run_eval.py [--no-embed] [--sweep] [--show-errors]
```

## 5. Invariants

1. **Every router can decline.** A router with no abstention path is not a valid
   implementation of the protocol, regardless of its accuracy.
2. **Every decision carries a `reason`.** A bare agent name is undiagnosable
   when the route is wrong.
3. **Routers train only on card text.** No separate labelled file. If routing
   needs data an agent did not publish about itself, the card is wrong.
4. **Evaluation tasks are phrased unlike the card examples.** Same intent,
   different words. A router scored on its training strings measures nothing.
5. **`UNROUTABLE` is distinct from `FAILED`.** Nothing went wrong when a task is
   unroutable; the router declined to guess.
6. **An agent that raises fails the task, not the process.**
7. **The test suite performs no network I/O and downloads no model.**
8. **A tuned threshold is reported with the caveat that it was tuned on the set
   it is measured against.**

## 6. Metrics — and why four

| metric | definition | what it catches |
|---|---|---|
| accuracy | correct ÷ in-scope tasks (declining counts as a miss) | overall usefulness |
| precision | correct ÷ in-scope tasks *claimed* | quality when it commits |
| abstention | correctly refused ÷ out-of-scope tasks | **whether it can say no at all** |
| balanced | mean of accuracy and abstention | kills both degenerate strategies |

`balanced` exists because a router that always answers scores 0 on abstention
and one that never answers scores 0 on accuracy. Neither cheat can look good.

## 7. Verification

| | |
|---|---|
| tests | 30, `pytest -q`, ~5 s, no network |
| coverage | protocol round-trip, orchestrator's three terminal states, `KeywordRouter`, `LearnedRouter` |
| not covered | `EmbeddingRouter` — needs a 90 MB download; exercised by `run_eval.py` |
| CI | GitHub Actions, `numpy scikit-learn pytest` |

`TestRouterContract` is parametrised over both testable routers, so the same
five assertions run against each and the harness can swap them blind.

## 8. Results this repo produced

5 agents · 8 skills · 48 held-out tasks (40 in scope, 8 not)

| router | accuracy | precision | abstention | balanced | ms/task |
|---|---|---|---|---|---|
| keyword | 0.625 | 0.625 | **0.000** | 0.312 | 0.04 |
| learned | 0.500 | 0.909 | 0.750 | 0.625 | 4.01 |
| **embedding** | **0.850** | **0.919** | **0.875** | **0.863** | 41.55 |

```
keyword      wrong 15 · declined  0 · claimed out-of-scope 8
learned      wrong  2 · declined 18 · claimed out-of-scope 2
embedding    wrong  3 · declined  3 · claimed out-of-scope 1
```

**Finding:** the abstention threshold moves the result more than the model
choice. On the embedding router, 0.30 vs 0.35 is 9 points of balanced score.

## 9. Known limitations

- **48 tasks.** One task moves accuracy by 2.5 points.
- **The 0.30 threshold was tuned on the evaluation set.** There is no separate
  development set. Trust the *shape* of the sweep curve, not the optimum.
- **Five agents, cleanly separated.** Real deployments have agents with
  overlapping capabilities, which is the hard case and is not represented.
- **Synthetic cards.** Written for this repo, so they are better-described than
  cards in the wild. Routing quality is downstream of card quality.
- **English only.**

## 10. Extending it

Add an agent: append a card to `data/agents.json`. No code change.

Add a router: implement `fit` and `route`, append it in `scripts/run_eval.py`.
`TestRouterContract` will apply to it if you add it to the parametrise list.

Deploy this for real: replace the callable handlers in `Orchestrator.handlers`
with HTTP calls to each card's `url`, and move `min_confidence` into config.

## 11. Related

- Post: [Accuracy is the wrong metric for an agent router](https://ketanshukla.dev/blog/accuracy-is-the-wrong-metric-for-a-router)
- Related: [One agent found 0 of 36. A crew found 36 of 36, for half the tokens.](https://ketanshukla.dev/blog/one-agent-found-zero-of-thirty-six)
- Decision history: [CHATLOG.md](CHATLOG.md)
