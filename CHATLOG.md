# CHATLOG — a2a-task-router

> A record of the decisions that produced this repo: what was chosen, what was
> rejected, and what a measurement changed. Written from the session it was
> built in, not reconstructed afterwards.

---

## 7 Sep 2026 — the whole build

Built in one session with Claude Opus 5, directed through a plan → build →
measure → revise loop. One of three repos built that day to cover MCP, A2A, RAG
and model APIs with work that genuinely required PyTorch and scikit-learn rather
than importing them decoratively.

### The brief

Cover **A2A** — agent-to-agent — with something that needs a learned model for a
real reason, not an ornamental one.

Routing is that reason. Deciding which of several agents should handle an
incoming task is a classification problem with a genuine abstention requirement,
and abstention is where the interesting engineering is.

### Decision: learn from capability cards, not a labelled file

The A2A idea worth stealing even without adopting the spec: **an agent's
capability declaration is data it publishes about itself.** So routing can be
learned from it.

`training_pairs()` flattens every skill name, description and example into
`(text, agent-name)` rows. All three routers fit on exactly that and nothing
else.

Consequence, which is the point: adding a sixth agent means publishing a card.
No retraining pipeline, no annotation step, no router edit.

Consequence, which is the cost: routing quality is downstream of how well agents
describe themselves. Same property that makes an MCP tool description the most
consequential line in a server.

### Decision: build the baseline properly

**Rejected:** a naive substring match, which would have made the learned routers
look good for free.

`KeywordRouter` weights each agent's terms by inverse document frequency *across
agents*, so a word every agent uses counts for almost nothing. Without that, the
agent with the longest card wins every ambiguous task.

A comparison against deliberately bad code proves nothing. The baseline had to
lose on its merits — and it does, structurally: it has no mechanism to decline.

### Decision: eight tasks that nothing should claim

The evaluation set contains `"what is the capital of Portugal"`,
`"write me a limerick about a badger"` and six more, all labelled
`expected: null`.

This is the measurement most routing demos leave out, and it produced the
repo's headline. Without those eight rows, the keyword router looks like a
reasonable option.

### Decision: four metrics, because one hides the answer

Accuracy alone ranked keyword (0.625) above learned (0.500). That ordering is
arithmetically correct and operationally backwards.

Added precision, abstention, and a balanced score that is the mean of accuracy
and abstention — specifically so that "always answer" and "never answer" both
score 0.5 and neither looks good.

### First run

```
keyword    accuracy 0.625  abstention 0.000  balanced 0.312
learned    accuracy 0.500  abstention 0.750  balanced 0.625
embedding  accuracy 0.675  abstention 0.875  balanced 0.775
```

The keyword router claimed **8 of 8** out-of-scope tasks. Its abstention rate is
not low, it is zero — there is no code path that declines.

The learned router had the opposite failure: precision 0.909, but it declined 18
of 40 real tasks. Safe and useless.

### Revision: sweep the threshold, and find the real finding

The learned routers both take a `min_confidence`. `0.35` had been chosen by
guessing. Adding `--sweep` to vary it produced the most useful table in the repo:

```
embedding   0.25 → balanced 0.812
            0.30 → balanced 0.863   ← best
            0.35 → balanced 0.775   ← the guess
            0.40 → balanced 0.738
```

**The guess cost 17 points of accuracy** — a larger swing than switching from
scikit-learn to PyTorch would have produced at the wrong threshold.

Changed the `EmbeddingRouter` default to 0.30, with a code comment saying it
came from the sweep rather than from judgement.

### Decision: state the overfitting rather than hide it

0.30 was chosen on the same 48 tasks it is reported against. With no separate
development set that is overfitting, and quoting 0.863 without saying so would
be a misleading number.

Both the README and the SPEC say the trustworthy signal is the *shape* of the
curve — steeply falling accuracy past the knee — not the specific optimum.

### Decision: `UNROUTABLE` is not `FAILED`

A task nothing claimed is a distinct terminal state. Nothing went wrong; the
router declined to guess. Collapsing the two loses exactly the distinction you
need when looking at why work stopped — the same reasoning as marking a failed
background run explicitly rather than leaving it stuck in `running`.

### Things that went wrong

- **Nested shell escaping mangled the sweep code** on first write: `\n` inside a
  heredoc inside a Python string became a real newline and produced an
  unterminated string literal. Rebuilt the block line by line instead.
- **Character n-grams were not in the first version** of `LearnedRouter`. Added
  after a typo test (`revew this pul reqest`) failed. `char_wb` 3–5 grams fixed
  it and the test stayed.

### Deliberately not done

- No transport. Handlers are Python callables; HTTP would obscure the subject.
- No auth. Real, and a different repo.
- No multi-hop delegation or task cancellation.
- No claim of A2A conformance — three ideas borrowed, not a compliant server.

---

## Conventions for future entries

Each entry should record **what changed, what it was measured against, and what
was rejected.** An entry that only says what was added is a commit message and
belongs in git instead.
