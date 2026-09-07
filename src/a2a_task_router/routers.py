"""Three routers with one interface, so the harness can compare them fairly.

  KeywordRouter   the baseline everybody writes first
  LearnedRouter   TF-IDF + logistic regression, fitted on the agent cards
  EmbeddingRouter PyTorch sentence embeddings, nearest skill by cosine

The baseline is not a straw man. It is written properly — multi-word phrases,
weighted by specificity, drawn from the same card data the learned router
trains on — because a comparison against a deliberately bad baseline proves
nothing.

Every router returns a RoutingDecision with a confidence, and every router can
decline. Declining is the behaviour that matters in production: a router that
always answers will confidently send a payroll question to the code reviewer.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .protocol import AgentCard, RoutingDecision

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def training_pairs(cards: list[AgentCard]) -> tuple[list[str], list[str]]:
    """Flatten the cards into (text, agent-name) pairs.

    Every router learns from this and nothing else. No separate labelled file,
    no annotation step — the agent's own published examples are the labels,
    which is the whole argument for putting examples in the card.
    """
    texts: list[str] = []
    labels: list[str] = []
    for card in cards:
        for skill in card.skills:
            texts.append(f"{skill.name}. {skill.description}")
            labels.append(card.name)
            for example in skill.examples:
                texts.append(example)
                labels.append(card.name)
    return texts, labels


# --------------------------------------------------------------------------- #
#  Baseline
# --------------------------------------------------------------------------- #


@dataclass
class KeywordRouter:
    """Score each agent by how many of its card's distinctive terms appear.

    Terms are weighted by inverse document frequency across agents, so a word
    used by every agent ("the", "data", "request") counts for almost nothing
    and a word used by one counts fully. Without that weighting the longest
    card wins every ambiguous task, which is the failure mode that makes naive
    keyword routing look worse than it is.
    """

    name: str = "keyword"
    min_score: float = 0.5

    _weights: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)

    def fit(self, cards: list[AgentCard]) -> KeywordRouter:
        texts, labels = training_pairs(cards)

        per_agent: dict[str, Counter[str]] = {}
        for text, label in zip(texts, labels, strict=True):
            per_agent.setdefault(label, Counter()).update(set(tokenize(text)))

        agents_with = Counter()
        for counts in per_agent.values():
            agents_with.update(counts.keys())

        n = len(per_agent)
        self._weights = {
            agent: {
                term: count * (n / agents_with[term])
                for term, count in counts.items()
            }
            for agent, counts in per_agent.items()
        }
        return self

    def route(self, text: str) -> RoutingDecision:
        terms = set(tokenize(text))
        scores = {
            agent: sum(weights.get(t, 0.0) for t in terms)
            for agent, weights in self._weights.items()
        }
        if not scores:
            return RoutingDecision(None, 0.0, "no agents registered")

        best, best_score = max(scores.items(), key=lambda kv: kv[1])
        total = sum(scores.values()) or 1.0
        confidence = best_score / total

        if best_score < self.min_score:
            return RoutingDecision(None, confidence, "no distinctive term matched")
        return RoutingDecision(best, confidence, f"matched terms scoring {best_score:.1f}")


# --------------------------------------------------------------------------- #
#  Learned — scikit-learn
# --------------------------------------------------------------------------- #


@dataclass
class LearnedRouter:
    """TF-IDF over character and word n-grams into a logistic regression.

    Character n-grams are in there on purpose: they survive typos, plurals and
    the compound words that appear constantly in technical requests, where a
    pure word model sees "rerank" and "reranking" as unrelated.

    `min_confidence` is the abstention threshold, and it is a real decision
    rather than a default. Set it to zero and the router always answers, which
    on out-of-scope input means always answering wrongly.
    """

    name: str = "learned"
    min_confidence: float = 0.35

    _pipeline: object = field(default=None, repr=False)
    _classes: list[str] = field(default_factory=list, repr=False)

    def fit(self, cards: list[AgentCard]) -> LearnedRouter:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline

        texts, labels = training_pairs(cards)

        features = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ])
        self._pipeline = Pipeline([
            ("features", features),
            # Balanced class weights because agents publish uneven numbers of
            # examples, and the router should not prefer the chatty one.
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)),
        ])
        self._pipeline.fit(texts, labels)
        self._classes = list(self._pipeline.named_steps["clf"].classes_)
        return self

    def route(self, text: str) -> RoutingDecision:
        if self._pipeline is None:
            return RoutingDecision(None, 0.0, "router not fitted")

        probs = self._pipeline.predict_proba([text])[0]
        best = int(probs.argmax())
        confidence = float(probs[best])

        if confidence < self.min_confidence:
            return RoutingDecision(
                None, confidence, f"top class only {confidence:.2f} confident"
            )
        return RoutingDecision(
            self._classes[best], confidence, f"p={confidence:.2f}"
        )


# --------------------------------------------------------------------------- #
#  Embedding — PyTorch
# --------------------------------------------------------------------------- #


@dataclass
class EmbeddingRouter:
    """Nearest published example by cosine similarity, using a sentence encoder.

    No training step at all: the card's examples are embedded once and a task
    is routed to whichever agent owns the closest one. That makes adding an
    agent a deployment rather than a retraining, which is the property that
    matters if agents come and go.

    The cost is that it cannot learn what distinguishes two agents — it only
    knows what each looks like on its own.
    """

    name: str = "embedding"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Chosen by the sweep in scripts/run_eval.py --sweep, not guessed. At 0.35
    # this router declines 11 of 40 real tasks; at 0.30 it declines 3, for one
    # extra wrong route. See the README for the caveat about picking it on the
    # same set it is reported on.
    min_confidence: float = 0.30

    _model: object = field(default=None, repr=False)
    _vectors: object = field(default=None, repr=False)
    _labels: list[str] = field(default_factory=list, repr=False)

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, cards: list[AgentCard]) -> EmbeddingRouter:
        model = self._load()
        texts, self._labels = training_pairs(cards)
        self._vectors = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return self

    def route(self, text: str) -> RoutingDecision:
        if self._vectors is None:
            return RoutingDecision(None, 0.0, "router not fitted")

        model = self._load()
        vec = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self._vectors @ vec
        best = int(sims.argmax())
        confidence = float(sims[best])

        if confidence < self.min_confidence:
            return RoutingDecision(
                None, confidence, f"nearest example only {confidence:.2f} similar"
            )
        return RoutingDecision(
            self._labels[best], confidence, f"cosine={confidence:.2f}"
        )
