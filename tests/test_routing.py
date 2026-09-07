"""Covers the protocol, the orchestrator, and the two routers that need no
model download. EmbeddingRouter is exercised by scripts/run_eval.py."""

import json

import pytest

from a2a_task_router.evaluate import EvalTask, evaluate
from a2a_task_router.protocol import (
    AgentCard,
    Message,
    RoutingDecision,
    Skill,
    Task,
    TaskState,
)
from a2a_task_router.registry import Orchestrator, Registry, load_cards
from a2a_task_router.routers import KeywordRouter, LearnedRouter, training_pairs


@pytest.fixture(scope="module")
def cards():
    return load_cards()


# --------------------------------------------------------------------------- #
#  Protocol
# --------------------------------------------------------------------------- #


class TestProtocol:
    def test_card_survives_a_json_round_trip(self, cards):
        card = cards[0]
        assert AgentCard.from_dict(json.loads(json.dumps(card.to_dict()))) == card

    def test_message_text_joins_only_the_text_parts(self):
        assert Message.user("hello there").text == "hello there"

    def test_tasks_get_distinct_ids(self):
        a, b = Task(Message.user("x")), Task(Message.user("y"))
        assert a.task_id != b.task_id

    def test_task_starts_submitted_and_unassigned(self):
        task = Task(Message.user("x"))
        assert task.state is TaskState.SUBMITTED
        assert task.assigned_to is None

    def test_unroutable_is_distinct_from_failed(self):
        """Nothing went wrong when a task is unroutable — the router declined
        to guess. Collapsing the two states loses that."""
        assert TaskState.UNROUTABLE != TaskState.FAILED

    def test_decision_without_an_agent_is_not_routed(self):
        assert RoutingDecision(None, 0.1, "unsure").routed is False


class TestTrainingPairs:
    def test_every_skill_and_example_becomes_one_labelled_row(self, cards):
        texts, labels = training_pairs(cards)
        expected = sum(1 + len(s.examples) for c in cards for s in c.skills)
        assert len(texts) == len(labels) == expected

    def test_labels_are_agent_names(self, cards):
        _, labels = training_pairs(cards)
        assert set(labels) <= {c.name for c in cards}


# --------------------------------------------------------------------------- #
#  Routers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("factory", [KeywordRouter, LearnedRouter])
class TestRouterContract:
    """Both routers must satisfy the same contract, so the harness can swap
    them without knowing which it holds."""

    def test_routes_an_obvious_task(self, factory, cards):
        router = factory().fit(cards)
        assert router.route("review this pull request for bugs").agent == "code-review-agent"

    def test_confidence_is_a_probability_like_number(self, factory, cards):
        router = factory().fit(cards)
        decision = router.route("why is my invoice higher this month")
        assert 0.0 <= decision.confidence <= 1.0

    def test_every_decision_carries_a_reason(self, factory, cards):
        router = factory().fit(cards)
        assert router.route("anything at all").reason

    def test_declines_gibberish(self, factory, cards):
        """The behaviour that matters in production: a router that always
        answers will confidently send nonsense to a real agent."""
        router = factory().fit(cards)
        assert router.route("zxqw plort mimsy borogove").routed is False

    def test_unfitted_router_declines_rather_than_crashing(self, factory):
        assert factory().route("anything").routed is False


class TestKeywordRouter:
    def test_idf_weighting_stops_the_longest_card_winning(self, cards):
        """Terms shared by every agent are worth nearly nothing. Without that,
        the agent with the most examples wins every ambiguous task."""
        router = KeywordRouter().fit(cards)
        assert router.route("find 30 minutes with the team").agent == "scheduling-agent"


class TestLearnedRouter:
    def test_threshold_is_what_produces_abstention(self, cards):
        """Same input, two thresholds. At zero the router always answers —
        which is exactly how a confident wrong route happens."""
        text = "zxqw plort mimsy borogove"
        assert LearnedRouter(min_confidence=0.35).fit(cards).route(text).routed is False
        assert LearnedRouter(min_confidence=0.0).fit(cards).route(text).routed is True

    def test_character_ngrams_survive_a_typo(self, cards):
        router = LearnedRouter().fit(cards)
        assert router.route("revew this pul reqest").agent == "code-review-agent"


# --------------------------------------------------------------------------- #
#  Orchestrator
# --------------------------------------------------------------------------- #


class TestOrchestrator:
    @pytest.fixture
    def orchestrator(self, cards):
        registry = Registry(cards=list(cards))
        handlers = {c.name: (lambda t, n=c.name: f"{n} handled it") for c in cards}
        return Orchestrator(registry=registry, router=LearnedRouter(), handlers=handlers)

    def test_completes_a_routed_task(self, orchestrator):
        task = orchestrator.submit("review my pull request for security holes")
        assert task.state is TaskState.COMPLETED
        assert task.assigned_to == "code-review-agent"
        assert "handled it" in task.history[-1].text

    def test_unroutable_task_is_not_a_failure(self, orchestrator):
        task = orchestrator.submit("zxqw plort mimsy borogove")
        assert task.state is TaskState.UNROUTABLE
        assert task.assigned_to is None
        assert task.error

    def test_a_raising_agent_fails_the_task_rather_than_the_process(self, cards):
        def explode(_task):
            raise RuntimeError("upstream timeout")

        orchestrator = Orchestrator(
            registry=Registry(cards=list(cards)),
            router=LearnedRouter(),
            handlers={"billing-agent": explode},
        )
        task = orchestrator.submit("why was I charged twice this month")
        assert task.state is TaskState.FAILED
        assert "upstream timeout" in task.error

    def test_missing_handler_fails_loudly(self, cards):
        orchestrator = Orchestrator(
            registry=Registry(cards=list(cards)), router=LearnedRouter(), handlers={}
        )
        task = orchestrator.submit("draft the postmortem for the outage")
        assert task.state is TaskState.FAILED
        assert "no handler" in task.error

    def test_fits_lazily_on_first_submit(self, orchestrator):
        assert orchestrator._fitted is False
        orchestrator.submit("book a room for Tuesday")
        assert orchestrator._fitted is True


class TestRegistry:
    def test_adding_the_same_name_replaces_rather_than_duplicates(self):
        skill = Skill("s", "S", "d", ())
        registry = Registry()
        registry.add(AgentCard("a", "one", "u", "1", (skill,)))
        registry.add(AgentCard("a", "two", "u", "2", (skill,)))
        assert registry.names == ["a"]
        assert registry.get("a").description == "two"

    def test_remove(self, cards):
        registry = Registry(cards=list(cards)).remove("billing-agent")
        assert "billing-agent" not in registry.names


# --------------------------------------------------------------------------- #
#  Scoring
# --------------------------------------------------------------------------- #


class TestEvaluate:
    def test_a_router_that_always_declines_scores_zero_accuracy(self, cards):
        class Mute:
            name = "mute"

            def fit(self, _cards):
                return self

            def route(self, _text):
                return RoutingDecision(None, 0.0, "never")

        result = evaluate(
            Mute(), [EvalTask("a", "billing-agent"), EvalTask("b", None)]
        )
        assert result.accuracy == 0.0
        assert result.abstention == 1.0
        # The balanced score is what stops that degenerate strategy looking good.
        assert result.balanced == 0.5

    def test_precision_ignores_declines_but_accuracy_does_not(self):
        class Half:
            name = "half"

            def fit(self, _cards):
                return self

            def route(self, text):
                if text == "a":
                    return RoutingDecision("billing-agent", 0.9, "ok")
                return RoutingDecision(None, 0.1, "unsure")

        result = evaluate(
            Half(), [EvalTask("a", "billing-agent"), EvalTask("b", "billing-agent")]
        )
        assert result.precision == 1.0
        assert result.accuracy == 0.5
