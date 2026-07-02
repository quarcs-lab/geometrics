"""Tests for the pedagogy layer: the explainer registry and its shipped topics.

Interpretation strings are asserted on stable substrings (not whole strings) to stay robust
to small wording changes.
"""

from __future__ import annotations

import pytest

from geometrics.pedagogy import Explainer, explain, list_topics

# --- explainer registry --------------------------------------------------------------


def test_list_topics_is_sorted_and_nonempty():
    topics = list_topics()
    assert topics == sorted(topics)
    assert {
        "pearson",
        "spearman",
        "correlation_vs_causation",
        "beta_convergence",
        "sigma_convergence",
        "convergence_clubs",
    } <= set(topics)


def test_every_topic_builds_markdown():
    for topic in list_topics():
        exp = explain(topic)
        assert isinstance(exp, Explainer)
        md = exp.to_markdown()
        assert md.startswith("### ")
        assert "**What it is.**" in md
        assert exp._repr_markdown_() == md


def test_explain_alias_resolves():
    assert explain("convergence").topic == "beta_convergence"
    assert explain("sigma").topic == "sigma_convergence"
    assert explain("phillips_sul").topic == "convergence_clubs"
    assert explain("log_t").topic == "convergence_clubs"


def test_explain_unknown_raises_with_available():
    with pytest.raises(KeyError) as excinfo:
        explain("not_a_topic")
    msg = str(excinfo.value)
    assert "not_a_topic" in msg
    assert "beta_convergence" in msg  # the message lists available topics


def test_interpretable_defaults_raise():
    from geometrics.pedagogy import Interpretable

    class Dummy(Interpretable):
        pass

    d = Dummy()
    with pytest.raises(NotImplementedError, match="interpret"):
        d.interpret()
    with pytest.raises(NotImplementedError, match="explain"):
        d.explain()
    with pytest.raises(NotImplementedError, match="tidy"):
        d.tidy()
    with pytest.raises(NotImplementedError, match="glance"):
        d.glance()
