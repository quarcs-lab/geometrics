"""The concept-explainer registry: ``Explainer`` plus ``explain`` / ``list_topics``.

An *explainer* is data-independent teaching content about a method or concept ("what is a
fixed effect / when do I use it / what are the pitfalls"). It is structured (not a bare
string) so apps can render each section separately, while notebooks get a rich
``_repr_markdown_``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Explainer", "explain", "list_topics", "register_topic"]


@dataclass(frozen=True)
class Explainer:
    """Structured teaching content for one method or concept.

    Parameters
    ----------
    topic
        Canonical key (e.g. ``"beta_convergence"``).
    title
        Human-readable title (e.g. ``"Beta convergence"``).
    what
        One-paragraph "what it is".
    when_to_use
        One-paragraph "when to use it".
    caveats
        Bullet-point pitfalls / things to watch for.
    see_also
        Related topic keys.
    references
        Short citations (e.g. textbook chapters).
    """

    topic: str
    title: str
    what: str
    when_to_use: str
    caveats: tuple[str, ...] = ()
    see_also: tuple[str, ...] = ()
    references: tuple[str, ...] = field(default=())

    def to_markdown(self) -> str:
        """Render the explainer as a Markdown string (for apps, notebooks and docs)."""
        parts = [f"### {self.title}", "", f"**What it is.** {self.what}", ""]
        parts += [f"**When to use it.** {self.when_to_use}"]
        if self.caveats:
            parts += ["", "**Watch out for.**"]
            parts += [f"- {c}" for c in self.caveats]
        if self.see_also:
            parts += ["", f"*See also:* {', '.join(self.see_also)}"]
        if self.references:
            parts += ["", f"*References:* {'; '.join(self.references)}"]
        return "\n".join(parts)

    def _repr_markdown_(self) -> str:  # Jupyter rich display
        return self.to_markdown()

    def __str__(self) -> str:
        return self.to_markdown()


_TOPICS: dict[str, Explainer] = {}
_ALIASES: dict[str, str] = {}


def register_topic(explainer: Explainer, *, aliases: tuple[str, ...] = ()) -> None:
    """Register ``explainer`` under its ``topic`` key (and any ``aliases``).

    Parameters
    ----------
    explainer
        The explainer to register.
    aliases
        Alternative keys that resolve to this topic (e.g. ``"sigma"`` ->
        ``"sigma_convergence"``).
    """
    _TOPICS[explainer.topic] = explainer
    for alias in aliases:
        _ALIASES[alias] = explainer.topic


def explain(topic: str, *, lang: str = "en") -> Explainer:
    """Return the :class:`Explainer` for a method or concept.

    Parameters
    ----------
    topic
        A topic key or alias (see :func:`list_topics`).
    lang
        Language code. Only ``"en"`` ships today; the parameter is reserved so that adding
        translations later is non-breaking.

    Returns
    -------
    Explainer
        The matching explainer.

    Raises
    ------
    KeyError
        If ``topic`` is unknown; the message lists the available topics.

    Examples
    --------
    ```python
    from geometrics.pedagogy import explain

    explain("beta_convergence").title
    ```
    """
    key = _ALIASES.get(topic, topic)
    if key not in _TOPICS:
        available = ", ".join(list_topics())
        raise KeyError(f"unknown topic {topic!r}; available topics: {available}")
    return _TOPICS[key]


def list_topics() -> list[str]:
    """Return the sorted list of canonical topic keys (for app menus and docs)."""
    return sorted(_TOPICS)
