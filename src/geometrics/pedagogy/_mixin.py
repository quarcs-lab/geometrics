"""The ``Interpretable`` mixin shared by geometrics' result dataclasses.

It declares a small, uniform, broom-style surface — ``interpret`` (plain-language reading of
*these* results), ``explain`` (the concept explainer for the method), and ``tidy`` /
``glance`` (per-term and one-row summary frames). Concrete result classes override the
methods that make sense for them; the defaults raise an informative ``NotImplementedError``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from geometrics.pedagogy._registry import Explainer

__all__ = ["Interpretable"]


class Interpretable:
    """Mixin adding ``interpret`` / ``explain`` / ``tidy`` / ``glance`` to result objects."""

    def interpret(self, *, lang: str = "en") -> str:
        """Return a plain-language, Markdown reading of these results."""
        raise NotImplementedError(
            f"interpret() is not available for {type(self).__name__}"
        )

    def explain(self, *, lang: str = "en") -> Explainer:
        """Return the concept explainer for the method that produced these results."""
        raise NotImplementedError(
            f"explain() is not available for {type(self).__name__}"
        )

    def tidy(self) -> pd.DataFrame:
        """Return a tidy, per-term/row data frame (broom-style ``tidy``)."""
        raise NotImplementedError(f"tidy() is not available for {type(self).__name__}")

    def glance(self) -> pd.DataFrame:
        """Return a one-row, model-level summary data frame (broom-style ``glance``)."""
        raise NotImplementedError(
            f"glance() is not available for {type(self).__name__}"
        )
