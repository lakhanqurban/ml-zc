"""Custom sklearn transformers for the hotel booking cancellation model.

Kept in its own module so that the pickled pipeline can be unpickled by
``serve.py`` / ``predict.py`` without importing ``train.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CountryGrouper(BaseEstimator, TransformerMixin):
    """Collapse the long tail of country codes into a single ``Other`` bucket.

    The dataset contains 178 origin countries, but a handful of them account for
    almost every booking and the rest are one-off rows that only add noise (and
    overfitting opportunity) to a one-hot encoding.

    The top-N countries are learned during ``fit``, which happens on the
    training split only, so no information from validation or test leaks into
    the model.

    Parameters
    ----------
    n_top : int
        How many most frequent countries to keep as their own category.
    other_label : str
        Category assigned to every remaining country.
    """

    def __init__(self, n_top: int = 5, other_label: str = "Other"):
        self.n_top = n_top
        self.other_label = other_label

    @staticmethod
    def _as_series(X) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].astype(str)
        return pd.Series(np.asarray(X).ravel()).astype(str)

    def fit(self, X, y=None) -> "CountryGrouper":
        series = self._as_series(X)
        counts = series.value_counts()
        self.top_countries_ = list(counts.index[: self.n_top])
        return self

    def transform(self, X) -> np.ndarray:
        series = self._as_series(X)
        grouped = series.where(series.isin(self.top_countries_), self.other_label)
        # keep it 2D so it can be chained into a OneHotEncoder
        return grouped.to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(["country"])
