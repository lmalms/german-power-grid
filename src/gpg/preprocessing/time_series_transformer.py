from typing import Tuple

from copy import deepcopy
import numpy as np


class TimeSeriesTransformer:

    def __init__(self, n_lags: int, horizon: int):
        self.n_lags = n_lags
        self.horizon = horizon

    def fit(self, y: np.ndarray) -> None:
        pass

    def fit_transform(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.transform(y)

    def transform(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ts = deepcopy(y)
        n_folds = int((len(y) - self.horizon) / self.n_lags)
        X = np.empty((n_folds, self.n_lags, 1), dtype=np.float32)
        y = np.empty((n_folds, self.n_lags, self.horizon), dtype=np.float32)

        for i in range(n_folds):
            X[i, :] = ts[i * self.n_lags: (i + 1) * self.n_lags].reshape(-1, 1)
            for j in range(self.n_lags):
                y[i, j, :] = ts[i * self.n_lags + j: i * self.n_lags + j + self.horizon]

        return X, y

