from typing import Optional

from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_percentage_error


class CrossValidationExperiment:

    def __init__(
            self,
            n_splits: int,
            max_train_samples: Optional[int] = None
    ):
        self.n_splits = n_splits
        self.max_train_samples = max_train_samples

    def run(self, forecaster, data: np.ndarray) -> pd.DataFrame:
        data = self._validate_and_format_data(data=data)
        scores = defaultdict(pd.DataFrame)

        with tqdm(total=self.n_splits, leave=True, desc="running split:") as t:
            for split, (train_idx, test_idx) in enumerate(
                    TimeSeriesSplit(self.n_splits, max_train_size=self.max_train_samples).split(X=data)
            ):
                # Update progress bar
                t.set_description(f"running split: {split}/{self.n_splits}", refresh=True)

                # Train / test split
                y_train, y_test = data[train_idx], data[test_idx]

                # Fit model
                forecaster.fit(y=np.squeeze(y_train))

                # Walk forward validation
                n_folds = int(len(y_test) - forecaster.horizon)
                predicted = np.zeros(shape=(n_folds, forecaster.horizon))
                actual = np.zeros(shape=(n_folds, forecaster.horizon))
                for i in range(n_folds):
                    predicted[i, :] = forecaster.predict()
                    actual[i, :] = np.squeeze(y_test[i: i + forecaster.horizon])
                    forecaster.update(y=np.squeeze(y_test)[i:i + 1])

                scores[split] = pd.DataFrame(
                    data={
                        f"{metric.__name__}": [metric(actual[:, h], predicted[:, h]) for h in range(forecaster.horizon)]
                        for metric in (r2_score, mean_absolute_percentage_error)
                    },
                    index=pd.Series(np.arange(1, forecaster.horizon + 1), name="horizon"),
                )

                # Iterate progress bar
                t.update()

        # Compute aggregate scores
        return pd.concat(scores).reset_index(level="horizon").groupby("horizon").agg(["mean", "sem"])

    def _validate_and_format_data(self, data: np.ndarray) -> np.ndarray:
        # TODO: add data validation and formatting
        return data