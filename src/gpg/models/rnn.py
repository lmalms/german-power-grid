from typing import Optional
import numpy as np
import tensorflow as tf

from gpg.preprocessing.time_series_transformer import TimeSeriesTransformer


class RNNForecaster:

    def __init__(
            self,
            n_layers: int,
            n_units: int,
            horizon: int = 10,
            n_lags: int = 50,
            activation: str = "tanh",
            learning_rate: float = 1e-03,
            apply_layer_normalization: bool = False
    ):
        self.n_layers = n_layers
        self.n_units = n_units
        self.horizon = horizon
        self.n_lags = n_lags
        self.activation = activation
        self.learning_rate = learning_rate
        self.apply_layer_normalization = apply_layer_normalization
        self._validate_model()
        self._model = self._build_model()
        self._history = None

    def _build_model(self) -> tf.keras.models.Model:
        # Build model with layer normalization
        if self.apply_layer_normalization:
            pass

        # Build model without layer normalization
        else:
            model = tf.keras.models.Sequential([
                tf.keras.layers.SimpleRNN(
                    self.n_units, activation=self.activation, input_shape=[None, 1], return_sequences=True
                ),
            ])
            for i in range(self.n_layers - 1):
                model.add(tf.keras.layers.SimpleRNN(self.n_units, activation=self.activation, return_sequences=True))

            model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.horizon)))

        model.compile(
            loss=tf.keras.losses.mean_squared_error,
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        return model

    def fit(
            self,
            y_train: np.ndarray,
            y_valid: Optional[np.ndarray] = None,
            epochs: int = 20,
    ) -> None:
        # Format time series
        transformer = TimeSeriesTransformer(n_lags=self.n_lags, horizon=self.horizon)
        X_train, y_train = transformer.fit_transform(y_train)
        X_valid, y_valid = transformer.transform(y_valid) if y_valid is not None else (None, None)

        # Assign model history
        self._history = X_train[-1:, :, :]

        # Fit model
        self._model.fit(
            X_train, y_train, epochs=epochs, validation_data=((X_valid, y_valid) if y_valid is not None else None)
        )

    def predict(self) -> np.ndarray:
        return self._model.predict(self._history)[:, -1, :].flatten()

    def update(self) -> None:
        pass  # TODO

    def _validate_model(self):
        pass
