from typing import Callable, Optional

from copy import deepcopy
import numpy as np
from scipy.optimize import minimize


class ExponentialSmoothingForecaster:

    def __init__(
            self,
            has_trend: bool = True,
            trend_is_damped: bool = False,
            has_season: bool = True,
            seasonality_is_additive: bool = True,
            apply_autocorrection: bool = True,
            period: int = 1,
            horizon: int = 1,
            tol: Optional[float] = None,
            method: Optional[str] = None,
            max_iter: Optional[int] = None
    ) -> None:
        self.has_trend = has_trend
        self.trend_is_damped = trend_is_damped
        self.has_season = has_season
        self.seasonality_is_additive = seasonality_is_additive
        self.apply_autocorrection = apply_autocorrection
        self.period = period
        self.horizon = horizon
        self.tol = tol if tol is not None else 1e-06
        self.method = method if method is not None else "Nelder-Mead"
        self.max_iter = max_iter
        self._alpha = None  # level coefficient
        self._beta = None  # trend coefficient
        self._gamma = None  # season coefficient
        self._phi = None  # trend damping coefficient
        self._lambda = None  # AR correction coefficient
        self._current = None
        self._level = None
        self._prev_level = None
        self._trend = None
        self._season = None
        # self._validate_model()  # TODO

    def fit(self, y: np.ndarray) -> None:
        # self._validate_fit_data  # TODO
        self._optimize_params(y)

    def predict(self, compute_confidence_intervals: bool = False) -> np.ndarray:  # TODO: add confidence intervals
        h = np.arange(1, self.horizon + 1)
        y_hat = self._level * np.ones(self.horizon)
        if self.has_trend:
            if self.trend_is_damped:
                y_hat += np.cumsum(self._phi ** h) * self._trend
            else:
                y_hat += h * self._trend

        if self.has_season:
            if self.seasonality_is_additive:
                y_hat = y_hat + np.array(self._season)[(h - 1) % self.period]
            else:
                y_hat *= np.array(self._season)[(h - 1) % self.period]

        if self.apply_autocorrection:
            y_hat += (self._lambda ** h) + (self._current - (self._prev_level + self._season[0]))

        return y_hat

    def update(self, y: np.ndarray) -> None:
        # self._validate_update_data  # TODO
        for value in y:
            self._step(y=value)

    def _step(self, y: float) -> None:
        previous_level = deepcopy(self._level)
        previous_trend = deepcopy(self._trend)

        # Update level
        self._step_level(y)

        # Update trend
        if self.has_trend:
            self._step_trend(previous_level)

        # Update season
        if self.has_season:
            self._step_season(y, previous_level, previous_trend)

    def _step_level(self, y: float) -> None:

        # Seasonal and trend terms
        if self.has_trend and self.has_season:

            if self.trend_is_damped and self.seasonality_is_additive:
                self._level = self._alpha * (y - self._season[0]) \
                              + (1 - self._alpha) * (self._level + self._phi * self._trend)

            elif self.trend_is_damped:
                self._level = self._alpha * (y / self._season[0]) \
                              + (1 - self._alpha) * (self._level + self._phi * self._trend)

            elif self.seasonality_is_additive:
                self._level = self._alpha * (y - self._season[0]) + (1 - self._alpha) * (self._level + self._trend)

            else:
                self._level = self._alpha * (y / self._season[0]) \
                              + (1 - self._alpha) * (self._level + self._phi * self._trend)

        # Trend but no seasonal terms
        elif self.has_trend:
            if self.trend_is_damped:
                self._level = self._alpha * y + (1 - self._alpha) * (self._level + self._phi * self._trend)

            else:
                self._level = self._alpha * y + (1 - self._alpha) * (self._level + self._trend)

        # Seasonal but no trend terms
        elif self.has_season:
            if self.seasonality_is_additive:
                self._level = self._alpha * (y - self._season[0]) + (1 - self._alpha) * self._level

            else:
                self._level = self._alpha * (y / self._season[0]) + (1 - self._alpha) * self._level

        # No trend and no seasonal terms
        else:
            self._level = self._alpha * y + (1 - self._alpha) * self._level

    def _step_trend(self, level: float) -> None:

        if self.trend_is_damped:
            self._trend = self._beta * (self._level - level) + (1 - self._beta) * self._phi * self._trend

        else:
            self._trend = self._beta * (self._level - level) + (1 - self._beta) * self._trend

    def _step_season(self, y: float, level: float, trend: float) -> None:
        # Seasonal and trend terms
        if self.has_trend:
            if self.trend_is_damped and self.seasonality_is_additive:
                self._season.append(
                    self._gamma * (y - level - self._phi * trend) + (1 - self._gamma) * self._season.pop(0)
                )
            elif self.trend_is_damped:  # and seasonality_is_multiplicative
                self._season.append(
                    self._gamma * (y / (level + self._phi * trend)) + (1 - self._gamma) * self._season.pop(0)
                )
            elif self.seasonality_is_additive:  # and no damping
                self._season.append(
                    self._gamma * (y - level - trend) + (1 - self._gamma) * self._season.pop(0)
                )
            else:  # multiplicative and no damping
                self._season.append(
                    self._gamma * (y / (level + trend)) + (1 - self._gamma) * self._season.pop(0)
                )
        # Only seasonal terms
        else:
            if self.seasonality_is_additive:
                self._season.append(self._gamma * (y - level) + (1 - self._gamma) * self._season.pop(0))
            else:
                self._season.append(self._gamma * (y / level) + (1 - self._gamma) * self._season.pop(0))

    def _initialize(self, y: np.ndarray) -> None:

        assert len(y) > self.period, f"Need at least {2 * self.period} data points to initialise model."

        # Level
        self._current = y[2 * self.period]
        self._level = np.mean(y[self.period: 2 * self.period])
        self._prev_level = np.mean(y[self.period - 1: 2 * self.period - 1])

        # Trend
        if self.has_trend:
            self._trend = (np.sum(y[self.period: 2 * self.period]) - np.sum(y[:self.period])) / self.period ** 2

        # Season
        if self.has_season:
            if self.seasonality_is_additive:
                self._season = (y[:self.period] - self._level).tolist()
            else:
                self._season = (y[:self.period] / self._level).tolist()

    def _optimize_params(self, y: np.ndarray):
        # Initialise terms (level, trend, season)
        self._initialize(y)

        # Infer number of parameters

        n_params = (
            1  # level
            + (1 if self.has_trend else 0)
            + (1 if self.trend_is_damped else 0)
            + (1 if self.has_season else 0)
            + (1 if self.apply_autocorrection else 0)
        )

        # Optimize parameters
        res = minimize(
            self._make_objective(y),
            x0=np.random.uniform(size=(n_params,)),
            method=self.method,
            tol=self.tol,
            options=dict(maxiter=self.max_iter) if self.max_iter is not None else None
        )

        # Assign optimal parameters to self._alpha, self._beta, etc.
        self._unpack_params(res.x)

    def _make_objective(self, y: np.ndarray) -> Callable[[np.ndarray], float]:

        def objective(params: np.ndarray):
            self._unpack_params(params)
            n_splits = int(len(y) - 2 * self.period - self.horizon)
            errors = np.zeros(shape=(n_splits, self.horizon))
            for n in range(n_splits):
                start, stop = n + 2*self.period, n + 2*self.period + self.horizon
                errors[n] = y[start:stop] - self.predict()
                self._step(y=float(y[start]))

            return np.mean(errors**2)

        return objective

    def _unpack_params(self, params: np.ndarray) -> None:
        params = np.clip(params, a_min=0., a_max=1.)
        if self.has_trend and self.has_season and self.apply_autocorrection:
            if self.trend_is_damped:
                self._alpha, self._beta, self._phi, self._gamma, self._lambda = params
            else:
                self._alpha, self._beta, self._gamma, self._lambda = params
        elif self.has_trend and self.has_season:
            if self.trend_is_damped:
                self._alpha, self._beta, self._phi, self._gamma = params
            else:
                self._alpha, self._beta, self._gamma = params
        elif self.has_trend and self.apply_autocorrection:
            if self.trend_is_damped:
                self._alpha, self._beta, self._phi, self._lambda = params
            else:
                self._alpha, self._beta, self._lambda = params
        elif self.has_season and self.apply_autocorrection:
            self._alpha, self._gamma, self._lambda = params
        elif self.has_trend:
            if self.trend_is_damped:
                self._alpha, self._beta, self._phi = params
            else:
                self._alpha, self._beta = params
        elif self.has_season:
            self._alpha, self._gamma = params
        elif self.apply_autocorrection:
            self._alpha, self._lambda = params
        else:
            self._alpha = params[0]
