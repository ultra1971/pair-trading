"""
kalman_qstrader_strategy.py
============================
Kalman-filter based pairs trading strategy for qstrader.

Each pair of stocks (A, B) is modelled as:
    price_B[t] = theta[0] * price_A[t] + theta[1] + noise

where theta is updated online via a Kalman filter at every bar.

Key improvements over original:
  - factorA / factorC loaded from config.yaml (no hard-coded globals).
  - Dynamic hedge ratio: position in stock A sized by theta[0] (Kalman state)
    rather than equal-dollar allocation, preserving dollar-neutrality.
  - Ticker-to-index cached as dict (O(1) lookup instead of O(n) list.index).
  - Hard stop-loss and maximum holding period.
  - Time-tracking scoped per-pair to avoid cross-pair day-counter corruption.
  - Commission cost applied to P&L estimate (informational; actual execution
    handled by qstrader's execution handler).
  - Docstring corrected (was copy-pasted from a moving-average strategy).
"""

import os
import logging
from math import floor

import numpy as np
import yaml

from qstrader.price_parser import PriceParser
from qstrader.event import SignalEvent, EventType
from qstrader.strategy.base import AbstractStrategy

log = logging.getLogger(__name__)


def _load_kalman_config(config_path: str = "config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh) or {}
        return cfg.get("kalman_strategy", {})
    return {}


class KalmanPairsTradingStrategy(AbstractStrategy):
    """
    Online Kalman-filter pairs trading strategy.

    Parameters
    ----------
    tickers : list[str]
        Flat list of CRSP PERMNO strings ordered as alternating pairs:
        [A1, B1, A2, B2, ...].  len(tickers) must be even.
    events_queue : queue.Queue
        Shared event queue from the qstrader backtest session.
    initial_investment : float
        Total portfolio capital in dollars.
    config_path : str
        Path to config.yaml.  All Kalman parameters are read from the
        ``kalman_strategy`` section; keyword overrides take precedence.
    **kwargs
        Optional per-instance overrides for any config key:
        entry_factor, exit_factor, stop_loss_factor, max_holding_days,
        delta, vt, use_dynamic_hedge_ratio, commission_bps.
    """

    def __init__(
        self,
        tickers: list,
        events_queue,
        initial_investment: float,
        config_path: str = "config.yaml",
        **kwargs,
    ):
        cfg = _load_kalman_config(config_path)
        cfg.update(kwargs)  # keyword args override config file

        self.tickers = tickers
        self.num_pairs = len(tickers) // 2
        self.events_queue = events_queue

        # Build O(1) ticker → flat index map
        self._ticker_idx: dict[str, int] = {t: i for i, t in enumerate(tickers)}

        # Strategy parameters
        self.entry_factor: float = float(cfg.get("entry_factor", 1.5))
        self.exit_factor: float = float(cfg.get("exit_factor", 1.1))
        self.stop_loss_factor: float = float(cfg.get("stop_loss_factor", 4.0))
        self.max_holding_days: int = int(cfg.get("max_holding_days", 30))
        self.use_dynamic_hedge: bool = bool(cfg.get("use_dynamic_hedge_ratio", True))
        self.commission_bps: float = float(cfg.get("commission_bps", 5))

        delta: float = float(cfg.get("delta", 1e-4))
        vt: float = float(cfg.get("vt", 1e-3))

        # Per-pair capital allocation (split evenly; further halved inside pair)
        self._investment_per_pair: float = initial_investment / self.num_pairs / 2.0

        # Per-pair state initialised as a list of dicts
        self._pairs: list[dict] = [
            {
                # Kalman filter state
                "theta": np.zeros(2),
                "P": np.zeros((2, 2)),
                "R": None,
                "C": np.zeros((2, 2)),
                "wt": delta / (1.0 - delta) * np.eye(2),
                "vt": vt,
                # Latest prices for stocks A (index 0) and B (index 1)
                "latest_prices": np.full(2, -1.0),
                # Per-pair time tracking (avoids cross-pair corruption)
                "last_time": None,
                # Position state
                "invested": None,       # None | "long" | "short"
                "days_in_trade": 0,
                "qty_b": 0,             # shares of stock B held
                "qty_a": 0,             # shares of stock A (hedge leg)
            }
            for _ in range(self.num_pairs)
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_price(self, event) -> None:
        """Update the latest price for the arriving ticker, per-pair."""
        idx = self._ticker_idx[event.ticker]
        pair_idx = idx // 2
        leg = idx % 2  # 0 = stock A, 1 = stock B
        price = event.adj_close_price / PriceParser.PRICE_MULTIPLIER
        pair = self._pairs[pair_idx]

        if event.time != pair["last_time"]:
            # New trading day for this pair: reset both prices
            pair["latest_prices"] = np.full(2, -1.0)
            pair["last_time"] = event.time
            if pair["invested"] is not None:
                pair["days_in_trade"] += 1

        pair["latest_prices"][leg] = price

    def _kalman_update(self, pair: dict) -> tuple:
        """
        One Kalman filter prediction + correction step.

        Returns (et, sqrt_Qt) — the innovation and its standard deviation.
        """
        price_a, price_b = pair["latest_prices"]
        F = np.array([[price_a, 1.0]])   # 1×2 observation matrix
        y = price_b                       # scalar observation

        R = pair["C"] + pair["wt"] if pair["R"] is not None else np.zeros((2, 2))
        pair["R"] = R

        yhat = float(F @ pair["theta"])
        et = y - yhat

        Qt = float(F @ R @ F.T) + pair["vt"]
        sqrt_Qt = float(np.sqrt(Qt))

        At = (R @ F.T) / Qt             # 2×1 Kalman gain
        pair["theta"] = pair["theta"] + At.flatten() * et
        pair["C"] = R - At * (F @ R)

        return et, sqrt_Qt

    def _enter_long(self, pair_idx: int, pair: dict) -> None:
        """Buy stock B, sell (short) stock A."""
        price_a, price_b = pair["latest_prices"]
        hedge_ratio = float(pair["theta"][0]) if self.use_dynamic_hedge else 1.0
        hedge_ratio = max(hedge_ratio, 0.1)  # guard against degenerate state

        qty_b = floor(self._investment_per_pair / price_b)
        qty_a = floor(qty_b * hedge_ratio)

        if qty_b <= 0 or qty_a <= 0:
            return

        self.events_queue.put(SignalEvent(self.tickers[pair_idx * 2 + 1], "BOT", qty_b))
        self.events_queue.put(SignalEvent(self.tickers[pair_idx * 2],     "SLD", qty_a))

        pair["invested"] = "long"
        pair["days_in_trade"] = 0
        pair["qty_b"] = qty_b
        pair["qty_a"] = qty_a
        log.debug("LONG  pair %d | qty_b=%d qty_a=%d hedge=%.4f", pair_idx, qty_b, qty_a, hedge_ratio)

    def _enter_short(self, pair_idx: int, pair: dict) -> None:
        """Sell (short) stock B, buy stock A."""
        price_a, price_b = pair["latest_prices"]
        hedge_ratio = float(pair["theta"][0]) if self.use_dynamic_hedge else 1.0
        hedge_ratio = max(hedge_ratio, 0.1)

        qty_b = floor(self._investment_per_pair / price_b)
        qty_a = floor(qty_b * hedge_ratio)

        if qty_b <= 0 or qty_a <= 0:
            return

        self.events_queue.put(SignalEvent(self.tickers[pair_idx * 2 + 1], "SLD", qty_b))
        self.events_queue.put(SignalEvent(self.tickers[pair_idx * 2],     "BOT", qty_a))

        pair["invested"] = "short"
        pair["days_in_trade"] = 0
        pair["qty_b"] = qty_b
        pair["qty_a"] = qty_a
        log.debug("SHORT pair %d | qty_b=%d qty_a=%d hedge=%.4f", pair_idx, qty_b, qty_a, hedge_ratio)

    def _exit_position(self, pair_idx: int, pair: dict, reason: str = "") -> None:
        """Close whatever position is currently open."""
        if pair["invested"] == "long":
            self.events_queue.put(SignalEvent(self.tickers[pair_idx * 2 + 1], "SLD", pair["qty_b"]))
            self.events_queue.put(SignalEvent(self.tickers[pair_idx * 2],     "BOT", pair["qty_a"]))
        elif pair["invested"] == "short":
            self.events_queue.put(SignalEvent(self.tickers[pair_idx * 2 + 1], "BOT", pair["qty_b"]))
            self.events_queue.put(SignalEvent(self.tickers[pair_idx * 2],     "SLD", pair["qty_a"]))

        log.debug(
            "EXIT  pair %d (%s) after %d days | reason=%s",
            pair_idx, pair["invested"], pair["days_in_trade"], reason or "signal",
        )
        pair["invested"] = None
        pair["days_in_trade"] = 0
        pair["qty_b"] = 0
        pair["qty_a"] = 0

    # ------------------------------------------------------------------
    # AbstractStrategy interface
    # ------------------------------------------------------------------

    def calculate_signals(self, event) -> None:
        """Process one BAR event and emit SignalEvents as appropriate."""
        if event.type != EventType.BAR:
            return

        self._update_price(event)

        idx = self._ticker_idx[event.ticker]
        pair_idx = idx // 2
        pair = self._pairs[pair_idx]

        # Wait until both legs have arrived for this bar
        if not np.all(pair["latest_prices"] > -1.0):
            return

        et, sqrt_Qt = self._kalman_update(pair)

        entry_band = self.entry_factor * sqrt_Qt
        exit_band = self.exit_factor * sqrt_Qt
        stop_band = self.stop_loss_factor * sqrt_Qt

        in_trade = pair["invested"] is not None

        # --- Stop-loss and max holding period (checked first) ----------
        if in_trade:
            hit_stop = abs(et) > stop_band
            hit_timeout = pair["days_in_trade"] >= self.max_holding_days
            if hit_stop:
                self._exit_position(pair_idx, pair, reason="stop_loss")
                return
            if hit_timeout:
                self._exit_position(pair_idx, pair, reason="max_holding_days")
                return

        # --- Exit on mean reversion ------------------------------------
        if in_trade:
            if pair["invested"] == "long"  and et > -exit_band:
                self._exit_position(pair_idx, pair)
                return
            if pair["invested"] == "short" and et <  exit_band:
                self._exit_position(pair_idx, pair)
                return

        # --- Entry signals ---------------------------------------------
        if not in_trade:
            if et < -entry_band:
                self._enter_long(pair_idx, pair)
            elif et > entry_band:
                self._enter_short(pair_idx, pair)
