"""
kalman_qstrader_backtest.py
============================
Runs the Kalman-filter pairs trading backtest via qstrader.

Usage
-----
    python kalman_qstrader_backtest.py
    python kalman_qstrader_backtest.py --config config.yaml
    python kalman_qstrader_backtest.py --testing
"""

import logging
import os

import click
import yaml

from qstrader import settings
from qstrader.compat import queue
from qstrader.price_parser import PriceParser
from qstrader.price_handler.yahoo_daily_csv_bar import YahooDailyCsvBarPriceHandler
from qstrader.strategy import Strategies, DisplayStrategy
from qstrader.position_sizer.naive import NaivePositionSizer
from qstrader.risk_manager.example import ExampleRiskManager
from qstrader.portfolio_handler import PortfolioHandler
from qstrader.compliance.example import ExampleCompliance
from qstrader.execution_handler.ib_simulated import IBSimulatedExecutionHandler
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading_session.backtest import Backtest

from kalman_qstrader_strategy import KalmanPairsTradingStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        with open(config_path) as fh:
            return yaml.safe_load(fh) or {}
    return {}


def build_ticker_list(cfg: dict) -> list[str]:
    """Flatten active_pairs from config into alternating [A, B, A, B, ...] list."""
    pairs = cfg.get("pair_selection", {}).get("active_pairs", None)
    if pairs:
        return [permno for pair in pairs for permno in pair]
    # Fallback to the six pairs used in the original study
    return [
        "43350", "82651",
        "44644", "90458",
        "24969", "24985",
        "42585", "83621",
        "60186", "81095",
        "16548", "81577",
    ]


def run(config, testing: bool, tickers: list[str], output_csv: str, config_path: str) -> dict:
    events_queue = queue.Queue()
    csv_dir = config.CSV_DATA_DIR

    cfg = load_config(config_path)
    initial_capital = float(
        cfg.get("kalman_strategy", {}).get("initial_capital", 1_000_000)
    )
    initial_equity = PriceParser.parse(initial_capital)

    price_handler = YahooDailyCsvBarPriceHandler(csv_dir, events_queue, tickers)

    strategy = KalmanPairsTradingStrategy(
        tickers, events_queue, initial_capital, config_path=config_path
    )
    strategy = Strategies(strategy, DisplayStrategy())

    position_sizer = NaivePositionSizer()
    risk_manager = ExampleRiskManager()

    portfolio_handler = PortfolioHandler(
        initial_equity, events_queue, price_handler,
        position_sizer, risk_manager,
    )

    compliance = ExampleCompliance(config)

    execution_handler = IBSimulatedExecutionHandler(
        events_queue, price_handler, compliance
    )

    statistics = TearsheetStatistics(config, portfolio_handler, title="Kalman Pairs Strategy")

    backtest = Backtest(
        price_handler, strategy,
        portfolio_handler, execution_handler,
        position_sizer, risk_manager,
        statistics, initial_equity,
    )

    log.info("Starting backtest with %d tickers (%d pairs).", len(tickers), len(tickers) // 2)
    results = backtest.simulate_trading(testing=testing)

    cum_returns = results["cum_returns"]
    cum_returns.to_csv(output_csv, header=["cum_returns"])
    log.info("Cumulative returns written to %s", output_csv)

    statistics.save("output")
    log.info("Tearsheet saved to output/")

    return results


@click.command()
@click.option(
    "--config-path",
    default="config.yaml",
    show_default=True,
    help="Path to config.yaml.",
)
@click.option(
    "--qstrader-config",
    default=settings.DEFAULT_CONFIG_FILENAME,
    show_default=True,
    help="qstrader settings file.",
)
@click.option(
    "--testing/--no-testing",
    default=False,
    show_default=True,
    help="Enable qstrader testing mode (smaller dataset).",
)
@click.option(
    "--output-csv",
    default="6pair_kalman.csv",
    show_default=True,
    help="Filename for cumulative returns CSV output.",
)
def main(config_path: str, qstrader_config: str, testing: bool, output_csv: str) -> None:
    cfg = load_config(config_path)
    tickers = build_ticker_list(cfg)
    log.info("Tickers: %s", tickers)

    qs_config = settings.from_file(qstrader_config, testing)
    run(qs_config, testing, tickers, output_csv, config_path)


if __name__ == "__main__":
    main()
