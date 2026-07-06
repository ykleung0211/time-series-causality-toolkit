"""Compatibility entry point for the toolkit examples."""

from cases_study.environmental_health_case import main as environmental_health_main
from cases_study.macro_regime_case import main as macro_regime_main
from cases_study.market_regime_case import main as market_regime_main
from cases_study.yfinance_case import main as finance_main


def main() -> None:
    """Prompt for which example workflow to run."""
    print("Choose an example case study:")
    print("1) Finance: Yahoo Finance data")
    print("2) Environmental/health: ozone concentration vs mortality")
    print("3) Market regime: SPY vs VIX")
    print("4) Macro regime: inflation vs unemployment")
    choice = input("Example [1/4]: ").strip() or "1"

    if choice == "2":
        environmental_health_main()
    elif choice == "3":
        market_regime_main()
    elif choice == "4":
        macro_regime_main()
    else:
        finance_main()


if __name__ == "__main__":
    main()
