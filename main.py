"""Entry point for the time-series causality toolkit."""

from __future__ import annotations

from src.workflows import run_general_workflow


def run_full_analysis_cli() -> None:
    """Launch the full interactive workflow from the command line."""
    run_general_workflow()


def main() -> None:
    """Backward-compatible script entry point."""
    run_full_analysis_cli()


if __name__ == "__main__":
    main()
