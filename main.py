"""Entry point for the time-series causality toolkit."""

from __future__ import annotations

from src.workflows import run_general_workflow


def run_full_analysis_cli() -> None:
    """Launch the full interactive workflow from the command line."""
    run_general_workflow()


def main() -> None:
    """Main entry point for the time-series causality toolkit CLI."""
    # It is split into a separate function so that it can be called from other entry points, such as a GUI or a web app.
    run_full_analysis_cli()


if __name__ == "__main__":
    main()
