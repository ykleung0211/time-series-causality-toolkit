"""Entry point for the time-series causality toolkit."""

from src.workflows import run_general_workflow


def main() -> None:
    """Launch the general workflow."""
    run_general_workflow()


if __name__ == "__main__":
    main()
