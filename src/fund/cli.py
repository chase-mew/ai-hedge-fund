"""Command line entry point. Commands arrive in Phase 3."""

import typer

app = typer.Typer(help="An autonomous multi asset AI fund.")


@app.callback()
def main() -> None:
    """An autonomous multi asset AI fund. Claude proposes, code disposes."""


if __name__ == "__main__":
    app()
