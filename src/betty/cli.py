"""Betty CLI — entry point for all betty commands."""

import click

from betty import __version__


@click.group()
@click.version_option(version=__version__, prog_name="betty")
def main():
    """Betty - a peer programming agent for Claude Code."""


@main.command()
def status():
    """Show Betty daemon status."""
    click.echo("Betty is not running.")


if __name__ == "__main__":
    main()
