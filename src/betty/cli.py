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


@main.group()
def config():
    """View and manage Betty configuration."""


@config.command("show")
def config_show():
    """Display current configuration."""
    from betty.config import load_config

    cfg = load_config()
    click.echo(f"[llm]")
    click.echo(f"  model = {cfg.llm.model}")
    if cfg.llm.api_base:
        click.echo(f"  api_base = {cfg.llm.api_base}")
    click.echo(f"  api_key = {'***' if cfg.llm.api_key else '(not set)'}")
    click.echo()
    click.echo(f"[delegation]")
    click.echo(f"  autonomy_level = {cfg.delegation.autonomy_level}")
    click.echo(f"  auto_approve_read_tools = {cfg.delegation.auto_approve_read_tools}")
    click.echo(f"  confidence_threshold = {cfg.delegation.confidence_threshold}")
    click.echo()
    click.echo(f"[escalation]")
    click.echo(f"  escalation_mode = {cfg.escalation.escalation_mode}")
    click.echo(f"  telegram_token = {'***' if cfg.escalation.telegram_token else '(not set)'}")
    click.echo(f"  telegram_chat_id = {cfg.escalation.telegram_chat_id or '(not set)'}")


@config.command("llm-preset")
@click.argument("preset_name", required=False)
def config_llm_preset(preset_name):
    """Apply a predefined LLM configuration preset.

    Without arguments, lists available presets.
    """
    from betty.config import get_llm_presets, load_config, save_config

    presets = get_llm_presets()

    if not preset_name:
        click.echo("Available LLM presets:")
        for name, preset in presets.items():
            click.echo(f"  {name:20s} {preset['description']}")
        click.echo()
        click.echo("Usage: betty config llm-preset <name>")
        return

    if preset_name not in presets:
        click.echo(f"Unknown preset: {preset_name}", err=True)
        click.echo(f"Available: {', '.join(presets.keys())}", err=True)
        raise SystemExit(1)

    preset = presets[preset_name]
    cfg = load_config()
    cfg.llm.model = preset["model"]
    cfg.llm.api_base = preset.get("api_base")
    save_config(cfg)
    click.echo(f"LLM preset '{preset_name}' applied: model={preset['model']}")


if __name__ == "__main__":
    main()
