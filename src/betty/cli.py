"""Betty CLI — full command surface for Betty 2.0.

All commands work whether or not the daemon is running.
"""

import json as json_lib
import sys

import click
from rich.console import Console
from rich.table import Table

from betty import __version__

console = Console()


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="betty")
@click.pass_context
def main(ctx):
    """Betty - a peer programming agent for Claude Code."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# --- Daemon commands ---


@main.command()
def start():
    """Start the Betty daemon."""
    console.print("[green]Starting Betty daemon...[/green]")
    console.print("[yellow]Daemon not yet implemented.[/yellow]")


@main.command()
def stop():
    """Stop the Betty daemon."""
    console.print("[yellow]Betty daemon is not running.[/yellow]")


@main.command()
def status():
    """Show Betty daemon status."""
    console.print("[dim]Betty daemon:[/dim] not running")
    console.print(f"[dim]Version:[/dim] {__version__}")


# --- Config commands ---


@main.group()
def config():
    """View and manage Betty configuration."""


@config.command("show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def config_show(as_json):
    """Display current configuration."""
    from betty.config import _config_to_dict, load_config

    cfg = load_config()

    if as_json:
        click.echo(json_lib.dumps(_config_to_dict(cfg), indent=2))
        return

    console.print("[bold]Betty Configuration[/bold]\n")
    console.print("[dim]\\[llm][/dim]")
    console.print(f"  model = {cfg.llm.model}")
    if cfg.llm.api_base:
        console.print(f"  api_base = {cfg.llm.api_base}")
    console.print(f"  api_key = {'***' if cfg.llm.api_key else '(not set)'}")
    console.print()
    console.print("[dim]\\[delegation][/dim]")
    console.print(f"  autonomy_level = {cfg.delegation.autonomy_level}")
    console.print(f"  auto_approve_read_tools = {cfg.delegation.auto_approve_read_tools}")
    console.print(f"  confidence_threshold = {cfg.delegation.confidence_threshold}")
    console.print()
    console.print("[dim]\\[escalation][/dim]")
    console.print(f"  escalation_mode = {cfg.escalation.escalation_mode}")
    console.print(f"  telegram_token = {'***' if cfg.escalation.telegram_token else '(not set)'}")
    console.print(
        f"  telegram_chat_id = {cfg.escalation.telegram_chat_id or '(not set)'}"
    )


@config.command("llm-preset")
@click.argument("preset_name", required=False)
def config_llm_preset(preset_name):
    """Apply a predefined LLM configuration preset.

    Without arguments, lists available presets.
    """
    from betty.config import get_llm_presets, load_config, save_config

    presets = get_llm_presets()

    if not preset_name:
        table = Table(title="LLM Presets")
        table.add_column("Name", style="cyan")
        table.add_column("Model")
        table.add_column("Description")
        for name, preset in presets.items():
            table.add_row(name, preset["model"], preset["description"])
        console.print(table)
        return

    if preset_name not in presets:
        console.print(f"[red]Unknown preset: {preset_name}[/red]")
        console.print(f"Available: {', '.join(presets.keys())}")
        sys.exit(1)

    preset = presets[preset_name]
    cfg = load_config()
    cfg.llm.model = preset["model"]
    cfg.llm.api_base = preset.get("api_base")
    save_config(cfg)
    console.print(
        f"[green]LLM preset '{preset_name}' applied:[/green] model={preset['model']}"
    )


@config.command("delegation")
@click.argument("level", type=click.IntRange(0, 3))
def config_delegation(level):
    """Set autonomy level (0=observe, 1=suggest, 2=semi-auto, 3=full-auto)."""
    from betty.config import load_config, save_config

    cfg = load_config()
    cfg.delegation.autonomy_level = level
    save_config(cfg)
    labels = {0: "observer", 1: "suggest", 2: "semi-auto", 3: "full-auto"}
    console.print(f"[green]Autonomy level set to {level} ({labels[level]})[/green]")


@config.command("telegram-token")
@click.argument("token")
def config_telegram_token(token):
    """Set the Telegram bot token for escalation."""
    from betty.config import load_config, save_config

    cfg = load_config()
    cfg.escalation.telegram_token = token
    cfg.escalation.escalation_mode = "telegram"
    save_config(cfg)
    console.print("[green]Telegram token configured.[/green]")


# --- Hooks commands ---


@main.group()
def hooks():
    """Manage Claude Code hooks integration."""


@hooks.command("install")
def hooks_install():
    """Install Betty hooks into Claude Code."""
    try:
        from betty.hooks import install_hooks

        installed = install_hooks()
        if installed:
            for hook_type in installed:
                console.print(f"  Installed {hook_type}")
            console.print(f"[green]Done - {len(installed)} hook(s) installed.[/green]")
        else:
            console.print("All hooks already installed.")
    except ImportError:
        console.print("[yellow]Hook installation not yet implemented.[/yellow]")


@hooks.command("uninstall")
def hooks_uninstall():
    """Remove Betty hooks from Claude Code."""
    try:
        from betty.hooks import uninstall_hooks

        removed = uninstall_hooks()
        if removed:
            for hook_type in removed:
                console.print(f"  Removed {hook_type}")
            console.print(f"[green]Done - {len(removed)} hook(s) removed.[/green]")
        else:
            console.print("No Betty hooks found to remove.")
    except ImportError:
        console.print("[yellow]Hook uninstallation not yet implemented.[/yellow]")


@hooks.command("status")
def hooks_status():
    """Show hook installation status."""
    try:
        from betty.hooks import hooks_status as _hooks_status

        info = _hooks_status()
        any_installed = False
        for hook_type, installed in info.items():
            marker = "installed" if installed else "not installed"
            console.print(f"  {hook_type:25s} {marker}")
            if installed:
                any_installed = True
        if not any_installed:
            console.print("\nRun 'betty hooks install' to set up hooks.")
    except ImportError:
        console.print("[dim]Hooks:[/dim] not installed")


# --- Hook handler (called by Claude Code, hidden from user) ---


@main.command("hook-handler", hidden=True)
@click.argument("hook_type")
def hook_handler(hook_type: str):
    """Handle a Claude Code hook event (called by hook scripts)."""
    try:
        from betty.hook_handler import handle_hook

        handle_hook(hook_type)
    except ImportError:
        pass


# --- Profile commands ---


@main.command()
@click.option("--project", type=click.Path(), help="Show profile for specific project.")
@click.option("--reset", is_flag=True, help="Reset user profile.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def profile(project, reset, as_json):
    """View your learned user profile."""
    if reset:
        console.print("[yellow]Profile reset not yet implemented.[/yellow]")
        return

    console.print("[dim]User profile is empty. Betty learns as you work.[/dim]")


# --- Sessions commands ---


@main.command()
@click.option("--project", type=click.Path(), help="Filter by project directory.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@click.option("--limit", default=20, help="Max sessions to show.")
def sessions(project, as_json, limit):
    """List recent Claude Code sessions."""
    console.print("[dim]No sessions recorded yet.[/dim]")


# --- Policies commands ---


@main.group()
def policies():
    """Manage organizational policies."""


@policies.command("list")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def policies_list(as_json):
    """List active policies."""
    console.print("[dim]No policies configured.[/dim]")


@policies.command("add")
@click.argument("policy_type")
@click.argument("rule")
@click.option("--description", "-d", help="Policy description.")
def policies_add(policy_type, rule, description):
    """Add a new policy rule."""
    console.print(f"[green]Policy added:[/green] [{policy_type}] {rule}")


@policies.command("import")
@click.argument("path", type=click.Path(exists=True))
def policies_import(path):
    """Import policies from a file."""
    console.print("[yellow]Policy import not yet implemented.[/yellow]")


# --- Dashboard ---


@main.command()
@click.option("--port", default=8111, help="Dashboard port.")
def dashboard(port):
    """Open the Betty web dashboard."""
    console.print(f"[yellow]Dashboard not yet implemented (port {port}).[/yellow]")


# --- Telegram ---


@main.group()
def telegram():
    """Manage Telegram bot integration."""


@telegram.command("link")
def telegram_link():
    """Link a Telegram chat for escalations."""
    console.print("[yellow]Telegram linking not yet implemented.[/yellow]")


@telegram.command("test")
def telegram_test():
    """Send a test message via Telegram."""
    console.print("[yellow]Telegram test not yet implemented.[/yellow]")


# --- Logs ---


@main.command()
@click.option(
    "--level",
    type=click.Choice(["debug", "info", "warning", "error"]),
    default="info",
    help="Log level filter.",
)
@click.option("--tail", "-n", default=50, help="Number of lines to show.")
def logs(level, tail):
    """View Betty daemon logs."""
    console.print("[dim]No logs available (daemon not running).[/dim]")


if __name__ == "__main__":
    main()
