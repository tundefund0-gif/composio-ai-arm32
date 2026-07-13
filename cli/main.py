#!/usr/bin/env python3
"""Zen Agent CLI — rich chat interface with 1,000+ tools."""
from __future__ import annotations

import logging
import shutil
import sys
import threading
from typing import Optional

import typer

from core.agent import ZenAgent
from core.llm_client import LLMResponse

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("zen-agent")

app = typer.Typer(name="zen", help="Zen Agent — AI assistant with 1,000+ Composio tools", add_completion=False, no_args_is_help=True)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context, user: str = typer.Option("cli-user", "--user", "-u", help="User ID"),
         session: Optional[str] = typer.Option(None, "--session", "-s", help="Existing session ID"),
         oneshot: Optional[str] = typer.Option(None, "--oneshot", "-1", help="Single question"),
         no_sandbox: bool = typer.Option(False, "--no-sandbox"), verbose: bool = typer.Option(False, "--verbose", "-v")):
    if verbose: logging.getLogger("zen-agent").setLevel(logging.INFO)
    if ctx.invoked_subcommand is not None: return
    try:
        agent = ZenAgent(user_id=user, session_id=session, enable_sandbox=not no_sandbox)
    except Exception as e:
        typer.secho(f"❌ Failed to init: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    cols, _ = shutil.get_terminal_size()
    typer.secho("┌" + "─" * (cols - 2) + "┐", fg=typer.colors.BRIGHT_BLACK)
    typer.secho(f"│ 🤖  Zen Agent  │  User: {user}  │  Session: {agent.session_id[:16]}…", fg=typer.colors.CYAN, bold=True)
    typer.secho("│ Commands: /quit /clear /info", fg=typer.colors.BRIGHT_BLACK)
    typer.secho("└" + "─" * (cols - 2) + "┘", fg=typer.colors.BRIGHT_BLACK)
    print()

    if oneshot:
        _handle_oneshot(agent, oneshot)
        return
    _loop(agent)


def _handle_oneshot(agent: ZenAgent, q: str):
    try:
        resp = agent.chat(q)
        if isinstance(resp, LLMResponse):
            print(resp.content)
    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)


def _loop(agent: ZenAgent):
    while True:
        try:
            inp = typer.prompt("You", prompt_suffix=" > ", err=True)
        except typer.Abort:
            print()
            break
        if not inp.strip(): continue
        c = inp.strip().lower()
        if c in ("/quit", "/exit", "/q"): typer.secho("👋 Goodbye!", fg=typer.colors.GREEN); break
        if c in ("/clear", "/reset"): agent.clear_history(); typer.secho("🧹 Cleared.", fg=typer.colors.YELLOW); continue
        if c in ("/info", "/session"):
            for k, v in agent.get_info().items(): typer.secho(f"  {k}: {v}", fg=typer.colors.BRIGHT_BLACK)
            continue
        if c.startswith("/"): typer.secho(f" Unknown: {c}", fg=typer.colors.RED); continue

        try:
            print()
            resp = agent.chat(inp)
            if isinstance(resp, LLMResponse):
                print(resp.content)
                if resp.reasoning:
                    print()
                    typer.secho("── 🤔 Reasoning ──", fg=typer.colors.BRIGHT_BLACK)
                    print(resp.reasoning[:500])
            print()
        except Exception as e:
            typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)


@app.command()
def tools(query: str = typer.Argument(..., help="Search query"), user: str = typer.Option("cli-user", "--user", "-u")):
    """Search Composio tools."""
    agent = ZenAgent(user_id=user, enable_sandbox=False)
    try:
        r = agent._composio.search_tools(agent.session_id, query)
        schemas = r.get("data", {}).get("tool_schemas", {})
        if not schemas: typer.secho("No tools found.", fg=typer.colors.YELLOW); return
        typer.secho(f"\nFound {len(schemas)} tool(s):\n", fg=typer.colors.CYAN, bold=True)
        for slug, info in list(schemas.items())[:20]:
            tk = info.get("toolkit", "")
            desc = info.get("description", "")[:100]
            typer.secho(f"  • {slug}", fg=typer.colors.GREEN)
            if tk: typer.secho(f"    [{tk}]", fg=typer.colors.BRIGHT_MAGENTA)
            if desc: typer.secho(f"    {desc}", fg=typer.colors.BRIGHT_BLACK)
        print()
    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)


@app.command()
def session(user: str = "cli-user", create: bool = typer.Option(False, "-c", "--create"),
            show: Optional[str] = typer.Option(None, "-s", "--show")):
    """Manage sessions."""
    if create:
        a = ZenAgent(user_id=user)
        typer.secho(f"Session: {a.session_id}", fg=typer.colors.GREEN)
    elif show:
        a = ZenAgent(user_id=user, session_id=show)
        for k, v in a.get_info().items(): typer.secho(f"  {k}: {v}")
    else:
        typer.secho("Use --create or --show <id>", fg=typer.colors.YELLOW)


if __name__ == "__main__":
    app()
