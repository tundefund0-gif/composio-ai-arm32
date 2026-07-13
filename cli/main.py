"""Zen Agent CLI — interactive, oneshot, tools search, token stats."""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
from rich import box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.agent import ZenAgent
from core.llm_client import LLMResponse

app = typer.Typer(help="Zen Agent — AI assistant with 1,000+ Composio tools")
console = Console()


def _get_agent(user_id: str, session_id: Optional[str] = None) -> ZenAgent:
    try:
        return ZenAgent(user_id=user_id, session_id=session_id)
    except Exception as e:
        console.print(f"[red]Failed to initialize agent:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def interactive(
    user: str = typer.Option("cli-user", "--user", "-u", help="User ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Existing session ID"),
    no_sandbox: bool = typer.Option(False, "--no-sandbox", help="Disable sandbox"),
):
    """Interactive chat mode with streaming, colored output & token stats."""
    agent = _get_agent(user, session)
    console.print(Panel.fit(
        f"[bold cyan]🧠 Zen Agent[/bold cyan]\n"
        f"Model: {agent._llm.model} | Session: {agent.session_id[:20]}... | "
        f"Max tokens: {agent._llm.max_tokens:,}",
        border_style="cyan",
    ))
    console.print("[dim]Type /clear to reset, /info for stats, /quit to exit[/dim]\n")

    while True:
        try:
            prompt = console.input("[bold green]You:[/bold green] ")
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt.strip():
            continue
        if prompt.strip().lower() in ("/quit", "/exit", "/q"):
            break
        if prompt.strip().lower() == "/clear":
            agent.clear_history()
            console.print("[yellow]Conversation cleared.[/yellow]")
            continue
        if prompt.strip().lower() == "/info":
            info = agent.get_info()
            t = Table(box=box.SIMPLE)
            t.add_column("Key", style="cyan")
            t.add_column("Value")
            for k, v in info.items():
                t.add_row(k, str(v))
            console.print(t)
            continue

        # Stream response with spinner
        spinner = Spinner("dots", text="[yellow]Thinking...[/yellow]")
        with Live(spinner, refresh_per_second=10, console=console):
            full = ""
            reasoning = ""
            start = time.time()
            for token in agent.chat(prompt, stream=True):
                if token.startswith("__reasoning__"):
                    reasoning += token[13:]
                else:
                    full += token
            elapsed = time.time() - start

        if reasoning:
            with console.status("[dim]Reasoning complete[/dim]"):
                pass

        # Print response as markdown
        console.print()
        md = Markdown(full.strip())
        console.print(Panel(md, border_style="blue", title="[bold]AI[/bold]", title_align="left"))
        console.print(f"[dim]Response time: {elapsed:.1f}s | "
                      f"Length: {len(full):,} chars | "
                      f"History: {len(agent._messages) // 2} turns[/dim]")
        console.print()


@app.command()
def oneshot(
    question: str = typer.Argument(..., help="Single question"),
    user: str = typer.Option("cli-user", "--user", "-u"),
    session: Optional[str] = typer.Option(None, "--session", "-s"),
):
    """Ask a single question (non-streaming)."""
    agent = _get_agent(user, session)
    start = time.time()
    with console.status("[yellow]Processing...[/yellow]"):
        resp = agent.chat(question)
    elapsed = time.time() - start

    if isinstance(resp, LLMResponse):
        md = Markdown(resp.content.strip())
        console.print(Panel(md, border_style="blue", title="[bold]Response[/bold]", title_align="left"))
        if resp.input_tokens or resp.output_tokens:
            console.print(f"[dim]Time: {elapsed:.1f}s | "
                          f"Input: {resp.input_tokens:,} | "
                          f"Output: {resp.output_tokens:,} | "
                          f"Model: {resp.model}[/dim]")
    else:
        console.print(str(resp)[:2000])


@app.command()
def tools(
    query: str = typer.Argument("", help="Search query"),
    user: str = typer.Option("cli-user", "--user", "-u"),
):
    """Search Composio tools."""
    agent = _get_agent(user)
    if query:
        with console.status(f"[yellow]Searching for '{query}'...[/yellow]"):
            result = agent._composio.search_tools(agent.session_id, query)
        data = result.get("data", {})
        results = data.get("results", data.get("tools", []))
        if isinstance(results, list) and results:
            t = Table(box=box.SIMPLE)
            t.add_column("#", style="dim")
            t.add_column("Tool", style="cyan")
            t.add_column("Description")
            for i, r in enumerate(results[:30], 1):
                name = r.get("slug", r.get("name", "?"))
                desc = r.get("description", "")[:100]
                t.add_row(str(i), name, desc)
            console.print(t)
        else:
            console.print("[yellow]No tools found.[/yellow]")
    else:
        with console.status("[yellow]Loading tools...[/yellow]"):
            result = agent._composio.list_all_tools(page=1, page_size=50)
        items = result.get("items", [])
        if items:
            t = Table(box=box.SIMPLE)
            t.add_column("Slug", style="cyan")
            t.add_column("App")
            t.add_column("Description")
            for i in items[:30]:
                slug = i.get("slug", "?")
                app = (i.get("toolkit", {}) or {}).get("name", i.get("app", ""))
                desc = (i.get("description", "") or "")[:80]
                t.add_row(slug, app, desc)
            console.print(t)
            console.print(f"[dim]Showing 30 of {len(items)} tools[/dim]")
        else:
            console.print("[yellow]No tools loaded.[/yellow]")


@app.command()
def session(
    action: str = typer.Argument("info", help="Action: info, reset"),
    user: str = typer.Option("cli-user", "--user", "-u"),
):
    """Manage sessions."""
    agent = _get_agent(user)
    if action == "reset":
        agent.clear_history()
        console.print("[green]Session history cleared.[/green]")
    else:
        info = agent.get_info()
        t = Table(box=box.SIMPLE)
        t.add_column("Key", style="cyan")
        t.add_column("Value")
        for k, v in info.items():
            t.add_row(k, str(v))
        console.print(t)


if __name__ == "__main__":
    app()
