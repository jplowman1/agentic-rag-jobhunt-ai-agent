"""
ask.py — interactive Q&A against your indexed professional profile.

Useful for ad-hoc questions during a job search or demo:
  "What gaps do I have for a Staff ML Engineer role?"
  "Summarise my experience with RAG pipelines."
  "How should I position my DoD clearance for a commercial AI role?"

Usage:
    python -m scripts.ask                        # interactive REPL
    python -m scripts.ask "your question here"   # single-shot
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

_CONFIG_PATH = Path("config/preferences.yaml")

SYSTEM_PROMPT = ChatPromptTemplate.from_template("""\
You are a career advisor helping a senior AI/ML engineer with their job search.
You have access to excerpts from their professional profile below.
Answer the question clearly and concisely. When relevant, cite specific experience
from the profile. If the profile doesn't contain enough information, say so.

## Profile Excerpts
{profile}

## Question
{question}
""")


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _get_llm(cfg: dict):
    provider = cfg["llm"]["provider"]
    model_id = cfg["llm"]["model_id"]
    region = cfg["llm"].get("region", "us-east-1")
    temperature = cfg["llm"].get("temperature", 0)

    if provider == "bedrock":
        from langchain_aws import ChatBedrock
        return ChatBedrock(
            model_id=model_id,
            region_name=region,
            model_kwargs={"temperature": temperature},
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_id, temperature=temperature)
    raise ValueError(f"Unknown provider: {provider!r}")


def build_chain():
    from src.retrieve import get_retriever
    cfg = _load_config()
    llm = _get_llm(cfg)
    retriever = get_retriever(k=8)

    def format_docs(docs):
        return "\n\n---\n\n".join(d.page_content for d in docs)

    return (
        {"profile": retriever | format_docs, "question": RunnablePassthrough()}
        | SYSTEM_PROMPT
        | llm
        | StrOutputParser()
    )


def ask(chain, question: str):
    console.print("[dim]Retrieving profile context...[/dim]")
    answer = chain.invoke(question)
    console.print()
    console.print(Panel(
        Markdown(answer),
        title=f"[bold cyan]{question[:80]}[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()
    return answer


def repl(chain):
    console.print(Panel.fit(
        "[bold cyan]Job Search Assistant — Ask Mode[/bold cyan]\n"
        "[dim]Ask anything about your profile, experience, or job fit.\n"
        "Type [bold]exit[/bold] or press Ctrl+C to quit.[/dim]",
        border_style="cyan",
    ))
    console.print()

    while True:
        try:
            question = console.input("[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        try:
            ask(chain, question)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]\n")


def main():
    try:
        chain = build_chain()
    except Exception as e:
        console.print(f"[red]Failed to initialise pipeline: {e}[/red]")
        sys.exit(1)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        ask(chain, question)
    else:
        repl(chain)


if __name__ == "__main__":
    main()
