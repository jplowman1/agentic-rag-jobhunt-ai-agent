"""
score_jobs.py — main CLI entry point.

Evaluates one or more job descriptions against your profile and resumes,
ranks them by composite score, optionally generates tailored resume versions,
and saves results to data/output/.

Usage:
    python -m scripts.score_jobs                          # all JDs in data/job_descriptions/
    python -m scripts.score_jobs --jd path/to/jd.txt     # single JD
    python -m scripts.score_jobs --arrangement remote --salary 210000
    python -m scripts.score_jobs --no-tailor              # skip resume tailoring
    python -m scripts.score_jobs --mlflow-ui              # open MLflow UI after run
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from src.jd_parser import extract_jd_metadata
from src.rag_pipeline import run_pipeline
from src.resume_manager import (
    get_best_resume,
    load_resume_texts,
    save_tailored_resume,
)

console = Console()

_JD_DIR = Path("data/job_descriptions")
_OUTPUT_DIR = Path("data/output")
_SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_jd(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    if suffix == ".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(str(path)).paragraphs if p.text.strip())
    return path.read_text(encoding="utf-8", errors="ignore")


def collect_jd_files(jd_arg: str | None) -> list[Path]:
    if jd_arg:
        p = Path(jd_arg)
        if not p.exists():
            console.print(f"[red]File not found: {jd_arg}[/red]")
            sys.exit(1)
        return [p]
    if not _JD_DIR.exists():
        return []
    return [p for p in sorted(_JD_DIR.iterdir()) if p.suffix.lower() in _SUPPORTED_EXTENSIONS]


# ---------------------------------------------------------------------------
# Score colouring helpers
# ---------------------------------------------------------------------------

def _score_color(score: float) -> str:
    if score >= 75:
        return "bright_green"
    if score >= 55:
        return "yellow"
    return "red"


def _rec_color(rec: str) -> str:
    if "Strong" in rec:
        return "bright_green"
    if "Possible" in rec:
        return "yellow"
    return "red"


def _score_bar(score: float, width: int = 20) -> str:
    filled = int(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_header(n_jds: int, n_resumes: int):
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Job Search Assistant[/bold cyan]\n"
        f"[dim]Scoring {n_jds} job description(s) against {n_resumes} resume(s)[/dim]",
        border_style="cyan",
    ))
    console.print()


def print_progress(label: str, composite: float | None = None, error: str | None = None):
    if error:
        console.print(f"  [red]✗[/red] {label}  [red]{error}[/red]")
    elif composite is not None:
        color = _score_color(composite)
        console.print(f"  [green]✓[/green] {label}  [{color}]{composite:.0f}[/{color}]")
    else:
        console.print(f"  [dim]→[/dim] {label} ...", end=" ")


def print_metadata(meta: dict):
    parts = []
    if meta.get("company"):
        parts.append(f"[bold]{meta['company']}[/bold]")
    if meta.get("role_title"):
        parts.append(meta["role_title"])
    if meta.get("location"):
        parts.append(f"📍 {meta['location']}")
    if meta.get("work_arrangement"):
        parts.append(f"🏠 {meta['work_arrangement'].title()}")
    sal_min = meta.get("salary_min")
    sal_max = meta.get("salary_max")
    if sal_min and sal_max:
        parts.append(f"💰 ${sal_min:,}–${sal_max:,}")
    elif sal_min:
        parts.append(f"💰 ${sal_min:,}+")
    if parts:
        console.print("  " + "  │  ".join(parts), style="dim")


def print_rankings_table(results: list[dict]):
    table = Table(
        title="[bold]Job Match Rankings[/bold]",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold white",
        border_style="bright_blue",
        header_style="bold bright_blue",
    )
    table.add_column("#",             width=3,  justify="center")
    table.add_column("Job",           min_width=28)
    table.add_column("Composite",     width=12, justify="center")
    table.add_column("Fit",           width=6,  justify="right")
    table.add_column("Pref",          width=6,  justify="right")
    table.add_column("Work",          width=6,  justify="right")
    table.add_column("Comp",          width=6,  justify="right")
    table.add_column("Recommendation", min_width=16)

    for i, r in enumerate(results, 1):
        composite = r["composite_score"]
        rec = r.get("recommendation", "")
        cc = _score_color(composite)
        rc = _rec_color(rec)
        bar = f"[{cc}]{_score_bar(composite, 10)}[/{cc}] [{cc}]{composite:.0f}[/{cc}]"
        table.add_row(
            f"[bold]{i}[/bold]",
            r["jd_label"],
            bar,
            f"[{_score_color(r['semantic_fit'])}]{r['semantic_fit']}[/{_score_color(r['semantic_fit'])}]",
            f"{r['preference_score']:.0f}",
            f"{r.get('work_arrangement_score', 0):.0f}",
            f"{r.get('compensation_score', 0):.0f}",
            f"[{rc}]{rec}[/{rc}]",
        )

    console.print()
    console.print(table)


def print_detail(r: dict, rank: int):
    rec = r.get("recommendation", "")
    rc = _rec_color(rec)
    cc = _score_color(r["composite_score"])

    header = (
        f"[bold white]#{rank}  {r['jd_label']}[/bold white]  "
        f"[{rc}]{rec}[/{rc}]"
    )
    scores = (
        f"  Composite [{cc}]{r['composite_score']:.0f}[/{cc}]  │"
        f"  Fit [white]{r['semantic_fit']}[/white]  │"
        f"  Pref [white]{r['preference_score']:.0f}[/white]  │"
        f"  Work [white]{r.get('work_arrangement_score', 0):.0f}[/white]  │"
        f"  Comp [white]{r.get('compensation_score', 0):.0f}[/white]"
    )

    meta = r.get("meta", {})
    meta_parts = []
    if meta.get("company"):     meta_parts.append(meta["company"])
    if meta.get("role_title"):  meta_parts.append(meta["role_title"])
    if meta.get("location"):    meta_parts.append(f"📍 {meta['location']}")
    if meta.get("work_arrangement"): meta_parts.append(f"🏠 {meta['work_arrangement'].title()}")
    sal_min, sal_max = meta.get("salary_min"), meta.get("salary_max")
    if sal_min and sal_max:     meta_parts.append(f"💰 ${sal_min:,}–${sal_max:,}")
    elif sal_min:               meta_parts.append(f"💰 ${sal_min:,}+")

    body_lines = [scores]
    if meta_parts:
        body_lines.append(f"[dim]{'  │  '.join(meta_parts)}[/dim]")
    body_lines.append("")

    if r.get("summary"):
        body_lines += [f"[italic]{r['summary']}[/italic]", ""]

    if r.get("strengths"):
        body_lines.append("[bold green]Strengths[/bold green]")
        for s in r["strengths"]:
            body_lines.append(f"  [green]✓[/green] {s}")
        body_lines.append("")

    if r.get("gaps"):
        body_lines.append("[bold red]Gaps[/bold red]")
        for g in r["gaps"]:
            body_lines.append(f"  [red]✗[/red] {g}")
        body_lines.append("")

    if r.get("tailored_resume_path"):
        body_lines.append(f"[bold]Tailored resume →[/bold] [cyan]{r['tailored_resume_path']}[/cyan]")

    body_lines.append(f"[dim]MLflow run: {r.get('mlflow_run_id', 'n/a')}[/dim]")

    console.print(Panel(
        "\n".join(body_lines),
        title=header,
        border_style=cc,
        padding=(0, 1),
    ))


def print_summary_footer(results: list[dict], output_path: Path):
    strong = sum(1 for r in results if "Strong" in r.get("recommendation", ""))
    possible = sum(1 for r in results if "Possible" in r.get("recommendation", ""))
    poor = sum(1 for r in results if "Poor" in r.get("recommendation", ""))
    console.print()
    console.print(
        f"[dim]Summary: "
        f"[bright_green]{strong} Strong[/bright_green]  "
        f"[yellow]{possible} Possible[/yellow]  "
        f"[red]{poor} Poor[/red]  │  "
        f"Results → {output_path}[/dim]"
    )


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_summary(results: list[dict]) -> Path:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = _OUTPUT_DIR / f"scored_jobs_{ts}.json"
    summary = [{k: v for k, v in r.items() if k != "tailored_resume"} for r in results]
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Score job descriptions against your profile.")
    parser.add_argument("--jd", help="Path to a single JD file (default: all in data/job_descriptions/)")
    parser.add_argument("--arrangement", help="Work arrangement hint: remote | hybrid | onsite")
    parser.add_argument("--salary", type=int, help="Stated annual base salary (USD)")
    parser.add_argument("--no-tailor", action="store_true", help="Skip resume tailoring")
    parser.add_argument("--mlflow-ui", action="store_true", help="Launch MLflow UI after run")
    args = parser.parse_args()

    jd_files = collect_jd_files(args.jd)
    if not jd_files:
        console.print("[red]No job description files found.[/red]")
        sys.exit(1)

    resumes = load_resume_texts()
    if not resumes:
        console.print("[yellow]Warning: no resume files found in data/resumes/[/yellow]")

    print_header(len(jd_files), len(resumes))

    all_results = []
    for jd_path in jd_files:
        jd_label = jd_path.stem
        print_progress(jd_label)

        try:
            jd_text = load_jd(jd_path)
            resume_texts = [r["text"] for r in resumes]

            # Auto-extract metadata unless caller provided explicit overrides
            meta = extract_jd_metadata(jd_text)
            arrangement = args.arrangement or meta.get("work_arrangement")
            # Use midpoint of salary range if available; fall back to min
            salary = args.salary or (
                ((meta["salary_min"] + meta["salary_max"]) // 2)
                if meta.get("salary_min") and meta.get("salary_max")
                else meta.get("salary_min")
            )
            print_metadata(meta)

            result = run_pipeline(
                jd_text=jd_text,
                resume_texts=resume_texts,
                jd_label=jd_label,
                work_arrangement=arrangement,
                salary=salary,
                tailor_top_n=0 if not args.no_tailor else -1,
            )
            result["meta"] = meta
            result["jd_label"] = jd_label
            result["tailored_resume_path"] = None

            if result.get("tailored_resume") and not args.no_tailor:
                best = get_best_resume(resumes, jd_text)
                saved = save_tailored_resume(
                    tailored_text=result["tailored_resume"],
                    base_resume_name=best["name"],
                    jd_label=jd_label,
                    mlflow_run_id=result["mlflow_run_id"],
                )
                result["tailored_resume_path"] = str(saved)

            all_results.append(result)
            print_progress(jd_label, composite=result["composite_score"])

        except Exception as e:
            print_progress(jd_label, error=str(e))

    if not all_results:
        console.print("[red]No results produced.[/red]")
        sys.exit(1)

    all_results.sort(key=lambda r: r["composite_score"], reverse=True)

    print_rankings_table(all_results)
    console.print()

    for rank, r in enumerate(all_results, 1):
        print_detail(r, rank)

    out_path = save_summary(all_results)
    print_summary_footer(all_results, out_path)

    if args.mlflow_ui:
        console.print("\n[bold]Launching MLflow UI → http://localhost:5000[/bold]")
        subprocess.Popen(["mlflow", "ui"])


if __name__ == "__main__":
    main()
