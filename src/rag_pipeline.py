"""
RAG pipeline — LangChain LCEL chains + Amazon Bedrock + MLflow tracking.

Main entry point: run_pipeline(jd_text, resume_texts)

Flow per job description:
  1. Retrieve profile chunks relevant to the JD
  2. LLM scores semantic fit and produces gap analysis
  3. Preference scorer adds work-arrangement / comp / role-priority score
  4. Composite score ranks the job
  5. Everything is logged to MLflow
"""
from __future__ import annotations

import textwrap
from typing import Any

import mlflow
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.config_loader import load_config
from src.job_scorer import compute_composite_score, compute_preference_score
from src.retrieve import get_retriever


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

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

    raise ValueError(f"Unknown LLM provider: {provider!r}")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

FIT_PROMPT = ChatPromptTemplate.from_template(textwrap.dedent("""\
    You are a senior technical recruiter evaluating a candidate for a role.

    ## Candidate Profile (retrieved excerpts)
    {profile}

    ## Job Description
    {jd}

    Evaluate the match and respond in EXACTLY this format (no extra text):

    SCORE: <integer 0-100>
    RECOMMENDATION: <Strong Match | Possible Match | Poor Match>
    STRENGTHS:
    - <bullet>
    - <bullet>
    GAPS:
    - <bullet or "None identified">
    SUMMARY: <2-3 sentence overall assessment>
"""))

TAILOR_PROMPT = ChatPromptTemplate.from_template(textwrap.dedent("""\
    You are a professional resume writer helping a candidate tailor their resume
    for a specific job description.

    ## Original Resume
    {resume}

    ## Job Description
    {jd}

    ## Gaps Identified
    {gaps}

    Rewrite the resume's EXPERIENCE BULLETS only (do not change dates, titles,
    or education). Reframe existing experience to align with the JD's language
    and priorities. Do not fabricate experience that isn't in the original resume.
    Return only the rewritten experience section.
"""))


# ---------------------------------------------------------------------------
# Chain builders
# ---------------------------------------------------------------------------

def _build_fit_chain(llm, retriever):
    """LCEL chain: {jd} → retrieve profile → score fit."""
    def format_docs(docs):
        return "\n\n---\n\n".join(d.page_content for d in docs)

    return (
        {
            "profile": retriever | format_docs,
            "jd": RunnablePassthrough(),
        }
        | FIT_PROMPT
        | llm
        | StrOutputParser()
    )


def _build_tailor_chain(llm):
    """LCEL chain: {resume, jd, gaps} → tailored experience section."""
    return TAILOR_PROMPT | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_fit_response(raw: str) -> dict:
    result = {"score": 0, "recommendation": "", "strengths": [], "gaps": [], "summary": "", "raw": raw}
    current = None
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("SCORE:"):
            try:
                result["score"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("RECOMMENDATION:"):
            result["recommendation"] = line.split(":", 1)[1].strip()
        elif line.startswith("SUMMARY:"):
            result["summary"] = line.split(":", 1)[1].strip()
        elif line.startswith("STRENGTHS:"):
            current = "strengths"
        elif line.startswith("GAPS:"):
            current = "gaps"
        elif line.startswith("- ") and current in ("strengths", "gaps"):
            result[current].append(line[2:])
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    jd_text: str,
    resume_texts: list[str],
    jd_label: str = "job",
    work_arrangement: str | None = None,
    salary: int | None = None,
    tailor_top_n: int = 1,
) -> dict[str, Any]:
    """
    Evaluate a job description against the indexed profile + supplied resumes.

    Args:
        jd_text:          Full text of the job description.
        resume_texts:     List of resume variant texts to consider for tailoring.
        jd_label:         Short label for MLflow logging (e.g. company name).
        work_arrangement: 'remote', 'hybrid', 'onsite', or None.
        salary:           Stated annual base salary, or None.
        tailor_top_n:     Generate tailored resume only if semantic score >= this rank.
                          Pass 0 to always tailor, -1 to never tailor.

    Returns:
        {
            semantic_fit: int,
            recommendation: str,
            strengths: list[str],
            gaps: list[str],
            summary: str,
            preference_score: float,
            composite_score: float,
            tailored_resume: str | None,
            mlflow_run_id: str,
        }
    """
    cfg = load_config()

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=jd_label) as run:
        # Log inputs
        mlflow.log_param("jd_label", jd_label)
        mlflow.log_param("llm_model", cfg["llm"]["model_id"])
        mlflow.log_param("embedding_model", cfg["embeddings"]["model_id"])
        mlflow.log_param("work_arrangement", work_arrangement or "unknown")
        mlflow.log_param("salary", salary or "unknown")
        mlflow.log_text(jd_text, "jd_input.txt")

        llm = _get_llm(cfg)
        retriever = get_retriever(k=20)

        # --- Semantic fit ---
        fit_chain = _build_fit_chain(llm, retriever)
        raw_fit = fit_chain.invoke(jd_text)
        fit = _parse_fit_response(raw_fit)

        mlflow.log_metric("semantic_fit_score", fit["score"])
        mlflow.log_text(raw_fit, "fit_response.txt")

        # --- Preference score ---
        pref = compute_preference_score(jd_text, work_arrangement, salary)
        mlflow.log_metric("preference_score", pref["preference_score"])
        mlflow.log_metric("work_arrangement_score", pref["work_arrangement_score"])
        mlflow.log_metric("compensation_score", pref["compensation_score"])
        mlflow.log_metric("role_priority_score", pref["role_priority_score"])

        # --- Composite ---
        composite = compute_composite_score(fit["score"], pref["preference_score"])
        mlflow.log_metric("composite_score", composite)

        # --- Resume tailoring ---
        tailored = None
        if tailor_top_n != -1 and fit["score"] >= 60:
            best_resume = resume_texts[0] if resume_texts else ""
            if best_resume:
                tailor_chain = _build_tailor_chain(llm)
                tailored = tailor_chain.invoke({
                    "resume": best_resume,
                    "jd": jd_text,
                    "gaps": "\n".join(f"- {g}" for g in fit["gaps"]) or "None identified",
                })
                mlflow.log_text(tailored, "tailored_resume.txt")

        result = {
            "semantic_fit": fit["score"],
            "recommendation": fit["recommendation"],
            "strengths": fit["strengths"],
            "gaps": fit["gaps"],
            "summary": fit["summary"],
            "preference_score": pref["preference_score"],
            "work_arrangement_score": pref["work_arrangement_score"],
            "compensation_score": pref["compensation_score"],
            "role_priority_score": pref["role_priority_score"],
            "composite_score": composite,
            "tailored_resume": tailored,
            "mlflow_run_id": run.info.run_id,
        }

        mlflow.log_dict(
            {k: v for k, v in result.items() if k != "tailored_resume"},
            "result_summary.json",
        )

    return result
