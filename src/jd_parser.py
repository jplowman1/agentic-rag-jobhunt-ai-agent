"""
JD metadata extractor — uses the LLM to pull structured facts from raw JD text.

This is the "agentic extraction" step: instead of requiring the user to manually
pass --arrangement and --salary flags, the LLM reads the JD and extracts them.

Extracted fields:
    company         str | None   e.g. "Leidos"
    role_title      str | None   e.g. "Senior AI Engineer"
    work_arrangement str | None  "remote" | "hybrid" | "onsite" | None
    location        str | None   e.g. "McLean, VA" or "Remote"
    salary_min      int | None   annual base USD
    salary_max      int | None   annual base USD
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "preferences.yaml"

_EXTRACT_PROMPT = ChatPromptTemplate.from_template("""\
Extract structured metadata from the job description below.
Respond with ONLY a valid JSON object — no markdown, no explanation.

Fields to extract:
- company: employer name (string or null)
- role_title: job title (string or null)
- work_arrangement: one of "remote", "hybrid", "onsite", or null if unclear
- location: city/state or "Remote" (string or null)
- salary_min: minimum annual base salary in USD as integer, or null if not stated
- salary_max: maximum annual base salary in USD as integer, or null if not stated

Job Description:
{jd}
""")


def _get_llm(cfg: dict):
    provider = cfg["llm"]["provider"]
    model_id = cfg["llm"]["model_id"]
    region = cfg["llm"].get("region", "us-east-1")

    if provider == "bedrock":
        from langchain_aws import ChatBedrock
        return ChatBedrock(
            model_id=model_id,
            region_name=region,
            model_kwargs={"temperature": 0},
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_id, temperature=0)
    raise ValueError(f"Unknown provider: {provider!r}")


def extract_jd_metadata(jd_text: str) -> dict:
    """
    Parse a JD and return a metadata dict.
    Falls back gracefully — any field that can't be extracted is None.
    """
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    llm = _get_llm(cfg)
    chain = _EXTRACT_PROMPT | llm | StrOutputParser()

    try:
        raw = chain.invoke({"jd": jd_text[:4000]})  # truncate to keep prompt small
        # Strip any accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        data = json.loads(raw)
    except Exception:
        data = {}

    return {
        "company":          data.get("company"),
        "role_title":       data.get("role_title"),
        "work_arrangement": data.get("work_arrangement"),
        "location":         data.get("location"),
        "salary_min":       data.get("salary_min"),
        "salary_max":       data.get("salary_max"),
    }
