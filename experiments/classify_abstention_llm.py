"""Classify each judge=incorrect prediction via two LLMs (Azure gpt-4.1-mini + Claude Haiku 4.5).

Output: abstention_taxonomy_llm.csv with columns
  experiment, phase, query_id, pred, regex_label, gpt4mini_label, claude_label,
  majority_label, all_agree.

Scope: main-paper experiments only (BCP seed 42/7/100/gpt-4o, QASPER-Oracle,
QASPER-RAG, FinanceBench-LC, A1/A2 ablations). Smoke runs excluded.

Concurrency: 10 concurrent requests per provider.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv

from hgc.paths import results_root

CODE_ROOT = Path(__file__).resolve().parents[1]
RESULTS = results_root(CODE_ROOT)
TAXONOMY_CSV = CODE_ROOT / "analysis" / "abstention_taxonomy.csv"
OUT_CSV = CODE_ROOT / "analysis" / "abstention_taxonomy_llm.csv"

load_dotenv(CODE_ROOT / ".env")

MAIN_EXPERIMENTS = {
    "bcp_gpt41_seed42",
    "bcp_gpt41_seed7",
    "bcp_gpt41_seed100",
    "bcp_gpt4o",
    "base_qasper_oracle",
    "base_qasper_rag",
    "base_finbench_lc",
    "bcp_a1_g2off",
    "bcp_a1_g3off",
    "bcp_a2_5pct",
    "bcp_a2_50pct",
}

CONCURRENCY = 10

PROMPT = """You classify answers produced by an information-seeking AI system.

Given a QUESTION, the GOLD answer (reference), and the PREDICTED answer that the system produced, decide which of these two categories the prediction falls into:

- HONEST_ABSTENTION: the prediction explicitly refuses to answer, admits the answer is unknown or not available, says the source document does not contain the answer, or otherwise declines to commit to a specific answer. The prediction may or may not be accompanied by a partial guess, but its primary stance is "I cannot answer" / "information not available".
- CONFIDENT_WRONG: the prediction commits to a specific answer that is different from the gold answer (hallucinated, wrong entity, wrong number, unrelated content, etc.). It does NOT admit it is unknown.

Output exactly one token: HONEST_ABSTENTION or CONFIDENT_WRONG. No other text."""


def load_incorrect_rows() -> list[dict]:
    with open(TAXONOMY_CSV) as f:
        rows = list(csv.DictReader(f))
    # Only main experiments and only rows the judge said were incorrect
    rows = [
        r
        for r in rows
        if r["experiment"] in MAIN_EXPERIMENTS and r["judgment_correct"].lower() == "false"
    ]
    # Read the full prediction text (the taxonomy CSV only has a 140-char preview)
    enriched = []
    for r in rows:
        tp = RESULTS / r["experiment"] / r["phase"] / f"trajectory_q{r['query_id']}.json"
        try:
            d = json.loads(tp.read_text())
        except Exception:
            continue
        enriched.append(
            {
                "experiment": r["experiment"],
                "phase": r["phase"],
                "query_id": r["query_id"],
                "question": (d.get("question") or "").strip(),
                "gold": (d.get("gold") or "").strip(),
                "pred": (d.get("pred") or "").strip(),
                "regex_label": r["category"],
            }
        )
    return enriched


def build_user_message(row: dict) -> str:
    return f"QUESTION:\n{row['question']}\n\nGOLD:\n{row['gold']}\n\nPREDICTED:\n{row['pred']}"


def normalise(label: str) -> str:
    lab = label.strip().upper()
    if "ABSTAIN" in lab:
        return "honest_abstention"
    if "WRONG" in lab or "CONFIDENT" in lab:
        return "confident_wrong"
    # Fallback: regex sniff the label
    if re.search(r"\babstain|abstention|cannot|unknown|refus", lab, re.IGNORECASE):
        return "honest_abstention"
    return "confident_wrong"


async def azure_classify(client: httpx.AsyncClient, row: dict, sem: asyncio.Semaphore) -> str:
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    deployment = os.environ["AZURE_OPENAI_SUB_DEPLOYMENT"]
    version = os.environ["AZURE_OPENAI_API_VERSION"]
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}"
    payload = {
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": build_user_message(row)},
        ],
        "temperature": 0,
        "max_tokens": 16,
    }
    headers = {"api-key": os.environ["AZURE_OPENAI_API_KEY"]}
    async with sem:
        for attempt in range(4):
            try:
                r = await client.post(url, json=payload, headers=headers, timeout=60)
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
                return normalise(text)
            except Exception as e:
                if attempt == 3:
                    return f"error:{type(e).__name__}"
                await asyncio.sleep(1.5**attempt)
    return "error:retries"


async def anthropic_classify(client: httpx.AsyncClient, row: dict, sem: asyncio.Semaphore) -> str:
    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": os.environ["ANTHROPIC_SUB_MODEL"],
        "max_tokens": 16,
        "system": PROMPT,
        "messages": [
            {"role": "user", "content": build_user_message(row)},
        ],
    }
    headers = {
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    async with sem:
        for attempt in range(4):
            try:
                r = await client.post(url, json=payload, headers=headers, timeout=60)
                r.raise_for_status()
                text = r.json()["content"][0]["text"]
                return normalise(text)
            except Exception as e:
                if attempt == 3:
                    return f"error:{type(e).__name__}"
                await asyncio.sleep(1.5**attempt)
    return "error:retries"


async def main() -> None:
    rows = load_incorrect_rows()
    print(f"Loaded {len(rows)} incorrect rows from main experiments")

    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient() as client:
        az_tasks = [azure_classify(client, r, sem) for r in rows]
        cl_tasks = [anthropic_classify(client, r, sem) for r in rows]
        az_results, cl_results = await asyncio.gather(
            asyncio.gather(*az_tasks),
            asyncio.gather(*cl_tasks),
        )

    def majority(labels: list[str]) -> str:
        votes = [l for l in labels if not l.startswith("error")]
        if not votes:
            return "unknown"
        counts = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        return max(counts, key=counts.get)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(
            [
                "experiment",
                "phase",
                "query_id",
                "regex_label",
                "gpt4mini_label",
                "claude_label",
                "majority_label",
                "all_agree",
                "pred_preview",
            ]
        )
        for r, az, cl in zip(rows, az_results, cl_results):
            triple = [r["regex_label"], az, cl]
            maj = majority(triple)
            all_agree = len(set(triple)) == 1 and not any(x.startswith("error") for x in triple)
            w.writerow(
                [
                    r["experiment"],
                    r["phase"],
                    r["query_id"],
                    r["regex_label"],
                    az,
                    cl,
                    maj,
                    str(all_agree).lower(),
                    r["pred"][:140].replace("\n", " "),
                ]
            )

    # Agreement summary
    from collections import Counter

    pair_regex_gpt = sum(1 for r, az in zip(rows, az_results) if r["regex_label"] == az) / len(rows)
    pair_regex_cl = sum(1 for r, cl in zip(rows, cl_results) if r["regex_label"] == cl) / len(rows)
    pair_gpt_cl = sum(1 for az, cl in zip(az_results, cl_results) if az == cl) / len(rows)
    all_three = sum(
        1 for r, az, cl in zip(rows, az_results, cl_results) if r["regex_label"] == az == cl
    ) / len(rows)
    print()
    print(f"regex vs gpt-4.1-mini   agreement: {pair_regex_gpt * 100:.1f}%")
    print(f"regex vs claude-haiku   agreement: {pair_regex_cl * 100:.1f}%")
    print(f"gpt-4.1-mini vs claude  agreement: {pair_gpt_cl * 100:.1f}%")
    print(f"all three agree:                   {all_three * 100:.1f}%")
    print()
    print(f"gpt-4.1-mini label counts: {dict(Counter(az_results))}")
    print(f"claude-haiku  label counts: {dict(Counter(cl_results))}")
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
