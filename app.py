"""
Streamlit UI for the Coal Mine Accidents Q&A bot.

All retrieval / prompting / Gemini logic now lives in agent.py (single source of truth):
    dense FAISS + BM25 + topic-tag list -> RRF -> cross-encoder rerank (+ tag/year/name boosts)
    -> Gemini with exponential backoff and fallback model.
This file only adds the chat UI and the optional RAGAS-style LLM-judge evaluation.
Keep agent.py in the same folder as this file.
"""
import json
import os
import random
import re
import time

import streamlit as st

# agent.py reads the key from the environment: bridge Streamlit secrets -> env BEFORE importing it.
try:
    if "GEMINI_API_KEY" in st.secrets and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

import agent  # noqa: E402
from google.genai import errors, types  # noqa: E402

NO_ANSWER = agent.NO_ANSWER
JUDGE_MODEL = agent.PRIMARY_MODEL  # ideally a different/stronger model than the generator

METRIC_LABELS = {
    "context_precision": "Context precision",
    "context_recall": "Context recall",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "answer_correctness": "Answer correctness",
}


# =========================
# Cached resources
# =========================
@st.cache_resource(show_spinner="Loading index and models…")
def load_resources():
    return agent.load_resources()


# =========================
# Pipeline (mirrors agent.ask, but keeps the hits for display + evaluation)
# =========================
def answer_query(query: str):
    """Returns (answer, hits). Gemini failures are caught by the caller."""
    topics, _, is_list = agent.analyse_query(query)
    hits = agent.retrieve(query, agent.LIST_K if is_list else agent.FINAL_K, is_list)

    if not hits or (hits[0][2] < agent.MIN_RERANK_SCORE and not topics):
        return f"{NO_ANSWER} I couldn't find a relevant passage.", hits

    tokens = agent.LIST_MAX_TOKENS if is_list else agent.QA_MAX_TOKENS
    answer = agent.ask_gemini(agent.build_qa_prompt(hits), f"Question: {query}", max_tokens=tokens)
    return answer, hits


# =========================
# Judge call (JSON mode) with the same backoff policy as agent.py
# =========================
def judge_generate(prompt: str, max_tokens: int = 3072) -> str:
    client = agent._get_client()
    for attempt in range(agent.MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                ),
            )
            return resp.text or ""
        except errors.APIError as e:
            if e.code not in agent.RETRYABLE_CODES or attempt == agent.MAX_RETRIES - 1:
                raise
            delay = min(agent.MAX_DELAY, agent.BASE_DELAY * (2 ** attempt)) + random.uniform(0, 1)
            time.sleep(delay)


# =========================
# Evaluation metrics (LLM-as-judge, RAGAS-style)
# =========================
JUDGE_SCHEMA = """Return ONLY valid JSON (no markdown) in exactly this shape:
{
  "chunk_relevance": [1, 0, ...],
  "answer_claims": [{"claim": "...", "supported": 1}],
  "answer_relevancy": 0.0,
  "reference_statements": [{"statement": "...", "supported": 1}],
  "answer_correctness": 0.0
}
Use [] for reference_statements and null for answer_correctness when no reference answer is given."""


def build_judge_prompt(query, hits, answer, reference):
    chunks = "\n\n".join(f"[Chunk {i + 1}] ({src})\n{txt}" for i, (txt, src, _) in enumerate(hits))
    ref_block = f"\nREFERENCE (ground-truth) ANSWER:\n{reference}\n" if reference else ""
    ref_task = (
        "4. reference_statements: split the REFERENCE answer into atomic statements; supported = 1 only if "
        "the statement can be attributed to the retrieved chunks, else 0.\n"
        "5. answer_correctness: 0.0-1.0, how well the GENERATED answer agrees with the REFERENCE answer "
        "(facts and numbers must match).\n" if reference else ""
    )
    return (
        "You are a strict, impartial evaluator of a RAG system.\n\n"
        f"QUESTION:\n{query}\n\n"
        f"RETRIEVED CONTEXT CHUNKS (in retrieval order):\n{chunks}\n\n"
        f"GENERATED ANSWER:\n{answer}\n{ref_block}\n"
        "TASKS\n"
        f"1. chunk_relevance: for each of the {len(hits)} chunks, 1 if it contains information useful for "
        "answering the question, else 0. Exactly one item per chunk, in order.\n"
        "2. answer_claims: split the GENERATED answer into atomic factual claims; supported = 1 only if the "
        "claim can be directly inferred from the chunks, else 0.\n"
        "3. answer_relevancy: 0.0-1.0, how directly and completely the answer addresses the question "
        "(ignore whether it is correct).\n"
        f"{ref_task}\n{JUDGE_SCHEMA}"
    )


def parse_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return json.loads(m.group(0)) if m else {}


def context_precision(relevance):
    """Rank-aware precision (RAGAS Context Precision@K): rewards relevant chunks ranked higher."""
    total_rel = sum(relevance)
    if total_rel == 0:
        return 0.0
    seen, score = 0, 0.0
    for k, v in enumerate(relevance, start=1):
        if v:
            seen += 1
            score += seen / k
    return score / total_rel


def ratio(items):
    vals = [int(bool(x.get("supported", 0))) for x in items if isinstance(x, dict)]
    return (sum(vals) / len(vals)) if vals else None


def clamp01(v):
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return None


def evaluate_rag(query, hits, answer, reference=""):
    """Returns a dict of metrics (None = not computable). Uses one judge LLM call."""
    refused = answer.startswith(NO_ANSWER)
    has_ref = bool(reference.strip())
    data = parse_json(judge_generate(build_judge_prompt(query, hits, answer, reference.strip())))

    rel = [int(bool(x)) for x in data.get("chunk_relevance", [])][: len(hits)]
    rel += [0] * (len(hits) - len(rel))

    return {
        "chunk_relevance": rel,
        "context_precision": context_precision(rel),
        "context_recall": ratio(data.get("reference_statements") or []) if has_ref else None,
        "faithfulness": None if refused else ratio(data.get("answer_claims") or []),
        "answer_relevancy": None if refused else clamp01(data.get("answer_relevancy")),
        "answer_correctness": clamp01(data.get("answer_correctness")) if has_ref else None,
    }


def fmt(v):
    return "n/a" if v is None else f"{v:.2f}"


def render_metrics(m):
    if not m:
        return
    with st.expander("📊 RAG metrics", expanded=False):
        cols = st.columns(5)
        for col, key in zip(cols, METRIC_LABELS):
            col.metric(METRIC_LABELS[key], fmt(m.get(key)))
        st.caption(
            f"Retrieval: {m.get('t_retrieval', 0):.2f}s · Generation: {m.get('t_generation', 0):.2f}s · "
            f"Top rerank score: {m.get('top_rerank', float('nan')):.2f}"
        )
        if m.get("error"):
            st.warning(f"Judge evaluation failed: {m['error']}")
        st.caption(
            "LLM-judged (RAGAS-style). Context recall & answer correctness need a reference answer "
            "(sidebar). Faithfulness/relevancy are n/a when the bot refuses to answer."
        )


# =========================
# UI
# =========================
st.set_page_config(page_title="Coal Mine Accidents Q&A", page_icon="⛏️", layout="wide")
st.title("⛏️ Coal Mine Accidents Q&A")
st.caption("RAG chatbot over DGMS SANKET – Key Evaluation of Trends in Coal Mine Accidents "
           "(Hybrid BM25 + FAISS + topic tags · RRF · Cross-encoder rerank + boosts · Gemini)")

if not os.getenv("GEMINI_API_KEY"):
    st.error("❌ `GEMINI_API_KEY` not found in Streamlit secrets or environment.")
    st.stop()

with st.sidebar:
    st.header("Evaluation")
    eval_on = st.toggle("Compute RAG metrics (extra Gemini call)", value=False)
    reference = st.text_area(
        "Reference answer (optional)",
        placeholder="Paste the ground-truth answer for your NEXT question to also get "
                    "context recall and answer correctness.",
        height=140,
    )
    avg_box = st.container()

try:
    load_resources()
except Exception as e:
    st.error(str(e))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_metrics(msg.get("metrics"))

if query := st.chat_input("Ask a question about coal mine accidents…"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    metrics = None
    hits = []
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            t0 = time.perf_counter()
            try:
                answer, hits = answer_query(query)
            except errors.APIError as e:
                answer = f"Sorry, the Gemini API request failed ({e.code}). Please try again in a moment."
            except Exception as e:
                answer = f"Sorry, something went wrong: {e}"
            t_total = time.perf_counter() - t0

        st.markdown(answer)

        failed = answer.startswith("Sorry,")
        if eval_on and hits and not failed:
            with st.spinner("Evaluating…"):
                try:
                    metrics = evaluate_rag(query, hits, answer, reference)
                except Exception as e:
                    metrics = {"error": str(e)}
            # retrieval + generation are now one pipeline call; report the total as generation time
            metrics.update(t_retrieval=0.0, t_generation=t_total, top_rerank=hits[0][2])

        rel = (metrics or {}).get("chunk_relevance")
        with st.expander("Sources"):
            for i, (txt, src, score) in enumerate(hits):
                mark = "" if not rel else ("✅ " if rel[i] else "❌ ")
                st.markdown(f"{mark}**{src}** (score: `{score:.2f}`)")
                st.caption(txt[:300] + ("…" if len(txt) > 300 else ""))
        render_metrics(metrics)

    st.session_state.messages.append({"role": "assistant", "content": answer, "metrics": metrics})

# Running averages across this session (sidebar)
scored = [m["metrics"] for m in st.session_state.messages if m.get("metrics") and not m["metrics"].get("error")]
if scored:
    with avg_box:
        st.subheader(f"Session averages ({len(scored)} answers)")
        for key, label in METRIC_LABELS.items():
            vals = [m[key] for m in scored if m.get(key) is not None]
            st.metric(label, f"{sum(vals) / len(vals):.2f}" if vals else "n/a")
