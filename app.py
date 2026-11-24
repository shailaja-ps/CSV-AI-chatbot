"""CSV AI Chatbot: ask questions about any CSV in plain English.

Flow: Gemini turns the question into a structured JSON "plan" (it never sees
your full data, only the schema + a small sample). The plan is validated, then
Pandas runs the calculation directly on the uploaded dataframe.
"""
import ast
import builtins
import json
import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

MODEL = "gemini-3.8-flash"

AGGS = ["mean", "sum", "min", "max", "median", "count", "nunique", "std"]
OPERATIONS = AGGS + [
    "describe", "value_counts", "groupby", "top_n", "bottom_n",
    "filter_rows", "correlation", "row_count", "custom_code",
]

DATE_FREQS = {
    "year": lambda s: s.dt.year,
    "month": lambda s: s.dt.to_period("M").astype(str),
    "quarter": lambda s: s.dt.to_period("Q").astype(str),
    "weekday": lambda s: s.dt.day_name(),
}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# --- guardrails for LLM-written pandas code (custom_code) ---
BANNED_NAMES = {
    "open", "eval", "exec", "compile", "__import__", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "input", "breakpoint", "os", "sys", "subprocess",
    "importlib", "builtins",
}
BANNED_ATTRS = {
    "eval", "query", "format", "format_map", "io", "testing", "load", "save", "savetxt",
    "loadtxt", "genfromtxt", "fromfile", "tofile", "memmap", "system", "popen",
    "to_csv", "to_pickle", "to_excel", "to_json", "to_parquet", "to_sql", "to_html",
    "to_clipboard", "to_hdf", "to_feather", "to_stata", "to_latex", "to_markdown",
    "to_xml", "to_orc",
}
SAFE_BUILTIN_NAMES = [
    "len", "min", "max", "sum", "abs", "round", "sorted", "range", "list", "dict", "set",
    "tuple", "str", "int", "float", "bool", "enumerate", "zip", "any", "all", "isinstance",
    "reversed", "map", "filter",
]
SAFE_BUILTINS = {n: getattr(builtins, n) for n in SAFE_BUILTIN_NAMES}
NUMERIC_OPS = {"mean", "sum", "median", "std", "correlation"}
FILTER_OPS = ["==", "!=", ">", "<", ">=", "<=", "contains"]

SYSTEM_PROMPT = f"""You convert a user's question about a pandas DataFrame into a JSON plan.
Respond with ONLY a JSON object with these keys:
  "operation": one of {OPERATIONS}
  "column": target column name or null
  "column2": second column (only for "correlation") or null
  "group_by": column to group by (only for "groupby") or null
  "agg": one of {AGGS} (only for "groupby") or null
  "group_freq": one of ["year","month","quarter","weekday"] when group_by is a date
     column and the user wants time-based grouping, otherwise null
  "code": ONLY for operation "custom_code": Python using pandas (pd) and numpy (np) on a
     DataFrame named df. No imports, no file or network access, do not print. Assign the
     final answer to a variable named result (a scalar, Series or DataFrame).
  "n": integer (for top_n / bottom_n / filter_rows preview size) or null
  "filters": list of {{"column": str, "op": one of {FILTER_OPS}, "value": any}} (may be empty)
  "unsupported_reason": string if the question cannot be answered with these
     operations or the needed columns don't exist, otherwise null
Use exact column names from the schema. Never invent columns.
Prefer the specific operations. Use "custom_code" only for multi-step questions, derived
columns, percentages or ratios, OR conditions, or grouping by multiple columns.
Date columns appear with dtype datetime64.
If a conversation is provided, the new question may be a follow-up ("of them", "those",
"and how many...", "what about..."). Carry over the earlier plan's filters and combine
them with the new condition. Always return a complete, self-contained plan."""


# ---------- helpers ----------
def detect_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns that look like dates into datetime64."""
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].dropna().astype(str)
        if s.empty:
            continue
        sample = s.head(200)
        if sample.str.len().max() > 40:
            continue
        if sample.str.contains(r"[-/:]|[A-Za-z]{3}").mean() < 0.9:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() >= 0.9:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
    return df


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    import io
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        # first try delimiter sniffing (handles ; and tab), then plain comma
        for kwargs in ({"sep": None, "engine": "python"}, {}):
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc, **kwargs)
                return detect_dates(df)
            except UnicodeDecodeError:
                break
            except Exception:
                continue
    raise ValueError("Couldn't read this file as a CSV.")


def schema_text(df: pd.DataFrame) -> str:
    def describe_col(c):
        line = f"- {c} ({df[c].dtype})"
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c]):
            vals = df[c].dropna().unique()
            if 0 < len(vals) <= 15:
                line += f" values: {[str(v) for v in vals]}"
        return line

    cols = "\n".join(describe_col(c) for c in df.columns)
    sample = df.head(3).to_string(index=False)
    return f"Columns:\n{cols}\n\nSample rows:\n{sample}"


def summarize_result(result) -> str:
    if isinstance(result, (pd.DataFrame, pd.Series)):
        return f"table with {len(result)} rows, first rows:\n{result.head(3).to_string()}"
    return str(result)


def build_context(history, max_msgs: int = 8) -> str:
    """Compact text of recent turns so Gemini can resolve 'of them', 'those', etc."""
    lines = []
    for m in history[-max_msgs:]:
        if m["role"] == "user":
            lines.append(f"User asked: {m['content']}")
        elif m.get("plan"):
            lines.append(f"Plan used: {json.dumps(m['plan'])}")
            lines.append(f"Result: {summarize_result(m['result'])}")
        else:
            lines.append(f"Assistant replied: {m['content']}")
    return "\n".join(lines)


def get_plan(client, df: pd.DataFrame, question: str, context: str = "") -> dict:
    convo = f"\n\nConversation so far:\n{context}" if context else ""
    resp = client.models.generate_content(
        model=MODEL,
        contents=f"{schema_text(df)}{convo}\n\nNew question: {question}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0,
        ),
    )
    return json.loads(resp.text)


def resolve_col(df: pd.DataFrame, name):
    """Case-insensitive column lookup. Returns the real name or raises."""
    if name is None:
        return None
    lookup = {c.lower().strip(): c for c in df.columns}
    key = str(name).lower().strip()
    if key not in lookup:
        raise ValueError(f"Column '{name}' doesn't exist. Available: {', '.join(df.columns)}")
    return lookup[key]


def require_numeric(df, col, op):
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ValueError(f"'{op}' needs a numeric column, but '{col}' is {df[col].dtype}.")


def apply_filters(df: pd.DataFrame, filters) -> pd.DataFrame:
    for f in filters or []:
        col = resolve_col(df, f.get("column"))
        op, val = f.get("op"), f.get("value")
        if op not in FILTER_OPS:
            raise ValueError(f"Unsupported filter operator '{op}'.")
        s = df[col]
        if op == "contains":
            df = df[s.astype(str).str.contains(str(val), case=False, na=False)]
            continue
        if pd.api.types.is_numeric_dtype(s):
            try:
                val = float(val)
            except (TypeError, ValueError):
                raise ValueError(f"'{col}' is numeric but filter value '{val}' isn't.")
        elif pd.api.types.is_datetime64_any_dtype(s):
            try:
                val = pd.to_datetime(val)
            except (TypeError, ValueError):
                raise ValueError(f"'{col}' is a date column but '{val}' isn't a valid date.")
        else:
            s = s.astype(str)
            val = str(val)
        mask = {
            "==": s == val, "!=": s != val, ">": s > val,
            "<": s < val, ">=": s >= val, "<=": s <= val,
        }[op]
        df = df[mask]
    return df


def validate_code(code: str):
    """Static checks on LLM-written code before it is allowed to run."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"The generated code has a syntax error: {e}")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
                             ast.ClassDef, ast.AsyncFunctionDef)):
            raise ValueError("The generated code used a statement that isn't allowed.")
        if isinstance(node, ast.Attribute):
            a = node.attr
            if a.startswith("_") or a.startswith("read_") or a in BANNED_ATTRS:
                raise ValueError(f"The generated code used a blocked attribute: '{a}'.")
        if isinstance(node, ast.Name) and (node.id in BANNED_NAMES or node.id.startswith("__")):
            raise ValueError(f"The generated code used a blocked name: '{node.id}'.")


def run_custom_code(df: pd.DataFrame, code):
    if not code or not isinstance(code, str):
        raise ValueError("No code was generated for this question.")
    validate_code(code)
    g = {"__builtins__": SAFE_BUILTINS, "df": df.copy(), "pd": pd, "np": np}
    try:
        exec(compile(code, "<generated>", "exec"), g)
    except Exception as e:
        raise ValueError(f"The generated code failed ({type(e).__name__}: {e}). Try rephrasing.")
    if "result" not in g:
        raise ValueError("The generated code didn't produce a result.")
    res = g["result"]
    if isinstance(res, (pd.DataFrame, pd.Series)):
        if len(res) == 0:
            raise ValueError("The calculation returned no rows.")
        return res.head(200)
    if isinstance(res, np.generic):
        res = res.item()
    if isinstance(res, float):
        res = round(res, 4)
    return res if isinstance(res, (int, float, str, bool)) else str(res)


def execute(df: pd.DataFrame, plan: dict):
    """Validate the plan, then run it with Pandas. Returns (result, explanation)."""
    if plan.get("unsupported_reason"):
        raise ValueError(plan["unsupported_reason"])

    op = plan.get("operation")
    if op not in OPERATIONS:
        raise ValueError(f"Unsupported operation '{op}'. Supported: {', '.join(OPERATIONS)}")

    if op == "custom_code":
        return run_custom_code(df, plan.get("code")), "Custom pandas analysis"

    df = apply_filters(df, plan.get("filters"))
    if df.empty:
        raise ValueError("No rows match those filters.")

    col = resolve_col(df, plan.get("column"))
    n = int(plan.get("n") or 5)

    if op == "row_count":
        return len(df), "Number of rows"
    if op == "describe":
        return (df[[col]] if col else df).describe(include="all").T, "Statistical summary"
    if op == "filter_rows":
        return df.head(n if plan.get("n") else 20), f"Matching rows ({len(df)} total)"

    if col is None:
        raise ValueError("That operation needs a column, but none was identified.")

    if op == "value_counts":
        return df[col].value_counts().head(n if plan.get("n") else 10).to_frame("count"), f"Value counts for {col}"
    if op in ("top_n", "bottom_n"):
        require_numeric(df, col, op)
        fn = df.nlargest if op == "top_n" else df.nsmallest
        return fn(n, col), f"{'Top' if op == 'top_n' else 'Bottom'} {n} by {col}"
    if op == "correlation":
        col2 = resolve_col(df, plan.get("column2"))
        if col2 is None:
            raise ValueError("Correlation needs two columns.")
        require_numeric(df, col, op)
        require_numeric(df, col2, op)
        return round(df[col].corr(df[col2]), 4), f"Correlation between {col} and {col2}"
    if op == "groupby":
        grp = resolve_col(df, plan.get("group_by"))
        agg = plan.get("agg")
        if grp is None or agg not in AGGS:
            raise ValueError("Group-by needs a group column and a valid aggregation.")
        if agg in NUMERIC_OPS:
            require_numeric(df, col, agg)
        key, freq = df[grp], plan.get("group_freq")
        if freq:
            if freq not in DATE_FREQS:
                raise ValueError(f"Unsupported date grouping '{freq}'.")
            if not pd.api.types.is_datetime64_any_dtype(key):
                raise ValueError(f"'{grp}' isn't a date column, so I can't group it by {freq}.")
            key = DATE_FREQS[freq](key)
        res = df.groupby(key)[col].agg(agg)
        if freq == "weekday":
            res = res.reindex([d for d in WEEKDAYS if d in res.index])
        elif freq:
            res = res.sort_index()
        else:
            res = res.sort_values(ascending=False)
        res = res.to_frame(f"{agg}_{col}")
        res.index.name = f"{grp} ({freq})" if freq else grp
        return res, f"{agg} of {col} grouped by {grp}" + (f" per {freq}" if freq else "")

    # simple aggregations
    if op in NUMERIC_OPS:
        require_numeric(df, col, op)
    val = getattr(df[col], op)()
    return (round(val, 4) if isinstance(val, float) else val), f"{op} of {col}"


def render_result(result, explanation):
    st.caption(explanation)
    if isinstance(result, (pd.DataFrame, pd.Series)):
        st.dataframe(result, use_container_width=True)
    else:
        st.metric(label=explanation, value=str(result))


# ---------- UI ----------
st.set_page_config(page_title="CSV AI Chatbot", page_icon="📊", layout="wide")
st.title("📊 CSV AI Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Gemini API key", type="password",
                            value=os.getenv("GEMINI_API_KEY", ""))
    uploaded = st.file_uploader("Upload a CSV", type="csv")
    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()
    if st.session_state.history:
        export = [{"role": m["role"], "content": m["content"]} for m in st.session_state.history]
        st.download_button("Download chat (JSON)", json.dumps(export, indent=2),
                           "chat_history.json", "application/json")

if not uploaded:
    st.info("Upload a CSV in the sidebar to get started.")
    st.stop()

try:
    df = load_csv(uploaded.getvalue())
except ValueError as e:
    st.error(str(e))
    st.stop()

tab_chat, tab_data = st.tabs(["💬 Chat", "🔎 Data overview"])

with tab_data:
    st.subheader("Live preview")
    st.dataframe(df.head(50), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    st.subheader("Column types")
    types_df = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "non_null": df.notna().sum().values,
        "missing": df.isna().sum().values,
        "unique": df.nunique().values,
    })
    st.dataframe(types_df, use_container_width=True, hide_index=True)
    st.subheader("Statistical summary")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

with tab_chat:
    for m in st.session_state.history:
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m.get("result") is not None:
                render_result(m["result"], m["explanation"])

    question = st.chat_input("Ask something about your data, e.g. 'average salary by department'")
    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            if not api_key:
                msg = "Add your Gemini API key in the sidebar first."
                st.warning(msg)
                st.session_state.history.append({"role": "assistant", "content": msg})
            else:
                try:
                    with st.spinner("Thinking..."):
                        client = genai.Client(api_key=api_key)
                        context = build_context(st.session_state.history[:-1])
                        plan = get_plan(client, df, question, context)
                        result, explanation = execute(df, plan)
                    st.write("Here's what I found:")
                    render_result(result, explanation)
                    with st.expander("How this was computed"):
                        if plan.get("operation") == "custom_code":
                            st.code(plan.get("code", ""), language="python")
                        else:
                            st.json(plan)
                    st.session_state.history.append({
                        "role": "assistant", "content": "Here's what I found:",
                        "result": result, "explanation": explanation, "plan": plan,
                    })
                except ValueError as e:
                    msg = f"I can't do that: {e}"
                    st.warning(msg)
                    st.session_state.history.append({"role": "assistant", "content": msg})
                except Exception as e:  # API / JSON errors
                    msg = f"Something went wrong talking to Gemini: {e}"
                    st.error(msg)
                    st.session_state.history.append({"role": "assistant", "content": msg})