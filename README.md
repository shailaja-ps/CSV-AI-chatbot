# CSV AI Chatbot

Ask questions about any CSV in plain English. Gemini decides *what* to compute; Pandas does the actual calculation on your data.

**Stack:** Python, Pandas, Gemini API, Streamlit

## Features
- Natural-language querying of arbitrary CSVs (LLM -> structured plan -> Pandas execution)
- Streamlit UI: live data preview, column type analysis, statistical summaries, persistent chat history
- Validation of operations and columns, so unsupported or invalid queries get a clear message instead of wrong numbers

- Auto-detected date columns (filter by date, group by year/month/quarter/weekday)
- Multi-step questions via a guarded code-execution fallback: Gemini writes pandas code, which is statically checked (no imports, file/network access or dunder tricks) and run with restricted builtins on a copy of the data. The generated code is always shown. This is a guardrail, not a hardened sandbox, so don't expose it publicly with untrusted users
- Handles utf-8/latin-1 encodings and comma/semicolon/tab delimiters

## Run it
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key   # or paste it in the sidebar
streamlit run app.py
```
Get a key at https://aistudio.google.com/apikey

## How it works
1. Gemini sees only the schema + 3 sample rows, never the full dataset.
2. It returns a JSON plan: operation, column(s), group-by, filters.
3. The plan is validated (whitelisted operations, real columns, numeric type checks).
4. Pandas executes it on the full dataframe and the result is displayed.

## Example questions
- "What's the average of price?"
- "Top 5 rows by revenue"
- "Total sales grouped by region"
- "How many rows where age > 30?"
- "Correlation between height and weight"
