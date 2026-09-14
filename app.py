import streamlit as st
import pandas as pd
import requests
import json
import difflib

st.set_page_config(page_title="Decision Brief Generator", layout="wide")

st.title("📊 Decision Brief Generator")
st.caption("Turns raw growth funnel data into a business-ready decision brief: 3 key takeaways + a recommended action.")

REQUIRED_SCHEMA = {
    "week": "The week number, period index, or date of the record.",
    "channel": "The acquisition/marketing channel name (e.g. Paid Search, Organic, Email, Referral).",
    "spend": "Marketing spend / cost attributed to that channel in that period.",
    "visitors": "Number of website visitors / sessions / traffic from that channel.",
    "signups": "Number of new signups / new users / registrations acquired.",
    "activated_users": "Number of signups who became activated / engaged / active users.",
    "retained_users_w4": "Number of users still active / retained roughly 4 weeks later.",
    "revenue": "Revenue generated attributable to that channel and period.",
}
SYNONYMS = {
    "week": ["week", "wk", "period", "date", "week_number"],
    "channel": ["channel", "source", "medium", "marketing channel", "acquisition channel"],
    "spend": ["spend", "cost", "budget", "ad spend", "marketing spend", "ad_cost"],
    "visitors": ["visitors", "visits", "sessions", "traffic", "impressions", "clicks"],
    "signups": ["signups", "sign ups", "new users", "registrations", "leads", "new_signups"],
    "activated_users": ["activated", "activation", "engaged users", "active users", "activated_users"],
    "retained_users_w4": ["retained", "retention", "returning users", "wk4 retained", "retained_users"],
    "revenue": ["revenue", "sales", "income", "earnings", "total revenue"],
}

# ---------- SIDEBAR: AI SETTINGS (shared by column-mapping AND brief generation) ----------
with st.sidebar:
    st.header("⚙️ AI settings")
    st.caption("Optional. Without a key, the app still works using rule-based fallbacks for both column-mapping and the brief.")
    provider = st.selectbox("Provider", ["None (rule-based fallback)", "Anthropic (Claude)", "OpenAI (GPT)"])
    api_key = st.text_input("API key (session-only, never stored)", type="password") if provider != "None (rule-based fallback)" else None
    model_name = None
    if provider == "Anthropic (Claude)":
        model_name = st.text_input("Model", value="claude-sonnet-4-5-20250929")
    elif provider == "OpenAI (GPT)":
        model_name = st.text_input("Model", value="gpt-4o-mini")


def call_llm(prompt, provider, api_key, model_name):
    """Returns raw text response from the chosen LLM. Raises on failure."""
    if provider == "Anthropic (Claude)":
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": model_name, "max_tokens": 800, "messages": [{"role": "user", "content": prompt}]},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(b["text"] for b in data["content"] if b["type"] == "text")
    elif provider == "OpenAI (GPT)":
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"},
            json={"model": model_name, "messages": [{"role": "user", "content": prompt}], "max_tokens": 800},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    else:
        raise ValueError("No provider configured")


def heuristic_column_mapping(uploaded_cols):
    """Fuzzy-match uploaded column names to the required schema without any AI call."""
    mapping = {}
    used = set()
    lower_map = {c.lower().strip(): c for c in uploaded_cols}
    for req_col, syns in SYNONYMS.items():
        best = None
        for lc, orig in lower_map.items():
            if orig in used:
                continue
            if lc == req_col or any(s == lc or s in lc or lc in s for s in syns):
                best = orig
                break
        if best is None:
            candidates = [c for c in uploaded_cols if c not in used]
            close = difflib.get_close_matches(req_col, candidates, n=1, cutoff=0.5)
            if not close:
                for s in syns:
                    close = difflib.get_close_matches(s, candidates, n=1, cutoff=0.6)
                    if close:
                        break
            if close:
                best = close[0]
        mapping[req_col] = best
        if best:
            used.add(best)
    return mapping


def llm_column_mapping(uploaded_cols, sample_rows, provider, api_key, model_name):
    """Ask the LLM to map uploaded columns to REQUIRED_SCHEMA. Returns dict or raises."""
    schema_desc = "\n".join(f"- {k}: {v}" for k, v in REQUIRED_SCHEMA.items())
    prompt = f"""You are a data-mapping assistant. I have a dataset with these columns:
{list(uploaded_cols)}

Sample rows (as JSON):
{json.dumps(sample_rows, default=str)}

I need to map each of the following REQUIRED fields to the best-matching column name from the dataset above
(or null if there is truly no matching column):
{schema_desc}

Respond with ONLY a valid JSON object, no markdown fences, no commentary, in this exact form:
{{"week": "<uploaded column name or null>", "channel": "...", "spend": "...", "visitors": "...",
"signups": "...", "activated_users": "...", "retained_users_w4": "...", "revenue": "..."}}"""
    raw = call_llm(prompt, provider, api_key, model_name)
    cleaned = raw.strip().strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    return json.loads(cleaned)


def coerce_week(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.9:
        return numeric
    dates = pd.to_datetime(series, errors="coerce")
    if dates.notna().mean() > 0.9:
        return dates.rank(method="dense").astype(int)
    codes, _ = pd.factorize(series)
    return codes + 1


# ---------- 1. DATA INPUT ----------
st.header("1. Load data")
data_source = st.radio("Data source", ["Use sample growth dataset", "Upload your own CSV"], horizontal=True)

df = None

if data_source == "Use sample growth dataset":
    df = pd.read_csv("growth_metrics_sample.csv")
    st.info("Using a self-created sample dataset: 10 weeks of activity across 5 acquisition channels.")

else:
    uploaded = st.file_uploader("Upload any CSV with growth funnel data — column names don't need to match exactly.", type="csv")
    if uploaded:
        raw_df = pd.read_csv(uploaded)
        st.write("Detected columns:", list(raw_df.columns))

        if set(REQUIRED_SCHEMA.keys()).issubset(set(raw_df.columns)):
            df = raw_df
            st.success("Columns already match the required schema exactly — no mapping needed.")
        else:
            st.warning("Columns don't match the required schema exactly. Proposing a mapping below — review and confirm before continuing.")

            proposed = None
            if provider != "None (rule-based fallback)" and api_key:
                try:
                    sample_rows = raw_df.head(3).to_dict(orient="records")
                    proposed = llm_column_mapping(raw_df.columns.tolist(), sample_rows, provider, api_key, model_name)
                    st.caption("Mapping proposed by AI. Please verify — check each field below.")
                except Exception as e:
                    st.error(f"AI mapping failed ({e}). Falling back to keyword-based matching.")
                    proposed = heuristic_column_mapping(raw_df.columns.tolist())
            else:
                proposed = heuristic_column_mapping(raw_df.columns.tolist())
                st.caption("Mapping proposed by keyword matching (no AI key set). Please verify — check each field below.")

            st.subheader("Confirm column mapping")
            confirmed = {}
            options = ["-- none --"] + list(raw_df.columns)
            cols_ui = st.columns(2)
            for i, (req_col, desc) in enumerate(REQUIRED_SCHEMA.items()):
                guess = proposed.get(req_col) if proposed else None
                default_idx = options.index(guess) if guess in options else 0
                with cols_ui[i % 2]:
                    confirmed[req_col] = st.selectbox(
                        f"{req_col}  ·  {desc}", options, index=default_idx, key=f"map_{req_col}"
                    )

            if st.button("✅ Confirm mapping & continue"):
                missing = [k for k, v in confirmed.items() if v == "-- none --"]
                if missing:
                    st.error(f"These required fields have no column assigned: {missing}. Fix the mapping above and confirm again.")
                    st.stop()
                mapped_df = pd.DataFrame()
                for req_col, up_col in confirmed.items():
                    mapped_df[req_col] = raw_df[up_col]
                mapped_df["week"] = coerce_week(mapped_df["week"])
                for num_col in ["spend", "visitors", "signups", "activated_users", "retained_users_w4", "revenue"]:
                    mapped_df[num_col] = pd.to_numeric(mapped_df[num_col], errors="coerce").fillna(0)
                st.session_state["mapped_df"] = mapped_df
                st.success("Mapping confirmed. Scroll down for computed metrics.")

            df = st.session_state.get("mapped_df")

if df is None:
    st.warning("Upload a CSV (and confirm its column mapping) or switch to the sample dataset to continue.")
    st.stop()

with st.expander("View data being used"):
    st.dataframe(df, use_container_width=True)

# ---------- 2. COMPUTE METRICS ----------
st.header("2. Computed metrics")

agg = df.groupby("channel").agg(
    total_spend=("spend", "sum"),
    total_visitors=("visitors", "sum"),
    total_signups=("signups", "sum"),
    total_activated=("activated_users", "sum"),
    total_retained=("retained_users_w4", "sum"),
    total_revenue=("revenue", "sum"),
).reset_index()

agg["cac"] = (agg["total_spend"] / agg["total_signups"]).replace([float("inf")], 0).round(2)
agg["signup_rate_pct"] = (agg["total_signups"] / agg["total_visitors"] * 100).round(2)
agg["activation_rate_pct"] = (agg["total_activated"] / agg["total_signups"] * 100).round(2)
agg["retention_rate_pct"] = (agg["total_retained"] / agg["total_activated"] * 100).round(2)
agg["roas"] = (agg["total_revenue"] / agg["total_spend"]).replace([float("inf")], float("nan")).round(2)

st.dataframe(agg, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Signups by channel (total)")
    st.bar_chart(agg.set_index("channel")["total_signups"])
with col2:
    st.subheader("Weekly signup trend")
    weekly = df.groupby(["week", "channel"])["signups"].sum().unstack()
    st.line_chart(weekly)

first_half = df[df["week"] <= df["week"].median()]
second_half = df[df["week"] > df["week"].median()]
trend = (
    second_half.groupby("channel")["signups"].sum() - first_half.groupby("channel")["signups"].sum()
).round(0).to_dict()

metrics_summary = agg[[
    "channel", "total_spend", "total_signups", "cac",
    "activation_rate_pct", "retention_rate_pct", "total_revenue", "roas"
]].to_markdown(index=False)

# ---------- 3. GENERATE DECISION BRIEF ----------
st.header("3. Generate decision brief")


def build_brief_prompt(metrics_md, trend_dict):
    return f"""You are a sharp, numbers-driven Growth Manager. Below is a per-channel performance summary
from a growth funnel (spend, signups, CAC, activation rate, retention rate, revenue, ROAS),
plus the change in signups between the first and second half of the period (positive = growing, negative = declining):

METRICS TABLE:
{metrics_md}

SIGNUP TREND (2nd half minus 1st half): {trend_dict}

Write a decision brief with:
1. Exactly 3 key takeaways, each one sentence, each citing a specific number from the data.
2. Exactly 1 recommended action a Growth Manager should take next week, framed as a concrete decision
   (e.g. reallocate budget, pause a channel, double down on a channel), justified by the data.

Be direct and specific. No generic advice. No preamble."""


def rule_based_brief(agg_df, trend_dict):
    best_cac = agg_df.loc[agg_df["cac"].idxmin()]
    best_roas_row = agg_df.dropna(subset=["roas"])
    best_roas = best_roas_row.loc[best_roas_row["roas"].idxmax()] if not best_roas_row.empty else None
    worst_retention = agg_df.loc[agg_df["retention_rate_pct"].idxmin()]
    declining = {k: v for k, v in trend_dict.items() if v < 0}

    lines = [
        f"1. **{best_cac['channel']}** has the lowest CAC at **${best_cac['cac']}**, "
        f"making it the most capital-efficient acquisition channel this period."
    ]
    if best_roas is not None:
        lines.append(
            f"2. **{best_roas['channel']}** delivers the highest ROAS at **{best_roas['roas']}x**, "
            f"generating the most revenue per dollar spent."
        )
    lines.append(
        f"3. **{worst_retention['channel']}** has the weakest week-4 retention at "
        f"**{worst_retention['retention_rate_pct']}%**, indicating a leaky activation-to-retention funnel."
    )

    if declining:
        worst_decline_channel = min(declining, key=declining.get)
        action = (
            f"**Recommended action:** Reallocate a portion of spend away from **{worst_decline_channel}** "
            f"(signups down {abs(declining[worst_decline_channel]):.0f} week-over-week) toward "
            f"**{best_cac['channel']}**, which combines low CAC with strong retention — and investigate the "
            f"onboarding flow for **{worst_retention['channel']}** to fix the retention leak."
        )
    else:
        action = (
            f"**Recommended action:** Increase budget allocation to **{best_cac['channel']}** given its low CAC "
            f"and strong retention, while auditing onboarding for **{worst_retention['channel']}** to close its "
            f"retention gap."
        )
    lines.append(action)
    return "\n\n".join(lines)


if st.button("🚀 Generate Decision Brief", type="primary"):
    with st.spinner("Generating brief..."):
        try:
            if provider != "None (rule-based fallback)" and api_key:
                brief = call_llm(build_brief_prompt(metrics_summary, trend), provider, api_key, model_name)
            else:
                brief = rule_based_brief(agg, trend)
                if provider != "None (rule-based fallback)":
                    st.warning("No API key entered — showing the rule-based brief instead.")
        except Exception as e:
            st.error(f"AI call failed ({e}). Showing rule-based brief instead.")
            brief = rule_based_brief(agg, trend)

    st.subheader("📋 Decision Brief")
    st.markdown(brief)

st.divider()
st.caption(
    "Built for a Growth Manager application challenge. Uploaded CSVs with non-matching column names are "
    "auto-mapped to the required schema — via AI when a key is set, via keyword matching otherwise — and "
    "you confirm the mapping before any metrics are computed. The brief itself is written by an LLM "
    "(bring your own API key, used only in-session) with a transparent rule-based fallback."
)
