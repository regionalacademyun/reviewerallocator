import re
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import STOPWORDS, WordCloud

from .allocation_logic import compute_planning_metrics

UNIVERSITY_STOPWORDS = {
    "university", "universitat", "universidad", "universite", "universität",
    "school", "institute", "institut", "college", "faculty", "centre", "center",
    "academy", "of", "the", "for", "and", "in", "at", "de", "di", "la", "le",
    "del", "della", "state", "main", "current", "enrolment",
}
FIELD_STOPWORDS = {
    "studies", "study", "program", "programme", "master", "masters", "bachelor",
    "bachelors", "phd", "degree", "specialization", "specialisation", "semester",
    "second", "first", "equivalent", "field", "current", "level", "of", "the",
    "and", "for", "in", "to", "based",
}
PLOTLY_PALETTE = [
    "#2563EB", "#0EA5E9", "#10B981", "#8B5CF6",
    "#F59E0B", "#EF4444", "#14B8A6", "#F97316",
]
YES_VALUES = {"yes", "y", "true", "1"}
NO_VALUES = {"no", "n", "false", "0"}


def app_styles():
    st.markdown(
        '''
        <style>
        .main .block-container {padding-top: 1rem; padding-bottom: 1.8rem; max-width: 1500px;}
        .hero {
            border-radius:22px; padding:1.2rem 1.3rem;
            background:linear-gradient(135deg,#eef6ff,#ffffff,#eefdf6);
            border:1px solid rgba(148,163,184,0.25);
            box-shadow: 0 12px 28px rgba(15,23,42,0.08);
            margin-bottom:1rem;
        }
        .step {
            border-radius:18px; padding:1rem 1.05rem;
            background:linear-gradient(135deg,#ffffff,#f8fbff);
            border:1px solid rgba(148,163,184,0.18);
            box-shadow:0 8px 18px rgba(15,23,42,0.04);
            margin-bottom:1rem;
        }
        .explain {font-size:0.96rem; color:#475569;}
        .mini-sub {font-size:0.88rem; color:#64748b; margin-top:-0.25rem;}
        </style>
        ''',
        unsafe_allow_html=True,
    )


def _clean_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="object")
    s = df[column].fillna("").astype(str).str.strip()
    return s[s != ""]


def _tokenize_text(values: Iterable[str], stopwords: set[str]) -> str:
    tokens: List[str] = []
    for val in values:
        pieces = re.findall(r"[A-Za-zÀ-ÿ0-9\-']+", str(val))
        for token in pieces:
            lowered = token.lower()
            if lowered in stopwords or len(lowered) <= 1:
                continue
            tokens.append(token)
    return " ".join(tokens)


def _metric_subtext(text: str):
    st.markdown(f"<div class='mini-sub'>{text}</div>", unsafe_allow_html=True)


def _normalize_yes_no(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip().str.lower()
    out = pd.Series("No response", index=s.index)
    out[s.isin(YES_VALUES)] = "Yes"
    out[s.isin(NO_VALUES)] = "No"
    return out


def _yes_no_counts(df: pd.DataFrame, column: str, title: str, key: str):
    if column not in df.columns:
        st.info(f"There is no data for {title}.")
        return
    raw = df[column].fillna("").astype(str).str.strip()
    if raw.eq("").all():
        st.info(f"There is no data for {title}.")
        return
    mapped = _normalize_yes_no(raw)
    counts = mapped.value_counts().reset_index()
    counts.columns = ["Response", "Count"]
    counts["Percent"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)
    fig = px.pie(
        counts,
        names="Response",
        values="Count",
        hole=0.45,
        color="Response",
        color_discrete_sequence=PLOTLY_PALETTE,
        title=title,
    )
    fig.update_traces(
        textinfo="label+percent",
        customdata=counts[["Count", "Percent"]].values,
        hovertemplate="<b>%{label}</b><br>Count: %{customdata[0]}<br>Share: %{customdata[1]}%<extra></extra>",
    )
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=20, r=20, t=60, b=20), showlegend=True)
    st.plotly_chart(fig, width="stretch", key=key)


def _scholarship_subgroup_chart(df: pd.DataFrame, scholarship_col: str, attend_col: str, key: str):
    if scholarship_col not in df.columns:
        st.info("There is no scholarship data.")
        return
    scholarship_status = _normalize_yes_no(df[scholarship_col])
    subgroup = df[scholarship_status.eq("Yes")].copy()
    if subgroup.empty:
        st.info("No scholarship applicants were found in the file.")
        return
    if attend_col not in subgroup.columns:
        st.info("The file does not contain the follow-up question about participating without scholarship.")
        return
    attend_status = _normalize_yes_no(subgroup[attend_col])
    counts = attend_status.value_counts().reset_index()
    counts.columns = ["Response", "Count"]
    counts["Percent"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)
    fig = px.pie(
        counts,
        names="Response",
        values="Count",
        hole=0.45,
        color="Response",
        color_discrete_sequence=PLOTLY_PALETTE,
        title="Among scholarship applicants: would still participate without scholarship?",
    )
    fig.update_traces(
        textinfo="label+percent",
        customdata=counts[["Count", "Percent"]].values,
        hovertemplate="<b>%{label}</b><br>Count: %{customdata[0]}<br>Share within scholarship group: %{customdata[1]}%<extra></extra>",
    )
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=20, r=20, t=60, b=20), showlegend=True)
    st.plotly_chart(fig, width="stretch", key=key)
    total_n = len(df)
    applied_n = len(subgroup)
    applied_pct = round(applied_n / total_n * 100, 1) if total_n else 0
    st.caption(f"{applied_n} of {total_n} applicants applied for scholarship ({applied_pct}%). The chart above looks only at that scholarship subgroup.")


def _distribution_chart(df: pd.DataFrame, column: str, title: str, color: str, key: str):
    s = _clean_series(df, column)
    if s.empty:
        st.info(f"There is no data for {column}.")
        return
    counts = s.value_counts().reset_index()
    counts.columns = [column, "Count"]
    counts["Percent"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)
    counts = counts.sort_values(["Count", column], ascending=[True, True])
    counts["Label"] = counts["Count"].astype(str) + " (" + counts["Percent"].astype(str) + "%)"
    height = max(440, 28 * len(counts))
    fig = px.bar(counts, x="Count", y=column, orientation="h", text="Label", title=title)
    fig.update_traces(
        marker_color=color,
        textposition="outside",
        cliponaxis=False,
        customdata=counts[["Percent"]].values,
        hovertemplate=f"<b>%{{y}}</b><br>Count: %{{x}}<br>Share: %{{customdata[0]}}%<extra></extra>",
    )
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=20, r=40, t=60, b=20), yaxis_title="", xaxis_title="Applicants", showlegend=False)
    st.plotly_chart(fig, width="stretch", key=key)


def _program_mix_chart(df: pd.DataFrame, key: str):
    s = _clean_series(df, "Program")
    if s.empty:
        st.info("There is no data for Programme mix.")
        return
    counts = s.value_counts().reset_index()
    counts.columns = ["Program", "Count"]
    counts["Percent"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)
    fig = px.pie(counts, names="Program", values="Count", hole=0.45, title="Programme mix", color_discrete_sequence=PLOTLY_PALETTE)
    fig.update_traces(
        textinfo="label+percent",
        customdata=counts[["Count", "Percent"]].values,
        hovertemplate="<b>%{label}</b><br>Count: %{customdata[0]}<br>Share: %{customdata[1]}%<extra></extra>",
    )
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, width="stretch", key=key)


def _country_map(df: pd.DataFrame, show_residence: bool, show_citizenship: bool, key: str):
    parts = []
    if show_residence and "Country of Residence" in df.columns:
        tmp = df[["Country of Residence"]].copy(); tmp.columns = ["country"]; tmp["type"] = "Residence"; parts.append(tmp)
    if show_citizenship and "Country of Citizenship" in df.columns:
        tmp = df[["Country of Citizenship"]].copy(); tmp.columns = ["country"]; tmp["type"] = "Citizenship"; parts.append(tmp)
    if not parts:
        st.info("Turn on residence and/or citizenship to show the map.")
        return
    merged = pd.concat(parts, ignore_index=True)
    merged["country"] = merged["country"].fillna("").astype(str).str.strip()
    merged = merged[merged["country"] != ""]
    if merged.empty:
        st.info("There is no country data to show on the map.")
        return
    counts = merged.groupby(["country", "type"]).size().reset_index(name="count")
    counts["Percent"] = (counts["count"] / counts["count"].sum() * 100).round(1)
    fig = px.choropleth(
        counts,
        locations="country",
        locationmode="country names",
        color="count",
        hover_name="country",
        hover_data={"type": True, "count": True, "Percent": True},
        color_continuous_scale="Blues",
        title="Where applicants are from",
    )
    fig.update_layout(height=520, template="plotly_white", margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, width="stretch", key=key)


def _word_cloud(df: pd.DataFrame, columns: List[str], title: str, stopwords: set[str]):
    values: List[str] = []
    for col in columns:
        if col in df.columns:
            values.extend(df[col].fillna("").astype(str).tolist())
    text = _tokenize_text(values, STOPWORDS.union(stopwords))
    if not text.strip():
        st.info(f"There is no text to show for {title}.")
        return
    wc = WordCloud(width=1400, height=560, background_color="white", collocations=False, stopwords=STOPWORDS.union(stopwords), colormap="viridis", prefer_horizontal=0.9).generate(text)
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=18, pad=12)
    st.pyplot(fig, clear_figure=True)


def render_dashboard(applicants_df: pd.DataFrame, reviewers_df: pd.DataFrame, shortlist_only: bool):
    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("## Dashboard")
    st.markdown("<div class='explain'>This page gives a simple big-picture view. It shows how many applicants you have, where they are from, how many asked for scholarship support, whether they would still join without a scholarship, study-level mix, repeat applicants, and what a fair workload would look like if everyone on the RAUN team helped.</div>", unsafe_allow_html=True)

    planning = compute_planning_metrics(applicants_df, reviewers_df, shortlist_only=shortlist_only)
    applicants = max(planning["Applicants"], 1)
    reviewer_total = max(len(reviewers_df), 1)
    scholarship_pct = planning["Scholarship Applicants"] / applicants * 100 if applicants else 0
    pre_pct = planning["Preselection Demand"] / applicants * 100 if applicants else 0
    int_pct = planning["Interview Demand"] / applicants * 100 if applicants else 0
    active_pct = planning["Active Reviewers"] / reviewer_total * 100 if reviewer_total else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Applicants", planning["Applicants"], help="How many people are in the applicant file.")
        _metric_subtext("100% of the loaded applicant file")
    with c2:
        st.metric("Scholarship applicants", planning["Scholarship Applicants"], help="How many applicants answered yes to the scholarship question.")
        _metric_subtext(f"{scholarship_pct:.1f}% of all applicants")
    with c3:
        st.metric("Ideal preselection per reviewer", planning["Suggested Avg Preselection per Reviewer"], help="If active RAUN reviewers shared the work evenly, this is the average number of preselection reviews each person would take.")
        _metric_subtext(f"Preselection demand = {pre_pct:.1f}% of applicant count")
    with c4:
        st.metric("Ideal interview per reviewer", planning["Suggested Avg Interview per Reviewer"], help="If active reviewers shared interview work evenly, this is the average number each person would take.")
        _metric_subtext(f"Interview demand = {int_pct:.1f}% of applicant count")
    with c5:
        st.metric("Active reviewers", planning["Active Reviewers"], help="How many reviewers are currently switched on in the reviewer table.")
        _metric_subtext(f"{active_pct:.1f}% of the current reviewer table")

    st.markdown("### Applicant snapshot")
    display_cols = [c for c in ["Applicant ID", "Full Name", "Program", "University", "Field", "Country of Residence", "Country of Citizenship", "Applied for scholarship?", "Would still participate without scholarship?", "Graduate Track Applicant?", "Previously applied to RAUN?", "Previous interview invitation?", "Accessibility requirements?"] if c in applicants_df.columns]
    st.dataframe(applicants_df[display_cols], width="stretch", height=320)

    st.markdown("### Geography")
    m1, m2 = st.columns([1, 1])
    with m1:
        show_residence = st.checkbox("Show country of residence", value=True, help="Turn this on to count applicants by where they currently live.")
    with m2:
        show_citizenship = st.checkbox("Show country of citizenship", value=True, help="Turn this on to count applicants by citizenship. You can switch on both at the same time.")
    _country_map(applicants_df, show_residence, show_citizenship, key="dashboard_country_map")

    st.markdown("### Applicant composition")
    g1, g2, g3 = st.columns(3)
    with g1:
        _program_mix_chart(applicants_df, key="dashboard_program_mix")
    with g2:
        _yes_no_counts(applicants_df, "Applied for scholarship?", "Scholarship application", key="dashboard_scholarship_pie")
    with g3:
        _scholarship_subgroup_chart(applicants_df, "Applied for scholarship?", "Would still participate without scholarship?", key="dashboard_without_scholarship_subgroup")

    st.markdown("### Graduate track and previous application history")
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        _yes_no_counts(applicants_df, "Graduate Track Applicant?", "Graduate Track applicants", key="dashboard_grad_track_pie")
    with h2:
        _yes_no_counts(applicants_df, "Previously applied to RAUN?", "Repeat applicants", key="dashboard_repeat_pie")
    with h3:
        _yes_no_counts(applicants_df, "Previous interview invitation?", "Reached interview last time?", key="dashboard_prev_interview_pie")
    with h4:
        _yes_no_counts(applicants_df, "Accessibility requirements?", "Accessibility / support needs", key="dashboard_accessibility_pie")

    st.markdown("### Universities and fields")
    g1, g2 = st.columns(2)
    with g1:
        _distribution_chart(applicants_df, "University", "All universities", color=PLOTLY_PALETTE[0], key="dashboard_universities_bar")
    with g2:
        _distribution_chart(applicants_df, "Field", "All fields", color=PLOTLY_PALETTE[2], key="dashboard_fields_bar")

    w1, w2 = st.columns(2)
    with w1:
        _word_cloud(applicants_df, ["University", "University (All)"], "University cloud", stopwords=UNIVERSITY_STOPWORDS)
    with w2:
        _word_cloud(applicants_df, ["Field", "Field (All)", "Background Tags"], "Field and topic cloud", stopwords=FIELD_STOPWORDS)

    st.markdown('</div>', unsafe_allow_html=True)
