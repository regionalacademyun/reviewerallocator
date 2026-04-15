import math
import random
import unicodedata
from typing import List

import pandas as pd

from .data_io import safe_text

YES_VALUES = {"yes", "y", "true", "1"}


def _normalize_name(name: str) -> str:
    return unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("utf-8").lower()


def build_default_reviewers_df(
    team_members: List[str],
    applicants_df: pd.DataFrame | None = None,
    shortlist_only: bool = True,
    use_uniform_baseline: bool = False,
) -> pd.DataFrame:
    tag_map = {
        "Beatrice Contino": "global politics, climate migration, migration, human rights, rule of law, democratic principles, development aid, global governance, international organizations, multilingual",
        "Laura María García": "economics, management, organization, industrial engineering, data analysis, education, public policy, mental health, public health, crime, corruption, labour, occupational safety, quantitative research",
        "Thi Hoang": "human trafficking, anti-human trafficking, organized crime, technology, cybercrime, migration, forced labour, modern slavery, child protection, crime policy, digital governance, criminal justice",
        "Roman Hoffmann": "economics, sociology, development, poverty, health, environment, climate change, migration, livelihoods, refugees, policy interventions, vulnerability, adaptation",
        "Mariia Kostetckaia (Masha)": "sustainability, sustainable development, european union, international relations, global studies, asylum, refugees, migration, policy",
        "Alice Mazzetto": "international relations, migration, refugees, mediterranean governance, mental health, underrepresented groups, eu policy, volunteer work",
        "Samar Momin": "engineering, earthquake engineering, structural dynamics, disaster risk reduction, resilience, seismic risk, public health, contaminated medicines, regulation, data analysis, infrastructure, risk assessment",
        "Ghinwa Moujaes": "data science, generative ai, large language models, asset management, investment, innovation policy, economic geography, local economic development, methodology, evaluation",
        "Florian Müller": "economics, business, international management, renewable energy, energy policy, public investment, management",
        "Ivy Omondi": "peace and security, conflict, refugees, gender, lgbtq, trafficking in persons, diplomacy, victim-centred approach",
        "Isabel Sáenz Hernández": "sociology, migration, multilingualism, education, inequality, immigrant background, language, inclusion, accessibility, digital inequality, social demography, human rights",
        "Berkay Öztürk": "migration, diaspora, remittances, identity, policy, global governance, youth inclusion, security, international relations, civil society, refugee participation, sport",
        "Martina Pardy": "economics, economic geography, inequality, globalization, regional development, migration, women’s participation, electoral reform, governance",
        "Mary Peloche": "law, political science, refugee protection, migration policy, national security, public administration, international relations, governance",
        "Nicola Jansen": "law",
        "Cecilia Vera Lagomarsino": "international relations, media freedom, osce, migration, migration law, gender equality, elections, democratic governance, european affairs, human rights",
        "Billy Batware": "transnational organized crime, unodc, international security, conflict analysis, development, human rights, sustainable development, youth empowerment, education, leadership, diplomacy, international relations",
    }

    members = sorted([safe_text(x) for x in team_members if safe_text(x)], key=_normalize_name)
    df = pd.DataFrame({"Reviewer Name": members})
    df["Active"] = True

    if use_uniform_baseline and applicants_df is not None and not applicants_df.empty and len(df) > 0:
        n_reviewers = len(df)

        pre_demand = len(applicants_df) * 2
        int_demand = compute_interview_demand(applicants_df, shortlist_only=shortlist_only)

        pre_base = pre_demand // n_reviewers
        pre_rem = pre_demand % n_reviewers

        int_base = int_demand // n_reviewers
        int_rem = int_demand % n_reviewers

        df["Preselection Capacity"] = pre_base
        if pre_rem > 0:
            df.loc[df.index[:pre_rem], "Preselection Capacity"] += 1

        df["Interview Capacity"] = int_base
        if int_rem > 0:
            df.loc[df.index[:int_rem], "Interview Capacity"] += 1
    else:
        df["Preselection Capacity"] = 10
        df["Interview Capacity"] = 5

    df["Preselection Flexible"] = False
    df["Preselection Flex Limit"] = df["Preselection Capacity"]
    df["Background Tags"] = df["Reviewer Name"].map(tag_map).fillna("")

    return df[
        [
            "Reviewer Name",
            "Active",
            "Preselection Capacity",
            "Interview Capacity",
            "Preselection Flexible",
            "Preselection Flex Limit",
            "Background Tags",
        ]
    ]


def parse_tags(value):
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return {x.strip().lower() for x in str(value).split(",") if x.strip()}


def reviewers_to_dict(reviewers_df: pd.DataFrame):
    reviewers = {}
    for _, row in reviewers_df.iterrows():
        if not bool(row.get("Active", True)):
            continue

        name = safe_text(row.get("Reviewer Name"))
        if not name:
            continue

        pre_cap = max(0, int(row.get("Preselection Capacity", 0)))
        int_cap = max(0, int(row.get("Interview Capacity", 0)))
        pre_flexible = bool(row.get("Preselection Flexible", False))
        flex_limit = max(pre_cap, int(row.get("Preselection Flex Limit", pre_cap))) if pre_flexible else pre_cap

        reviewers[name] = {
            "pre_cap": pre_cap,
            "pre_assigned": 0,
            "int_cap": int_cap,
            "int_assigned": 0,
            "pre_flexible": pre_flexible,
            "pre_flex_limit": flex_limit,
            "tags": parse_tags(row.get("Background Tags", "")),
        }
    return reviewers


def match_bonus(overlap_count: int, strength: str) -> float:
    if strength == "Off":
        return 0.0
    if strength == "Low":
        return overlap_count * 0.75
    if strength == "Medium":
        return overlap_count * 1.5
    return overlap_count * 0.75


def score_pair(candidate_tags, reviewer_tags, assigned_count, capacity, rng, strength="Low"):
    overlap = len(candidate_tags.intersection(reviewer_tags))
    bonus = match_bonus(overlap, strength)
    load_ratio = assigned_count / capacity if capacity > 0 else assigned_count
    noise = rng.uniform(0, 0.15)
    return bonus - (load_ratio * 2.0) + noise


def compute_interview_demand(applicants_df: pd.DataFrame, shortlist_only=True):
    if applicants_df is None or applicants_df.empty:
        return 0

    if shortlist_only:
        if "Shortlisted" in applicants_df.columns:
            shortlisted = applicants_df["Shortlisted"].astype(str).str.strip().str.lower().isin(YES_VALUES)
            if shortlisted.any():
                return int(shortlisted.sum())

        if "Interview Required" in applicants_df.columns:
            required = applicants_df["Interview Required"].astype(str).str.strip().str.lower().isin(YES_VALUES)
            if required.any():
                return int(required.sum())

        return len(applicants_df)

    return len(applicants_df)


def compute_planning_metrics(applicants_df: pd.DataFrame, reviewers_df: pd.DataFrame, shortlist_only=True):
    n_applicants = len(applicants_df)
    pre_demand = n_applicants * 2
    interview_needed = compute_interview_demand(applicants_df, shortlist_only=shortlist_only)

    active_reviewers = reviewers_df[reviewers_df["Active"] == True].copy()
    n_active = len(active_reviewers)

    pre_cap = int(pd.to_numeric(active_reviewers["Preselection Capacity"], errors="coerce").fillna(0).sum())
    int_cap = int(pd.to_numeric(active_reviewers["Interview Capacity"], errors="coerce").fillna(0).sum())

    flex_only_extra = 0
    if "Preselection Flexible" in active_reviewers.columns:
        flex_mask = active_reviewers["Preselection Flexible"] == True
        base_sum = int(pd.to_numeric(active_reviewers.loc[flex_mask, "Preselection Capacity"], errors="coerce").fillna(0).sum())
        limit_sum = int(pd.to_numeric(active_reviewers.loc[flex_mask, "Preselection Flex Limit"], errors="coerce").fillna(0).sum())
        flex_only_extra = max(0, limit_sum - base_sum)

    pre_gap = pre_cap - pre_demand
    pre_gap_with_flex = (pre_cap + flex_only_extra) - pre_demand
    int_gap = int_cap - interview_needed

    avg_pre_needed = math.ceil(pre_demand / n_active) if n_active > 0 else 0
    avg_int_needed = math.ceil(interview_needed / n_active) if n_active > 0 else 0

    scholarship_count = 0
    if "Applied for scholarship?" in applicants_df.columns:
        scholarship_count = int(
            applicants_df["Applied for scholarship?"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(YES_VALUES)
            .sum()
        )

    return {
        "Applicants": n_applicants,
        "Active Reviewers": n_active,
        "Preselection Demand": pre_demand,
        "Interview Demand": interview_needed,
        "Preselection Capacity": pre_cap,
        "Interview Capacity": int_cap,
        "Preselection Flex Extra": flex_only_extra,
        "Preselection Capacity With Flex": pre_cap + flex_only_extra,
        "Preselection Gap": pre_gap,
        "Preselection Gap With Flex": pre_gap_with_flex,
        "Interview Gap": int_gap,
        "Suggested Avg Preselection per Reviewer": avg_pre_needed,
        "Suggested Avg Interview per Reviewer": avg_int_needed,
        "Scholarship Applicants": scholarship_count,
    }


def allocate_preselection(applicants_df, reviewers_df, seed=42, match_strength="Low"):
    rng = random.Random(seed)
    reviewers = reviewers_to_dict(reviewers_df)

    allocations = []
    exceptions = []

    shuffled = applicants_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    for _, row in shuffled.iterrows():
        aid = int(row["Applicant ID"])
        full_name = row["Full Name"]
        candidate_tags = parse_tags(row.get("Background Tags", ""))

        selected = []
        used_flexible = []
        used_emergency = []

        eligible = []
        for name, rv in reviewers.items():
            if rv["pre_assigned"] < rv["pre_cap"]:
                s = score_pair(candidate_tags, rv["tags"], rv["pre_assigned"], max(rv["pre_cap"], 1), rng, match_strength)
                eligible.append((name, s))
        eligible = sorted(eligible, key=lambda x: x[1], reverse=True)

        for name, _ in eligible:
            if name not in selected and len(selected) < 2:
                selected.append(name)

        if len(selected) < 2:
            fallback = []
            for name, rv in reviewers.items():
                if name not in selected and rv["pre_assigned"] < rv["pre_cap"]:
                    cap = max(rv["pre_cap"], 1)
                    s = -(rv["pre_assigned"] / cap * 2.0) + rng.uniform(0, 0.15)
                    fallback.append((name, s))
            fallback = sorted(fallback, key=lambda x: x[1], reverse=True)

            while len(selected) < 2 and fallback:
                selected.append(fallback.pop(0)[0])

        if len(selected) < 2:
            flex = []
            for name, rv in reviewers.items():
                if name not in selected and rv["pre_flexible"] and rv["pre_assigned"] < rv["pre_flex_limit"]:
                    flex_limit = max(rv["pre_flex_limit"], 1)
                    overload = max(rv["pre_assigned"] - rv["pre_cap"], 0)
                    s = -((rv["pre_assigned"] / flex_limit) * 2.0 + overload * 0.5) + rng.uniform(0, 0.15)
                    flex.append((name, s))
            flex = sorted(flex, key=lambda x: x[1], reverse=True)

            while len(selected) < 2 and flex:
                chosen = flex.pop(0)[0]
                selected.append(chosen)
                used_flexible.append(chosen)

        if len(selected) < 2:
            emergency = []
            for name, rv in reviewers.items():
                if name not in selected and rv["pre_flexible"]:
                    over_flex = max(rv["pre_assigned"] - rv["pre_flex_limit"], 0)
                    over_base = max(rv["pre_assigned"] - rv["pre_cap"], 0)
                    penalty = (rv["pre_assigned"] * 0.25) + (over_base * 0.8) + (over_flex * 1.2)
                    s = -penalty + rng.uniform(0, 0.15)
                    emergency.append((name, s))
            emergency = sorted(emergency, key=lambda x: x[1], reverse=True)

            while len(selected) < 2 and emergency:
                chosen = emergency.pop(0)[0]
                selected.append(chosen)
                used_emergency.append(chosen)

        if len(selected) < 2:
            last = []
            for name, rv in reviewers.items():
                if name not in selected:
                    s = -(rv["pre_assigned"] * 0.5) + rng.uniform(0, 0.15)
                    last.append((name, s))
            last = sorted(last, key=lambda x: x[1], reverse=True)

            while len(selected) < 2 and last:
                chosen = last.pop(0)[0]
                selected.append(chosen)
                used_emergency.append(chosen)

        r1 = selected[0] if len(selected) >= 1 else None
        r2 = selected[1] if len(selected) >= 2 else None

        if r1:
            reviewers[r1]["pre_assigned"] += 1
        if r2:
            reviewers[r2]["pre_assigned"] += 1

        if used_flexible:
            exceptions.append({
                "Applicant ID": aid,
                "Full Name": full_name,
                "Issue Type": "Flexible fallback used",
                "Details": f"Flexible reviewer(s): {', '.join(used_flexible)}",
            })
        if used_emergency:
            exceptions.append({
                "Applicant ID": aid,
                "Full Name": full_name,
                "Issue Type": "Emergency overflow used",
                "Details": f"Emergency reviewer(s): {', '.join(used_emergency)}",
            })

        allocations.append({
            "Applicant ID": aid,
            "Full Name": full_name,
            "Program": row.get("Program", ""),
            "University": row.get("University", ""),
            "Field": row.get("Field", ""),
            "Preselection Reviewer 1": r1,
            "Preselection Reviewer 2": r2,
            "Used Flexible Reviewer": bool(used_flexible),
            "Used Emergency Overflow": bool(used_emergency),
        })

    alloc_df = pd.DataFrame(allocations).sort_values("Applicant ID").reset_index(drop=True) if allocations else pd.DataFrame()

    load_rows = []
    for name, rv in reviewers.items():
        overload = max(rv["pre_assigned"] - rv["pre_cap"], 0)
        over_flex = max(rv["pre_assigned"] - rv["pre_flex_limit"], 0)
        util = round((rv["pre_assigned"] / rv["pre_cap"] * 100), 1) if rv["pre_cap"] > 0 else 0
        load_rows.append({
            "Reviewer Name": name,
            "Preselection Assigned": rv["pre_assigned"],
            "Preselection Capacity": rv["pre_cap"],
            "Preselection Flexible": rv["pre_flexible"],
            "Preselection Flex Limit": rv["pre_flex_limit"],
            "Remaining Preselection Capacity": rv["pre_cap"] - rv["pre_assigned"],
            "Remaining To Flex Limit": rv["pre_flex_limit"] - rv["pre_assigned"],
            "Over Capacity By": overload,
            "Over Flex Limit By": over_flex,
            "Preselection Utilization %": util,
        })

    loads_df = pd.DataFrame(load_rows).sort_values(["Preselection Assigned", "Reviewer Name"], ascending=[False, True]).reset_index(drop=True) if load_rows else pd.DataFrame()

    return alloc_df, loads_df, pd.DataFrame(exceptions)


def allocate_interviews(
    applicants_df,
    prealloc_df,
    reviewers_df,
    seed=42,
    match_strength="Low",
    allow_same_reviewer_exception=True,
    shortlist_only=True,
):
    rng = random.Random(seed + 1000)
    reviewers = reviewers_to_dict(reviewers_df)

    for name in reviewers:
        reviewers[name]["int_assigned"] = 0

    work_df = applicants_df.copy()

    if shortlist_only:
        if "Shortlisted" in work_df.columns:
            work_df = work_df[work_df["Shortlisted"].astype(str).str.strip().str.lower().isin(YES_VALUES)]
        elif "Interview Required" in work_df.columns:
            work_df = work_df[work_df["Interview Required"].astype(str).str.strip().str.lower().isin(YES_VALUES)]

    merged = pd.merge(
        work_df,
        prealloc_df[["Applicant ID", "Preselection Reviewer 1", "Preselection Reviewer 2"]],
        on="Applicant ID",
        how="left",
    )

    results = []
    exceptions = []

    for _, row in merged.iterrows():
        aid = int(row["Applicant ID"])
        full_name = row["Full Name"]
        candidate_tags = parse_tags(row.get("Background Tags", ""))

        forbidden = {row.get("Preselection Reviewer 1"), row.get("Preselection Reviewer 2")}
        eligible = []

        for name, rv in reviewers.items():
            if rv["int_assigned"] < rv["int_cap"] and name not in forbidden:
                s = score_pair(candidate_tags, rv["tags"], rv["int_assigned"], max(rv["int_cap"], 1), rng, match_strength)
                eligible.append((name, s))
        eligible = sorted(eligible, key=lambda x: x[1], reverse=True)

        interviewer = None
        used_exception = False
        reason = ""

        if eligible:
            interviewer = eligible[0][0]
        elif allow_same_reviewer_exception:
            fallback = []
            for name, rv in reviewers.items():
                if rv["int_assigned"] < rv["int_cap"]:
                    s = score_pair(candidate_tags, rv["tags"], rv["int_assigned"], max(rv["int_cap"], 1), rng, match_strength)
                    fallback.append((name, s))
            fallback = sorted(fallback, key=lambda x: x[1], reverse=True)

            if fallback:
                interviewer = fallback[0][0]
                if interviewer in forbidden:
                    used_exception = True
                    reason = "Interview reviewer matched one of the preselection reviewers because no alternative reviewer had remaining interview capacity."
            else:
                reason = "No interview reviewer available."
        else:
            reason = "No eligible non-preselection interview reviewer available."

        if interviewer:
            reviewers[interviewer]["int_assigned"] += 1

        if reason:
            exceptions.append({
                "Applicant ID": aid,
                "Full Name": full_name,
                "Issue Type": "Interview allocation exception",
                "Details": reason,
            })

        results.append({
            "Applicant ID": aid,
            "Full Name": full_name,
            "Preselection Reviewer 1": row.get("Preselection Reviewer 1"),
            "Preselection Reviewer 2": row.get("Preselection Reviewer 2"),
            "Interview Reviewer": interviewer,
            "Interview Conflict Exception": used_exception,
            "Program": row.get("Program", ""),
            "University": row.get("University", ""),
            "Field": row.get("Field", ""),
        })

    result_df = pd.DataFrame(results).sort_values("Applicant ID").reset_index(drop=True) if results else pd.DataFrame()

    load_rows = []
    for name, rv in reviewers.items():
        util = round((rv["int_assigned"] / rv["int_cap"] * 100), 1) if rv["int_cap"] > 0 else 0
        load_rows.append({
            "Reviewer Name": name,
            "Interview Assigned": rv["int_assigned"],
            "Interview Capacity": rv["int_cap"],
            "Remaining Interview Capacity": rv["int_cap"] - rv["int_assigned"],
            "Interview Utilization %": util,
        })

    loads_df = pd.DataFrame(load_rows).sort_values(["Interview Assigned", "Reviewer Name"], ascending=[False, True]).reset_index(drop=True) if load_rows else pd.DataFrame()

    return result_df, loads_df, pd.DataFrame(exceptions)


def combined_workload_stats(preloads_df, intloads_df):
    if preloads_df is None or preloads_df.empty:
        preloads_df = pd.DataFrame(columns=[
            "Reviewer Name", "Preselection Assigned", "Preselection Capacity",
            "Preselection Flex Limit", "Over Capacity By", "Over Flex Limit By"
        ])
    if intloads_df is None or intloads_df.empty:
        intloads_df = pd.DataFrame(columns=[
            "Reviewer Name", "Interview Assigned", "Interview Capacity"
        ])

    combined = pd.merge(
        preloads_df[[
            "Reviewer Name", "Preselection Assigned", "Preselection Capacity",
            "Preselection Flex Limit", "Over Capacity By", "Over Flex Limit By"
        ]],
        intloads_df[[
            "Reviewer Name", "Interview Assigned", "Interview Capacity"
        ]],
        on="Reviewer Name",
        how="outer",
    ).fillna(0)

    for col in [
        "Preselection Assigned", "Preselection Capacity", "Preselection Flex Limit",
        "Over Capacity By", "Over Flex Limit By", "Interview Assigned", "Interview Capacity"
    ]:
        combined[col] = combined[col].astype(int)

    combined["Total Assigned"] = combined["Preselection Assigned"] + combined["Interview Assigned"]
    combined["Total Base Capacity"] = combined["Preselection Capacity"] + combined["Interview Capacity"]
    combined["Total Flex Capacity"] = combined["Preselection Flex Limit"] + combined["Interview Capacity"]

    combined["Total Utilization % (Base)"] = combined.apply(
        lambda r: round((r["Total Assigned"] / r["Total Base Capacity"] * 100), 1)
        if r["Total Base Capacity"] > 0 else 0,
        axis=1,
    )
    combined["Total Utilization % (With Flex)"] = combined.apply(
        lambda r: round((r["Total Assigned"] / r["Total Flex Capacity"] * 100), 1)
        if r["Total Flex Capacity"] > 0 else 0,
        axis=1,
    )

    return combined.sort_values(
        ["Total Assigned", "Reviewer Name"],
        ascending=[False, True],
    ).reset_index(drop=True)
