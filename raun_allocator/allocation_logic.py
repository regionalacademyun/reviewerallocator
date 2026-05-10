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
        "Laura María García": "economics, management, organization, industrial engineering, data analysis, education, public policy, mental health, public health, crime, corruption, labour, occupational safety, quantitative research",
        "Thi Hoang": "human trafficking, anti-human trafficking, organized crime, technology, cybercrime, migration, forced labour, modern slavery, child protection, crime policy, digital governance, criminal justice",
        "Roman Hoffmann": "economics, sociology, development, poverty, health, environment, climate change, migration, livelihoods, refugees, policy interventions, vulnerability, adaptation",
        "Mariia Kostetckaia (Masha)": "sustainability, sustainable development, european union, international relations, global studies, asylum, refugees, migration, policy",
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
        pre_base = pre_demand // n_reviewers
        pre_rem = pre_demand % n_reviewers
        df["Preselection Capacity"] = pre_base
        if pre_rem > 0:
            df.loc[df.index[:pre_rem], "Preselection Capacity"] += 1
    else:
        df["Preselection Capacity"] = 10

    df["Preselection Flexible"] = False
    df["Preselection Flex Limit"] = df["Preselection Capacity"]
    df["Background Tags"] = df["Reviewer Name"].map(tag_map).fillna("")
    df["Reviewer Notes"] = ""

    return df[[
        "Reviewer Name", "Active", "Preselection Capacity", "Preselection Flexible",
        "Preselection Flex Limit", "Background Tags", "Reviewer Notes",
    ]]


def parse_tags(value):
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return {x.strip().lower() for x in str(value).split(",") if x.strip()}


def reviewers_to_dict(reviewers_df: pd.DataFrame):
    """Convert reviewer availability rows into strict allocation limits.

    Rules:
    - Active=False means no allocation at all.
    - Preselection Capacity is the strict normal/base limit.
    - If Preselection Flexible=True, Preselection Flex Limit is the strict total maximum.
    - If Preselection Flexible=False, total maximum is exactly Preselection Capacity.
    - There is no assignment beyond the flex limit in this preselection app.
    """
    reviewers = {}
    for _, row in reviewers_df.iterrows():
        if not bool(row.get("Active", True)):
            continue

        name = safe_text(row.get("Reviewer Name"))
        if not name:
            continue

        pre_cap = max(0, int(row.get("Preselection Capacity", 0)))
        pre_flexible = bool(row.get("Preselection Flexible", False))
        raw_flex_limit = max(0, int(row.get("Preselection Flex Limit", pre_cap)))
        flex_limit = raw_flex_limit if pre_flexible else pre_cap
        if pre_flexible:
            flex_limit = max(pre_cap, flex_limit)

        reviewers[name] = {
            "pre_cap": pre_cap,
            "pre_assigned": 0,
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


def compute_planning_metrics(applicants_df: pd.DataFrame, reviewers_df: pd.DataFrame, shortlist_only=True):
    n_applicants = len(applicants_df)
    pre_demand = n_applicants * 2

    active_reviewers = reviewers_df[reviewers_df["Active"] == True].copy() if "Active" in reviewers_df.columns else reviewers_df.copy()
    n_active = len(active_reviewers)

    pre_cap = int(pd.to_numeric(active_reviewers.get("Preselection Capacity", 0), errors="coerce").fillna(0).sum())

    flex_extra = 0
    flex_total = pre_cap
    if "Preselection Flexible" in active_reviewers.columns:
        flex_mask = active_reviewers["Preselection Flexible"] == True
        base_flexible = int(pd.to_numeric(active_reviewers.loc[flex_mask, "Preselection Capacity"], errors="coerce").fillna(0).sum()) if flex_mask.any() else 0
        limit_flexible = int(pd.to_numeric(active_reviewers.loc[flex_mask, "Preselection Flex Limit"], errors="coerce").fillna(0).sum()) if flex_mask.any() else 0
        flex_extra = max(0, limit_flexible - base_flexible)
        flex_total = pre_cap + flex_extra

    avg_pre_needed = math.ceil(pre_demand / n_active) if n_active > 0 else 0
    scholarship_count = 0
    if "Applied for scholarship?" in applicants_df.columns:
        scholarship_count = int(
            applicants_df["Applied for scholarship?"]
            .astype(str).str.strip().str.lower().isin(YES_VALUES).sum()
        )

    return {
        "Applicants": n_applicants,
        "Active Reviewers": n_active,
        "Preselection Demand": pre_demand,
        "Preselection Capacity": pre_cap,
        "Preselection Flex Extra": flex_extra,
        "Preselection Capacity With Flex": flex_total,
        "Preselection Gap": pre_cap - pre_demand,
        "Preselection Gap With Flex": flex_total - pre_demand,
        "Preselection Shortage With Flex": max(0, pre_demand - flex_total),
        "Suggested Avg Preselection per Reviewer": avg_pre_needed,
        "Scholarship Applicants": scholarship_count,
    }


def _pick_from_candidates(candidates, selected, needed):
    chosen = []
    for name, _score in candidates:
        if name not in selected and name not in chosen:
            chosen.append(name)
        if len(chosen) >= needed:
            break
    return chosen


def allocate_preselection(applicants_df, reviewers_df, seed=42, match_strength="Low"):
    rng = random.Random(seed)
    reviewers = reviewers_to_dict(reviewers_df)
    allocations = []
    exceptions = []

    if len(reviewers) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame([{
            "Applicant ID": "",
            "Full Name": "",
            "Issue Type": "No active reviewers",
            "Details": "No allocation was possible because no reviewer is active.",
        }])

    shuffled = applicants_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    for _, row in shuffled.iterrows():
        aid = int(row["Applicant ID"])
        full_name = row["Full Name"]
        candidate_tags = parse_tags(row.get("Background Tags", ""))
        selected = []
        phase_used = []

        # Phase 1: strict normal/base capacity only.
        base_candidates = []
        for name, rv in reviewers.items():
            if rv["pre_assigned"] < rv["pre_cap"]:
                score = score_pair(candidate_tags, rv["tags"], rv["pre_assigned"], max(rv["pre_cap"], 1), rng, match_strength)
                base_candidates.append((name, score))
        base_candidates = sorted(base_candidates, key=lambda x: x[1], reverse=True)
        for ch in _pick_from_candidates(base_candidates, selected, 2 - len(selected)):
            selected.append(ch)
            phase_used.append("Base capacity")

        # Phase 2: declared flexible capacity only, and only up to flex limit.
        if len(selected) < 2:
            flex_candidates = []
            for name, rv in reviewers.items():
                if name in selected:
                    continue
                if rv["pre_flexible"] and rv["pre_assigned"] < rv["pre_flex_limit"]:
                    flex_limit = max(rv["pre_flex_limit"], 1)
                    over_base = max(rv["pre_assigned"] - rv["pre_cap"], 0)
                    score = -((rv["pre_assigned"] / flex_limit) * 2.0 + over_base * 0.5) + rng.uniform(0, 0.15)
                    flex_candidates.append((name, score))
            flex_candidates = sorted(flex_candidates, key=lambda x: x[1], reverse=True)
            for ch in _pick_from_candidates(flex_candidates, selected, 2 - len(selected)):
                selected.append(ch)
                phase_used.append("Flexible capacity")

        r1 = selected[0] if len(selected) >= 1 else ""
        r2 = selected[1] if len(selected) >= 2 else ""

        if r1:
            reviewers[r1]["pre_assigned"] += 1
        if r2:
            reviewers[r2]["pre_assigned"] += 1

        used_flex = "Flexible capacity" in phase_used
        incomplete = len(selected) < 2

        if used_flex:
            exceptions.append({
                "Applicant ID": aid,
                "Full Name": full_name,
                "Issue Type": "Flexible capacity used",
                "Details": "At least one reviewer was assigned above base capacity, but still within their declared flex limit.",
            })
        if incomplete:
            exceptions.append({
                "Applicant ID": aid,
                "Full Name": full_name,
                "Issue Type": "Incomplete allocation",
                "Details": "Could not assign two distinct reviewers without exceeding the active reviewers' base/flex limits. Increase reviewer capacity or flex limits, or add more active reviewers.",
            })

        allocations.append({
            "Applicant ID": aid,
            "Full Name": full_name,
            "Program": row.get("Program", ""),
            "University": row.get("University", ""),
            "Field": row.get("Field", ""),
            "Preselection Reviewer 1": r1,
            "Preselection Reviewer 2": r2,
            "Used Flexible Capacity": used_flex,
            "Incomplete Allocation": incomplete,
        })

    alloc_df = pd.DataFrame(allocations).sort_values("Applicant ID").reset_index(drop=True)

    load_rows = []
    for name, rv in reviewers.items():
        over_base = max(rv["pre_assigned"] - rv["pre_cap"], 0)
        over_flex = max(rv["pre_assigned"] - rv["pre_flex_limit"], 0)
        util_base = round(rv["pre_assigned"] / rv["pre_cap"] * 100, 1) if rv["pre_cap"] > 0 else (100.0 if rv["pre_assigned"] > 0 else 0.0)
        util_flex = round(rv["pre_assigned"] / rv["pre_flex_limit"] * 100, 1) if rv["pre_flex_limit"] > 0 else (100.0 if rv["pre_assigned"] > 0 else 0.0)
        load_rows.append({
            "Reviewer Name": name,
            "Preselection Assigned": rv["pre_assigned"],
            "Preselection Capacity": rv["pre_cap"],
            "Preselection Flexible": rv["pre_flexible"],
            "Preselection Flex Limit": rv["pre_flex_limit"],
            "Remaining Base Capacity": rv["pre_cap"] - rv["pre_assigned"],
            "Remaining To Flex Limit": rv["pre_flex_limit"] - rv["pre_assigned"],
            "Over Capacity By": over_base,
            "Over Flex Limit By": over_flex,
            "Utilization % Base": util_base,
            "Utilization % Flex": util_flex,
        })

    loads_df = pd.DataFrame(load_rows).sort_values(["Preselection Assigned", "Reviewer Name"], ascending=[False, True]).reset_index(drop=True)

    # Add a clear top-level shortage warning when demand is impossible under declared limits.
    total_capacity_with_flex = sum(rv["pre_flex_limit"] for rv in reviewers.values())
    demand = len(applicants_df) * 2
    if total_capacity_with_flex < demand:
        exceptions = [{
            "Applicant ID": "",
            "Full Name": "SYSTEM",
            "Issue Type": "Capacity shortage",
            "Details": f"Demand is {demand} reviews, but active reviewers can provide only {total_capacity_with_flex} reviews including declared flexibility. {demand - total_capacity_with_flex} review slots remain impossible under the current availability file.",
        }] + exceptions

    return alloc_df, loads_df, pd.DataFrame(exceptions)


def combined_workload_stats(preloads_df, intloads_df=None):
    if preloads_df is None or preloads_df.empty:
        return pd.DataFrame()
    out = preloads_df.copy()
    out["Total Assigned"] = out["Preselection Assigned"]
    out["Total Base Capacity"] = out["Preselection Capacity"]
    out["Total Flex Capacity"] = out["Preselection Flex Limit"]
    out["Total Utilization % (Base)"] = out["Utilization % Base"]
    out["Total Utilization % (With Flex)"] = out["Utilization % Flex"]
    return out.sort_values(["Total Assigned", "Reviewer Name"], ascending=[False, True]).reset_index(drop=True)
