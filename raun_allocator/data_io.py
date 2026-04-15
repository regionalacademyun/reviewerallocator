import io
import re
from typing import Dict, List

import pandas as pd


def safe_text(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def sanitize_filename(text: str) -> str:
    text = re.sub(r"\s+", " ", safe_text(text))
    text = re.sub(r"[^\w\-. ]+", "", text, flags=re.UNICODE)
    return text.replace(" ", "_")


COLUMN_ALIASES: Dict[str, str] = {
    " Country of citizenship": "Country of citizenship",
    "Last Name(s)": "Surname",
    "Last name(s)": "Surname",
    "University of current enrolment (If more than one affiliation, please add below)": "University",
    "University of current enrolment (If more than one affiliation, please add below)": "University",
    "University of current enrolment": "University",
    "Current level of study (or equivalent)": "Current level of study",
    "Field of study (If more than one, please add below)": "Major/Field",
    "Field of study": "Major/Field",
    "Would you like to apply for a scholarship (tuition waiver)?": "Applied for scholarship?",
    "Would you like to apply for a scholarship (tuition waiver)? (Please apply only if you have NO other means to finance your participation)": "Applied for scholarship?",
    "Would you still be willing to participate in the Academy if you do NOT receive the scholarship?": "Would still participate without scholarship?",
    "Do you have any accessibility requirements or special needs that we should be aware of to support your participation in the RAUN programme?": "Accessibility requirements?",
    "If yes, please briefly describe your requirements": "Accessibility details",
    "Current professional status": "Current professional status",
    "Please briefly explain your motivation for applying under the Graduate Track and how you believe your experience adds value to the cohort.": "Graduate Track motivation",
    "Have you previously applied to the RAUN programme?": "Previously applied to RAUN?",
    "If yes, in which year(s) did you previously apply?": "Previous application years",
    "If yes, were you invited to an interview during your previous application?": "Previous interview invitation?",
    "Highest degree completed": "Highest degree completed",
    "Year of graduation (DD-MM-YYYY)": "Year of graduation",
}


def _bool_from_text(v, default=False):
    if pd.isna(v):
        return default
    s = str(v).strip().lower()
    if s in {"true", "yes", "y", "1"}:
        return True
    if s in {"false", "no", "n", "0"}:
        return False
    return default


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [safe_text(c) for c in df.columns]
    return df.rename(columns=COLUMN_ALIASES)


def _combine_nonempty(row: pd.Series, cols: List[str]) -> str:
    parts = [safe_text(row.get(c)) for c in cols if safe_text(row.get(c))]
    return " | ".join(parts)


def _normalize_applicant_df(df: pd.DataFrame) -> pd.DataFrame:
    df = _rename_columns(df)

    needed = [
        "First name(s)", "Middle name(s)", "Surname", "Gender", "Date of birth", "Age",
        "Country of residence", "Country of citizenship", "E-Mail", "Phone number with country code",
        "Are you currently enrolled in a Master or PhD program?", "MA/PHD", "Current level of study", "University",
        "University (Second affiliation, if applicable)", "University (Third affiliation, if applicable)",
        "Major/Field", "Field of study (2nd field of study, if applicable)", "Field of study (3rd field of study, if applicable)",
        "Area 1", "Area 2", "Area 3", "Applied for scholarship?",
        "Would still participate without scholarship?", "Accessibility requirements?", "Accessibility details",
        "Current professional status", "Graduate Track motivation", "Previously applied to RAUN?",
        "Previous application years", "Previous interview invitation?", "Highest degree completed", "Year of graduation",
    ]
    for col in needed:
        if col not in df.columns:
            df[col] = ""

    df["Applicant ID"] = range(1, len(df) + 1)
    df["Full Name"] = (
        df["First name(s)"].fillna("").astype(str) + " " +
        df["Middle name(s)"].fillna("").astype(str) + " " +
        df["Surname"].fillna("").astype(str)
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    # Program handling
    enrolled = df["Are you currently enrolled in a Master or PhD program?"].fillna("").astype(str).str.strip()
    current_level = df["Current level of study"].fillna("").astype(str).str.strip()
    ma_phd = df["MA/PHD"].fillna("").astype(str).str.strip()
    highest = df["Highest degree completed"].fillna("").astype(str).str.strip()

    program = ma_phd.where(ma_phd != "", current_level)
    graduate_track_mask = enrolled.str.contains("graduate track", case=False, na=False)
    program = program.where(program != "", highest)
    program = program.where(program != "", enrolled)
    program = program.replace({"I applied and I am waiting for the acceptance": "Pending acceptance"})
    program = program.astype(str).str.strip()

    df["Graduate Track Applicant?"] = graduate_track_mask.map({True: "Yes", False: "No"})
    df["Program"] = program
    df["Email"] = df["E-Mail"].fillna("").astype(str).str.strip()
    df["Country of Residence"] = df["Country of residence"].fillna("").astype(str).str.strip()
    df["Country of Citizenship"] = df["Country of citizenship"].fillna("").astype(str).str.strip()
    df["Country of Citizenship (All)"] = df.apply(
        lambda r: _combine_nonempty(r, [
            "Country of citizenship",
            "Country of citizenship (Second, if applicable)",
            "Country of citizenship (Third, if applicable)",
        ]),
        axis=1,
    ) if "Country of citizenship (Second, if applicable)" in df.columns else df["Country of Citizenship"]

    df["Field"] = df["Major/Field"].fillna("").astype(str).str.strip()
    df["Field (All)"] = df.apply(
        lambda r: _combine_nonempty(r, [
            "Major/Field",
            "Field of study (2nd field of study, if applicable)",
            "Field of study (3rd field of study, if applicable)",
        ]),
        axis=1,
    )
    df["University"] = df["University"].fillna("").astype(str).str.strip()
    df["University (All)"] = df.apply(
        lambda r: _combine_nonempty(r, [
            "University",
            "University (Second affiliation, if applicable)",
            "University (Third affiliation, if applicable)",
        ]),
        axis=1,
    )
    df["Applied for scholarship?"] = df["Applied for scholarship?"].fillna("").astype(str).str.strip()
    df["Would still participate without scholarship?"] = df["Would still participate without scholarship?"].fillna("").astype(str).str.strip()
    df["Accessibility requirements?"] = df["Accessibility requirements?"].fillna("").astype(str).str.strip()
    df["Accessibility details"] = df["Accessibility details"].fillna("").astype(str).str.strip()
    df["Current professional status"] = df["Current professional status"].fillna("").astype(str).str.strip()
    df["Previously applied to RAUN?"] = df["Previously applied to RAUN?"].fillna("").astype(str).str.strip()
    df["Previous application years"] = df["Previous application years"].fillna("").astype(str).str.strip()
    df["Previous interview invitation?"] = df["Previous interview invitation?"].fillna("").astype(str).str.strip()
    df["Highest degree completed"] = df["Highest degree completed"].fillna("").astype(str).str.strip()
    df["Year of graduation"] = df["Year of graduation"].fillna("").astype(str).str.strip()

    df["Background Tags"] = (
        df["Field"].fillna("").astype(str).str.strip() + ", " +
        df["Area 1"].fillna("").astype(str).str.strip() + ", " +
        df["Area 2"].fillna("").astype(str).str.strip() + ", " +
        df["Area 3"].fillna("").astype(str).str.strip()
    ).str.strip(", ").str.replace(r",\s*,", ",", regex=True)

    if "Interview Required" not in df.columns:
        df["Interview Required"] = "Yes"
    if "Shortlisted" not in df.columns:
        df["Shortlisted"] = ""

    keep_cols = [
        "Applicant ID", "Full Name", "Program", "University", "University (All)", "Field", "Field (All)", "Gender", "Age",
        "Country of Residence", "Country of Citizenship", "Country of Citizenship (All)", "Email",
        "Applied for scholarship?", "Would still participate without scholarship?",
        "Accessibility requirements?", "Accessibility details",
        "Graduate Track Applicant?", "Current professional status", "Highest degree completed", "Year of graduation",
        "Previously applied to RAUN?", "Previous application years", "Previous interview invitation?",
        "Background Tags", "Interview Required", "Shortlisted"
    ]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = ""
    out = df[keep_cols].copy()
    out = out[out["Full Name"].astype(str).str.strip() != ""].reset_index(drop=True)
    out["Applicant ID"] = range(1, len(out) + 1)
    return out


def load_applicants(uploaded_file) -> pd.DataFrame:
    attempts = [2, 1, 0]
    for header in attempts:
        try:
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file, header=header)
            norm = _normalize_applicant_df(df)
            if not norm.empty and norm["Full Name"].astype(str).str.strip().ne("").any():
                return norm
        except Exception:
            continue
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    df = pd.read_excel(uploaded_file)
    return _normalize_applicant_df(df)


def normalize_reviewers_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [safe_text(c) for c in df.columns]
    rename_map = {
        "Reviewer": "Reviewer Name",
        "Name": "Reviewer Name",
        "Flex Limit": "Preselection Flex Limit",
        "Preselection Flexible?": "Preselection Flexible",
    }
    df = df.rename(columns=rename_map)
    needed = {
        "Reviewer Name": "",
        "Active": True,
        "Preselection Capacity": 10,
        "Interview Capacity": 5,
        "Preselection Flexible": False,
        "Preselection Flex Limit": 14,
        "Background Tags": "",
    }
    for k, v in needed.items():
        if k not in df.columns:
            df[k] = v
    df["Reviewer Name"] = df["Reviewer Name"].fillna("").astype(str).str.strip()
    df = df[df["Reviewer Name"] != ""].copy()
    df["Active"] = df["Active"].apply(lambda x: _bool_from_text(x, True))
    df["Preselection Flexible"] = df["Preselection Flexible"].apply(lambda x: _bool_from_text(x, False))
    df["Preselection Capacity"] = pd.to_numeric(df["Preselection Capacity"], errors="coerce").fillna(10).astype(int)
    df["Interview Capacity"] = pd.to_numeric(df["Interview Capacity"], errors="coerce").fillna(5).astype(int)
    df["Preselection Flex Limit"] = pd.to_numeric(df["Preselection Flex Limit"], errors="coerce").fillna(df["Preselection Capacity"]).astype(int)
    df["Preselection Flex Limit"] = df[["Preselection Flex Limit", "Preselection Capacity"]].max(axis=1)
    df["Background Tags"] = df["Background Tags"].fillna("").astype(str)
    return df[["Reviewer Name", "Active", "Preselection Capacity", "Interview Capacity", "Preselection Flexible", "Preselection Flex Limit", "Background Tags"]].reset_index(drop=True)


def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            if df is None:
                df = pd.DataFrame()
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    output.seek(0)
    return output.getvalue()
