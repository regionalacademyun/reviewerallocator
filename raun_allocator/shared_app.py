import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from . import user_config as cfg
from .allocation_logic import (
    allocate_preselection,
    build_default_reviewers_df,
    combined_workload_stats,
    compute_planning_metrics,
)
from .dashboard_components import app_styles, render_dashboard
from .data_io import load_applicants, normalize_reviewers_input, to_excel_bytes, make_streamlit_safe_df
from .google_sheets_io import read_google_sheet, write_google_sheet


def get_logo_path():
    for p in cfg.LOGO_CANDIDATES:
        if p.exists():
            return p
    return None


def login_screen():
    app_styles()
    left, right = st.columns([1, 1.4])
    with left:
        logo = get_logo_path()
        if logo:
            st.image(str(logo), width=120)
        st.markdown('<div class="hero">', unsafe_allow_html=True)
        st.title(cfg.APP_TITLE)
        st.caption(cfg.APP_SUBTITLE)
        st.markdown("<div class='explain'>Choose your name and enter the password to open the app. If your name is not listed, choose <b>New user / guest reviewer</b> and type your name.</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="step">', unsafe_allow_html=True)
        login_options = [""] + cfg.RAUN_TEAM_MEMBERS + [cfg.NEW_USER_LABEL]
        selected_user = st.selectbox("Choose your name", login_options, index=0)
        custom_user = ""
        if selected_user == cfg.NEW_USER_LABEL:
            custom_user = st.text_input("Type your full name")
        password = st.text_input("Password", type="password")
        if st.button("Login", type="primary", width="stretch"):
            final_user = custom_user.strip() if selected_user == cfg.NEW_USER_LABEL else selected_user
            if not final_user:
                st.warning("Please choose or type your name.")
            elif password == cfg.APP_PASSWORD:
                st.session_state.logged_in = True
                st.session_state.username = final_user
                st.rerun()
            else:
                st.error("Wrong password")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()


def make_reviewer_template_bytes(applicants_df):
    template_df = build_default_reviewers_df(
        cfg.RAUN_TEAM_MEMBERS,
        applicants_df=applicants_df,
        use_uniform_baseline=True,
    )
    return to_excel_bytes({"Reviewer Availability Input": template_df})


def glossy_header(username: str):
    st.markdown(
        f"""
        <div class="hero">
            <h2 style="margin-bottom:0.25rem;">Hello {username.split()[0] if username else 'there'}!</h2>
            <div class="explain">This is now the <b>preselection allocation app</b>. It can work offline from an uploaded file or connect to a live Google Sheet. Interview allocation is intentionally separated for a later dedicated app.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ensure_state():
    defaults = {
        "reviewers_working_df": None,
        "reviewer_source_label": "Uniform RAUN baseline table",
        "applicants_df": None,
        "prealloc_df": None,
        "preloads_df": None,
        "preexceptions_df": None,
        "source_mode": "Offline upload",
        "connected_sheet_ref": "",
        "connected_worksheet": "Sheet1",
        "local_service_account_file": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def make_preselection_plot(loads_df: pd.DataFrame):
    if loads_df is None or loads_df.empty:
        return None
    bar_df = loads_df.sort_values("Preselection Assigned", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bar_df["Reviewer Name"],
        y=bar_df["Preselection Assigned"],
        name="Assigned",
        marker_color="#2563EB",
        hovertemplate="<b>%{x}</b><br>Assigned: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=bar_df["Reviewer Name"], y=bar_df["Preselection Capacity"],
        mode="lines+markers", name="Base capacity", line=dict(color="#16A34A", dash="dash", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=bar_df["Reviewer Name"], y=bar_df["Preselection Flex Limit"],
        mode="lines+markers", name="Flex limit", line=dict(color="#F59E0B", dash="dot", width=3)
    ))
    fig.update_layout(
        height=460,
        template="plotly_white",
        title="Preselection workload: assigned vs declared limits",
        xaxis_title="Reviewer",
        yaxis_title="Assignments",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig





def make_capacity_limit_plot(loads_df: pd.DataFrame):
    """Show declared base capacity, optional flex room, and actual assignment in one readable chart."""
    if loads_df is None or loads_df.empty:
        return None

    df = loads_df.copy()
    for col in ["Preselection Assigned", "Preselection Capacity", "Preselection Flex Limit", "Over Capacity By", "Over Flex Limit By"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["Flex Room"] = (df["Preselection Flex Limit"] - df["Preselection Capacity"]).clip(lower=0)
    df["Remaining To Limit"] = (df["Preselection Flex Limit"] - df["Preselection Assigned"]).clip(lower=0)
    df = df.sort_values("Preselection Assigned", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Reviewer Name"],
        y=df["Preselection Capacity"],
        name="Declared base capacity",
        marker_color="#BFDBFE",
        hovertemplate="<b>%{x}</b><br>Base capacity: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=df["Reviewer Name"],
        y=df["Flex Room"],
        name="Optional flex room",
        marker_color="#FDE68A",
        hovertemplate="<b>%{x}</b><br>Additional flex room: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["Reviewer Name"],
        y=df["Preselection Assigned"],
        mode="markers+lines",
        name="Actually assigned",
        marker=dict(size=12, color="#2563EB", symbol="diamond"),
        line=dict(color="#2563EB", width=3),
        customdata=df[["Preselection Capacity", "Preselection Flex Limit", "Remaining To Limit", "Over Flex Limit By"]].values,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Assigned: %{y}<br>"
            "Base capacity: %{customdata[0]}<br>"
            "Final allowed limit: %{customdata[1]}<br>"
            "Remaining to final limit: %{customdata[2]}<br>"
            "Over final limit: %{customdata[3]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        barmode="stack",
        height=470,
        template="plotly_white",
        title="Reviewer capacity check: base capacity, optional flex room, and actual assignments",
        xaxis_title="Reviewer",
        yaxis_title="Number of preselection reviews",
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def main_app():
    app_styles()
    ensure_state()
    glossy_header(st.session_state.get("username", ""))

    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("## Step 1 — Choose data source and load applicants")
    st.markdown("<div class='explain'>Choose <b>Connected Google Sheet</b> only when you want the app to read a live sheet. The app cannot see anything unless you explicitly paste the sheet address or ID. Choose <b>Offline upload</b> to keep the workflow exactly like before.</div>", unsafe_allow_html=True)

    source_mode = st.radio(
        "Applicant data source",
        ["Offline upload", "Connected Google Sheet"],
        horizontal=True,
        index=0 if st.session_state.source_mode == "Offline upload" else 1,
        help="Offline uses an uploaded Excel/CSV. Connected mode reads a live Google Sheet using the sheet address you enter.",
    )
    st.session_state.source_mode = source_mode

    if source_mode == "Offline upload":
        applicant_file = st.file_uploader(
            "Upload applicant Excel or CSV",
            type=["xlsx", "csv"],
            help="Use the RAUN applicant sheet. The app can read the historical RAUN sheet format as well as a standard table.",
        )
        if applicant_file is not None:
            try:
                st.session_state.applicants_df = load_applicants(applicant_file)
                st.success(f"Applicant file loaded: {len(st.session_state.applicants_df)} applicants found.")
            except Exception as e:
                st.error(f"Could not read applicant file: {e}")
                st.markdown('</div>', unsafe_allow_html=True)
                return
    else:
        c1, c2 = st.columns([1.6, 1])
        with c1:
            sheet_ref = st.text_input(
                "Google Sheet address or ID",
                value=st.session_state.connected_sheet_ref,
                help="Paste the Google Sheet URL or the spreadsheet ID. The service account must have access to this sheet.",
            )
        with c2:
            worksheet = st.text_input(
                "Worksheet/tab name",
                value=st.session_state.connected_worksheet,
                help="Exact tab name, for example Sheet1 or Applicants.",
            )
        service_path = st.text_input(
            "Optional local JSON key path for testing only",
            value=st.session_state.local_service_account_file,
            type="password",
            help="Leave blank when using Streamlit secrets. Use only for local testing.",
        )
        st.session_state.connected_sheet_ref = sheet_ref
        st.session_state.connected_worksheet = worksheet
        st.session_state.local_service_account_file = service_path
        if st.button("Load live Google Sheet", type="primary", width="stretch"):
            try:
                raw_df = read_google_sheet(sheet_ref, worksheet, service_account_file=service_path or None)
                st.session_state.applicants_df = load_applicants(raw_df)
                st.success(f"Live Google Sheet loaded: {len(st.session_state.applicants_df)} applicants found.")
            except Exception as e:
                st.error(f"Could not read Google Sheet: {e}")
                st.markdown('</div>', unsafe_allow_html=True)
                return

    applicants_df = st.session_state.applicants_df
    st.markdown('</div>', unsafe_allow_html=True)

    if applicants_df is None or applicants_df.empty:
        st.info("Please load applicants to continue.")
        return

    if st.session_state.reviewers_working_df is None:
        st.session_state.reviewers_working_df = build_default_reviewers_df(
            cfg.RAUN_TEAM_MEMBERS,
            applicants_df=applicants_df,
            use_uniform_baseline=True,
        ).copy()
        st.session_state.reviewer_source_label = "Uniform RAUN baseline table"

    render_dashboard(applicants_df, st.session_state.reviewers_working_df.copy(), shortlist_only=True)

    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("## Step 2 — Generate the reviewer request file")
    st.markdown("<div class='explain'>This file starts from the current RAUN reviewer list and includes availability fields for preselection. If someone is unavailable, set <b>Active = FALSE</b>. The allocator will not assign any cases to inactive reviewers.</div>", unsafe_allow_html=True)
    req_bytes = make_reviewer_template_bytes(applicants_df)
    planning = compute_planning_metrics(applicants_df, st.session_state.reviewers_working_df.copy(), shortlist_only=True)
    st.info(
        f"Planning view: {planning['Applicants']} applicants require {planning['Preselection Demand']} preselection reviews. "
        f"The current active reviewer table has {planning['Preselection Capacity']} base slots and "
        f"{planning['Preselection Capacity With Flex']} slots with declared flexibility."
    )
    st.download_button(
        "Download reviewer request file",
        data=req_bytes,
        file_name="reviewer_request_template_preselection.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
        help="Send this file to the RAUN team and ask each reviewer to fill in their availability.",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("## Step 3 — Load or edit reviewer availability")
    st.markdown("<div class='explain'>The allocator strictly respects <b>Active</b>, <b>Preselection Capacity</b>, and <b>Preselection Flex Limit</b>. It never assigns a reviewer beyond their flex limit. If declared availability is not enough, the app leaves the remaining slots incomplete and explains the shortage.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1.3])
    with c1:
        reviewer_file = st.file_uploader("Upload reviewer availability Excel or CSV", type=["xlsx", "csv"], key="rev_upload")
    with c2:
        st.markdown("### Choose what to do")
        load_clicked = st.button("Use uploaded reviewer file", width="stretch", help="Make the uploaded reviewer file the current working reviewer table.")
        default_clicked = st.button("Use uniform RAUN baseline", width="stretch", help="Reset back to the built-in RAUN baseline based on demand.")
    if load_clicked:
        if reviewer_file is None:
            st.warning("Please upload the reviewer availability file first.")
        else:
            try:
                if reviewer_file.name.lower().endswith(".xlsx"):
                    new_df = pd.read_excel(reviewer_file)
                else:
                    new_df = pd.read_csv(reviewer_file)
                st.session_state.reviewers_working_df = normalize_reviewers_input(new_df)
                st.session_state.reviewer_source_label = f"Loaded from file: {reviewer_file.name}"
                st.success("The uploaded reviewer file is now the working reviewer table.")
            except Exception as e:
                st.error(f"Could not read reviewer file: {e}")
    if default_clicked:
        st.session_state.reviewers_working_df = build_default_reviewers_df(
            cfg.RAUN_TEAM_MEMBERS,
            applicants_df=applicants_df,
            use_uniform_baseline=True,
        ).copy()
        st.session_state.reviewer_source_label = "Uniform RAUN baseline table"
        st.success("The uniform RAUN baseline table is active again.")

    st.info(f"Current reviewer table source: {st.session_state.reviewer_source_label}")
    reviewers_df = st.data_editor(
        st.session_state.reviewers_working_df.copy(),
        width="stretch",
        height=460,
        num_rows="dynamic",
        key="reviewer_availability_editor",
        column_config={
            "Reviewer Name": st.column_config.TextColumn("Reviewer Name", help="Reviewer full name. You may add new reviewers as extra rows."),
            "Active": st.column_config.CheckboxColumn("Active", help="If FALSE, this reviewer receives no allocations. You do not need to set capacities to zero."),
            "Preselection Capacity": st.column_config.NumberColumn("Preselection Capacity", min_value=0, step=1, help="Strict normal limit. The allocator only exceeds this when Preselection Flexible is TRUE, and never beyond the flex limit."),
            "Preselection Flexible": st.column_config.CheckboxColumn("Preselection Flexible", help="If TRUE, this reviewer can receive extra cases up to the flex limit."),
            "Preselection Flex Limit": st.column_config.NumberColumn("Preselection Flex Limit", min_value=0, step=1, help="Total maximum after flexibility. If flexibility is off, the app treats this as equal to base capacity."),
            "Background Tags": st.column_config.TextColumn("Background Tags", help="Soft matching helper only. It does not block allocation."),
            "Reviewer Notes": st.column_config.TextColumn("Reviewer Notes", help="Optional notes for the admin."),
        },
    )
    st.session_state.reviewers_working_df = normalize_reviewers_input(reviewers_df)
    reviewers_df = st.session_state.reviewers_working_df.copy()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("## Step 4 — Generate preselection allocation")
    st.markdown("<div class='explain'>Each applicant needs two distinct preselection reviewers. The app uses base capacity first, then declared flexibility up to the reviewer’s flex limit. It does not force assignments beyond declared limits. If availability is not enough, it leaves the slot blank and records a clear exception.</div>", unsafe_allow_html=True)

    seed = st.number_input("Random seed", min_value=1, value=42, step=1, help="Use the same seed if you want the app to produce the same allocation again.")
    match_strength = st.selectbox("Background matching strength", ["Off", "Low", "Medium"], index=1, help="Soft bonus only. It never overrides capacity restrictions.")

    planning = compute_planning_metrics(applicants_df, reviewers_df, shortlist_only=True)
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Preselection demand", planning["Preselection Demand"], help="Two preselection reviews are needed per applicant.")
    p2.metric("Base capacity", planning["Preselection Capacity"], help="Total normal slots declared by active reviewers.")
    p3.metric("Capacity with flex", planning["Preselection Capacity With Flex"], help="Base capacity plus declared flexible capacity.")
    p4.metric("Shortage after flex", planning["Preselection Shortage With Flex"], help="How many review slots cannot be allocated without exceeding declared base/flex limits.")

    tab_pre, tab_stats, tab_sync = st.tabs(["Preselection Allocation", "Workload Stats", "Google Sheet Output"])

    with tab_pre:
        st.subheader("Preselection allocation")
        if st.button("Generate preselection allocation", type="primary", width="stretch"):
            prealloc_df, preloads_df, preexceptions_df = allocate_preselection(
                applicants_df=applicants_df,
                reviewers_df=reviewers_df,
                seed=int(seed),
                match_strength=match_strength,
            )
            st.session_state.prealloc_df = prealloc_df
            st.session_state.preloads_df = preloads_df
            st.session_state.preexceptions_df = preexceptions_df

        if st.session_state.prealloc_df is not None:
            st.dataframe(make_streamlit_safe_df(st.session_state.prealloc_df), width="stretch", height=420)
            fig = make_preselection_plot(st.session_state.preloads_df)
            if fig is not None:
                st.plotly_chart(fig, width="stretch", key="preselection_chart_tab")
            st.markdown("### Exceptions")
            if st.session_state.preexceptions_df is not None and not st.session_state.preexceptions_df.empty:
                st.dataframe(make_streamlit_safe_df(st.session_state.preexceptions_df), width="stretch", height=240)
            else:
                st.success("No preselection exceptions.")
            pre_excel = to_excel_bytes({
                "Preselection Allocation": st.session_state.prealloc_df,
                "Reviewer Loads": st.session_state.preloads_df,
                "Allocation Exceptions": st.session_state.preexceptions_df,
            })
            st.download_button("Download preselection Excel", pre_excel, "raun_preselection_allocation.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")

    with tab_stats:
        st.subheader("Workload stats")
        if st.session_state.preloads_df is None:
            st.info("Generate preselection first to see workload stats.")
        else:
            st.dataframe(make_streamlit_safe_df(st.session_state.preloads_df), width="stretch", height=300)
            st.markdown("### Capacity check")
            st.caption("Blue diamonds are actual assignments. The pale blue bar is the declared base capacity. The yellow extension is optional flex room. No reviewer should go above the top of their stacked bar.")
            cap_fig = make_capacity_limit_plot(st.session_state.preloads_df)
            if cap_fig is not None:
                st.plotly_chart(cap_fig, width="stretch", key="capacity_limit_chart_stats")

    with tab_sync:
        st.subheader("Write allocation results to Google Sheet")
        st.caption("This is optional. Nothing is written unless you press a write button.")
        if st.session_state.prealloc_df is None:
            st.info("Generate a preselection allocation first.")
        else:
            out_sheet_ref = st.text_input("Output Google Sheet address or ID", value=st.session_state.connected_sheet_ref, help="Paste the destination Google Sheet URL or ID.")
            service_path_out = st.text_input("Optional local JSON key path", value=st.session_state.local_service_account_file, type="password", key="output_service_path")
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Write allocation tab", width="stretch"):
                    try:
                        write_google_sheet(st.session_state.prealloc_df, out_sheet_ref, "Preselection Allocation", service_account_file=service_path_out or None)
                        st.success("Preselection Allocation written.")
                    except Exception as e:
                        st.error(f"Could not write allocation: {e}")
            with c2:
                if st.button("Write loads tab", width="stretch"):
                    try:
                        write_google_sheet(st.session_state.preloads_df, out_sheet_ref, "Reviewer Loads", service_account_file=service_path_out or None)
                        st.success("Reviewer Loads written.")
                    except Exception as e:
                        st.error(f"Could not write loads: {e}")
            with c3:
                if st.button("Write exceptions tab", width="stretch"):
                    try:
                        write_google_sheet(st.session_state.preexceptions_df, out_sheet_ref, "Allocation Exceptions", service_account_file=service_path_out or None)
                        st.success("Allocation Exceptions written.")
                    except Exception as e:
                        st.error(f"Could not write exceptions: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
