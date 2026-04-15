
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from . import user_config as cfg
from .allocation_logic import (
    allocate_interviews,
    allocate_preselection,
    build_default_reviewers_df,
    combined_workload_stats,
    compute_planning_metrics,
)
from .dashboard_components import app_styles, render_dashboard
from .data_io import load_applicants, normalize_reviewers_input, to_excel_bytes


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
        st.markdown("<div class='explain'>Choose your name and enter the password to open the app.</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="step">', unsafe_allow_html=True)
        selected_user = st.selectbox("Choose your name", [""] + cfg.RAUN_TEAM_MEMBERS, index=0)
        password = st.text_input("Password", type="password")
        if st.button("Login", type="primary", width="stretch"):
            if not selected_user:
                st.warning("Please choose your name.")
            elif password == cfg.APP_PASSWORD:
                st.session_state.logged_in = True
                st.session_state.username = selected_user
                st.rerun()
            else:
                st.error("Wrong password")
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()


def make_reviewer_template_bytes(applicants_df):
    template_df = build_default_reviewers_df(
        cfg.RAUN_TEAM_MEMBERS,
        applicants_df=applicants_df,
        shortlist_only=True,
        use_uniform_baseline=True,
    )
    return to_excel_bytes({"Reviewer Availability Input": template_df})


def glossy_header(username: str):
    st.markdown(
        f"""
        <div class="hero">
            <h2 style="margin-bottom:0.25rem;">Hello {username.split()[0] if username else 'there'}!</h2>
            <div class="explain">This app works in a simple order. First it reads the applicant file. Then it shows what the workload looks like. Then it helps you send a reviewer request file. After that, you load the reviewer availability file and generate allocations.</div>
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
        "intalloc_df": None,
        "intloads_df": None,
        "intexceptions_df": None,
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
    fig.update_layout(height=430, template="plotly_white", title="Preselection workload", xaxis_title="Reviewer", yaxis_title="Assignments", margin=dict(l=20, r=20, t=60, b=20))
    return fig


def make_interview_plot(loads_df: pd.DataFrame):
    if loads_df is None or loads_df.empty:
        return None
    bar_df = loads_df.sort_values("Interview Assigned", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bar_df["Reviewer Name"], y=bar_df["Interview Assigned"], name="Assigned", marker_color="#8B5CF6"
    ))
    fig.add_trace(go.Scatter(
        x=bar_df["Reviewer Name"], y=bar_df["Interview Capacity"], mode="lines+markers", name="Interview capacity", line=dict(color="#7C3AED", dash="dash", width=3)
    ))
    fig.update_layout(height=430, template="plotly_white", title="Interview workload", xaxis_title="Reviewer", yaxis_title="Assignments", margin=dict(l=20, r=20, t=60, b=20))
    return fig


def make_combined_workload_plot(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    plot_df = df.sort_values("Total Assigned", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plot_df["Reviewer Name"], y=plot_df["Total Assigned"], name="Total assigned", marker_color="#0EA5E9"
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["Reviewer Name"], y=plot_df["Total Base Capacity"], mode="lines+markers", name="Total base capacity", line=dict(color="#16A34A", dash="dash", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=plot_df["Reviewer Name"], y=plot_df["Total Flex Capacity"], mode="lines+markers", name="Total flex capacity", line=dict(color="#F59E0B", dash="dot", width=3)
    ))
    fig.update_layout(height=430, template="plotly_white", title="Combined workload", xaxis_title="Reviewer", yaxis_title="Assignments", margin=dict(l=20, r=20, t=60, b=20))
    return fig


def main_app():
    app_styles()
    ensure_state()
    glossy_header(st.session_state.get("username", ""))

    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("## Step 1 — Load the applicant file")
    st.markdown("<div class='explain'>Upload the applicant Excel file here. The app reads the names, programme, university, field, countries, scholarship status, and topic areas. Then it shows a simple dashboard and demand picture.</div>", unsafe_allow_html=True)
    applicant_file = st.file_uploader("Upload applicant Excel or CSV", type=["xlsx", "csv"], help="Use the RAUN applicant sheet. The app can read the historical row-wise RAUN sheet as well as a more standard table.")
    if applicant_file is not None:
        try:
            st.session_state.applicants_df = load_applicants(applicant_file)
            st.success(f"Applicant file loaded: {len(st.session_state.applicants_df)} applicants found.")
        except Exception as e:
            st.error(f"Could not read applicant file: {e}")
            st.markdown('</div>', unsafe_allow_html=True)
            return
    applicants_df = st.session_state.applicants_df
    st.markdown('</div>', unsafe_allow_html=True)
    if applicants_df is None or applicants_df.empty:
        st.info("Please upload the applicant file to continue.")
        return

    if st.session_state.reviewers_working_df is None:
        st.session_state.reviewers_working_df = build_default_reviewers_df(
            cfg.RAUN_TEAM_MEMBERS,
            applicants_df=applicants_df,
            shortlist_only=True,
            use_uniform_baseline=True,
        ).copy()
        st.session_state.reviewer_source_label = "Uniform RAUN baseline table"

    render_dashboard(applicants_df, st.session_state.reviewers_working_df.copy(), shortlist_only=True)

    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("## Step 2 — Generate the reviewer request file")
    st.markdown("<div class='explain'>This is the file you send to the RAUN team. It starts from the built-in team list and shows one row per reviewer, with spaces for capacity and flexibility. Reviewers can fill it in and send it back.</div>", unsafe_allow_html=True)
    req_bytes = make_reviewer_template_bytes(applicants_df)
    planning = compute_planning_metrics(applicants_df, st.session_state.reviewers_working_df.copy(), shortlist_only=True)
    st.info(
        f"Simple planning view: {planning['Applicants']} applicants → {planning['Preselection Demand']} preselection assignments and "
        f"{planning['Interview Demand']} interview assignments. If everyone helped evenly, the average would be "
        f"{planning['Suggested Avg Preselection per Reviewer']} preselection reviews and {planning['Suggested Avg Interview per Reviewer']} interviews per active reviewer."
    )
    st.download_button(
        "Download reviewer request file",
        data=req_bytes,
        file_name="reviewer_request_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
        help="Download this file, send it to the RAUN team, and ask each reviewer to fill in their availability.",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("## Step 3 — Load the reviewer availability file")
    st.markdown("<div class='explain'>After the team sends the filled file back, upload it here and click the load button. That file becomes the working reviewer table. You can still edit it afterwards if something changes.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1.3])
    with c1:
        reviewer_file = st.file_uploader("Upload reviewer availability Excel or CSV", type=["xlsx", "csv"], key="rev_upload")
    with c2:
        st.markdown("### Choose what to do")
        load_clicked = st.button("Use uploaded reviewer file", width="stretch", help="Take the file you uploaded and make it the current working reviewer table.")
        default_clicked = st.button("Use default RAUN table", width="stretch", help="Reset back to the built-in RAUN baseline and default capacities.")
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
            shortlist_only=True,
            use_uniform_baseline=True,
        ).copy()
        st.session_state.reviewer_source_label = "Uniform RAUN baseline table"
        st.success("The uniform RAUN baseline table is active again.")
    st.info(
        f"Current reviewer table source: {st.session_state.reviewer_source_label}. "
        "This is the table the allocator will use right now."
    )
    reviewers_df = st.data_editor(
        st.session_state.reviewers_working_df.copy(),
        width="stretch",
        height=460,
        num_rows="dynamic",
        column_config={
            "Active": st.column_config.CheckboxColumn("Active", help="Turn a reviewer on or off."),
            "Preselection Capacity": st.column_config.NumberColumn("Preselection Capacity", min_value=0, step=1, help="Normal preselection limit."),
            "Interview Capacity": st.column_config.NumberColumn("Interview Capacity", min_value=0, step=1, help="Normal interview limit."),
            "Preselection Flexible": st.column_config.CheckboxColumn("Preselection Flexible", help="If yes, this reviewer can take extra preselection cases."),
            "Preselection Flex Limit": st.column_config.NumberColumn("Preselection Flex Limit", min_value=0, step=1, help="The highest total preselection load this reviewer is willing to take before emergency overflow."),
            "Background Tags": st.column_config.TextColumn("Background Tags", help="Soft matching helper only. It does not block allocation."),
        },
    )
    st.session_state.reviewers_working_df = normalize_reviewers_input(reviewers_df)
    reviewers_df = st.session_state.reviewers_working_df.copy()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="step">', unsafe_allow_html=True)
    st.markdown("## Step 4 — Generate allocations")
    st.markdown("<div class='explain'>Now the app uses the working reviewer table you see above. It first creates preselection allocations. After that, it can create interview allocations. Flexible support is used before emergency overflow.</div>", unsafe_allow_html=True)

    seed = st.number_input("Random seed", min_value=1, value=42, step=1, help="Use the same seed if you want the app to produce the same allocation again.")
    match_strength = st.selectbox("Background matching strength", ["Off", "Low", "Medium"], index=1, help="This only gives a soft bonus. It never blocks assignment.")
    shortlist_only = st.checkbox("Interview only shortlisted / interview-required", value=True, help="When this is on, the interview allocator only works on shortlisted or interview-required applicants.")
    allow_same_reviewer_exception = st.checkbox("Allow interview exception if unavoidable", value=True, help="If there is no other reviewer left, the app may use a reviewer who was already in preselection. It will flag that as an exception.")

    planning = compute_planning_metrics(applicants_df, reviewers_df, shortlist_only=shortlist_only)
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Preselection demand", planning["Preselection Demand"], help="Two preselection reviews are needed per applicant.")
    p2.metric("Base preselection capacity", planning["Preselection Capacity"], help="Normal preselection slots declared by reviewers.")
    p3.metric("Flex-adjusted preselection capacity", planning["Preselection Capacity With Flex"], help="Base capacity plus flexible preselection support.")
    p4.metric("Interview capacity", planning["Interview Capacity"], help="Interview slots declared by reviewers.")

    tab_pre, tab_int, tab_stats = st.tabs(["Preselection", "Interview", "Workload Stats"])

    with tab_pre:
        st.subheader("Preselection allocation")
        st.caption("Each applicant gets two preselection reviewers. The app uses normal capacity first, then flexible support, then emergency overflow if it has to.")
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
            st.session_state.intalloc_df = None
            st.session_state.intloads_df = None
            st.session_state.intexceptions_df = None

        if st.session_state.prealloc_df is not None:
            st.dataframe(st.session_state.prealloc_df, width="stretch", height=420)
            fig = make_preselection_plot(st.session_state.preloads_df)
            if fig is not None:
                st.plotly_chart(fig, width="stretch", key="preselection_chart_tab")
            st.markdown("### Exceptions")
            if st.session_state.preexceptions_df is not None and not st.session_state.preexceptions_df.empty:
                st.dataframe(st.session_state.preexceptions_df, width="stretch", height=220)
            else:
                st.success("No preselection exceptions.")
            pre_excel = to_excel_bytes({
                "Preselection Allocation": st.session_state.prealloc_df,
                "Preselection Loads": st.session_state.preloads_df,
                "Preselection Exceptions": st.session_state.preexceptions_df,
            })
            st.download_button("Download preselection Excel", pre_excel, "raun_preselection_allocation.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")

    with tab_int:
        st.subheader("Interview allocation")
        if st.session_state.prealloc_df is None:
            st.info("Generate preselection first.")
        else:
            if st.button("Generate interview allocation", type="primary", width="stretch"):
                intalloc_df, intloads_df, intexceptions_df = allocate_interviews(
                    applicants_df=applicants_df,
                    prealloc_df=st.session_state.prealloc_df,
                    reviewers_df=reviewers_df,
                    seed=int(seed),
                    match_strength=match_strength,
                    allow_same_reviewer_exception=allow_same_reviewer_exception,
                    shortlist_only=shortlist_only,
                )
                st.session_state.intalloc_df = intalloc_df
                st.session_state.intloads_df = intloads_df
                st.session_state.intexceptions_df = intexceptions_df
            if st.session_state.intalloc_df is not None:
                st.dataframe(st.session_state.intalloc_df, width="stretch", height=420)
                fig = make_interview_plot(st.session_state.intloads_df)
                if fig is not None:
                    st.plotly_chart(fig, width="stretch", key="interview_chart_tab")
                st.markdown("### Exceptions")
                if st.session_state.intexceptions_df is not None and not st.session_state.intexceptions_df.empty:
                    st.dataframe(st.session_state.intexceptions_df, width="stretch", height=220)
                else:
                    st.success("No interview exceptions.")

    with tab_stats:
        st.subheader("Workload stats")
        if st.session_state.preloads_df is None:
            st.info("Generate preselection first to see workload stats.")
        else:
            st.markdown("### Preselection workload")
            st.dataframe(st.session_state.preloads_df, width="stretch", height=260)
            pre_fig = make_preselection_plot(st.session_state.preloads_df)
            if pre_fig is not None:
                st.plotly_chart(pre_fig, width="stretch", key="preselection_chart_stats")

            if st.session_state.intloads_df is not None:
                st.markdown("### Interview workload")
                st.dataframe(st.session_state.intloads_df, width="stretch", height=240)
                int_fig = make_interview_plot(st.session_state.intloads_df)
                if int_fig is not None:
                    st.plotly_chart(int_fig, width="stretch", key="interview_chart_stats")

            st.markdown("### Combined workload")
            combined = combined_workload_stats(st.session_state.preloads_df, st.session_state.intloads_df)
            st.dataframe(combined, width="stretch", height=360)
            combo_fig = make_combined_workload_plot(combined)
            if combo_fig is not None:
                st.plotly_chart(combo_fig, width="stretch", key="combined_chart_stats")

            util_fig = px.bar(
                combined.sort_values("Total Utilization % (With Flex)", ascending=False),
                x="Reviewer Name",
                y="Total Utilization % (With Flex)",
                color="Over Flex Limit By",
                color_continuous_scale="RdYlGn_r",
                title="Total reviewer utilization % (with flex)",
            )
            util_fig.update_layout(template="plotly_white", height=420, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(util_fig, width="stretch", key="utilization_chart_stats")
    st.markdown('</div>', unsafe_allow_html=True)
