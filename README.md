
# RAUN Reviewer Allocation Studio v4

This version fixes the historical RAUN applicant sheet format, which stores applicants in columns and field labels in rows.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Workflow

1. Load the applicant file
2. Review the dashboard and ideal workload
3. Download the reviewer request file
4. Load the filled reviewer availability file
5. Generate preselection allocation
6. Generate interview allocation
