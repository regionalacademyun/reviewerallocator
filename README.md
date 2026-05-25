# RAUN Preselection Allocation Studio

Preselection-only version of the RAUN allocation app.

## What changed

- Login includes a `New user / guest reviewer` option.
- Offline upload and connected Google Sheet modes are both supported.
- Interview allocation has been removed from this app; it should be handled in a separate interview allocation app.
- Dashboard includes gender, age range, and country bar charts, while keeping the university word cloud.
- Reviewer capacity restrictions are applied strictly.
- `Active = False` excludes a reviewer completely. You do not need to set capacity to 0.
- Flexible capacity is used only when `Preselection Flexible = True` and only up to `Preselection Flex Limit`.
- There is no forced emergency overflow in this version. If declared capacity and flex limits are not enough, the app records incomplete allocations instead of over-assigning reviewers.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Google Sheet setup

For local testing, enter the service-account JSON key path in the app when using connected mode.
For deployment, add the service account JSON under `[gcp_service_account]` in Streamlit secrets.

The app cannot read or write any Google Sheet unless the Google Sheet address/ID is entered in the app and the sheet has been shared with the service account email.


## Deployment hygiene

This v5 package suppresses harmless openpyxl metadata warnings from Excel files and converts display tables into Streamlit/Arrow-safe types before rendering. This keeps GitHub/Streamlit deployment logs cleaner without changing allocation logic.
