# CHP Steenwijk — TESPy Streamlit App

This package rebuilds the CHP performance app with a TESPy-based thermodynamic core while keeping the same Streamlit structure and visual layout.

## Files

- `app.py` — simple deployment entrypoint
- `app_tespy_same_structure.py` — main Streamlit app
- `tespy_engine_chp.py` — TESPy-based CHP model and scenario engine
- `requirements.txt` — Python dependencies for deployment
- `.streamlit/config.toml` — optional local/project Streamlit config

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push these files to your GitHub repo.
2. In Streamlit Community Cloud, deploy the repo and choose `app.py` as the main file.
3. Keep the Python version at the default unless you have a repo-wide reason to change it.
4. If you update `requirements.txt`, reboot or redeploy the app so dependencies are refreshed.

## Notes

- The thermodynamic core uses TESPy and models one extraction point.
- Optimization from the reference example is intentionally removed.
- LP expander scenarios are still included as operating cases, not as an optimizer.
