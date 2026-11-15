# AgX_Hackathon_UA2026

Spray or Wait? is a Streamlit demo that uses two-height weather observations to show orchard spray drift risk.

## Run locally

From the repository root, using `venv`:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run code/app.py
```

Or using Conda:

```sh
conda create -n agx-hackathon python=3.11 -y
conda activate agx-hackathon
python -m pip install -r requirements.txt
streamlit run code/app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). The sample data needed for the demo is included in `data/`; no API key is needed to run it.
