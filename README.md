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

## Refresh the SJER tower replay

The included SJER temperature and wind data cover August–October 2026 and are provisional NEON data. To download them again, put your NEON API token in `.env` as `NEON_TOKEN`, then run from the repository root:

```sh
for month in 2026-08 2026-09 2026-10; do
  python code/download_neon.py --site SJER --month "$month"
done
python code/prepare_data.py --site SJER --months 2026-08 2026-09 2026-10
```

The app reads the combined files in `data/processed/`. Its NEON replay uses tower levels near 1.5 m and 5.5 m; these observations are from SJER, not from an orchard.
