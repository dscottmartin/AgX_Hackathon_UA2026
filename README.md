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

## Refresh the SJER tower data

The included SJER temperature and wind data cover August–October 2026 and are provisional NEON data. To download them again, put your NEON API token in `.env` as `NEON_TOKEN`, then run from the repository root:

```sh
for month in 2026-08 2026-09 2026-10; do
  python code/download_neon.py --site SJER --month "$month"
done
python code/prepare_data.py --site SJER --months 2026-08 2026-09 2026-10
```

The command writes combined files in `data/processed/`. The current dashboard's NEON replay still points to the included SOAP 2025-06 files; the SJER files are available for a later replay switch.

## Push notification demo

Subscribe to `agX_evoAG` in the ntfy app. The dashboard uses that topic by default and automatically sends STOP or CAUTION for the selected reading once per orchard, timestamp, and status in the current browser session. You can edit the topic or turn off **Automatically notify on risk** in the sidebar. **Send push for this reading** sends any selected status manually. Notifications identify the data as a demo replay; the dashboard does not receive live sensor readings.
