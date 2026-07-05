import os
import json
import argparse
import subprocess
from google.cloud import storage
from datetime import datetime, timezone
from pathlib import Path

# ── Config from environment variables ─────────────────────
BUCKET_NAME = os.environ.get("BUCKET_NAME")
if not BUCKET_NAME:
    raise ValueError("Environment variable 'BUCKET_NAME' is required but not set.")

DATA_PATH    = os.environ.get("DATA_PATH", "data/config.json")
MAX_VERSIONS = 3

# Which dataset to run. Only this country is downloaded and trained.
COUNTRY = os.environ.get("COUNTRY", "England")
# Optional: restrict which models run, space-separated (e.g. "MPNN_LSTM").
# Empty string => let main.py use its default (all models).
MODELS = os.environ.get("MODELS", "").strip()

# Training hyperparams (overridable via env vars)
EPOCHS      = int(os.environ.get("EPOCHS", 300))
LR          = float(os.environ.get("LR", 0.001))
HIDDEN      = int(os.environ.get("HIDDEN", 64))
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE", 8))
DROPOUT     = float(os.environ.get("DROPOUT", 0.5))
WINDOW      = int(os.environ.get("WINDOW", 7))
GRAPH_WIN   = int(os.environ.get("GRAPH_WINDOW", 7))
EARLY_STOP  = int(os.environ.get("EARLY_STOP", 100))
START_EXP   = int(os.environ.get("START_EXP", 15))
AHEAD       = int(os.environ.get("AHEAD", 14))
SEP         = int(os.environ.get("SEP", 10))

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)


def download_data():
    """Download config.json and ONLY the requested country's data from GCS."""
    print("Downloading config from GCS...")
    bucket.blob(DATA_PATH).download_to_filename("config.json")

    with open("config.json") as f:
        config = json.load(f)

    country_idx = config["country_idx"]
    if COUNTRY not in country_idx:
        raise ValueError(
            f"COUNTRY '{COUNTRY}' not found in config country_idx "
            f"{list(country_idx.keys())}"
        )

    idx = country_idx[COUNTRY]
    # The on-disk / GCS folder name for this country (what main.py chdirs into).
    folder = config["countries"][idx]

    print(f"Downloading data for {COUNTRY} only (folder '{folder}', index {idx})...")
    os.makedirs("/data", exist_ok=True)

    # Only pull this one country's subtree instead of all of data/data/.
    prefix = f"data/data/{folder}/"
    n = 0
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("/"):
            continue
        # data/data/<folder>/graphs/x.csv -> /data/<folder>/graphs/x.csv
        dest = "/data/" + "/".join(blob.name.split("/")[2:])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        blob.download_to_filename(dest)
        n += 1
        print(f"Downloaded {blob.name} -> {dest}")

    if n == 0:
        raise RuntimeError(
            f"No files found under gs://{BUCKET_NAME}/{prefix}. "
            f"Check that config['countries'][{idx}] ('{folder}') matches the "
            f"GCS folder name for {COUNTRY}."
        )
    print(f"Downloaded {n} files for {COUNTRY}.")


def run_training():
    """Call main.py as a subprocess with args (England only)."""
    cmd = [
        "python", "main.py",
        "--epochs",       str(EPOCHS),
        "--lr",           str(LR),
        "--hidden",       str(HIDDEN),
        "--batch-size",   str(BATCH_SIZE),
        "--dropout",      str(DROPOUT),
        "--window",       str(WINDOW),
        "--graph-window", str(GRAPH_WIN),
        "--early-stop",   str(EARLY_STOP),
        "--start-exp",    str(START_EXP),
        "--ahead",        str(AHEAD),
        "--sep",          str(SEP),
        "--country",      COUNTRY,
    ]
    if MODELS:
        cmd += ["--models"] + MODELS.split()

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def upload_results():
    """Upload results CSVs, the run log, and trained model checkpoints.
    Returns version tag."""
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = Path("../results")

    for csv_file in results_dir.glob("results_*.csv"):
        dest = f"predictions/{version}/{csv_file.name}"
        bucket.blob(dest).upload_from_filename(str(csv_file))
        print(f"Uploaded {csv_file.name} -> gs://{BUCKET_NAME}/{dest}")

    # Also upload the timestamped run log so you can inspect progress after the fact.
    for log_file in results_dir.glob("run_*.log"):
        dest = f"predictions/{version}/{log_file.name}"
        bucket.blob(dest).upload_from_filename(str(log_file))
        print(f"Uploaded {log_file.name} -> gs://{BUCKET_NAME}/{dest}")

    # Trained model checkpoints: one best checkpoint per (model, shift),
    # written by main.py into MODELS_DIR (default ../results/models). Kept
    # under the same version folder so they're purged with their predictions.
    models_dir = Path(os.environ.get("MODELS_DIR", "../results/models"))
    if models_dir.exists():
        for ckpt in sorted(models_dir.glob("*.pth.tar")):
            if ckpt.name.startswith("model_best_"):
                continue  # per-fit scratch file left by a crashed run
            dest = f"predictions/{version}/models/{ckpt.name}"
            bucket.blob(dest).upload_from_filename(str(ckpt))
            print(f"Uploaded {ckpt.name} -> gs://{BUCKET_NAME}/{dest}")
    else:
        print(f"No {models_dir} directory found; skipping model upload.")

    return version


def purge_old_versions():
    """Keep only the latest MAX_VERSIONS under predictions/"""
    blobs = list(bucket.list_blobs(prefix="predictions/"))
    # Get unique version folders
    versions = sorted(set(
        b.name.split("/")[1] for b in blobs if len(b.name.split("/")) > 2
    ))
    to_delete = versions[:-MAX_VERSIONS]
    for v in to_delete:
        for blob in bucket.list_blobs(prefix=f"predictions/{v}/"):
            blob.delete()
            print(f"Purged old version: {blob.name}")


if __name__ == "__main__":
    download_data()
    run_training()
    version = upload_results()
    purge_old_versions()
    print(f"Pipeline complete. Version: {version}")