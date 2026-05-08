# train.py
import os
import json
import argparse
import subprocess
from google.cloud import storage
from datetime import datetime
from pathlib import Path

# ── Config from environment variables ─────────────────────
BUCKET_NAME  = os.environ["BUCKET_NAME"]
DATA_PATH    = os.environ.get("DATA_PATH", "data/config.json")
MAX_VERSIONS = 3

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
    """Download config.json and any data files from GCS"""
    print("Downloading config from GCS...")
    bucket.blob(DATA_PATH).download_to_filename("config.json")

def run_training():
    """Call main.py as a subprocess with args"""
    cmd = [
        "python", "main.py",
        "--epochs",      str(EPOCHS),
        "--lr",          str(LR),
        "--hidden",      str(HIDDEN),
        "--batch-size",  str(BATCH_SIZE),
        "--dropout",     str(DROPOUT),
        "--window",      str(WINDOW),
        "--graph-window",str(GRAPH_WIN),
        "--early-stop",  str(EARLY_STOP),
        "--start-exp",   str(START_EXP),
        "--ahead",       str(AHEAD),
        "--sep",         str(SEP),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def upload_results():
    """Upload results CSVs and return version tag"""
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("../results")

    for csv_file in results_dir.glob("results_*.csv"):
        dest = f"predictions/{version}/{csv_file.name}"
        bucket.blob(dest).upload_from_filename(str(csv_file))
        print(f"Uploaded {csv_file.name} → gs://{BUCKET_NAME}/{dest}")

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