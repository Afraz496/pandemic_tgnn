import os
import json
import argparse
import re
import subprocess
import time
from google.cloud import storage
from datetime import datetime, timezone
from pathlib import Path

# ── Config from environment variables ─────────────────────
BUCKET_NAME = os.environ.get("BUCKET_NAME")
if not BUCKET_NAME:
    raise ValueError("Environment variable 'BUCKET_NAME' is required but not set.")

DATA_PATH    = os.environ.get("DATA_PATH", "data/config.json")
MAX_VERSIONS = 3

# Where results/models get uploaded: gs://BUCKET/<RESULTS_PREFIX>/<version>/...
# Defaults to "predictions" (unchanged behavior for manual England/etc. runs).
# Self-serve per-user jobs should override this to something unique per job,
# e.g. "jobs/<job_id>/results" -- otherwise purge_old_versions() below would
# eventually delete OTHER users' results once more than MAX_VERSIONS pile up
# under the shared "predictions/" prefix.
RESULTS_PREFIX = os.environ.get("RESULTS_PREFIX", "predictions").strip("/")
# Only meaningful under the shared "predictions/" prefix; a job-specific
# RESULTS_PREFIX has exactly one version and nothing to purge.
PURGE_OLD_VERSIONS = os.environ.get("PURGE_OLD_VERSIONS", "true").lower() == "true"

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

# When set (self-serve jobs only -- unset for shared/manual runs), progress
# is reported to gs://BUCKET/jobs/<JOB_ID>/progress.json so the website can
# show real per-shift/per-fit progress instead of one frozen "Training"
# step for the whole ~1-2h run. main.py trains shifts 1..AHEAD-1 (see its
# `range(1, args.ahead)`), so that's the total step count here too.
JOB_ID = os.environ.get("JOB_ID", "").strip()
TOTAL_SHIFTS = max(AHEAD - 1, 1)

# Where country data (labels.csv + graphs/) lives, independent of where
# DATA_PATH's config.json sits. Defaults to the shared "data/data" prefix
# (unchanged behavior); self-serve per-user jobs should override this to
# something job-scoped, e.g. "jobs/<job_id>/data".
DATA_PREFIX = os.environ.get("DATA_PREFIX", "data/data").strip("/")

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
    prefix = f"{DATA_PREFIX}/{folder}/"
    # Strip exactly as many leading segments as DATA_PREFIX has, whatever
    # its depth -- was hardcoded to 2 (matching the original "data/data"
    # prefix) which silently broke for job-scoped prefixes like
    # "jobs/<job_id>/data" (3 segments): an extra "data/" survived into the
    # destination path, so preprocess.py's os.chdir(folder) couldn't find
    # it. Caught via a live end-to-end test.
    prefix_depth = len(DATA_PREFIX.split("/"))
    n = 0
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("/"):
            continue
        # <DATA_PREFIX>/<folder>/graphs/x.csv -> /data/<folder>/graphs/x.csv
        dest = "/data/" + "/".join(blob.name.split("/")[prefix_depth:])
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


# Matches main.py's inner tqdm bar, e.g.:
#   "  shift 3:  42%|████▏     | 19/45 [02:26<03:38,  8.39s/fit]"
_SHIFT_PROGRESS_RE = re.compile(r"shift (\d+):\s+\d+%\|.*?\|\s*(\d+)/(\d+)")
_PROGRESS_WRITE_INTERVAL_S = 5


def _write_progress(shift: int, fit: int, total_fits: int) -> None:
    if not JOB_ID:
        return  # shared/manual runs (no per-job progress consumer)
    fit_frac = (fit / total_fits) if total_fits else 0.0
    percent = min(99.0, ((shift - 1) + fit_frac) / TOTAL_SHIFTS * 100)
    payload = {
        "shift": shift,
        "total_shifts": TOTAL_SHIFTS,
        "fit": fit,
        "total_fits": total_fits,
        "percent": round(percent, 1),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        bucket.blob(f"jobs/{JOB_ID}/progress.json").upload_from_string(
            json.dumps(payload), content_type="application/json"
        )
    except Exception as exc:
        # Progress reporting is best-effort -- never let it fail the run.
        print(f"Warning: failed to write progress.json: {exc}")


def run_training():
    """Call main.py as a subprocess with args (England only), streaming its
    output live (so `kubectl logs` still shows everything as before) while
    parsing per-shift/per-fit progress out of it to report to GCS."""
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
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    last_write = 0.0
    for line in proc.stdout:
        print(line, end="", flush=True)
        match = _SHIFT_PROGRESS_RE.search(line)
        if match:
            now = time.time()
            if now - last_write >= _PROGRESS_WRITE_INTERVAL_S:
                shift, fit, total_fits = int(match.group(1)), int(match.group(2)), int(match.group(3))
                _write_progress(shift, fit, total_fits)
                last_write = now

    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def upload_results():
    """Upload results CSVs, the run log, and trained model checkpoints.
    Returns version tag."""
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = Path("../results")

    for csv_file in results_dir.glob("results_*.csv"):
        dest = f"{RESULTS_PREFIX}/{version}/{csv_file.name}"
        bucket.blob(dest).upload_from_filename(str(csv_file))
        print(f"Uploaded {csv_file.name} -> gs://{BUCKET_NAME}/{dest}")

    # Also upload the timestamped run log so you can inspect progress after the fact.
    for log_file in results_dir.glob("run_*.log"):
        dest = f"{RESULTS_PREFIX}/{version}/{log_file.name}"
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
            dest = f"{RESULTS_PREFIX}/{version}/models/{ckpt.name}"
            bucket.blob(dest).upload_from_filename(str(ckpt))
            print(f"Uploaded {ckpt.name} -> gs://{BUCKET_NAME}/{dest}")
    else:
        print(f"No {models_dir} directory found; skipping model upload.")

    return version


def purge_old_versions():
    """Keep only the latest MAX_VERSIONS under RESULTS_PREFIX/"""
    blobs = list(bucket.list_blobs(prefix=f"{RESULTS_PREFIX}/"))
    # Get unique version folders
    versions = sorted(set(
        b.name.split("/")[1] for b in blobs if len(b.name.split("/")) > 2
    ))
    to_delete = versions[:-MAX_VERSIONS]
    for v in to_delete:
        for blob in bucket.list_blobs(prefix=f"{RESULTS_PREFIX}/{v}/"):
            blob.delete()
            print(f"Purged old version: {blob.name}")


if __name__ == "__main__":
    download_data()
    run_training()
    version = upload_results()
    if PURGE_OLD_VERSIONS:
        purge_old_versions()
    print(f"Pipeline complete. Version: {version}")