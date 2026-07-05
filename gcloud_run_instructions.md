# Running `pandemic_tgnn` on GKE — England, on GPU

Step-by-step runbook: train the temporal GNN on the **England** dataset on a GKE
L4 GPU node, write results to GCS, tear down.

> Example values — replace with yours: project `gresp7` · zone `us-central1-b` ·
> cluster `gresp-cluster` · image `us-central1-docker.pkg.dev/gresp7/gresp/gresp` ·
> bucket `YOUR_BUCKET`.

---

## Before you start — 3 things that bit the last run

1. **Use the new code.** The image must contain the updated `train.py`, `main.py`,
   `preprocess.py`, and CUDA `Dockerfile`. Your earlier log ran the *old* code
   (all four countries, `Country IT`, no progress bars).
2. **The dataset key is `EN`, not `England`** (graph files are `EN_*.csv`).
   `COUNTRY=EN`.
3. **Bump the image tag every build** (`:v3`, `:v4`, …) and set
   `imagePullPolicy: Always`, or the node reuses the old cached image.

---

## 1. One-time setup

```bash
gcloud config set project gresp7
gcloud services enable container.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

# Artifact Registry repo (skip if it exists)
gcloud artifacts repositories create gresp --repository-format=docker --location=us-central1
```

Expected GCS layout:

```
gs://YOUR_BUCKET/data/config.json
gs://YOUR_BUCKET/data/data/England/england_labels.csv
gs://YOUR_BUCKET/data/data/England/graphs/EN_2020-03-13.csv ...
gs://YOUR_BUCKET/predictions/<version>/results_EN.csv   <- job writes this
```

---

## 2. Build the image (GPU)

From the folder holding the `Dockerfile` + `.py` files (`~/pandemic_tgnn/code`):

```bash
cd ~/pandemic_tgnn/code
gcloud builds submit --tag us-central1-docker.pkg.dev/gresp7/gresp/gresp:v3 .
```

The CUDA (`cu121`) wheels are multi-GB, so this build is slower than the CPU one —
that's expected. Add a `.dockerignore` next to the Dockerfile so `COPY . .` doesn't
ship junk:

```
.git
__pycache__/
*.pyc
.venv/
venv/
.env
data/
results/
```

---

## 3. Cluster + L4 GPU node pool

```bash
# (create the cluster first if it doesn't exist)
gcloud container clusters create gresp-cluster \
  --zone us-central1-b \
  --num-nodes=1 \
  --machine-type=n1-standard-4 \
  --scopes=gke-default,storage-rw

gcloud container node-pools create gpu-pool \
  --cluster gresp-cluster \
  --zone us-central1-b \
  --node-locations us-central1-a \
  --accelerator type=nvidia-l4,count=1,gpu-driver-version=default \
  --num-nodes=1 \
  --machine-type=g2-standard-4\
  --scopes=gke-default,storage-rw

kubectl get nodes   # wait until the gpu-pool node is Ready
```

---

## 4. `job.yaml`

Point `image` at the tag you just built. Set `COUNTRY=EN`. `MODELS` is optional —
set it to `MPNN_LSTM` to run just the TGNN (much faster); remove it to run all 6.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: pandemic-tgnn-train
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-l4
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      containers:
        - name: trainer
          image: us-central1-docker.pkg.dev/gresp7/gresp/gresp:v3
          imagePullPolicy: Always
          env:
            - name: BUCKET_NAME
              value: "YOUR_BUCKET"
            - name: COUNTRY
              value: "EN"
            - name: MODELS
              value: "MPNN_LSTM"     # optional; remove to run all models
          resources:
            limits:
              nvidia.com/gpu: 1
```

---

## 5. Deploy and watch

```bash
kubectl delete job pandemic-tgnn-train --ignore-not-found
kubectl apply -f ~/pandemic_tgnn/job.yaml

sleep 20 && kubectl logs -f \
  $(kubectl get pods --selector=job-name=pandemic-tgnn-train -o jsonpath='{.items[0].metadata.name}')
```

**First ~30s of logs should show the new code:**
- `Downloading data for EN only (folder 'England', index 2)...` — only `England/` files
- `Running: python main.py ... --country EN --models MPNN_LSTM`
- `Using device: cuda`  ← confirms the GPU is actually used
- `n_samples=... | n_nodes=...` then a tqdm bar `EN | MPNN_LSTM (shifts)`

If you see `Country IT` or `Using device: cpu`, jump to Troubleshooting.

---

## 6. Get results

```bash
gsutil ls gs://YOUR_BUCKET/predictions/
gsutil cp gs://YOUR_BUCKET/predictions/<version>/results_EN.csv .
gsutil cp gs://YOUR_BUCKET/predictions/<version>/run_EN.log .
```

---

## 7. Tear down (stop billing)

```bash
while true; do
  DONE=$(kubectl get job pandemic-tgnn-train -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}')
  FAIL=$(kubectl get job pandemic-tgnn-train -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}')
  if [ "$DONE" = "True" ]; then
    echo "Complete — deleting cluster."
    gcloud container clusters delete gresp-cluster --zone us-central1-b --quiet
    break
  fi
  if [ "$FAIL" = "True" ]; then echo "Job FAILED — check logs, then delete the cluster manually."; break; fi
  echo "Still running... $(date)"; sleep 300
done
```

To stop just the GPU cost but keep the cluster:
`gcloud container node-pools delete gpu-pool --cluster gresp-cluster --zone us-central1-b --quiet`

---

## 8. Troubleshooting

| Symptom | Fix |
| --- | --- |
| Log shows `Country IT` / all countries download / no `--country` | Old image ran. Rebuild with a **new tag**, keep `imagePullPolicy: Always`, delete + re-apply the job. |
| `Using device: cpu` on a GPU node | Image built without CUDA wheels, or pod didn't get the GPU. Confirm the `cu121` Dockerfile built, and that `job.yaml` has the GPU `limits` + `nodeSelector`. |
| `ModuleNotFoundError: tqdm` | Rebuild with the updated Dockerfile (it installs `tqdm`). |
| `ValueError: COUNTRY 'England' not found ...` | Use the key: `COUNTRY=EN`. Check keys with `gsutil cat gs://YOUR_BUCKET/data/config.json`. |
| `FileNotFoundError: graphs/EN_....csv` | Graph-file prefix must equal the `country_idx` key (`EN`). |
| Pod stuck `Pending` | GPU pool not Ready or no GPU quota. `kubectl get nodes`; request GPU quota if on free trial. |