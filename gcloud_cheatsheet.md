# gcloud / kubectl cheat sheet — pandemic_tgnn on GKE

Fill in your values: project `gresp7` · zone `us-central1-b` · cluster `gresp-cluster` ·
repo `us-central1-docker.pkg.dev/gresp7/gresp` · image `.../gresp/gresp` · bucket `YOUR_BUCKET`.

## Context / auth
```bash
gcloud config set project gresp7
gcloud config list                       # what am I pointed at?
gcloud auth list                         # who am I?
gcloud container clusters get-credentials gresp-cluster --zone us-central1-b   # wire up kubectl
```

## Artifact Registry (images)
```bash
gcloud artifacts repositories list --location=us-central1
gcloud artifacts docker images list us-central1-docker.pkg.dev/gresp7/gresp --include-tags
gcloud artifacts docker tags list  us-central1-docker.pkg.dev/gresp7/gresp/gresp
gcloud artifacts docker images delete us-central1-docker.pkg.dev/gresp7/gresp/gresp:v1   # cleanup
```

## Cloud Build (build + push)
```bash
cd ~/pandemic_tgnn/code
gcloud builds submit --tag us-central1-docker.pkg.dev/gresp7/gresp/gresp:v3 .   # bump tag each time!
gcloud builds list --limit=5
gcloud builds log <BUILD_ID>
```

## GKE cluster + node pools
```bash
gcloud container clusters list
gcloud container node-pools list --cluster gresp-cluster --zone us-central1-b

# L4 GPU pool
gcloud container node-pools create gpu-pool --cluster gresp-cluster \
  --zone us-central1-b --node-locations us-central1-a \
  --accelerator type=nvidia-l4,count=1,gpu-driver-version=default \
  --num-nodes=1 --machine-type=g2-standard-4

gcloud container node-pools delete gpu-pool --cluster gresp-cluster --zone us-central1-b --quiet
```

## kubectl (run + monitor the job)
```bash
kubectl get nodes                                   # is the GPU node Ready?
kubectl delete job pandemic-tgnn-train --ignore-not-found
kubectl apply -f ~/pandemic_tgnn/job.yaml
kubectl get pods --selector=job-name=pandemic-tgnn-train

# follow logs
kubectl logs -f $(kubectl get pods --selector=job-name=pandemic-tgnn-train -o jsonpath='{.items[0].metadata.name}')

kubectl describe pod <POD>            # why is it Pending / crashing?
kubectl describe pod <POD> | grep -i image:      # which image digest actually ran
kubectl get job pandemic-tgnn-train -o wide
```

## GCS (data + results)
```bash
gsutil ls gs://YOUR_BUCKET/predictions/
gsutil cp gs://YOUR_BUCKET/predictions/<version>/results_EN.csv .
gsutil cp gs://YOUR_BUCKET/predictions/<version>/run_EN.log .
gsutil cat gs://YOUR_BUCKET/data/config.json        # check country_idx keys
```

## Teardown (stop billing!)
```bash
gcloud container clusters delete gresp-cluster --zone us-central1-b --quiet   # deletes everything
# or just the GPU cost:
gcloud container node-pools delete gpu-pool --cluster gresp-cluster --zone us-central1-b --quiet
```

## Quick sanity flags for this job
- Env in job.yaml: `BUCKET_NAME`, `COUNTRY=EN`, optional `MODELS=MPNN_LSTM`.
- New build → new tag (`:v3`, `:v4`…) AND `imagePullPolicy: Always`.
- Good log lines: `Downloading data for EN only`, `--country EN`, `Using device: cuda`.