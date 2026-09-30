param([string]$Image = 'us-west2-docker.pkg.dev/thtank/di-validator/backend:20260921-2', [switch]$RestoreCatalog)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot

if ($RestoreCatalog) {
    & gcloud run jobs deploy di-validator-migrate --project=thtank --region=us-west2 --image=$Image `
        --service-account=di-validator@thtank.iam.gserviceaccount.com --network=scar-network --subnet=scar-us-west2 `
        --vpc-egress=private-ranges-only --set-secrets=DI_DATABASE_URL=di-validator-database-url:latest `
        '--add-volume=name=workspace,type=cloud-storage,bucket=thtank-di-validator' `
        '--add-volume-mount=volume=workspace,mount-path=/data' --memory=1Gi --cpu=1 --max-retries=0 `
        --command=python '--args=-m,scripts.cloud_snapshot,restore,/data/migration/catalog.json' --execute-now --wait --quiet
    if ($LASTEXITCODE -ne 0) { throw 'Cloud catalog migration failed' }
}

# Deployment manifest contains no plaintext credentials.
$manifest = Join-Path $projectRoot 'runtime/cloud-deploy/cloud-run.generated.yaml'
if ($Image -notmatch '^[a-zA-Z0-9./_:@-]+$') { throw 'Invalid container image reference' }
$definition = (Get-Content -LiteralPath (Join-Path $projectRoot 'deploy/cloud-run.yaml') -Raw) -replace '(?m)^      - image: .+$', "      - image: $Image"
[System.IO.File]::WriteAllText($manifest, $definition)
& gcloud run services replace $manifest --project=thtank --region=us-west2 --quiet
if ($LASTEXITCODE -ne 0) { throw 'Cloud Run deployment failed' }
# Only the dedicated Sites gateway identity may invoke the Cloud Run service.
& gcloud run services add-iam-policy-binding di-validator-api --project=thtank --region=us-west2 `
    --member=serviceAccount:di-validator-gateway@thtank.iam.gserviceaccount.com --role=roles/run.invoker --quiet
if ($LASTEXITCODE -ne 0) { throw 'Could not configure gateway ingress' }
& gcloud run services describe di-validator-api --project=thtank --region=us-west2 --format='value(status.url)'
