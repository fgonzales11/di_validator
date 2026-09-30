# Hosted DI Validator

The hosted frontend runs on OpenAI Sites. Its server-side gateway forwards API and notebook requests through Google API Gateway to a private Google Cloud Run service. The site is private to its owner. The restricted API key and application gateway credential are stored in Sites and Google Secret Manager; neither is included in browser JavaScript.

The workstation application still uses its local SQLite database and files. The hosted application uses a separate PostgreSQL database and a private Cloud Storage bucket. Uploads and experiments created in either environment do not automatically synchronize with the other.

## Resources

| Resource | Value |
| --- | --- |
| Sites project | `appgprj_6ab0bec8acf8819197663f9d2148dd50` |
| Frontend | `https://di-validator.n2wm787gnp.chatgpt.site` |
| Google Cloud project / region | `thtank` / `us-west2` |
| Cloud Run service | `di-validator-api` |
| Cloud Run URL | `https://di-validator-api-7u6s6ht2pq-wl.a.run.app` |
| API Gateway API / configuration | `di-validator` / `di-validator-v1` |
| API Gateway URL | `https://di-validator-cts7goc1.wl.gateway.dev` |
| Artifact Registry | `us-west2-docker.pkg.dev/thtank/di-validator/backend` |
| Cloud SQL database | `di_validator` on `scar-postgres` |
| Storage bucket | `gs://thtank-di-validator` |
| Service account | `di-validator@thtank.iam.gserviceaccount.com` |
| Gateway invoker identity | `di-validator-gateway@thtank.iam.gserviceaccount.com` |

Cloud Run uses the existing `scar-network` / `scar-us-west2` network to reach the database over its private address. The application database has a dedicated database user. PostgreSQL stores metadata, jobs, and logs; Cloud Storage stores source files, Parquet datasets, and result artifacts. SQLite is not used on the bucket mount.

The service keeps one instance available with 4 CPUs and 8 GiB of memory. CPU remains allocated between requests so queued analysis can run in a separate worker process. A PostgreSQL advisory lock permits only one worker across overlapping service revisions. Interrupted jobs retain their configuration and can be rerun. This configuration incurs ongoing Cloud Run, Cloud SQL, and storage charges while enabled.

## Updating the backend

From the repository root, build and push the image using `deploy/Dockerfile`. Update the image tag in `deploy/cloud-run.yaml`, then run `scripts/deploy_cloud.ps1` with the same image tag. Do not use `-RestoreCatalog` for routine updates: that option is only for initial migration into an empty application database.

To build without Docker Desktop, run `python -m scripts.build_cloud`. It archives only the declared application inputs, uploads them into the private bucket, and starts the build described in `deploy/cloud-build.yaml`. Wait for that build to succeed before deploying its image. Keep the image tag consistent across the build configuration and service manifest.

The image includes JupyterLite and its Python runtime, COMTRADE reference recordings, and CPU forecasting environments. The forecasting build explicitly checks that neither `huggingface_hub` nor `transformers` is installed in the optional provider environments. Existing model compatibility checks still apply; historical fitted models are not automatically portable across different Python or library versions. Train a new hosted run when an older model's environment is incompatible.

Credentials are read from Secret Manager at runtime. Do not place them in the image, source checkout, frontend build variables, or this document. Cloud Run requires IAM authentication, supplied by API Gateway's managed service identity. No service-account private key is created. The application's gateway token additionally protects data routes. End users access data through the private Sites frontend.

`deploy/api-gateway.yaml` declares the authenticated backend routing. After creating its API configuration, `python -m scripts.provision_api_bridge` enables the managed API, stores a key restricted to that API in Secret Manager, and creates the gateway. Organization policies restricting public IAM bindings and service-account keys remain in effect.

API Gateway limits individual requests and responses to 32 MB and does not stream them. For larger source files, upload into `gs://thtank-di-validator/sources/` with the authenticated Google Cloud CLI, then register the file in Datasets. Analysis and notebook reads use bounded data windows; the existing large AMI sources are already registered. Large exported artifacts remain accessible to the project owner in the private bucket. See [Google's payload limits](https://docs.cloud.google.com/api-gateway/docs/quotas).

## Updating the frontend

`python -m scripts.prepare_site` prepares the isolated source checkout under `runtime/cloud-deploy/site-source`. Commit and push that exact source state to the Sites repository, build it with the cloud environment, then run `python -m scripts.package_site` to package the completed output. Save the resulting archive with the full pushed commit SHA and publish the saved version using the private Sites deployment operation.

Sites runtime variables are `DI_BACKEND_URL` (the API Gateway URL), secret `DI_API_KEY`, and secret `DI_GATEWAY_TOKEN`. Updating them requires publishing a saved version to apply the new environment revision. Preserve the site's owner-only audience unless sharing is explicitly requested.

The Sites worker supplies `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` on both the application and notebook responses. JupyterLite needs this browser isolation for its shared-memory filesystem: without it, the Python kernel can start while failing to import notebook helper files. Keep notebook resources on the same authenticated origin and retain these headers when changing the gateway. See [JupyterLite's filesystem requirements](https://jupyterlite.readthedocs.io/en/latest/howto/content/python.html).

## Initial migration and recovery

`python -m scripts.cloud_snapshot export runtime/cloud-deploy/snapshot` makes a consistent copy of the local metadata database without changing the original. The exported JSON rewrites workspace paths for the mounted cloud workspace. `python -m scripts.upload_cloud_data` copies original data, normalized datasets, completed results, and the migration snapshot into the private bucket.

The migration job restores the snapshot transactionally and refuses to overwrite a populated database. It marks copied pending/running jobs as interrupted. For recovery, retain both the database backup and the bucket artifacts: neither alone contains the complete workspace. Original workstation files remain available independently.

Inspect service status with `gcloud run services describe di-validator-api --project=thtank --region=us-west2`. View service logs with `gcloud run services logs read di-validator-api --project=thtank --region=us-west2`. These commands do not require exposing the gateway credential.
