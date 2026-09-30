"""Prepare a source-only Sites checkout; artifacts are built from its committed state."""
import json
import shutil

from di_validator import store

root = store.ROOT
target = root/"runtime/cloud-deploy/site-source"
target.mkdir(parents=True, exist_ok=True)
for directory in ["frontend", "deploy"]:
    shutil.copytree(root/directory, target/directory, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("node_modules", "dist", "test-results", "playwright-report", "*.tsbuildinfo"))
(target/".openai").mkdir(exist_ok=True)
shutil.copyfile(root/".openai/hosting.json", target/".openai/hosting.json")
(target/"package.json").write_text(json.dumps({"name": "di-validator-site", "private": True, "type": "module",
    "scripts": {"build": "npm --prefix frontend ci && npm --prefix frontend run build -- --mode cloud && node deploy/build-sites.mjs"}}, indent=2))
(target/".gitignore").write_text("node_modules/\nfrontend/node_modules/\nfrontend/dist/\ndist/\n*.tsbuildinfo\n")
(target/"README.md").write_text("# DI Validator\n\nReact frontend hosted on Sites with an authenticated gateway to Google Cloud Run.\n\nBuild: `npm run build`. The original analysis workspace is maintained separately.\n")
print(target)
