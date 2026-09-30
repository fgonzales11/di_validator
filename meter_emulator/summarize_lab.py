"""Summarize recorded checks without rerunning or changing their verdicts."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    root = Path(__file__).resolve().parent.parent
    folder = root / "runtime/meter-lab/verification"
    reports = []
    for file in sorted([*folder.glob("fault_*.json"), *folder.glob("pv_*.json")]):
        record = json.loads(file.read_text())
        if isinstance(record, dict) and "case" in record and "checks" in record:
            reports.append(record)
    lines = [
        "# Meter Lab verification evidence",
        "",
        "These results describe the local ARM emulator. They do not establish field accuracy or physical-meter resource qualification.",
        "",
        "| Case | Saved run | Execution | Numerical | Expectations | Delivery |",
        "|---|---|---|---|---|---|",
    ]

    def verdict(record, category):
        checks = [c for c in record["checks"]["checks"] if c["category"] == category]
        return (
            "fail"
            if any(c["verdict"] == "fail" for c in checks)
            else "pass"
            if any(c["verdict"] == "pass" for c in checks)
            else "unavailable"
        )

    for r in reports:
        lines.append(
            f"| {r['case']} | `{r['run_id']}` | {r['state']} | {verdict(r, 'numerical')} | {verdict(r, 'expectation')} | {verdict(r, 'delivery')} |"
        )
    lines += [
        "",
        "## Resource observations",
        "",
        "Inherited agent limits remain 2% CPU, 2,048 KB RAM and 2,048 KB flash. Emulator RSS includes QEMU and diagnostic overhead. CPU is the maximum observed process CPU/actual elapsed-time ratio. The shared container ceiling is 65,000 KB RAM; it is not an agent allocation.",
        "",
        "| Case | Samples | Elapsed s | Peak RSS KB | CPU % | Persistent bytes | Stored bytes | SDK rejections | Bounded drops |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in reports:
        t = r["telemetry"]
        resources = t.get("resources", {})
        lines.append(
            f"| {r['case']} | {t.get('processed', 0)} | {t.get('elapsed_seconds', 0):.2f} | {resources.get('peak_rss_kb', 0)} | {resources.get('max_observed_cpu_percent', 0):.2f} | {t.get('persistent_bytes', 0)} | {t.get('stored_bytes', 0)} | {t.get('sdk_rejected', 0)} | {t.get('dropped', 0)} |"
        )
    overruns = [
        r["case"]
        for r in reports
        if any(r["telemetry"].get("resources", {}).get(key) for key in ["ram_overrun", "cpu_overrun"])
    ]
    if overruns:
        lines += [
            "",
            "Observed agent-policy CPU/RAM overruns: " + ", ".join(overruns) + ". Quotas were not increased.",
        ]
    lines += ["", "## Software and artifacts", ""]
    restart = folder / "worker-restart.json"
    if restart.exists():
        evidence = json.loads(restart.read_text())
        lines.append(
            f"- Worker restart: {evidence['verdict']}; paused sample count and recording clock remained fixed, the WSL process owner survived, and processing resumed (`{evidence['run_id']}`)."
        )
    for file in sorted(folder.glob("*-tests.xml")):
        root = ET.parse(file).getroot()
        suites = [root] if root.get("tests") is not None else list(root.findall("testsuite"))
        if suites:
            counts = {
                key: sum(int(s.get(key, 0)) for s in suites)
                for key in ["tests", "failures", "errors", "skipped"]
            }
            lines.append(
                f"- {file.name}: {counts['tests']} tests, {counts['failures']} failures, {counts['errors']} errors, {counts['skipped']} skipped. Test suites may overlap; do not add their counts."
            )
    build = folder / "frontend-build-result.json"
    if build.exists():
        result = json.loads(build.read_text(encoding="utf-8-sig"))
        lines.append(f"- Frontend build: exit {result['exit_code']} at {result['finished_utc']}.")
    preservation = folder / "preservation.json"
    if preservation.exists():
        result = json.loads(preservation.read_text())
        unchanged = all(item["matches_baseline"] for item in result["preserved"])
        arm = all(item["arm_eabi5"] for item in result["arm"].values())
        lines.append(
            f"- Original emulator binaries and production packages unchanged: {unchanged}; both replay binaries are ARM EABI5: {arm}."
        )
    lines += [
        "- Each case JSON contains independent numerical, expectation, delivery and structural checks. Its ZIP preserves inputs, references, actual diagnostics, acknowledgements, SDK completion records, SQLite rows and build provenance.",
        "- `preservation.json` audits the original emulator and production package checksums; `arm-build.log` records the isolated replay build.",
        "- Browser screenshots, downloads and Playwright reports document desktop/tablet behavior.",
        "- Known adverse PV false positives remain failed expectation checks even when numerical parity passes. Missing independent truth is unavailable, never an accuracy pass.",
        "",
    ]
    destination = folder / "REPORT.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
