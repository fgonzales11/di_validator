"""Disk-backed 24-hour / four-channel / 1 kHz acceptance benchmark.

Run: uv run python -m scripts.benchmark
Uses runtime/benchmark independently of the interactive application.
"""
import json
import os
import threading
import time

import psutil

from di_validator import store


def main(fresh=False):
    os.environ["DI_WORKSPACE"] = str(store.ROOT / "runtime" / "benchmark")
    from di_validator.events import run, synthetic
    from di_validator.query import recording_window
    store.initialize()
    process, peak, stopped = psutil.Process(), [0], threading.Event()
    def monitor():
        while not stopped.wait(.1):
            peak[0] = max(peak[0], process.memory_info().rss)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    started = time.perf_counter()
    existing = [] if fresh else [d for d in store.listing("dataset") if d.get("duration_seconds", 0) > 86399]
    dataset = existing[0] if existing else synthetic(dict(name="24-hour acceptance benchmark", duration_seconds=86400, sample_rate=1000))
    generation_seconds = time.perf_counter()-started
    generation_peak = peak[0]
    (store.workspace()/"generation.json").write_text(store.encode(dict(dataset_id=dataset["id"],
        generated=not bool(existing), seconds=generation_seconds, peak_rss_bytes=generation_peak)),encoding="utf-8")
    print(f"Indexed {dataset['valid_rows']:,} samples in {generation_seconds:.1f}s", flush=True)
    timings = []
    for start, end in [(0,86399.999),(9.5,10.5),(43000,43002),(80000,86000)]:
        measurements = []
        for repeat in range(3):
            began = time.perf_counter()
            output = recording_window(dataset["id"],start,end)
            measurements.append(time.perf_counter()-began)
            assert all(len(t["x"]) <= 10000 for t in output["traces"])
        timings.append(dict(start=start,end=end,seconds=measurements))
    started = time.perf_counter()
    result = run(dict(dataset_id=dataset["id"], name="24-hour native detector benchmark"))
    processing_seconds = time.perf_counter()-started
    stopped.set(); monitor_thread.join()
    report = dict(dataset_id=dataset["id"], native_samples=dataset["valid_rows"], channels=len(dataset["channels"]),
                  generation_seconds=generation_seconds, detector_seconds=processing_seconds,
                  generated=not bool(existing), generation_peak_rss_bytes=generation_peak,
                  peak_rss_bytes=peak[0], peak_rss_gib=peak[0]/2**30, chart_queries=timings,
                  bounded_memory_pass=peak[0]<8*2**30, warm_queries_under_2s=all(max(r["seconds"][1:])<2 for r in timings),
                  metrics=result["metrics"], platform=process.exe())
    path=store.workspace()/"benchmark.json"
    path.write_text(store.encode(report),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--fresh",action="store_true",help="Include fresh generation in the memory benchmark")
    main(parser.parse_args().fresh)
