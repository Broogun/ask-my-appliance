import sys
import os
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
from pipeline.figure_extractor import extract_figures  # noqa: E402

base_pdf = "data/lg"
base_out = "data/manual_figures/lg"

total_start = time.time()
summary = []
for cat in ["aircon", "fridge", "washer"]:
    for fname in sorted(os.listdir(os.path.join(base_pdf, cat))):
        prefix = fname.split("_")[0]
        model = fname[len(prefix) + 1:-4]  # strip "PREFIX_" and ".pdf"
        pdf_path = os.path.join(base_pdf, cat, fname)
        out_dir = os.path.join(base_out, cat, model)
        t0 = time.time()
        figs = extract_figures(pdf_path, out_dir)
        elapsed = time.time() - t0
        n_figs = sum(len(v) for v in figs.values())
        summary.append((cat, model, n_figs, round(elapsed, 1)))

print("\n=== SUMMARY ===")
for cat, model, n, t in summary:
    print(f"{cat:8s} {model:20s} figs={n:4d} time={t}s")
print(f"\nTOTAL TIME: {round(time.time() - total_start, 1)}s")
