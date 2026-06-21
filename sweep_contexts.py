"""Sweep llama3.1:8b across context windows 4k -> 128k using FLORES-200 English.

Per window: fill context with FLORES sentences, inject 5 markers at 5/25/50/75/95%,
then ask the model to recall them. Save results incrementally.
"""
import json
import os
import time
from pathlib import Path

from sequential_prompt import ContextDegradationTest

MODEL = "llama3.1:8b"
WINDOWS = [4096, 8192, 16384, 32768, 65536, 131072]
HERE = Path(__file__).parent
FLORES_DIR = HERE / "flores200_dataset"
RESULTS_FILE = HERE / "sweep_results.json"


def load_flores_english():
    dev = (FLORES_DIR / "dev" / "eng_Latn.dev").read_text(encoding="utf-8").splitlines()
    devtest = (FLORES_DIR / "devtest" / "eng_Latn.devtest").read_text(encoding="utf-8").splitlines()
    sents = [s.strip() for s in dev + devtest if s.strip()]
    print(f"[flores] loaded {len(sents)} English sentences "
          f"({sum(len(s) for s in sents)//4} approx tokens)")
    return sents


def chunks_for_window(sentences, window):
    # rough budget: utilization hits 100% around window tokens; add 30% slack
    approx_token_budget = int(window * 1.3)
    mean_tokens = max(1, sum(len(s) for s in sentences) // 4 // len(sentences))
    needed = approx_token_budget // mean_tokens
    if needed <= len(sentences):
        return sentences[:needed]
    # cycle if we need more
    return (sentences * ((needed // len(sentences)) + 1))[:needed]


def save_sweep(sweep):
    RESULTS_FILE.write_text(json.dumps(sweep, indent=2, ensure_ascii=False))


def run_window(window, sentences):
    print(f"\n{'='*70}\n=== WINDOW {window} ({window//1024}k) ===\n{'='*70}")
    chunks = chunks_for_window(sentences, window)
    print(f"[window {window}] feeding up to {len(chunks)} FLORES chunks")

    test = ContextDegradationTest(
        max_context_tokens=window,
        model=MODEL,
        test_name=f"llama31_8b_window_{window}",
    )

    start = time.time()
    last_print = 0
    for chunk_id, chunk in enumerate(chunks):
        util = test.add_chunk(chunk, chunk_id)
        # Reduce console spam for large windows
        if chunk_id - last_print >= 50 or util >= 1.0:
            last_print = chunk_id
        if util >= 1.0:
            print(f"[window {window}] context full at chunk {chunk_id}")
            break
    fill_time = time.time() - start

    print(f"[window {window}] running recall (this can take a while at large ctx)...")
    recall_start = time.time()
    recall_results = test.run_recall_test()
    recall_time = time.time() - recall_start

    summary = {
        "window": window,
        "fill_seconds": round(fill_time, 2),
        "recall_seconds": round(recall_time, 2),
        "tokens_at_full": test.current_tokens,
        "chunks_fed": len(test.messages),
        "injection_log": test.results["injection_log"],
        "recall_results": recall_results,
        "recall_response": test.results.get("recall_response", ""),
    }
    test.save_results()
    return summary


def main():
    sentences = load_flores_english()

    sweep = {
        "model": MODEL,
        "windows": [],
        "summary_table": [],
    }
    if RESULTS_FILE.exists():
        try:
            sweep = json.loads(RESULTS_FILE.read_text())
            done = {w["window"] for w in sweep["windows"]}
            print(f"[resume] already done: {sorted(done)}")
        except Exception:
            pass
    else:
        done = set()

    for window in WINDOWS:
        if window in {w["window"] for w in sweep["windows"]}:
            continue
        try:
            row = run_window(window, sentences)
            sweep["windows"].append(row)
            save_sweep(sweep)
        except Exception as e:
            print(f"[error window {window}] {type(e).__name__}: {e}")
            sweep["windows"].append({"window": window, "error": f"{type(e).__name__}: {e}"})
            save_sweep(sweep)
            # Continue to next window
            continue

    # Build comparison table
    print(f"\n{'='*70}\n=== SWEEP SUMMARY ===\n{'='*70}")
    print(f"{'window':>8} | {'tokens':>7} | {'fill_s':>7} | {'recall_s':>9} | recall @ 5/25/50/75/95%")
    table = []
    for row in sweep["windows"]:
        if "error" in row:
            print(f"{row['window']:>8} | ERROR: {row['error']}")
            table.append({"window": row["window"], "error": row["error"]})
            continue
        recall = row["recall_results"]
        pcts = [5.0, 25.0, 50.0, 75.0, 95.0]
        marks = []
        for p in pcts:
            hit = next((r for r in recall.values() if r["injected_at_pct"] == p), None)
            marks.append("Y" if hit and hit["recalled"] else "N" if hit else "-")
        row_str = " ".join(marks)
        print(f"{row['window']:>8} | {row['tokens_at_full']:>7} | {row['fill_seconds']:>7.1f} | {row['recall_seconds']:>9.1f} | {row_str}")
        table.append({
            "window": row["window"],
            "tokens": row["tokens_at_full"],
            "fill_s": row["fill_seconds"],
            "recall_s": row["recall_seconds"],
            "recall_by_pct": dict(zip(pcts, marks)),
        })
    sweep["summary_table"] = table
    save_sweep(sweep)
    print(f"\nFull results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
