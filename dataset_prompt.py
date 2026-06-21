"""Driver: feed FLORES-200 chunks until context is full, then test recall."""
from sequential_prompt import ContextDegradationTest, CONTEXT_WINDOW


def load_chunks(n=1000):
    """Try FLORES-200 from HuggingFace; fall back to a synthetic corpus if offline/unavailable."""
    try:
        from datasets import load_dataset
        # FLORES-200 dev split, English (Latin script)
        ds = load_dataset("Muennighoff/flores200", "eng_Latn", split="dev")
        return [ex["sentence"] for ex in ds.select(range(min(n, len(ds))))]
    except Exception as e:
        print(f"[warn] FLORES load failed ({e}); using synthetic corpus.")
        base = (
            "The quick brown fox jumps over the lazy dog near the river bank. "
            "Climate patterns across the Pacific shifted in measurable ways last year. "
            "Researchers documented increased rainfall and warmer surface temperatures. "
            "Local communities adapted by changing fishing schedules and crop rotations. "
        )
        return [f"Sentence {i}: {base}" for i in range(n)]


def run_full_test():
    chunks = load_chunks(1000)
    test = ContextDegradationTest(max_context_tokens=CONTEXT_WINDOW)

    for chunk_id, chunk in enumerate(chunks):
        util = test.add_chunk(chunk, chunk_id)
        if util >= 1.0:
            print("Context window at 100% — running recall test...")
            break
    else:
        print("Ran out of chunks before filling context — running recall anyway.")

    recall_results = test.run_recall_test()

    print("\n=== RECALL RESULTS ===")
    for phrase, result in recall_results.items():
        status = "RECALLED" if result["recalled"] else "LOST"
        print(
            f"[{status}] {phrase} "
            f"(injected at {result['injected_at_pct']:.1f}%, "
            f"position={result['position_tokens']} tokens)"
        )

    print("\n=== MODEL RECALL RESPONSE ===")
    print(test.results.get("recall_response", ""))

    test.save_results()


if __name__ == "__main__":
    run_full_test()
