"""Context window exhaustion rate experiment.

For each FLORES-200 language in the test set:
  1. Reset the conversation (system prompt only).
  2. Feed FLORES sentences one at a time as user turns.
  3. After every turn (user + assistant), record the *real* tokens consumed
     using Ollama's `prompt_eval_count` (full prompt) + `eval_count` (response).
  4. Stop when cumulative tokens >= num_ctx (window full).

Distinct from `sequential_prompt.py`'s recall test: that one estimates tokens
locally (chars / 4) and only calls the LLM once at the end. This script
measures the real per-turn token cost and exhaustion turn under actual chat
inference, so you can compare exhaustion rates across languages with different
tokenizer fertility (Latin vs. Devanagari vs. Khmer, etc.).

Output: `exhaustion_rate_results.json` — per-language list of
{turn, user_tokens_delta, assistant_tokens, cumulative_tokens, utilization}
plus the turn at which exhaustion happened.
"""
import json
import time
from pathlib import Path

import ollama

MODEL = "llama3.1:8b"
CONTEXT_WINDOW = 4096  # small window so exhaustion is reachable in minutes

LANGS = {
    "eng_Latn": "English",
    "vie_Latn": "Vietnamese",
    "tha_Thai": "Thai",
    "tam_Taml": "Tamil",
    "hin_Deva": "Hindi",
    "khm_Khmr": "Khmer",
}

HERE = Path(__file__).parent
RESULTS_FILE = HERE / "exhaustion_rate_results.json"

SYSTEM_PROMPT = (
    "You are a concise multilingual assistant. For each chunk the user sends, "
    "reply with one short sentence acknowledging the language and topic. "
    "Keep replies under 25 words."
)


def load_sentences(lang_code, n=2000):
    """Load FLORES-200 sentences for `lang_code`.

    Tries HuggingFace `Muennighoff/flores200` (same source as
    `dataset_prompt.py`); falls back to a local `flores200_dataset/` directory
    structure if HF is unavailable.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("Muennighoff/flores200", lang_code, split="dev")
        return [ex["sentence"] for ex in ds.select(range(min(n, len(ds))))]
    except Exception as e:
        local = HERE / "flores200_dataset"
        if local.exists():
            dev = (local / "dev" / f"{lang_code}.dev").read_text(encoding="utf-8").splitlines()
            devtest = (local / "devtest" / f"{lang_code}.devtest").read_text(encoding="utf-8").splitlines()
            return [s.strip() for s in (dev + devtest) if s.strip()][:n]
        raise RuntimeError(f"Could not load FLORES for {lang_code}: {e}")


def run_language(lang_code, lang_name, sentences, window):
    print(f"\n=== {lang_name} ({lang_code}) @ ctx={window} ===")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    turns = []
    cumulative = 0
    exhausted_turn = None

    t0 = time.time()
    for turn_id, sentence in enumerate(sentences):
        messages.append({"role": "user", "content": f"[Chunk {turn_id}] {sentence}"})

        resp = ollama.chat(
            model=MODEL,
            messages=messages,
            options={
                "num_ctx": window,
                "temperature": 0.2,
                "num_predict": 60,
            },
        )
        assistant_text = resp["message"]["content"]
        messages.append({"role": "assistant", "content": assistant_text})

        # `prompt_eval_count` is the total prompt size Ollama tokenized for
        # this call (system + every prior turn + this turn's user message).
        # `eval_count` is the number of tokens the model just generated.
        prompt_total = resp.get("prompt_eval_count", 0)
        assistant_tok = resp.get("eval_count", 0)

        # Tokens visible after this turn (what the *next* call's prompt would
        # see) = prompt this turn saw + tokens we just generated.
        new_cumulative = prompt_total + assistant_tok

        prev_cum = turns[-1]["cumulative_tokens"] if turns else 0
        # First turn's delta includes the one-time system prompt overhead.
        user_delta = prompt_total - prev_cum

        cumulative = new_cumulative
        util = cumulative / window

        turns.append({
            "turn": turn_id,
            "user_tokens_delta": user_delta,
            "assistant_tokens": assistant_tok,
            "prompt_eval_count_total": prompt_total,
            "cumulative_tokens": cumulative,
            "utilization": round(util, 4),
            "assistant_preview": assistant_text[:80].replace("\n", " "),
        })

        print(
            f"  turn {turn_id:>3}: +user={user_delta:>4} +asst={assistant_tok:>3} "
            f"cum={cumulative:>5} util={util*100:5.1f}%"
        )

        if cumulative >= window:
            exhausted_turn = turn_id
            print(f"  >>> EXHAUSTED at turn {turn_id} (cum {cumulative} >= window {window})")
            break

    elapsed = round(time.time() - t0, 2)
    return {
        "language": lang_name,
        "lang_code": lang_code,
        "window": window,
        "model": MODEL,
        "exhausted_turn": exhausted_turn,
        "total_turns": len(turns),
        "final_cumulative_tokens": cumulative,
        "elapsed_seconds": elapsed,
        "turns": turns,
    }


def main():
    state = {"model": MODEL, "window": CONTEXT_WINDOW, "rows": []}
    if RESULTS_FILE.exists():
        try:
            state = json.loads(RESULTS_FILE.read_text())
        except Exception:
            pass
    done = {r["lang_code"] for r in state["rows"] if r.get("total_turns", 0) > 0}

    for lang_code, lang_name in LANGS.items():
        if lang_code in done:
            print(f"[{lang_name}] skip (already recorded)")
            continue
        try:
            sentences = load_sentences(lang_code)
            row = run_language(lang_code, lang_name, sentences, CONTEXT_WINDOW)
            state["rows"].append(row)
            RESULTS_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"[{lang_name}] ERROR: {err}")
            state["rows"].append({
                "language": lang_name, "lang_code": lang_code,
                "window": CONTEXT_WINDOW, "error": err,
            })
            RESULTS_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))

    print(f"\n{'=' * 60}\nEXHAUSTION SUMMARY (window={CONTEXT_WINDOW})\n{'=' * 60}")
    print(f"{'language':>12} | {'turns_to_full':>13} | {'avg_tok/turn':>12} | {'elapsed_s':>9}")
    for r in state["rows"]:
        if "error" in r:
            print(f"{r['language']:>12} | ERROR: {r['error']}")
            continue
        avg = r["final_cumulative_tokens"] / max(1, r["total_turns"])
        ex = r["exhausted_turn"] if r["exhausted_turn"] is not None else "—"
        print(f"{r['language']:>12} | {str(ex):>13} | {avg:>12.1f} | {r['elapsed_seconds']:>9.1f}")


if __name__ == "__main__":
    main()
