import json
import ollama

MODEL = "llama3.1:8b"
CONTEXT_WINDOW = 8192

SECRET_PHRASES = {
    0.05: "MARKER_FIVE_PERCENT_AURORA",
    0.25: "MARKER_TWENTY_FIVE_PERCENT_HORIZON",
    0.50: "MARKER_FIFTY_PERCENT_ZENITH",
    0.75: "MARKER_SEVENTY_FIVE_PERCENT_SOLSTICE",
    0.95: "MARKER_NINETY_FIVE_PERCENT_APEX",
}


class ContextDegradationTest:
    def __init__(self, max_context_tokens=CONTEXT_WINDOW, model=MODEL, test_name="llama31_8b_degradation"):
        self.max_context_tokens = max_context_tokens
        self.model = model
        self.test_name = test_name
        self.current_tokens = 0
        self.messages = []
        self.injected_phrases = {}
        self.results = {
            "test_config": {
                "model": model,
                "max_context_tokens": max_context_tokens,
                "thresholds": list(SECRET_PHRASES.keys()),
            },
            "injection_log": [],
            "recall_results": {},
        }

    def calculate_tokens(self, text):
        # Rough estimate: 1 token ~ 4 characters
        return max(1, len(text) // 4)

    def get_context_utilization(self):
        return self.current_tokens / self.max_context_tokens

    def should_inject_phrase(self):
        util = self.get_context_utilization()
        for threshold, phrase in SECRET_PHRASES.items():
            if threshold not in self.injected_phrases and util >= threshold:
                return threshold, phrase
        return None, None

    def add_chunk(self, chunk_text, chunk_id):
        tokens = self.calculate_tokens(chunk_text)

        self.messages.append({
            "role": "user",
            "content": f"[Chunk {chunk_id}]\n{chunk_text}",
        })
        self.current_tokens += tokens

        threshold, phrase = self.should_inject_phrase()
        if threshold:
            assistant_msg = f"Processing. Secret marker: {phrase}. Continuing analysis."
            self.messages.append({"role": "assistant", "content": assistant_msg})
            self.current_tokens += self.calculate_tokens(assistant_msg)
            self.injected_phrases[threshold] = {
                "phrase": phrase,
                "token_count": self.current_tokens,
                "utilization_pct": threshold * 100,
            }
            self.results["injection_log"].append({
                "threshold": threshold,
                "phrase": phrase,
                "position_in_context": self.current_tokens,
            })
            print(f"  >>> INJECTED @ {threshold*100:.0f}%: {phrase}")

        util = self.get_context_utilization()
        # Only print every 50 chunks (or on injection) to keep large-window logs readable
        if chunk_id % 50 == 0 or util >= 1.0:
            print(
                f"Chunk {chunk_id}: {tokens} tokens | "
                f"Total: {self.current_tokens} | "
                f"Utilization: {util*100:.2f}%"
            )
        return util

    def run_recall_test(self):
        recall_prompt = (
            "Looking back at everything we've discussed, "
            "list any special marker phrases you remember encountering. "
            "Be specific and exact with the marker text."
        )
        self.messages.append({"role": "user", "content": recall_prompt})

        response = ollama.chat(
            model=self.model,
            messages=self.messages,
            options={
                "num_ctx": self.max_context_tokens,
                "temperature": 0.2,
                "num_predict": 500,
            },
        )
        recall_text = response["message"]["content"]

        for threshold, phrase_data in self.injected_phrases.items():
            phrase = phrase_data["phrase"]
            found = phrase in recall_text
            self.results["recall_results"][phrase] = {
                "injected_at_pct": threshold * 100,
                "recalled": found,
                "position_tokens": phrase_data["token_count"],
            }

        self.results["recall_response"] = recall_text
        return self.results["recall_results"]

    def save_results(self):
        output_file = f"{self.test_name}_results.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_file}")
