import os
import json
from openai import OpenAI

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
CONTEXT_WINDOW = 128000  # GPT-5's max context
SECRET_PHRASES = {
    0.05: "MARKER_FIVE_PERCENT_AURORA",
    0.25: "MARKER_TWENTY_FIVE_PERCENT_HORIZON", 
    0.50: "MARKER_FIFTY_PERCENT_ZENITH",
    0.75: "MARKER_SEVENTY_FIVE_PERCENT_SOLSTICE",
    0.95: "MARKER_NINETY_FIVE_PERCENT_APEX"
}

class ContextDegradationTest:
    def __init__(self, max_context_tokens, test_name="gpt5_degradation"):
        self.max_context_tokens = max_context_tokens
        self.test_name = test_name
        self.current_tokens = 0
        self.messages = []
        self.injected_phrases = {}
        self.results = {
            "test_config": {},
            "injection_log": [],
            "recall_results": {}
        }
    
    def calculate_tokens(self, text):
        """Estimate token count (use tiktoken for accuracy)"""
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4
    
    def get_context_utilization(self):
        """Return current context fill percentage"""
        return self.current_tokens / self.max_context_tokens
    
    def should_inject_phrase(self):
        """Check if we've crossed a threshold for injection"""
        util = self.get_context_utilization()
        for threshold, phrase in SECRET_PHRASES.items():
            if threshold not in self.injected_phrases and util >= threshold:
                return threshold, phrase
        return None, None
    
    def add_chunk(self, chunk_text, chunk_id):
        """Add a FLORES-200 chunk and check for phrase injection"""
        tokens = self.calculate_tokens(chunk_text)
        
        # Add the actual content
        self.messages.append({
            "role": "user",
            "content": f"[Chunk {chunk_id}]\n{chunk_text}"
        })
        self.current_tokens += tokens
        
        # Check if we need to inject a secret phrase at this threshold
        threshold, phrase = self.should_inject_phrase()
        if threshold:
            self.messages.append({
                "role": "assistant",
                "content": f"Processing. Secret marker: {phrase}. Continuing analysis."
            })
            self.injected_phrases[threshold] = {
                "phrase": phrase,
                "token_count": self.current_tokens,
                "utilization_pct": threshold * 100
            }
            self.results["injection_log"].append({
                "threshold": threshold,
                "phrase": phrase,
                "position_in_context": self.current_tokens
            })
        
        util = self.get_context_utilization()
        print(f"Chunk {chunk_id}: {tokens} tokens | "
              f"Total: {self.current_tokens} | "
              f"Utilization: {util*100:.2f}%")
        
        return util
    
    def run_recall_test(self):
        """At 100% context, test which secret phrases are retrievable"""
        recall_prompt = (
            "Looking back at everything we've discussed, "
            "list any special marker phrases you remember encountering. "
            "Be specific and exact with the marker text."
        )
        
        self.messages.append({
            "role": "user",
            "content": recall_prompt
        })
        
        response = client.chat.completions.create(
            model="gpt-5",  # Replace with actual model name when available
            messages=self.messages,
            temperature=0.2,  # Low temperature for consistency
            max_tokens=500
        )
        
        recall_text = response.choices[0].message.content
        
        # Check which phrases appear in the recall
        for threshold, phrase_data in self.injected_phrases.items():
            phrase = phrase_data["phrase"]
            found = phrase in recall_text
            self.results["recall_results"][phrase] = {
                "injected_at_pct": threshold * 100,
                "recalled": found,
                "position_tokens": phrase_data["token_count"]
            }
        
        self.results["recall_response"] = recall_text
        return self.results["recall_results"]
    
    def save_results(self):
        """Persist test results"""
        output_file = f"{self.test_name}_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_file}")
