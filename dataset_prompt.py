# Load FLORES-200 (example with HuggingFace)
from datasets import load_dataset

def run_full_test():
    # Load dataset
    flores = load_dataset("facebook/flores", "en_XX")  # English
    chunks = [ex["sentence"] for ex in flores["dev"][:1000]]
    
    # Initialize test
    test = ContextDegradationTest(max_context_tokens=128000)
    
    # Feed chunks sequentially until 100%
    chunk_id = 0
    for chunk in chunks:
        util = test.add_chunk(chunk, chunk_id)
        chunk_id += 1
        
        if util >= 1.0:  # Context is full
            print("Context window at 100%—running recall test...")
            break
    
    # Run recall test when context is full
    recall_results = test.run_recall_test()
    
    # Analyze results
    print("\n=== RECALL RESULTS ===")
    for phrase, result in recall_results.items():
        status = "✓ RECALLED" if result["recalled"] else "✗ LOST"
        print(f"{status} | {phrase} "
              f"(injected at {result['injected_at_pct']:.1f}%)")
    
    test.save_results()

if __name__ == "__main__":
    run_full_test()
