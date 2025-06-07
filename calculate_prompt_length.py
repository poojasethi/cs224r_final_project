import json

def analyze_prompt_lengths(jsonl_path):
    lengths = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            prompt = data.get("prompt", "")
            lengths.append(len(prompt.split()))

    if not lengths:
        print("No prompts found.")
        return

    min_len = min(lengths)
    max_len = max(lengths)
    avg_len = sum(lengths) / len(lengths)

    print(f"Min prompt length (word count): {min_len}")
    print(f"Max prompt length (word count): {max_len}")
    print(f"Average prompt length (word count): {avg_len:.2f}")

# Example usage
analyze_prompt_lengths("evaluation/input/ultrafeedback_heldout_prompt.json")
