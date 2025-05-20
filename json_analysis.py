import json
from collections import defaultdict
import matplotlib.pyplot as plt

def count_id_occurrences(input_path):
    # Load JSON data from the input file
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Count appearances
    count = defaultdict(int)

    for entry in data:
        unique_ids = set(entry["question"])  # To avoid double counting within the same question
        for q_id in unique_ids:
            count[q_id] += 1

    # Convert to normal dict
    result = dict(count)
    return result

if __name__ == "__main__":
    input_path = r"C:\Users\42pet\Desktop\Lookify\outfit-transformer\datasets\polyvore\nondisjoint\compatibility\train.json"  # <-- Change to your actual file path
    output = count_id_occurrences(input_path)

    # Sort by occurrences descending
    sorted_output = dict(sorted(output.items(), key=lambda item: item[1], reverse=True))

    # Pretty print to console
    print(json.dumps(sorted_output, indent=4))

    # Save sorted output to a file
    with open("output.json", "w") as f:
        json.dump(sorted_output, f, indent=4)

    # Plotting
    ids = list(sorted_output.keys())
    counts = list(sorted_output.values())

    plt.figure(figsize=(15, 6))
    plt.bar(range(len(ids)), counts, tick_label=ids)
    plt.xticks([], [])  # Hide x-axis labels if too many IDs
