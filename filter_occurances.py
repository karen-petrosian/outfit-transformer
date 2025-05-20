import json
import random
from collections import Counter


def load_data(path):
    """Load the list of outfits from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_data(data, path):
    """Save the list of outfits to a JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def filter_outfits_by_frequency(data, freq_threshold):
    """
    Remove any outfit that contains an item whose overall occurrence
    exceeds freq_threshold.
    """
    # 1) Count each item across all outfits
    freq = Counter()
    for outfit in data:
        freq.update(set(outfit['question']))
    # 2) Identify “too frequent” items
    too_freq_items = {item for item, cnt in freq.items() if cnt > freq_threshold}
    # 3) Filter out any outfit containing one of those items
    return [
        outfit
        for outfit in data
        if not any(item in too_freq_items for item in outfit['question'])
    ]


def balance_outfits(data):
    """
    Iteratively drop outfits so that all items end up with the same
    occurrence count (the minimum count among them).
    """
    outfits = data.copy()
    random.shuffle(outfits)  # randomize removal order

    # Compute the minimal target frequency
    freq = Counter()
    for outf in outfits:
        freq.update(set(outf['question']))
    target = min(freq.values())

    # Greedy loop: while some item is above target, drop one outfit containing it
    while True:
        freq = Counter()
        for outf in outfits:
            freq.update(set(outf['question']))
        # find the most frequent item
        item, cnt = freq.most_common(1)[0]
        if cnt <= target:
            break
        # drop the first outfit that contains this item
        for i, outf in enumerate(outfits):
            if item in outf['question']:
                outfits.pop(i)
                break

    return outfits


if __name__ == '__main__':
    # --- USER PARAMETERS ---
    input_path = r"C:\Users\42pet\Desktop\Lookify\outfit-transformer\datasets\polyvore\nondisjoint\compatibility\test.json"
    output_path = ("balanced_train_15"
                   ""
                   "0.json")

    # Any item appearing more than this will cause its outfits to be removed
    freq_threshold = 150

    # --- RUN ---
    data = load_data(input_path)
    # filtered = filter_outfits_by_frequency(data, freq_threshold)
    # balanced = balance_outfits(filtered)
    # save_data(balanced, output_path)

    print(f"Original outfits: {len(data)}")
    # print(f"After freq‐threshold filter: {len(filtered)}")
    # print(f"After balancing: {len(balanced)}")
    # print(f"Balanced dataset saved to {output_path}")
