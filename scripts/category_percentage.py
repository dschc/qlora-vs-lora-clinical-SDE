import json
from collections import defaultdict

def calculate_category_percentages(file_path):
    # Load JSON data from file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize a dictionary to count occurrences
    category_count = defaultdict(int)
    total_records = len(data)

    # Iterate through each record and count metadata categories
    for record in data:
        metadata = record.get('summary', {})
        for category, value in metadata.items():
            if value and value.strip().lower() != "n/a":
                category_count[category] += 1

    # Calculate presence percentages
    category_percentages = {
        category: (count / total_records) * 100 for category, count in category_count.items()
    }

    # Print the results in a nice format
    print("\nMetadata Presence Percentages:\n")
    print("+------------------------------+---------+")
    print("|         Category             |  %      |")
    print("+------------------------------+---------+")
    for category, percentage in sorted(category_percentages.items()):
        print(f"| {category.ljust(28)} | {percentage:6.2f}% |")
    print("+------------------------------+---------+")

    return category_percentages

# Example usage
if __name__ == "__main__":
    file_path = "/home/psig/elmtex_prabin/data/en/train.json"
    percentages = calculate_category_percentages(file_path)
