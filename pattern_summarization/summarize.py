import re

# Dictionary storing regular expressions to detect specific text patterns
format_compile_list = {
    "bold": r'\*\*(.*?)\*\*',              # Matches bold text format
    # Matches uppercase words (3 or more letters)
    # "uppercase": r'\b[A-Z]{3,}\b',
    "list": r'(?m)^\d+\.\s|^[*+-]\s',          # Matches list item format
    "exclamation": r'!',                 # Matches exclamation marks
    "link": r'http[^\)]*',  # Matches link format
    "emoji": re.compile(
        # r"\s*"  # Preceding spaces
        r"([\U0001F600-\U0001F64F]"  # Emoticons
        r"|[\U0001F300-\U0001F5FF]"  # Miscellaneous Symbols and Pictographs
        r"|[\U0001F680-\U0001F6FF]"  # Transport and Map Symbols
        r"|[\U0001F1E0-\U0001F1FF]"  # Flags (iOS)
        r"|[\U00002700-\U000027BF]"  # Dingbats
        r"|[\U0001F900-\U0001F9FF]"  # Supplemental Symbols and Pictographs
        r"|[\U0001FA70-\U0001FAFF]"  # Symbols and Pictographs Extended-A
        r"|[\U00002600-\U000026FF]"  # Miscellaneous Symbols
        r")",
        re.UNICODE
    )
}

def has_pattern(response, augment_type=None):
    """
    Check if a given response contains a specific pattern or matches a predefined type.

    Args:
        response (str): The text response to be checked for patterns.
        augment_type (str, optional): Specific type of pattern to look for.
            If None, it checks for all defined patterns in 'format_compile_list'.

    Returns:
        bool: True if the response matches the pattern(s), otherwise False.
    """
    try:
        if augment_type is None:
            for pattern in list(format_compile_list.values()):
                if re.search(pattern, response) is not None:
                    return True
            if response.startswith("Sure") or response.startswith("Certainly") or response.startswith("Of course"):
                return True

            return False

        if augment_type in list(format_compile_list.keys()):
            if re.search(format_compile_list[augment_type], response) is None:
                return False
        elif augment_type == "affirmative":
            return response.startswith("Sure") or response.startswith("Certainly") or response.startswith("Of course")

        return True
    except Exception as e:
        return False

from datasets import load_dataset

# List of dataset names to be analyzed
ds_names = [
    "1231czx/biased_ppo_step56_alpaca",
    '1231czx/biased_ppo_step56_uf'
]

for ds_name in ds_names:
    ds = load_dataset(ds_name, split="train")
    print(f"dataset: {ds_name}")

    patterns = ["bold", "list", "exclamation", "link", "emoji", "affirmative"]

    pattern_cnts = {pattern: 0 for pattern in patterns}
    unpattern_cnts = {pattern: 0 for pattern in patterns}

    for d in ds:
        response = d['responses'][0]
        for pattern in patterns:
            if has_pattern(response, pattern):
                pattern_cnts[pattern] += 1
    # Calculate the percentage of samples containing each pattern
    pattern_cnts = {pattern: pattern_cnts[pattern] * 100 / len(ds) for pattern in patterns}

    print(pattern_cnts)
