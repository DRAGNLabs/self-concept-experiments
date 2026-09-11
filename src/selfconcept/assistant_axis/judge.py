# Derived from safety-research/assistant-axis (https://github.com/safety-research/assistant-axis),
# MIT licensed. See the NOTICE file at the repository root for the full license text.
"""Parse a local judge model's numeric role-adherence score."""

import re


def parse_judge_score(response_text: str) -> int | None:
    """
    Parse the judge's response to extract the numerical score.

    Args:
        response_text: The judge model's response

    Returns:
        Integer score between 0-3, or None if parsing fails
    """
    if not response_text:
        return None

    # Look for numbers in the response
    numbers = re.findall(r'\b(\d+)\b', response_text.strip())

    if not numbers:
        return None

    try:
        score = int(numbers[0])
        if 0 <= score <= 3:
            return score
        return None
    except ValueError:
        return None
