# %% Script to contain the helper functions as part of the API response for Pirate Weather
# Alexander Rey. October 2025


def get_alert_field(details, index, default=""):
    """Return a field from an alert details sequence with a safe default.

    This helper safely extracts an item from a sequence-like `details` by
    index. If the index is out of range the provided `default` value is
    returned instead of raising an IndexError. The function is a small
    convenience used when parsing variable-length alert details.

    Args:
        details (Sequence): A sequence (e.g., list or tuple) containing
            alert detail values.
        index (int): The zero-based index of the desired field in
            `details`.
        default (Any): The value to return if `index` is out of range.
            Defaults to an empty string.

    Returns:
        Any: The value at `details[index]` when available, otherwise
        `default`.
    """

    return details[index] if len(details) > index else default
