def flatten_lists(lst):
    """Remove nested lists."""
    return [item for sublist in lst for item in sublist]
