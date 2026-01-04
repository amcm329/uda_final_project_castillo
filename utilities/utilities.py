# ============================================================
# We define utilities (timing)
# ============================================================

def format_seconds(seconds):
    """
    Formats elapsed seconds into a human-readable string.

    Args:
        seconds (float): Elapsed time in seconds.

    Returns:
        str: Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    rem = seconds - 60 * minutes
    return f"{minutes}m {rem:.2f}s"

