"""FSRS-6 Spaced Repetition Functions."""


def fsrs_retention(stability, days_elapsed):
    """Calculate current retention probability (0-1) using FSRS-6 forgetting curve."""
    if stability <= 0:
        return 0.0
    return (1 + days_elapsed / (9 * stability)) ** -1


def fsrs_new_stability(old_s, old_d, rating, elapsed_days):
    """Calculate new stability after a review/access.
    rating: 1=forgotten, 2=hard, 3=good, 4=easy
    """
    retention = fsrs_retention(old_s, elapsed_days)

    # FSRS-6 stability update
    if rating == 1:  # forgotten/lapsed
        # Stability drops significantly
        return max(0.1, old_s * 0.2)

    # Success path (rating 2-4)
    # Difficulty modifier: harder memories gain less stability
    d_mod = 1.0 - (old_d - 5) / 20  # range ~0.75-1.25

    # Rating modifier
    r_mod = {2: 0.8, 3: 1.0, 4: 1.3}[rating]

    # Retention modifier: lower retention = bigger stability gain (desirable difficulty)
    ret_mod = 1.0 + (1.0 - retention) * 0.5

    new_s = old_s * (1 + d_mod * r_mod * ret_mod)
    return min(new_s, 365.0)  # Cap at 1 year


def fsrs_new_difficulty(old_d, rating):
    """Update difficulty based on access quality.
    Easier accesses (high rating) lower difficulty.
    """
    # Mean reversion toward 5.0
    delta = {1: 1.5, 2: 0.5, 3: -0.2, 4: -0.8}[rating]
    new_d = old_d + delta
    # Mean reversion
    new_d = new_d * 0.9 + 5.0 * 0.1
    return max(0.1, min(10.0, new_d))


def fsrs_next_interval(stability, desired_retention=0.9):
    """Calculate days until retention drops to desired_retention."""
    if stability <= 0:
        return 0.1
    return stability * 9 * (1/desired_retention - 1)


def fsrs_auto_rating(category, access_count, priority):
    """Auto-determine rating based on how the memory is being used.
    Higher priority + frequent access = easier recall."""
    if access_count == 0:
        return 3  # first access = "good"
    if priority >= 8:
        return 4  # high priority = easy (actively used)
    if priority >= 5:
        return 3  # medium = good
    return 2  # low priority = hard (rarely needed)
