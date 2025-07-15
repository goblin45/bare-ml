def get_normalized_score_from_error(total_error: float, count: int) -> float:
    if count == 0:
        return 0
    avg_error = total_error / count
    normalized_score = 1 / (1 + avg_error)
    return 100 * normalized_score