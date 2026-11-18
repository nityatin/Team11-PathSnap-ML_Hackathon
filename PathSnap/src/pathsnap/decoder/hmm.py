import math


def emission_log_probability(probability, epsilon=1e-12):
    """Log probability of observing a GPS point given a road candidate."""
    probability = max(epsilon, min(1.0, float(probability)))
    return math.log(probability)


def transition_log_probability(
    previous,
    current,
    disconnected_penalty=8.0,
):
    """
    Basic directed-road transition model.

    Same arc: allowed.
    Connected directed arcs: allowed.
    Disconnected arcs: penalized.
    """
    previous_arc = int(previous["arc_id"])
    current_arc = int(current["arc_id"])

    if previous_arc == current_arc:
        return 0.0

    previous_end = int(previous["to_node_id"])
    current_start = int(current["from_node_id"])

    if previous_end == current_start:
        return 0.0

    return -float(disconnected_penalty)
