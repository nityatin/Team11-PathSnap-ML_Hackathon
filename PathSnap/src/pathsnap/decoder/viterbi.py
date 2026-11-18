from pathsnap.decoder.hmm import (
    emission_log_probability,
    transition_log_probability,
)


def decode_viterbi(
    candidate_groups,
    disconnected_penalty=8.0,
):
    """
    Select the most likely road candidate sequence.

    candidate_groups:
        List of lists. Each inner list contains candidates for one GPS point.

    Each candidate must contain:
        arc_id, from_node_id, to_node_id, probability
    """
    if not candidate_groups:
        return []

    if any(not group for group in candidate_groups):
        raise ValueError("Every GPS observation must have candidates.")

    scores = []
    backpointers = []

    # Initial observation
    first_scores = [
        emission_log_probability(c["probability"])
        for c in candidate_groups[0]
    ]

    scores.append(first_scores)
    backpointers.append([None] * len(first_scores))

    # Remaining observations
    for t in range(1, len(candidate_groups)):
        previous_group = candidate_groups[t - 1]
        current_group = candidate_groups[t]

        current_scores = []
        current_backpointers = []

        for current in current_group:
            emission = emission_log_probability(
                current["probability"]
            )

            possible_scores = [
                scores[t - 1][j]
                + transition_log_probability(
                    previous,
                    current,
                    disconnected_penalty,
                )
                + emission
                for j, previous in enumerate(previous_group)
            ]

            best_previous = max(
                range(len(possible_scores)),
                key=possible_scores.__getitem__,
            )

            current_scores.append(possible_scores[best_previous])
            current_backpointers.append(best_previous)

        scores.append(current_scores)
        backpointers.append(current_backpointers)

    # Backtrack
    last_index = max(
        range(len(scores[-1])),
        key=scores[-1].__getitem__,
    )

    selected = [None] * len(candidate_groups)
    selected[-1] = candidate_groups[-1][last_index]

    for t in range(len(candidate_groups) - 1, 0, -1):
        last_index = backpointers[t][last_index]
        selected[t - 1] = candidate_groups[t - 1][last_index]

    return selected
