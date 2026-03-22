"""Q*bert move planner — simulation-based search with route guidance."""

import random
from qbert.state import is_valid, MAX_ROW, NUM_CUBES, GameState, pos_to_cube_index
from qbert.sim import (
    make_sim_state, clone, simulate,
    UP, DOWN, LEFT, RIGHT, MOVE_DELTAS,
)
from qbert.strategy import select_mode, score_result, grid_dist, _find_coily

# Re-export constants used by main.py
MOVE_NAMES = {UP: "up-rt", DOWN: "dn-lt", LEFT: "up-lt", RIGHT: "dn-rt"}
MOVE_BUTTONS = {
    UP:    (":IN4", "P1 Up (Up-Right)"),
    DOWN:  (":IN4", "P1 Down (Down-Left)"),
    LEFT:  (":IN4", "P1 Left (Up-Left)"),
    RIGHT: (":IN4", "P1 Right (Down-Right)"),
}
COIN_BUTTON = (":IN1", "Coin 1")
START_BUTTON = (":IN1", "1 Player Start")

SEARCH_DEPTH = 6

# Optimal traversal route: 28 cubes in 28 hops (2 backtrack hops).
# Left edge down, bottom zigzag, right diagonal up, middle fill.
OPTIMAL_ROUTE = [
    (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),  # left edge
    (5, 0), (6, 1), (5, 1), (6, 2), (5, 2), (6, 3), (5, 3),  # bottom zigzag
    (6, 4), (5, 4), (6, 5), (5, 5), (6, 6),                    # bottom right
    (5, 5), (4, 4), (3, 3), (2, 2), (1, 1),                    # right diagonal
    (2, 1), (3, 1), (4, 2), (3, 2), (4, 3),                    # middle fill
]

# Recent position history for oscillation detection
_recent_positions = []
_MAX_RECENT = 8

# Hop counter for purple ball danger window
_hop_count = 0


def neighbors(r, c):
    """Return valid neighbor moves as list of (action, row, col)."""
    return [(a, r + dr, c + dc) for a, (dr, dc) in MOVE_DELTAS.items()
            if is_valid(r + dr, c + dc)]


def _generate_sequences(r, c, depth, max_depth, path):
    """Recursively generate all valid action sequences up to max_depth."""
    if depth >= max_depth:
        yield list(path)
        return
    for action, nr, nc in neighbors(r, c):
        path.append(action)
        yield from _generate_sequences(nr, nc, depth + 1, max_depth, path)
        path.pop()
    # Also yield partial sequences (shorter paths are valid)
    if depth > 0:
        yield list(path)


def _should_take_disc(state):
    """Check if Q*bert should take a disc right now."""
    coily = _find_coily(state)
    if not coily:
        return None
    coily_d = grid_dist(state.qbert[0], state.qbert[1], coily[0], coily[1])
    if coily_d > 3:
        return None
    for disc in state.discs:
        if state.qbert == disc.jump_from:
            return disc.direction
    return None


def _next_route_target(state):
    """Find the next target on the optimal route that still needs coloring.

    Returns the target position, or None if all route targets are colored.
    """
    for pos in OPTIMAL_ROUTE:
        idx = pos_to_cube_index(pos[0], pos[1])
        if idx is not None and state.cube_states[idx] != state.target_color:
            return pos
    return None


def reset_history():
    """Reset position history (call on death, level change, disc ride)."""
    global _recent_positions, _hop_count
    _recent_positions = []
    _hop_count = 0


def decide(state: GameState) -> tuple:
    """Pick the best action using simulation-based search.

    Returns (action, mode) where action is 0-3 and mode is a string.
    """
    global _recent_positions, _hop_count

    row, col = state.qbert
    if not is_valid(row, col):
        return DOWN, "collect"

    valid = neighbors(row, col)
    if not valid:
        return DOWN, "collect"

    # Disc override: if at disc launch point and Coily close, take it
    disc_action = _should_take_disc(state)
    if disc_action is not None:
        return disc_action, "lure"

    mode = select_mode(state)
    sim_state = make_sim_state(state)

    # Compute route target for scoring
    route_target = _next_route_target(state)

    # Collect scores per first action (take best future per action)
    action_scores = {}

    for seq in _generate_sequences(row, col, 0, SEARCH_DEPTH, []):
        if not seq:
            continue
        result = simulate(clone(sim_state), seq)
        score = score_result(result, mode, state, _recent_positions,
                             route_target)
        first = seq[0]
        if first not in action_scores or score > action_scores[first]:
            action_scores[first] = score

    if action_scores:
        best_score = max(action_scores.values())
        best_actions = [a for a, s in action_scores.items()
                        if s >= best_score - 1]
        best_action = random.choice(best_actions)
    else:
        best_action = valid[0][0]

    # Track position history
    _recent_positions.append((row, col))
    if len(_recent_positions) > _MAX_RECENT:
        _recent_positions.pop(0)

    return best_action, mode
