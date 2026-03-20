"""Q*bert move planner. Three modes: EXPLORE, EVADE, FINISH."""

from collections import deque
from qbert.state import is_valid, MAX_ROW, NUM_CUBES, GameState, Enemy
from qbert.predict import predict_coily, predict_coily_n, predict_ball_next

# Directions: action index → (delta_row, delta_col)
UP    = 0  # Up-Right on screen: (-1, 0) in grid
DOWN  = 1  # Down-Left: (+1, 0)
LEFT  = 2  # Up-Left: (-1, -1)
RIGHT = 3  # Down-Right: (+1, +1)

MOVE_DELTAS = {UP: (-1, 0), DOWN: (+1, 0), LEFT: (-1, -1), RIGHT: (+1, +1)}
MOVE_NAMES = {UP: "up-rt", DOWN: "dn-lt", LEFT: "up-lt", RIGHT: "dn-rt"}

# MAME I/O port mappings for Q*bert
MOVE_BUTTONS = {
    UP:    (":IN4", "P1 Up (Up-Right)"),
    DOWN:  (":IN4", "P1 Down (Down-Left)"),
    LEFT:  (":IN4", "P1 Left (Up-Left)"),
    RIGHT: (":IN4", "P1 Right (Down-Right)"),
}
COIN_BUTTON = (":IN1", "Coin 1")
START_BUTTON = (":IN1", "1 Player Start")


def neighbors(r, c):
    """Return valid (action, row, col) neighbors."""
    return [(a, r + dr, c + dc) for a, (dr, dc) in MOVE_DELTAS.items()
            if is_valid(r + dr, c + dc)]


def grid_dist(r1, c1, r2, c2):
    """Distance on Q*bert grid (accounts for diagonal movement)."""
    dr = r2 - r1
    dc = c2 - c1
    if dr >= 0 and dc >= 0:
        return max(dr, dc)
    elif dr <= 0 and dc <= 0:
        return max(-dr, -dc)
    else:
        return abs(dr) + abs(dc)


def nearest_unvisited(row, col, visited):
    """BFS to find nearest unvisited cube. Returns (pos, distance)."""
    seen = {(row, col)}
    q = deque([(row, col, 0)])
    while q:
        cr, cc, dist = q.popleft()
        for _, nr, nc in neighbors(cr, cc):
            if (nr, nc) not in seen:
                if not visited.get((nr, nc), False):
                    return (nr, nc), dist + 1
                seen.add((nr, nc))
                q.append((nr, nc, dist + 1))
    return None, 999


def get_dangers(state: GameState):
    """Compute dangerous positions and identify Coily."""
    dangers = set()
    coily = None
    coily_target = None  # what Coily is chasing

    for enemy in state.enemies:
        if enemy.harmless:
            continue
        pos = enemy.pos
        if not is_valid(pos[0], pos[1]):
            continue

        # Current and previous positions are dangerous (mid-hop catch)
        dangers.add(pos)
        if is_valid(enemy.prev_pos[0], enemy.prev_pos[1]):
            dangers.add(enemy.prev_pos)

        if enemy.going_up:
            # Coily — predict future positions
            coily = pos
            # Coily chases Q*bert's PREVIOUS position
            target = state.qbert_prev
            if coily == target:
                target = state.qbert  # fallback to current
            coily_target = target

            # Predict 1 and 2 hops ahead
            c1 = predict_coily(pos[0], pos[1], target[0], target[1])
            if is_valid(c1[0], c1[1]):
                dangers.add(c1)
                c2 = predict_coily(c1[0], c1[1], state.qbert[0], state.qbert[1])
                if is_valid(c2[0], c2[1]):
                    dangers.add(c2)
        else:
            # Ball — predict next bounce
            ball_next = predict_ball_next(pos[0], pos[1], enemy.direction_bits)
            if is_valid(ball_next[0], ball_next[1]):
                dangers.add(ball_next)

    return dangers, coily, coily_target


def sim_score(qr, qc, coily_pos, visited, depth, max_depth=4):
    """Simulate Q*bert + Coily lookahead. Returns a heuristic score."""
    if depth >= max_depth:
        if coily_pos and is_valid(coily_pos[0], coily_pos[1]):
            return grid_dist(qr, qc, coily_pos[0], coily_pos[1]) * 3
        return 10

    valid = neighbors(qr, qc)
    if not valid:
        return -200

    best = -999
    for _, nr, nc in valid:
        if coily_pos and (nr, nc) == coily_pos:
            continue

        next_coily = None
        if coily_pos and is_valid(coily_pos[0], coily_pos[1]):
            next_coily = predict_coily(coily_pos[0], coily_pos[1], qr, qc)
            if not is_valid(next_coily[0], next_coily[1]):
                next_coily = None
            elif next_coily == (nr, nc):
                continue  # would be caught

        s = 0
        if not visited.get((nr, nc), False):
            s += 30
        escape = len(neighbors(nr, nc))
        cd = grid_dist(nr, nc, coily_pos[0], coily_pos[1]) if coily_pos else 99
        if escape <= 1 and cd <= 4:
            s -= 100
        if next_coily:
            s += grid_dist(nr, nc, next_coily[0], next_coily[1]) * 2
        s += sim_score(nr, nc, next_coily, visited, depth + 1, max_depth) * 0.5
        best = max(best, s)

    return best if best > -999 else -200


def decide(state: GameState, visited: dict) -> int:
    """Pick the best action given current state. Returns action index (0-3).

    Three modes:
    - EXPLORE: no nearby enemies → BFS to nearest unvisited
    - EVADE: Coily within 4 → lookahead simulation
    - FINISH: ≤3 cubes left → rush remaining cubes when safe
    """
    row, col = state.qbert
    if not is_valid(row, col):
        return DOWN

    dangers, coily, coily_target = get_dangers(state)
    remaining = NUM_CUBES - sum(1 for v in visited.values() if v)

    valid = neighbors(row, col)
    if not valid:
        return DOWN

    coily_dist = grid_dist(row, col, coily[0], coily[1]) if coily else 99

    scored = []
    for action, nr, nc in valid:
        # HARD BLOCK: never jump to a dangerous square
        if (nr, nc) in dangers:
            scored.append((-1000, action))
            continue

        score = 0.0

        # New cube bonus — increases as completion approaches
        if not visited.get((nr, nc), False):
            score += 50 + max(0, (28 - remaining)) * 5

        # Escape route analysis
        escape = len(neighbors(nr, nc))

        if coily:
            nd = grid_dist(nr, nc, coily[0], coily[1])
            # Dead-end avoidance when Coily is close
            if escape <= 1 and coily_dist <= 4:
                score -= 200
            elif escape <= 1:
                score -= 10
            elif escape <= 2 and coily_dist <= 3:
                score -= 20

            # Coily prediction + distance scoring
            if coily_target and is_valid(coily[0], coily[1]):
                c1 = predict_coily(coily[0], coily[1], coily_target[0], coily_target[1])
                if is_valid(c1[0], c1[1]):
                    if c1 == (nr, nc):
                        score -= 100
                    else:
                        score += grid_dist(nr, nc, c1[0], c1[1]) * 3

            # Lookahead simulation
            score += sim_score(nr, nc, coily, visited, 1) * 0.6
        else:
            if escape <= 1:
                score -= 10

        # BFS pull toward unvisited cubes
        _, d_before = nearest_unvisited(row, col, visited)
        _, d_after = nearest_unvisited(nr, nc, visited)
        if d_after < d_before:
            score += 15

        scored.append((score, action))

    scored.sort(reverse=True)
    return scored[0][1] if scored else DOWN
