"""Q*bert move planner. Prioritizes corners/edges early, uses full ball paths."""

from collections import deque
from qbert.state import is_valid, MAX_ROW, NUM_CUBES, GameState, Enemy
from qbert.predict import predict_coily, predict_coily_n, predict_ball_next, predict_ball_path

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
    """Compute dangerous positions using full ball paths and Coily prediction."""
    dangers = set()
    ball_danger_times = {}
    coily = None
    coily_target = None

    for enemy in state.enemies:
        if enemy.harmless:
            continue
        pos = enemy.pos
        if not is_valid(pos[0], pos[1]):
            continue

        dangers.add(pos)
        if is_valid(enemy.prev_pos[0], enemy.prev_pos[1]):
            dangers.add(enemy.prev_pos)

        if enemy.going_up:
            coily = pos
            target = state.qbert_prev
            if coily == target:
                target = state.qbert
            coily_target = target

            c1 = predict_coily(pos[0], pos[1], target[0], target[1])
            if is_valid(c1[0], c1[1]):
                dangers.add(c1)
                c2 = predict_coily(c1[0], c1[1], state.qbert[0], state.qbert[1])
                if is_valid(c2[0], c2[1]):
                    dangers.add(c2)
        else:
            path = predict_ball_path(pos[0], pos[1], enemy.direction_bits)
            for i, fp in enumerate(path):
                if not is_valid(fp[0], fp[1]):
                    break
                dangers.add(fp)
                if fp not in ball_danger_times or i < ball_danger_times[fp]:
                    ball_danger_times[fp] = i

    return dangers, coily, coily_target, ball_danger_times


def sim_score(qr, qc, coily_pos, visited, depth, max_depth=3):
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
                continue

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

    Strategy:
    - When Coily is far (dist > 5) or absent: visit cubes, prefer corners/edges
    - When Coily is near (dist <= 5): prioritize escape + distance from Coily
    - Never enter dead ends when Coily is within 5
    - Always use full ball path predictions for danger
    """
    row, col = state.qbert
    if not is_valid(row, col):
        return DOWN

    dangers, coily, coily_target, ball_times = get_dangers(state)
    remaining = NUM_CUBES - sum(1 for v in visited.values() if v)

    valid = neighbors(row, col)
    if not valid:
        return DOWN

    coily_dist = grid_dist(row, col, coily[0], coily[1]) if coily else 99
    evading = coily_dist <= 5

    scored = []
    for action, nr, nc in valid:
        if (nr, nc) in dangers:
            scored.append((-1000, action))
            continue

        score = 0.0
        is_new = not visited.get((nr, nc), False)
        escape = len(neighbors(nr, nc))
        nd = grid_dist(nr, nc, coily[0], coily[1]) if coily else 99

        if evading:
            # EVADE MODE: survival first, cubes are a bonus
            # Distance from Coily is the primary objective
            score += nd * 15

            # Escape routes: critical when Coily is near
            if escape <= 1:
                score -= 500  # never enter dead ends while evading
            elif escape <= 2:
                score -= 50
            score += escape * 10  # more exits = better

            # Coily prediction
            if coily_target and is_valid(coily[0], coily[1]):
                c1 = predict_coily(coily[0], coily[1], coily_target[0], coily_target[1])
                if is_valid(c1[0], c1[1]):
                    if c1 == (nr, nc):
                        score -= 200
                    else:
                        score += grid_dist(nr, nc, c1[0], c1[1]) * 5

            # Lookahead
            score += sim_score(nr, nc, coily, visited, 1) * 0.8

            # Still grab new cubes if it doesn't compromise safety
            if is_new and nd > coily_dist:
                score += 30

        else:
            # EXPLORE MODE: Coily is far or absent — visit cubes aggressively
            if is_new:
                base = 50 + max(0, (28 - remaining)) * 5
                # Bonus for corners/edges — get them while it's safe
                if escape <= 1:
                    # Corner: only if no ball arriving soon
                    ball_threat = ball_times.get((nr, nc), 99)
                    if ball_threat <= 2:
                        score -= 100
                    else:
                        score += base + 120  # big bonus: grab corners early!
                elif escape <= 2:
                    score += base + 40  # edges get bonus too
                else:
                    score += base
            else:
                if escape <= 1:
                    score -= 50  # don't re-enter dead ends

            # BFS toward nearest unvisited
            _, d_before = nearest_unvisited(row, col, visited)
            _, d_after = nearest_unvisited(nr, nc, visited)
            if d_after < d_before:
                score += 20
            elif d_after > d_before:
                score -= 5

        scored.append((score, action))

    scored.sort(reverse=True)
    return scored[0][1] if scored else DOWN
