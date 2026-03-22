"""Q*bert move planner. Enemy prediction + safe BFS routing."""

from collections import deque
from qbert.state import is_valid, MAX_ROW, NUM_CUBES, GameState, Enemy
from qbert.predict import predict_coily, predict_ball_path, predict_ball_next

# Action constants
UP    = 0  # Up-Right: (-1, 0)
DOWN  = 1  # Down-Left: (+1, 0)
LEFT  = 2  # Up-Left: (-1, -1)
RIGHT = 3  # Down-Right: (+1, +1)

MOVE_DELTAS = {UP: (-1, 0), DOWN: (+1, 0), LEFT: (-1, -1), RIGHT: (+1, +1)}
MOVE_NAMES = {UP: "up-rt", DOWN: "dn-lt", LEFT: "up-lt", RIGHT: "dn-rt"}

MOVE_BUTTONS = {
    UP:    (":IN4", "P1 Up (Up-Right)"),
    DOWN:  (":IN4", "P1 Down (Down-Left)"),
    LEFT:  (":IN4", "P1 Left (Up-Left)"),
    RIGHT: (":IN4", "P1 Right (Down-Right)"),
}
COIN_BUTTON = (":IN1", "Coin 1")
START_BUTTON = (":IN1", "1 Player Start")

# Measured hop intervals (frames between grid-word updates)
QBERT_INTERVAL = 18
BALL_INTERVAL = 43
COILY_INTERVAL = 47


def neighbors(r, c):
    """Return valid neighbor moves as list of (action, row, col)."""
    return [(a, r + dr, c + dc) for a, (dr, dc) in MOVE_DELTAS.items()
            if is_valid(r + dr, c + dc)]


def grid_dist(r1, c1, r2, c2):
    """Minimum hops between two grid positions."""
    dr = r2 - r1
    dc = c2 - c1
    if dr >= 0 and dc >= 0:
        return max(dr, dc)
    if dr <= 0 and dc <= 0:
        return max(-dr, -dc)
    return abs(dr) + abs(dc)


def nearest_unvisited(row, col, visited):
    """BFS to find closest unvisited cube. Returns ((r,c), dist) or (None, 999)."""
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


# ---------------------------------------------------------------------------
# Enemy position prediction
# ---------------------------------------------------------------------------

def _predict_enemy_at_step(enemy, step, qbert_prev, qbert_pos):
    """Predict where an enemy will be after `step` Q*bert hops.

    Returns a set of positions the enemy occupies (current + each hop destination)
    to catch mid-hop collisions.
    """
    elapsed = step * QBERT_INTERVAL
    r, c = enemy.pos

    positions = {(r, c)}
    # Include previous position (ROM collision checks qbert vs enemy.prev)
    if is_valid(enemy.prev_pos[0], enemy.prev_pos[1]):
        positions.add(enemy.prev_pos)

    if enemy.etype == "coily":
        hops = elapsed // COILY_INTERVAL
        cr, cc = r, c
        target = qbert_prev if (cr, cc) != qbert_prev else qbert_pos
        for _ in range(hops):
            nr, nc = predict_coily(cr, cc, target[0], target[1])
            if not is_valid(nr, nc):
                break
            positions.add((nr, nc))
            cr, cc = nr, nc
            # Coily re-evaluates target each hop
            target = qbert_prev if (cr, cc) != qbert_prev else qbert_pos
    else:
        # Ball: fixed path from direction_bits
        hops = elapsed // BALL_INTERVAL
        br, bc = r, c
        bits = enemy.direction_bits
        for _ in range(min(hops, 7)):
            if bits & 1:
                nr, nc = br + 1, bc + 1
            else:
                nr, nc = br + 1, bc
            bits >>= 1
            if not is_valid(nr, nc):
                break
            positions.add((nr, nc))
            br, bc = nr, nc

    return positions


def _danger_set(state, step):
    """Compute set of dangerous positions at a given Q*bert hop step."""
    dangers = set()
    for e in state.enemies:
        if e.harmless:
            continue
        if not is_valid(e.pos[0], e.pos[1]):
            continue
        dangers |= _predict_enemy_at_step(
            e, step, state.qbert_prev, state.qbert
        )
    return dangers


# ---------------------------------------------------------------------------
# BFS route search
# ---------------------------------------------------------------------------

def _find_coily(state):
    """Return Coily's position or None."""
    for e in state.enemies:
        if e.etype == "coily" and not e.harmless and is_valid(e.pos[0], e.pos[1]):
            return e.pos
    return None


def _search_routes(state, visited, max_depth=7):
    """BFS up to max_depth hops. Returns list of (first_action, new_cubes,
    escape_routes, path_length, coily_distance) tuples."""
    routes = []
    start = state.qbert
    coily = _find_coily(state)

    # (row, col, step, first_action, visited_in_path set, new_cubes)
    q = deque()

    # Step 1: expand from start
    danger_1 = _danger_set(state, 1)
    for action, nr, nc in neighbors(start[0], start[1]):
        if (nr, nc) in danger_1:
            continue
        new = 1 if not visited.get((nr, nc), False) else 0
        coily_d = grid_dist(nr, nc, coily[0], coily[1]) if coily else 99
        escape = len(neighbors(nr, nc))
        routes.append((action, new, escape, 1, coily_d))
        path_set = frozenset({start, (nr, nc)})
        q.append((nr, nc, 1, action, path_set, new))

    while q:
        cr, cc, step, first_action, path_set, new_cubes = q.popleft()
        if step >= max_depth:
            continue

        next_step = step + 1
        danger = _danger_set(state, next_step)

        for _, nr, nc in neighbors(cr, cc):
            if (nr, nc) in danger:
                continue
            # Avoid revisiting recent path positions (loop avoidance)
            if (nr, nc) in path_set:
                continue

            new = new_cubes + (1 if not visited.get((nr, nc), False) else 0)
            coily_d = grid_dist(nr, nc, coily[0], coily[1]) if coily else 99
            escape = len(neighbors(nr, nc))
            routes.append((first_action, new, escape, next_step, coily_d))

            if next_step < max_depth:
                new_path = path_set | {(nr, nc)}
                q.append((nr, nc, next_step, first_action, new_path, new))

    return routes


# ---------------------------------------------------------------------------
# Main decision
# ---------------------------------------------------------------------------

def decide(state: GameState, visited: dict, qbert_prev_known=None, debug=False) -> int:
    """Pick the best action. Returns action 0-3."""
    row, col = state.qbert
    if not is_valid(row, col):
        return DOWN

    valid = neighbors(row, col)
    if not valid:
        return DOWN

    coily = _find_coily(state)
    cubes_done = NUM_CUBES - state.remaining_cubes

    # --- Disc: immediate use ---
    # If Q*bert is at a disc launch point and Coily is close, take the disc.
    if coily:
        coily_d = grid_dist(row, col, coily[0], coily[1])
        for disc in state.discs:
            if (row, col) == disc.jump_from and coily_d <= 2:
                return disc.direction

    # --- BFS route search ---
    routes = _search_routes(state, visited)

    # --- Disc: route toward disc when Coily is active and most cubes done ---
    disc_target = None
    coily_d = grid_dist(row, col, coily[0], coily[1]) if coily else 99
    if coily and state.discs:
        if coily_d <= 5 or cubes_done >= 20:
            best_dd = 999
            for disc in state.discs:
                dd = grid_dist(row, col, disc.jump_from[0], disc.jump_from[1])
                if dd < best_dd:
                    best_dd = dd
                    disc_target = disc.jump_from

    # --- Score routes ---
    if routes:
        action_scores = {}
        for first_action, new_cubes, escape, path_len, coily_dist in routes:
            score = (new_cubes * 100
                     + escape * 50
                     + coily_dist * 20
                     + path_len * 10)

            # Bonus for moving toward disc
            if disc_target and coily:
                dr, dc = MOVE_DELTAS[first_action]
                nr, nc = row + dr, col + dc
                d_before = grid_dist(row, col, disc_target[0], disc_target[1])
                d_after = grid_dist(nr, nc, disc_target[0], disc_target[1])
                if d_after < d_before:
                    score += 200 if cubes_done >= 20 else 100

            if first_action not in action_scores or score > action_scores[first_action]:
                action_scores[first_action] = score

        return max(action_scores, key=action_scores.get)

    # --- Fallback: no safe routes found ---
    # Pick move furthest from Coily that is not on an enemy's current position
    enemy_positions = set()
    for e in state.enemies:
        if not e.harmless and is_valid(e.pos[0], e.pos[1]):
            enemy_positions.add(e.pos)

    best_action = valid[0][0]
    best_score = -999
    for action, nr, nc in valid:
        score = 0
        if (nr, nc) in enemy_positions:
            score -= 500
        if coily:
            score += grid_dist(nr, nc, coily[0], coily[1]) * 20
        score += len(neighbors(nr, nc)) * 50
        if score > best_score:
            best_score = score
            best_action = action
    return best_action
