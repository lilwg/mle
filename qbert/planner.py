"""Q*bert move planner. Frame-accurate enemy simulation."""

from collections import deque
from qbert.state import is_valid, MAX_ROW, NUM_CUBES, GameState, Enemy
from qbert.predict import predict_coily, predict_ball_path, predict_ball_next

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

# Measured hop cycle lengths (frames between grid word changes)
QBERT_HOP_FRAMES = 18
COILY_HOP_FRAMES = 47
BALL_HOP_FRAMES = 39

SPAWN_POINTS = {(1, 0), (1, 1)}


def neighbors(r, c):
    return [(a, r + dr, c + dc) for a, (dr, dc) in MOVE_DELTAS.items()
            if is_valid(r + dr, c + dc)]


def grid_dist(r1, c1, r2, c2):
    dr = r2 - r1
    dc = c2 - c1
    if dr >= 0 and dc >= 0:
        return max(dr, dc)
    elif dr <= 0 and dc <= 0:
        return max(-dr, -dc)
    else:
        return abs(dr) + abs(dc)


def nearest_unvisited(row, col, visited):
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
# Frame-accurate enemy simulation
# ---------------------------------------------------------------------------

def _enemy_dangers_at_step(state, qbert_step, qbert_positions):
    """Predict enemy danger positions at a given Q*bert hop step.

    Conservative: assumes each enemy MIGHT hop during each Q*bert hop.
    Uses measured hop intervals:
    - Ball: 43 frames/hop → hops ~once per 2.4 Q*bert hops
    - Coily: 47 frames/hop → hops ~once per 2.6 Q*bert hops

    For safety, predicts both current AND next position for each enemy,
    since we don't know exactly when during the step the enemy hops.
    """
    dangers = set()

    for e in state.enemies:
        if e.harmless:
            continue
        if not is_valid(e.pos[0], e.pos[1]):
            continue

        r, c = e.pos

        if e.etype == "coily":
            # How many Coily hops in qbert_step Q*bert hops?
            coily_hops = int((qbert_step * QBERT_HOP_FRAMES) / COILY_HOP_FRAMES) + 1
            cr, cc = r, c
            dangers.add((cr, cc))
            for h in range(min(coily_hops, qbert_step)):
                target = state.qbert_prev
                if (cr, cc) == target:
                    target = state.qbert
                # Use Q*bert's predicted position for later hops
                if h < len(qbert_positions):
                    target = qbert_positions[h]
                nr, nc = predict_coily(cr, cc, target[0], target[1])
                if not is_valid(nr, nc):
                    break
                dangers.add((nr, nc))
                cr, cc = nr, nc
        else:
            # Ball: predict path using direction bits
            dangers.add((r, c))
            ball_hops = int((qbert_step * QBERT_HOP_FRAMES) / BALL_HOP_FRAMES) + 1
            br, bc = r, c
            bits = e.direction_bits
            for h in range(min(ball_hops, 7)):
                if bits & 1:
                    nr, nc = br + 1, bc + 1
                else:
                    nr, nc = br + 1, bc
                bits >>= 1
                if not is_valid(nr, nc):
                    break
                dangers.add((nr, nc))
                br, bc = nr, nc

    # Spawn points
    if state.spawn_countdown < QBERT_HOP_FRAMES * 3:
        dangers.update(SPAWN_POINTS)

    return dangers


def _find_coily(state):
    """Find Coily's position and chase target."""
    for e in state.enemies:
        if e.etype == "coily" and not e.harmless and is_valid(e.pos[0], e.pos[1]):
            target = state.qbert_prev
            if e.pos == target:
                target = state.qbert
            return e.pos, target
    return None, None


# ---------------------------------------------------------------------------
# Route search with frame-accurate simulation
# ---------------------------------------------------------------------------

def _find_safe_routes(state, visited, max_depth=7):
    """BFS to find routes that don't collide with any enemy.

    Uses frame-accurate enemy simulation at each step.
    """
    routes = []
    start = state.qbert
    coily, _ = _find_coily(state)

    q = deque()

    for action, nr, nc in neighbors(start[0], start[1]):
        qbert_path = [start, (nr, nc)]

        dangers = _enemy_dangers_at_step(state, 1, qbert_path)
        if (nr, nc) in dangers:
            continue

        new = 1 if not visited.get((nr, nc), False) else 0
        coily_d = grid_dist(nr, nc, coily[0], coily[1]) if coily else 99
        escape = len(neighbors(nr, nc))
        routes.append((action, new, escape, 1, coily_d))
        q.append((nr, nc, 1, action, qbert_path, new))

    while q:
        cr, cc, step, first_action, qbert_path, new_cubes = q.popleft()
        if step >= max_depth:
            continue

        for _, nr, nc in neighbors(cr, cc):
            next_step = step + 1
            new_path = qbert_path + [(nr, nc)]

            dangers = _enemy_dangers_at_step(state, next_step, new_path)
            if (nr, nc) in dangers:
                continue

            if (nr, nc) in qbert_path[-4:]:
                continue

            new = new_cubes + (1 if not visited.get((nr, nc), False) else 0)
            coily_d = grid_dist(nr, nc, coily[0], coily[1]) if coily else 99
            escape = len(neighbors(nr, nc))
            routes.append((first_action, new, escape, next_step, coily_d))

            if next_step < max_depth:
                q.append((nr, nc, next_step, first_action, new_path, new))

    return routes


# ---------------------------------------------------------------------------
# Main decision
# ---------------------------------------------------------------------------

def decide(state: GameState, visited: dict, qbert_prev_known=None, debug=False) -> int:
    """Pick the best action using frame-accurate enemy simulation."""
    row, col = state.qbert
    if not is_valid(row, col):
        return DOWN

    valid = neighbors(row, col)
    if not valid:
        return DOWN

    coily, coily_target = _find_coily(state)

    # Disc strategy: use when Coily is within 2 and we have discs
    cubes_done = NUM_CUBES - state.remaining_cubes
    if coily:
        coily_d = grid_dist(row, col, coily[0], coily[1])
        remaining_discs = len(state.discs)
        for disc in state.discs:
            if (row, col) != disc.jump_from:
                continue
            if coily_d <= 2:
                if remaining_discs >= 2 or cubes_done >= 20:
                    return disc.direction

    # Find safe routes using frame-accurate simulation
    routes = _find_safe_routes(state, visited, max_depth=7)

    if not routes:
        # Fallback: pick move furthest from all enemies
        best_action = valid[0][0]
        best_score = -999
        dangers = _enemy_dangers_at_step(state, 1, [(row, col)])
        for action, nr, nc in valid:
            score = 0
            if (nr, nc) in dangers:
                score -= 500
            if coily:
                score += grid_dist(nr, nc, coily[0], coily[1]) * 10
            score += len(neighbors(nr, nc)) * 5
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    # Route toward disc when Coily is active
    disc_target = None
    if coily and state.discs:
        if grid_dist(row, col, coily[0], coily[1]) <= 5 or cubes_done >= 20:
            best_dd = 999
            for disc in state.discs:
                dd = grid_dist(row, col, disc.jump_from[0], disc.jump_from[1])
                if dd < best_dd:
                    best_dd = dd
                    disc_target = disc.jump_from

    # Score routes
    action_scores = {}
    for first_action, new_cubes, escape, path_len, coily_d in routes:
        score = (new_cubes * 100
                 + escape * 50
                 + coily_d * 20
                 + path_len * 10)

        if disc_target and coily:
            dr2, dc2 = MOVE_DELTAS[first_action]
            nr2, nc2 = row + dr2, col + dc2
            d_before = grid_dist(row, col, disc_target[0], disc_target[1])
            d_after = grid_dist(nr2, nc2, disc_target[0], disc_target[1])
            if d_after < d_before:
                pull = 200 if cubes_done >= 20 else 100
                score += pull

        if first_action not in action_scores or score > action_scores[first_action]:
            action_scores[first_action] = score

    return max(action_scores, key=action_scores.get)
