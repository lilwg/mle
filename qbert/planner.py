"""Q*bert move planner. Plans multi-hop routes with full enemy simulation."""

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
# World simulation: advance all enemies by one hop
# ---------------------------------------------------------------------------

def _build_ball_positions(state):
    """Build list of (current_pos, full_future_path) for each active ball."""
    balls = []
    for e in state.enemies:
        if e.harmless or e.going_up:
            continue
        if not is_valid(e.pos[0], e.pos[1]):
            continue
        path = predict_ball_path(e.pos[0], e.pos[1], e.direction_bits)
        balls.append((e.pos, path))
    return balls


def _ball_positions_at_step(balls, step):
    """Return set of positions occupied by balls at a given future step."""
    positions = set()
    for cur, path in balls:
        if step == 0:
            positions.add(cur)
        elif step - 1 < len(path):
            p = path[step - 1]
            if is_valid(p[0], p[1]):
                positions.add(p)
            # Also include previous position (mid-hop catch)
            if step - 2 >= 0:
                prev = path[step - 2] if step >= 2 else cur
                if is_valid(prev[0], prev[1]):
                    positions.add(prev)
    return positions


def _coily_at_step(coily_pos, coily_target, qbert_positions, step):
    """Return Coily's position after `step` hops.

    qbert_positions is a list of Q*bert's position at each step (step 0 = current).
    Coily chases Q*bert's previous position (one step behind).
    """
    if coily_pos is None:
        return None
    r, c = coily_pos
    for i in range(step):
        # Coily chases Q*bert's position from one step ago
        if i == 0:
            target = coily_target
        else:
            # After first hop, chase where Q*bert was when Coily started this hop
            target = qbert_positions[i] if i < len(qbert_positions) else qbert_positions[-1]
        r, c = predict_coily(r, c, target[0], target[1])
        if not is_valid(r, c):
            return None
    return (r, c)


def _is_dangerous_at_step(pos, coily_now, coily_next, ball_positions):
    """Check if a position is dangerous (Coily or ball there).

    Checks both Coily's current position AND predicted next position,
    since collision happens if Q*bert shares a square with Coily at any point.
    """
    if coily_now and pos == coily_now:
        return True
    if coily_next and pos == coily_next:
        return True
    if pos in ball_positions:
        return True
    return False


# ---------------------------------------------------------------------------
# Route search: BFS over (position, step) with enemy simulation
# ---------------------------------------------------------------------------

def _find_safe_routes(qbert_pos, coily_pos, coily_target, balls, visited, max_depth=7):
    """BFS to find routes to unvisited cubes that don't collide with enemies.

    Returns list of (first_action, new_cubes, final_escape_routes, path_length)
    for each safe route found.
    """
    routes = []

    # BFS state: (row, col, step, first_action, path_positions, new_cubes_count)
    start = qbert_pos
    q = deque()

    for action, nr, nc in neighbors(start[0], start[1]):
        # Check step 1: Coily's current pos AND where Coily will be after 1 hop
        coily_0 = coily_pos  # Coily's current position
        coily_1 = _coily_at_step(coily_pos, coily_target, [start], 1)
        balls_1 = _ball_positions_at_step(balls, 1)
        if _is_dangerous_at_step((nr, nc), coily_0, coily_1, balls_1):
            continue
        new = 1 if not visited.get((nr, nc), False) else 0
        q.append((nr, nc, 1, action, [start, (nr, nc)], new))

        escape = len(neighbors(nr, nc))
        coily_d = 99
        if coily_pos:
            c1 = _coily_at_step(coily_pos, coily_target, [start, (nr, nc)], 1)
            if c1:
                coily_d = grid_dist(nr, nc, c1[0], c1[1])
        routes.append((action, new, escape, 1, coily_d))

    while q:
        cr, cc, step, first_action, path, new_cubes = q.popleft()
        if step >= max_depth:
            continue

        for _, nr, nc in neighbors(cr, cc):
            next_step = step + 1

            # Coily at previous step and at this step
            coily_prev = _coily_at_step(coily_pos, coily_target, path, step)
            coily_next = _coily_at_step(coily_pos, coily_target, path, next_step)
            balls_s = _ball_positions_at_step(balls, next_step)

            if _is_dangerous_at_step((nr, nc), coily_prev, coily_next, balls_s):
                continue

            # Don't revisit positions in the same path (no loops)
            if (nr, nc) in path[-4:]:  # only check recent to allow some backtracking
                continue

            new = new_cubes + (1 if not visited.get((nr, nc), False) else 0)
            new_path = path + [(nr, nc)]

            escape = len(neighbors(nr, nc))
            # Compute Coily distance at endpoint for scoring
            coily_d = 99
            if coily_pos:
                c_end = _coily_at_step(coily_pos, coily_target, new_path, next_step)
                if c_end:
                    coily_d = grid_dist(nr, nc, c_end[0], c_end[1])
            routes.append((first_action, new, escape, next_step, coily_d))

            if next_step < max_depth:
                q.append((nr, nc, next_step, first_action, new_path, new))

    return routes


# ---------------------------------------------------------------------------
# Main decision function
# ---------------------------------------------------------------------------

def decide(state: GameState, visited: dict) -> int:
    """Pick the best action by searching safe multi-hop routes.

    For each possible first move, explores routes up to 7 hops deep,
    simulating Coily chase and ball bounces at each step. Picks the
    first move that leads to the best safe route.
    """
    row, col = state.qbert
    if not is_valid(row, col):
        return DOWN

    valid = neighbors(row, col)
    if not valid:
        return DOWN

    # Build enemy state for simulation
    coily = None
    coily_target = None
    for e in state.enemies:
        if e.going_up and not e.harmless and is_valid(e.pos[0], e.pos[1]):
            coily = e.pos
            coily_target = state.qbert_prev
            if coily == coily_target:
                coily_target = state.qbert
            break

    balls = _build_ball_positions(state)
    remaining = NUM_CUBES - sum(1 for v in visited.values() if v)

    # Find all safe routes
    routes = _find_safe_routes(
        (row, col), coily, coily_target, balls, visited, max_depth=7
    )

    if not routes:
        # No safe routes found — pick the move furthest from Coily as last resort
        best_action = DOWN
        best_dist = -1
        for action, nr, nc in valid:
            d = grid_dist(nr, nc, coily[0], coily[1]) if coily else 0
            if d > best_dist:
                best_dist = d
                best_action = action
        return best_action

    # Score each first_action by the best route it leads to
    action_scores = {}
    for first_action, new_cubes, escape, path_len, coily_d in routes:
        # New cubes are valuable but not at the cost of getting trapped.
        # Escape routes at endpoint and distance from Coily prevent traps.
        score = (new_cubes * 200
                 + escape * 30
                 + coily_d * 15
                 - path_len * 2)
        if first_action not in action_scores or score > action_scores[first_action]:
            action_scores[first_action] = score

    best_action = max(action_scores, key=action_scores.get)
    return best_action
