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
# Enemy simulation
# ---------------------------------------------------------------------------

def _build_ball_positions(state):
    """Build list of (current_pos, full_future_path) for each non-Coily enemy."""
    balls = []
    for e in state.enemies:
        if e.harmless:
            continue
        # Skip Coily — it's handled separately via chase simulation.
        # Use etype (from tracker votes), NOT going_up (unreliable when stationary).
        if e.etype == "coily":
            continue
        if not is_valid(e.pos[0], e.pos[1]):
            continue
        path = predict_ball_path(e.pos[0], e.pos[1], e.direction_bits)
        balls.append((e.pos, path))
    return balls


SPAWN_POINTS = {(1, 0), (1, 1)}

def _ball_positions_at_step(balls, step, spawn_imminent=False):
    """Return set of positions where balls are dangerous at a given step.

    Includes both the ball's position AND its previous position
    (mid-hop collision, matching ROM $BD1E logic).
    If spawn_imminent, also includes spawn points (1,0) and (1,1).
    """
    positions = set()
    if spawn_imminent:
        positions.update(SPAWN_POINTS)
    for cur, path in balls:
        if step == 0:
            positions.add(cur)
        else:
            idx = step - 1
            if idx < len(path):
                p = path[idx]
                if is_valid(p[0], p[1]):
                    positions.add(p)
                prev = path[idx - 1] if idx >= 1 else cur
                if is_valid(prev[0], prev[1]):
                    positions.add(prev)
    return positions


def _simulate_coily(coily_pos, coily_target, qbert_path, num_steps):
    """Simulate Coily's positions for each Q*bert step.

    Uses 1:1 hop ratio (conservative). Coily actually hops slower than
    Q*bert, but collision detection runs mid-animation, so treating
    Coily as same-speed prevents close encounters.
    """
    if coily_pos is None:
        return [None] * (num_steps + 1)

    positions = [coily_pos]
    r, c = coily_pos
    for i in range(num_steps):
        if i == 0:
            target = coily_target
        else:
            target = qbert_path[i] if i < len(qbert_path) else qbert_path[-1]
        r, c = predict_coily(r, c, target[0], target[1])
        if not is_valid(r, c):
            positions.extend([None] * (num_steps - i))
            break
        positions.append((r, c))

    return positions


def _collides_with_coily(qbert_from, qbert_to, coily_before, coily_after):
    """Check collision between Q*bert and Coily during a hop.

    ROM $BD1E checks:
      1. Q*bert dest vs Coily dest (both land on same square)
      2. Q*bert dest vs Coily prev (Q*bert lands where Coily was)
      3. Q*bert prev vs Coily dest (Coily lands where Q*bert was = swap)
    """
    if coily_before is None and coily_after is None:
        return False
    if coily_after and qbert_to == coily_after:
        return True
    if coily_before and qbert_to == coily_before:
        return True
    if coily_after and qbert_from == coily_after:
        return True
    return False


# ---------------------------------------------------------------------------
# Route search
# ---------------------------------------------------------------------------

def _find_safe_routes(qbert_pos, coily_pos, coily_target, coily_zone, balls, visited, max_depth=7, debug=False, spawn_imminent=False):
    """BFS to find routes that don't collide with any enemy.

    At each step, simulates Coily's chase and ball bounces, checking the
    full ROM collision logic (current vs current, current vs previous,
    previous vs current) for both Coily and balls.

    Returns list of (first_action, new_cubes, escape_routes, path_len, coily_dist)
    """
    routes = []
    start = qbert_pos

    # Pre-simulate Coily for max_depth steps assuming Q*bert stands still.
    # We'll refine per-path, but this gives us a baseline.
    # For the BFS we need per-path Coily simulation — do it inline.

    q = deque()

    for action, nr, nc in neighbors(start[0], start[1]):
        # Block any move into Coily's reach zone
        if (nr, nc) in coily_zone:
            continue

        # Check collision with balls
        balls_1 = _ball_positions_at_step(balls, 1, spawn_imminent)
        if (nr, nc) in balls_1:
            continue

        path = [start, (nr, nc)]
        new = 1 if not visited.get((nr, nc), False) else 0

        coily_d = 99
        if coily_pos:
            coily_d = grid_dist(nr, nc, coily_pos[0], coily_pos[1])

        escape = len(neighbors(nr, nc))
        routes.append((action, new, escape, 1, coily_d, (nr, nc), coily_pos))
        q.append((nr, nc, 1, action, path, new))

    while q:
        cr, cc, step, first_action, path, new_cubes = q.popleft()
        if step >= max_depth:
            continue

        for _, nr, nc in neighbors(cr, cc):
            next_step = step + 1
            new_path = path + [(nr, nc)]

            # Simulate Coily for this path
            coily_steps = _simulate_coily(coily_pos, coily_target, new_path, next_step)
            coily_before = coily_steps[step]       # Coily before this hop
            coily_after = coily_steps[next_step]   # Coily after this hop

            # ROM collision check with Coily
            if _collides_with_coily((cr, cc), (nr, nc), coily_before, coily_after):
                continue

            # Ball collision check
            balls_s = _ball_positions_at_step(balls, next_step, spawn_imminent)
            if (nr, nc) in balls_s:
                continue

            # Don't revisit recent positions (prevent short loops)
            if (nr, nc) in path[-4:]:
                continue

            new = new_cubes + (1 if not visited.get((nr, nc), False) else 0)

            escape = len(neighbors(nr, nc))
            coily_d = 99
            if coily_after:
                coily_d = grid_dist(nr, nc, coily_after[0], coily_after[1])

            routes.append((first_action, new, escape, next_step, coily_d,
                           (nr, nc), coily_after))

            if next_step < max_depth:
                q.append((nr, nc, next_step, first_action, new_path, new))

    return routes


def _can_escape_coily(qr, qc, coily_pos, coily_target, depth=6):
    """Simulate Q*bert fleeing from Coily for `depth` hops.

    Q*bert picks the move that maximizes distance + escape routes.
    Coily uses the chase algorithm. Returns True if Q*bert survives.
    """
    cr, cc = coily_pos
    target = coily_target

    for i in range(depth):
        # Coily hops
        ncr, ncc = predict_coily(cr, cc, target[0], target[1])
        if not is_valid(ncr, ncc):
            return True  # Coily fell off
        cr, cc = ncr, ncc
        target = (qr, qc)  # after first hop, chase Q*bert's current

        if (cr, cc) == (qr, qc):
            return False  # caught

        # Q*bert picks best escape
        best = None
        best_score = -1
        for _, nr, nc in neighbors(qr, qc):
            if (nr, nc) == (cr, cc):
                continue
            d = grid_dist(nr, nc, cr, cc)
            exits = len(neighbors(nr, nc))
            s = d * 10 + exits
            if s > best_score:
                best_score = s
                best = (nr, nc)

        if best is None:
            return False
        qr, qc = best
        if (cr, cc) == (qr, qc):
            return False

    return True


# ---------------------------------------------------------------------------
# Main decision
# ---------------------------------------------------------------------------

def decide(state: GameState, visited: dict, qbert_prev_known=None, debug=False) -> int:
    """Pick the best action by searching safe multi-hop routes."""
    row, col = state.qbert
    if not is_valid(row, col):
        return DOWN

    valid = neighbors(row, col)
    if not valid:
        return DOWN

    # Use Python-tracked prev if available (set at known time), fall back to RAM
    qb_prev = qbert_prev_known if qbert_prev_known else state.qbert_prev
    coily = None
    coily_target = None
    for e in state.enemies:
        if e.etype == "coily" and not e.harmless and is_valid(e.pos[0], e.pos[1]):
            coily = e.pos
            coily_target = qb_prev
            if coily == coily_target:
                coily_target = state.qbert
            break

    # Use disc when Coily is within 1 hop — escape to (0,0).
    # Strategy: save last disc for endgame (>= 22 cubes done).
    remaining_discs = len(state.discs)
    cubes_done = NUM_CUBES - state.remaining_cubes
    if coily:
        coily_d = grid_dist(row, col, coily[0], coily[1])
        for disc in state.discs:
            if (row, col) != disc.jump_from:
                continue
            if coily_d <= 2:
                # Use disc freely if we have 2+, or if endgame
                if remaining_discs >= 2 or cubes_done >= 20:
                    return disc.direction

    balls = _build_ball_positions(state)
    # Spawn at (1,0)/(1,1) happens when $0085 countdown hits 0.
    # With Q*bert hopping in ~22 frames, avoid spawn points when countdown < 25.
    # Block spawn points when a spawn could happen within 2 Q*bert hops (~36 frames)
    spawn_imminent = state.spawn_countdown < 45

    # Coily danger zone. Block current position + all possible next hops.
    # Even though Coily hops slower than Q*bert, collision detection
    # checks mid-hop positions too — can't safely hop adjacent to Coily.
    coily_zone = set()
    if coily:
        coily_zone.add(coily)
        cr, cc = coily
        for dr, dc in [(-1, -1), (-1, 0), (1, 0), (1, 1)]:
            p = (cr + dr, cc + dc)
            if is_valid(p[0], p[1]):
                coily_zone.add(p)

    if debug and coily:
        cd = grid_dist(row, col, coily[0], coily[1])
        if cd <= 3:
            print(f"    DBG: Q@({row},{col}) prev={qb_prev} coily={coily} "
                  f"target={coily_target} d={cd} "
                  f"enemies=[{', '.join(f'{e.etype}@{e.pos}' for e in state.enemies)}]")

    routes = _find_safe_routes(
        (row, col), coily, coily_target, coily_zone, balls, visited, max_depth=7,
        debug=debug and coily is not None and grid_dist(row, col, coily[0], coily[1]) <= 3,
        spawn_imminent=spawn_imminent,
    )

    if debug and coily and grid_dist(row, col, coily[0], coily[1]) <= 3:
        if not routes:
            print(f"    DBG: NO SAFE ROUTES — using fallback")
        else:
            for a in set(r[0] for r in routes):
                best = max((r for r in routes if r[0] == a),
                           key=lambda r: r[1] * 200 + r[2] * 30 + r[4] * 15)
                dr, dc = MOVE_DELTAS[a]
                dest = (row + dr, col + dc)
                print(f"    DBG: {MOVE_NAMES[a]}→{dest} cubes={best[1]} "
                      f"esc={best[2]} coily_d={best[4]}")

    if not routes:
        # No safe routes — use Coily zone to avoid, maximize distance.
        best_action = valid[0][0]
        best_score = -999
        for action, nr, nc in valid:
            score = 0
            if coily:
                if (nr, nc) in coily_zone:
                    score -= 500
                score += grid_dist(nr, nc, coily[0], coily[1]) * 10
            # Avoid balls
            balls_now = _ball_positions_at_step(balls, 0, spawn_imminent)
            balls_next = _ball_positions_at_step(balls, 1, spawn_imminent)
            if (nr, nc) in balls_now or (nr, nc) in balls_next:
                score -= 200
            score += len(neighbors(nr, nc)) * 5
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    # Route toward disc when Coily is active.
    # More aggressive in endgame — always pull toward disc when cubes >= 20.
    disc_target = None
    if coily and state.discs:
        coily_d = grid_dist(row, col, coily[0], coily[1])
        if coily_d <= 5 or cubes_done >= 20:
            best_dd = 999
            for disc in state.discs:
                dd = grid_dist(row, col, disc.jump_from[0], disc.jump_from[1])
                if dd < best_dd:
                    best_dd = dd
                    disc_target = disc.jump_from

    # Score each first_action by its best route
    action_scores = {}
    for first_action, new_cubes, escape, path_len, coily_d, endpoint, coily_end in routes:
        # Route value: cubes matter but survival matters more.
        # A 1-step dead-end (escape=1) with 1 cube should lose to
        # a 5-step route (escape=4) with 1 cube.
        score = (new_cubes * 100
                 + escape * 50
                 + coily_d * 20
                 + path_len * 10)  # longer safe routes = more options

        # Bonus for moving toward a disc when Coily is chasing
        if disc_target and coily:
            dr2, dc2 = MOVE_DELTAS[first_action]
            nr2, nc2 = row + dr2, col + dc2
            d_before = grid_dist(row, col, disc_target[0], disc_target[1])
            d_after = grid_dist(nr2, nc2, disc_target[0], disc_target[1])
            if d_after < d_before:
                # Stronger pull in endgame
                pull = 200 if cubes_done >= 20 else 100
                score += pull

        if first_action not in action_scores or score > action_scores[first_action]:
            action_scores[first_action] = score

    return max(action_scores, key=action_scores.get)
