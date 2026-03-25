"""Q*bert agent: depth-6 search with unified scoring and frame-accurate simulation."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mle import MameEnv
from qbert.state import (
    QBERT_RAM, read_state, is_valid, EnemyTracker,
    NUM_CUBES, MAX_ROW, pos_to_cube_index, Disc,
)
from qbert.sim import MOVE_DELTAS, UP, DOWN, LEFT, RIGHT
from qbert.predict import predict_coily
from qbert.planner import neighbors, MOVE_BUTTONS, COIN_BUTTON, START_BUTTON
from qbert.spawn import is_spawn_imminent, spawn_danger_at, frames_until_next_spawn

ROMS_PATH = "/Users/pat/mame/roms"

# ── Timing constants (measured from ROM) ────────────────────────────
# All enemies hop when their animation counter reaches 16, not 0.
# Effective frames until hop = raw_anim - 16.

BUTTON_HOLD = 6         # frames to hold direction button
HOP_FRAMES = 18         # frames per Q*bert hop cycle
COILY_RELOAD = 46       # Measured: Coily reload=32, wait 32→16 (16f) + 30f flight = 46
BALL_RELOAD = 43         # Measured: Ball reload=28, wait 28→16 (12f) + 30f flight = 43
DISC_STALL_THRESHOLD = 5 # hops without progress before routing to disc
SEARCH_DEPTH = 6         # lookahead depth (6 hops ≈ 168 frames ≈ 3.6 Coily hops)
COILY_HOP_FRAMES = 46    # frames per Coily hop cycle
QBERT_HOP_FRAMES = 28    # frames per Q*bert hop cycle (18 flight + 10 grounded)

# Disc positions now read directly from RAM $0ECC table by state.py.
# No hardcoded positions needed. The table is auto-consumed when discs are used.

# ── Position quality (precomputed) ─────────────────────────────────
# Score each grid position by escape route richness.
# Interior (4 neighbors) = 40, edge (3) = 20, narrow (2) = 0, dead-end (1) = -200
POS_QUALITY = {}
for _r in range(MAX_ROW + 1):
    for _c in range(_r + 1):
        _n = sum(1 for _dr, _dc in [(- 1, 0), (1, 0), (-1, -1), (1, 1)]
                 if is_valid(_r + _dr, _c + _dc))
        if _n <= 1:
            POS_QUALITY[(_r, _c)] = -200   # dead-end: (6,0), (6,6)
        elif _n == 2:
            POS_QUALITY[(_r, _c)] = -80 if _r == 6 else 0
        elif _n == 3:
            POS_QUALITY[(_r, _c)] = 20
        else:
            POS_QUALITY[(_r, _c)] = 40

# Recent positions for oscillation detection (module-level, reset on death/level)
_recent_positions = []
_MAX_RECENT = 10


# ── Helpers ─────────────────────────────────────────────────────────

def grid_dist(r1, c1, r2, c2):
    dr, dc = r2 - r1, c2 - c1
    if dr >= 0 and dc >= 0:
        return max(dr, dc)
    if dr <= 0 and dc <= 0:
        return max(-dr, -dc)
    return abs(dr) + abs(dc)


def wait_until_ready(env):
    """Wait until Q*bert can accept input (qb_anim >= 16)."""
    for _ in range(35):
        data = env.step()
        if data.get("qb_anim", 0) >= 16:
            return data
    return data


def wait_until_landed(env):
    """Wait for Q*bert to finish hopping (qb_anim >= 16)."""
    return wait_until_ready(env)


def wait_for_level_start(env, tracker):
    """Wait for a new level to become playable.
    Poll until Q*bert at top with anim>=16, then test hop to confirm."""
    data = env.step()
    for _ in range(900):
        data = env.step()
        state = read_state(data, tracker)
        pos = state.qbert
        if (is_valid(pos[0], pos[1]) and pos[0] <= 1
                and state.lives > 0
                and data.get("qb_anim", 0) >= 16):
            # Try a hop — if it works, level is ready
            gw_before = (data.get("qb_gw0", 0), data.get("qb_gw1", 0))
            env.step_n(":IN4", "P1 Right (Down-Right)", BUTTON_HOLD)
            for _ in range(25):
                data = env.step()
                if (data.get("qb_gw0", 0), data.get("qb_gw1", 0)) != gw_before:
                    data = wait_until_landed(env)
                    return data, read_state(data, tracker)
    return data, read_state(data, tracker)


def generate_sequences(r, c, depth, max_depth, path):
    """Generate all valid action sequences of exactly max_depth hops."""
    if depth >= max_depth:
        yield list(path)
        return
    for action, nr, nc in neighbors(r, c):
        path.append(action)
        yield from generate_sequences(nr, nc, depth + 1, max_depth, path)
        path.pop()






def is_sequence_safe(state, actions, data=None):
    """Frame-perfect simulation using ROM-accurate tick logic.
    Optionally includes cy=0 entities from raw RAM for better prediction."""
    from qbert.frame_sim import simulate_sequence, make_entity, HOP_TRIGGER, BALL_RELOAD, COILY_RELOAD
    extra = None
    if data is not None:
        from qbert.state import gw_to_pos
        from qbert.collision import classify_entity_rom
        extra = []
        # Find all visible enemy positions to avoid duplicates
        visible = {e.pos for e in state.enemies if not e.harmless}
        for n in range(10):
            fl = data.get(f"e{n}_flags", 0)
            st = data.get(f"e{n}_st", 0)
            cy = data.get(f"e{n}_coll_y", 0)
            if st == 0 or fl == 0 or cy != 0:
                continue  # only pick up cy=0 entities (invisible to state reader)
            ft = fl & 0x06
            _, harmless, _ = classify_entity_rom(fl, ft, ft)
            if harmless:
                continue
            gw0 = data.get(f"e{n}_gw0", 0)
            gw1 = data.get(f"e{n}_gw1", 0)
            epos = gw_to_pos(gw0, gw1)
            if not is_valid(epos[0], epos[1]):
                continue
            if epos in visible:
                continue  # already in state.enemies
            pw0 = data.get(f"e{n}_pw0", 0)
            pw1 = data.get(f"e{n}_pw1", 0)
            eprev = gw_to_pos(pw0, pw1)
            anim = data.get(f"e{n}_anim", 0)
            dbits = data.get(f"e{n}_dir", 0)
            # Determine etype
            upper = fl & 0xE0
            if upper == 0x60:
                going_up = is_valid(eprev[0], eprev[1]) and epos[0] < eprev[0]
                etype = "coily" if (going_up or fl in (0x62, 0x68, 0x6a)) else "ball"
            else:
                etype = "ball"
            # In-flight entity: anim 0-5 means about to land
            if anim == 0 or (1 <= anim <= 5):
                sim_anim = -1  # about to land
            elif anim <= HOP_TRIGGER:
                sim_anim = HOP_TRIGGER
            else:
                sim_anim = anim
            extra.append(make_entity(epos, eprev, sim_anim, etype, dbits, fl))
    return simulate_sequence(state, actions, extra)






# ── Scoring ─────────────────────────────────────────────────────────

def walk_path(start, actions):
    """Walk an action sequence and return list of positions visited."""
    positions = [start]
    r, c = start
    for a in actions:
        dr, dc = MOVE_DELTAS[a]
        r, c = r + dr, c + dc
        positions.append((r, c))
    return positions


def predict_coily_along_path(coily_pos, qbert_positions):
    """Predict Coily's position as Q*bert moves through a sequence.
    Coily chases qbert_prev which updates after each Q*bert hop."""
    cr, cc = coily_pos
    hop_budget = 0
    for qb_prev in qbert_positions[:-1]:  # each element is prev after that hop
        hop_budget += QBERT_HOP_FRAMES
        while hop_budget >= COILY_HOP_FRAMES:
            hop_budget -= COILY_HOP_FRAMES
            nr, nc = predict_coily(cr, cc, qb_prev[0], qb_prev[1])
            if is_valid(nr, nc):
                cr, cc = nr, nc
    return (cr, cc)


def score_sequence(state, actions, alive, target, coily_pos,
                   disc_target, hops_stuck, data):
    """Score a candidate action sequence. Higher is better.

    Components:
    - Survival (binary gate: alive = 1M base, dead = steps * 100)
    - Cubes colored along path (deduplicated)
    - Distance to target at final position
    - Position quality (escape routes) — strong penalty for dead-ends
    - Coily distance at end of sequence (when Coily active)
    - Disc proximity (when luring Coily, increases with stuck count)
    - Oscillation penalty (avoid revisiting recent positions)
    """
    if not alive:
        return len(actions) * 100

    positions = walk_path(state.qbert, actions)
    final_pos = positions[-1]

    score = 1_000_000  # alive base

    # Cubes colored along path (deduplicated — each cube counted once)
    colored = set()
    for pos in positions[1:]:  # skip starting position
        idx = pos_to_cube_index(pos[0], pos[1])
        if idx is not None and idx not in colored and state.cube_states[idx] != state.target_color:
            colored.add(idx)
    score += len(colored) * 500

    # Distance to target (lower = better)
    if target:
        d = grid_dist(final_pos[0], final_pos[1], target[0], target[1])
        score -= d * 120

    # Position quality at final position + worst bottleneck in path
    pq_final = POS_QUALITY.get(final_pos, 0)
    pq_worst = min(POS_QUALITY.get(p, 0) for p in positions[1:])
    if coily_pos:
        predicted_coily = predict_coily_along_path(coily_pos, positions)
        cd = grid_dist(final_pos[0], final_pos[1],
                       predicted_coily[0], predicted_coily[1])
        pq_mult = 3 if cd <= 2 else 1
        score += pq_final * pq_mult
        # Penalize paths that pass through dead-end corners
        if pq_worst <= -200:
            score += pq_worst  # -200 penalty for traversing dead-end
        score += cd * 150
    else:
        score += pq_final
        if pq_worst <= -200:
            score += pq_worst

    # Disc proximity (pull toward disc when Coily is active)
    if disc_target and coily_pos:
        dd = grid_dist(final_pos[0], final_pos[1],
                       disc_target[0], disc_target[1])
        disc_weight = 80 + hops_stuck * 20
        score -= dd * disc_weight

    # Oscillation penalty
    for pos in positions[1:]:
        if pos in _recent_positions:
            idx_r = _recent_positions.index(pos)
            recency = len(_recent_positions) - idx_r
            score -= recency * 20

    return score


# ── Decision logic ──────────────────────────────────────────────────

def find_coily(state):
    """Find hatched Coily only. Exclude pre-hatch purple balls.
    Coily is confirmed by: fl=0x68 seen (tracker), or upward movement.
    Purple balls (fl=0x60) bouncing down are NOT Coily yet."""
    for e in state.enemies:
        if e.etype != "coily" or e.harmless or not is_valid(e.pos[0], e.pos[1]):
            continue
        # Exclude any fl=0x60 that's bouncing down — still a purple ball
        if e.flags == 0x60 and not e.going_up:
            continue
        return e.pos
    return None


def find_coily_raw(data):
    """Find Coily including cy=0 entities by scanning RAW RAM.
    This catches in-flight Coily that the state reader filters out.
    Returns (pos, prev_pos) or (None, None)."""
    from qbert.state import gw_to_pos
    for n in range(10):
        flags = data.get(f"e{n}_flags", 0)
        st = data.get(f"e{n}_st", 0)
        if st == 0 or flags == 0:
            continue
        # Coily family: flags & 0xE0 == 0x60
        if (flags & 0xE0) != 0x60:
            continue
        gw0 = data.get(f"e{n}_gw0", 0)
        gw1 = data.get(f"e{n}_gw1", 0)
        pos = gw_to_pos(gw0, gw1)
        if not is_valid(pos[0], pos[1]):
            continue
        pw0 = data.get(f"e{n}_pw0", 0)
        pw1 = data.get(f"e{n}_pw1", 0)
        prev = gw_to_pos(pw0, pw1)
        # Only return if going UP (confirmed Coily, not pre-hatch ball)
        if is_valid(prev[0], prev[1]) and pos[0] < prev[0]:
            return pos, prev
        # Or if flags indicate hatched (0x62, 0x68, 0x6a)
        if flags in (0x62, 0x68, 0x6a):
            return pos, prev
    return None, None


def pick_target(state, coily_pos=None):
    """Pick next cube to color, considering distance and position safety.
    Prioritizes (6,6) early (safe diagonal from start)."""
    qr, qc = state.qbert
    # Strategic priority: (6,6) first when uncolored (safe diagonal)
    idx66 = pos_to_cube_index(6, 6)
    if idx66 is not None and state.cube_states[idx66] != state.target_color:
        return (6, 6)
    best_score, best = -999, None
    for row in range(MAX_ROW + 1):
        for col in range(row + 1):
            if (row, col) == (qr, qc):
                continue  # skip current position
            idx = pos_to_cube_index(row, col)
            if idx is None or state.cube_states[idx] == state.target_color:
                continue
            d = grid_dist(qr, qc, row, col)
            score = -d * 3  # prefer closer
            score += POS_QUALITY.get((row, col), 0) / 10
            # Corners: dangerous when Coily active, great when safe
            if POS_QUALITY.get((row, col), 0) <= -200:
                if coily_pos:
                    score -= 8
                else:
                    score += 5
            if score > best_score:
                best_score = score
                best = (row, col)
    return best


def nearest_disc(state, pos):
    """Return (disc, distance) for the nearest available disc."""
    best_d, best = 999, None
    for disc in state.discs:
        dd = grid_dist(pos[0], pos[1], disc.jump_from[0], disc.jump_from[1])
        if dd < best_d:
            best_d, best = dd, disc
    return best, best_d


def decide(state, hops_since_progress, data=None):
    """Pick the best action via depth-6 search with unified scoring."""
    global _recent_positions
    row, col = state.qbert
    if not is_valid(row, col):
        return DOWN

    valid = neighbors(row, col)
    if not valid:
        return DOWN

    coily = find_coily(state)
    coily_d = grid_dist(row, col, coily[0], coily[1]) if coily else 99

    # ── Disc override: guaranteed Coily kill ──
    if coily and state.discs and coily_d <= 2:
        for disc in state.discs:
            if (row, col) == disc.jump_from:
                if coily_d <= 1:
                    return disc.direction
                if hops_since_progress >= DISC_STALL_THRESHOLD:
                    return disc.direction

    # ── Compute targets ──
    target = pick_target(state, coily_pos=coily)
    disc_target = None
    if coily and state.discs:
        disc, _ = nearest_disc(state, (row, col))
        if disc:
            disc_target = disc.jump_from
            # Urgent disc routing: when stuck, hop toward disc immediately
            if hops_since_progress >= DISC_STALL_THRESHOLD:
                for a, r, c in valid:
                    if (r, c) == disc.jump_from:
                        return a  # 1 hop from disc, take it now

    # ── Search: score all depth-6 sequences ──
    action_scores = {}  # first_action -> best_score
    for seq in generate_sequences(row, col, 0, SEARCH_DEPTH, []):
        alive = is_sequence_safe(state, seq, data)
        s = score_sequence(state, seq, alive, target, coily,
                           disc_target, hops_since_progress, data)
        first = seq[0]
        if first not in action_scores or s > action_scores[first]:
            action_scores[first] = s

    if action_scores:
        best_action = max(action_scores, key=action_scores.get)
    else:
        # No sequences generated (shouldn't happen on valid grid)
        best_action = valid[0][0]

    # ── Update position history ──
    _recent_positions.append((row, col))
    if len(_recent_positions) > _MAX_RECENT:
        _recent_positions.pop(0)

    return best_action


# ── Game loop ───────────────────────────────────────────────────────

def execute_hop(env, action, tracker):
    """Press direction button and wait for Q*bert to land.
    Real-time safety: check current positions AND predict imminent arrivals."""
    dr, dc = MOVE_DELTAS[action]
    data = env.step()
    state = read_state(data, tracker)
    dest = (state.qbert[0] + dr, state.qbert[1] + dc)
    qpos = state.qbert
    # Spawn danger: don't hop to (1,0)/(1,1) when spawn is imminent
    if spawn_danger_at(data, dest, threshold=25):
        return None
    # Bottom-row caution: balls accumulate at rows 5-6. If a ball is
    # directly above the destination and heading down, it could arrive
    # during our hop.
    if dest[0] >= 5:
        for e in state.enemies:
            if e.harmless or not is_valid(e.pos[0], e.pos[1]):
                continue
            if e.etype in ("ball", "coily") and e.pos[0] == dest[0] - 1:
                # Ball one row above, could hop to our destination
                if e.pos[1] == dest[1] or e.pos[1] == dest[1] - 1:
                    return None
    # Check RAW data for ANY dangerous entity at destination (including cy=0).
    # The state reader filters cy=0 entities, but they can still kill on landing.
    from qbert.state import gw_to_pos
    from qbert.collision import classify_entity_rom
    for n in range(10):
        fl = data.get(f"e{n}_flags", 0)
        st_raw = data.get(f"e{n}_st", 0)
        if st_raw == 0 or fl == 0:
            continue
        ft = fl & 0x06
        _, harmless, _ = classify_entity_rom(fl, ft, ft)
        if harmless:
            continue
        gw0 = data.get(f"e{n}_gw0", 0)
        gw1 = data.get(f"e{n}_gw1", 0)
        epos = gw_to_pos(gw0, gw1)
        if not is_valid(epos[0], epos[1]):
            continue
        cy_raw = data.get(f"e{n}_coll_y", 0)
        # Entity AT destination or AT current pos — block
        if epos == dest:
            return None
        if epos == qpos and cy_raw == 0:
            return None  # entity landing on us RIGHT NOW
        # Ball/purple ball 1 row above destination heading down — will arrive
        # during our hop. Check both possible ball landing positions.
        if epos[0] == dest[0] - 1 and cy_raw == 0:
            # In-flight ball about to land 1 row above, next hop goes to dest row
            dbits = data.get(f"e{n}_dir", 0)
            ball_next_l = (epos[0] + 1, epos[1])
            ball_next_r = (epos[0] + 1, epos[1] + 1)
            if ball_next_l == dest or ball_next_r == dest:
                return None  # ball will hop to dest after landing
        # Also check ball AT dest row but hasn't landed (cy=0, same row)
        if epos[0] == dest[0] and cy_raw == 0:
            if grid_dist(epos[0], epos[1], dest[0], dest[1]) <= 1:
                return None  # nearby cy=0 entity on same row
        # Coily within 1 hop — predict chase
        if (fl & 0xE0) == 0x60 and grid_dist(epos[0], epos[1], dest[0], dest[1]) <= 1:
            pw0 = data.get(f"e{n}_pw0", 0)
            pw1 = data.get(f"e{n}_pw1", 0)
            eprev = gw_to_pos(pw0, pw1)
            going_up = is_valid(eprev[0], eprev[1]) and epos[0] < eprev[0]
            if going_up or fl in (0x62, 0x68, 0x6a):
                coily_next = predict_coily(epos[0], epos[1], qpos[0], qpos[1])
                if coily_next == dest:
                    return None
    for e in state.enemies:
        if e.harmless or not is_valid(e.pos[0], e.pos[1]):
            continue
        # Enemy AT destination
        if e.pos == dest:
            return None
        # Cross-match
        if e.pos == qpos and e.prev_pos == dest:
            return None
        # Imminent arrival: enemy within 1 hop of dest whose next
        # position would be our destination. Enemy hop cycle is ~43 frames,
        # Q*bert's vulnerability window is ~25 frames. Any enemy with
        # anim <= 28 (ball reload) could arrive in time.
        d = grid_dist(e.pos[0], e.pos[1], dest[0], dest[1])
        if d <= 1 and e.anim > 0:
            if e.etype == "coily":
                next_pos = predict_coily(e.pos[0], e.pos[1],
                                         state.qbert_prev[0], state.qbert_prev[1])
                if next_pos == dest:
                    return None
            elif e.etype == "ball":
                for npos in [(e.pos[0]+1, e.pos[1]), (e.pos[0]+1, e.pos[1]+1)]:
                    if npos == dest:
                        return None
    port, field = MOVE_BUTTONS[action]
    # Split hop: press 3 frames, re-check, press remaining 3.
    # This catches enemies that moved during the first 3 frames.
    env.step_n(port, field, 3)
    data2 = env.step()
    state2 = read_state(data2, tracker)
    # Re-check with fresh positions
    for e in state2.enemies:
        if e.harmless or not is_valid(e.pos[0], e.pos[1]):
            continue
        if e.pos == dest:
            # Enemy arrived at dest during first 3 frames — too late to
            # fully abort (button already held) but we can stop pressing.
            # Q*bert may still hop but at least we detected it.
            return wait_until_landed(env)
        d = grid_dist(e.pos[0], e.pos[1], dest[0], dest[1])
        if d <= 1 and e.anim > 0:
            if e.etype == "coily":
                np2 = predict_coily(e.pos[0], e.pos[1],
                                    state2.qbert_prev[0], state2.qbert_prev[1])
                if np2 == dest:
                    # Can still abort if Q*bert hasn't started flying
                    if data2.get("qb_anim", 0) >= 16:
                        return None  # abort — Q*bert still grounded
            elif e.etype == "ball":
                for npos in [(e.pos[0]+1, e.pos[1]), (e.pos[0]+1, e.pos[1]+1)]:
                    if npos == dest and data2.get("qb_anim", 0) >= 16:
                        return None  # abort
    env.step_n(port, field, BUTTON_HOLD - 3)
    return wait_until_landed(env)


def execute_disc(env, action, tracker, used_discs, disc):
    """Take a disc ride: wait until ready, press, wait for arrival at (0,0)."""
    port, field = MOVE_BUTTONS[action]
    data = wait_until_ready(env)
    state = read_state(data, tracker)
    # Verify Q*bert is still at the disc launch position
    if state.qbert != disc.jump_from:
        return None, None  # Q*bert moved during wait — abort
    env.step_n(port, field, BUTTON_HOLD)
    used_discs.add(disc.side)
    # Poll until Q*bert arrives at (0,0) — no fixed wait
    data = env.step()
    for _ in range(300):
        data = env.step()
        state = read_state(data, tracker)
        if is_valid(state.qbert[0], state.qbert[1]) and state.qbert[0] <= 1:
            break
    data = wait_until_landed(env)
    return data, read_state(data, tracker)


def run():
    env = MameEnv(ROMS_PATH, "qbert", QBERT_RAM, render=True, sound=False,
                  throttle=True)
    tracker = EnemyTracker()

    # Start game (randomize wait to vary MAME RNG seed)
    import random, time
    import os
    env.wait(600 + int.from_bytes(os.urandom(2)) % 200)
    env.step_n(*COIN_BUTTON, 15)
    env.wait(180)
    env.step_n(*START_BUTTON, 5)

    # Wait for game to start
    data = env.step()
    state = read_state(data, tracker)
    for _ in range(900):
        if 0 < state.lives <= 5 and is_valid(state.qbert[0], state.qbert[1]):
            break
        data = env.step()
        state = read_state(data, tracker)

    print(f"Game started: lives={state.lives}, Q*bert at {state.qbert}")

    prev_lives = state.lives
    hops = 0
    used_discs = set()
    level_active = False
    hops_since_progress = 0
    last_cubes = 0
    prev_cube_snapshot = tuple(state.cube_states)
    current_level = 1
    # Discs now read from RAM $0ECC table automatically by state.py

    try:
        for _ in range(50000):
            state = read_state(data, tracker)
            cubes = NUM_CUBES - state.remaining_cubes

            # ── Level tracking ──
            if state.remaining_cubes > 0:
                level_active = True

            if level_active and state.remaining_cubes == 0:
                print(f"\n  === LEVEL {current_level} COMPLETE at hop {hops}! ===\n")
                level_active = False
                current_level += 1
                used_discs = set()  # discs reset each level
                hops_since_progress = 0
                last_cubes = 0
                prev_cube_snapshot = ()
                _recent_positions.clear()
                tracker.reset()
                data, state = wait_for_level_start(env, tracker)
                prev_lives = state.lives
                hops = 1
                # Capture level start screenshot to identify disc positions
                try:
                    env.request_frame()
                    fd = env.step()
                    if "frame" in fd:
                        import numpy as np
                        from PIL import Image
                        raw = np.frombuffer(fd["frame"], dtype=np.uint8)
                        bpp = 3 if len(raw) == 256 * 240 * 3 else 4
                        pixels = raw[:256*240*bpp].reshape(256, 240, bpp)
                        if bpp == 4:
                            img = Image.fromarray(pixels[:, :, 2::-1])
                        else:
                            img = Image.fromarray(pixels)
                        img.save(f"level_{current_level}_start.png")
                    data = fd
                except Exception:
                    pass
                # Discs now read from RAM $0ECC table automatically by state.py
                state = read_state(data, tracker)
                disc_info = ", ".join(f"{d.side}@r{d.row}" for d in state.discs)
                print(f"  Level {current_level}: lives={state.lives} "
                      f"Q*bert={state.qbert} discs: {disc_info}")
                continue

            # ── Death handling ──
            if state.lives < prev_lives:
                killers = ""
                for e in state.enemies:
                    killers += (f"\n    s{e.slot}:{e.etype}@{e.pos}"
                                f" fl={e.flags:#x} a={e.anim}"
                                f" prev={e.prev_pos} cy={e.coll_y}"
                                f"{' HARMLESS' if e.harmless else ''}")
                # Raw slot dump — ALWAYS on death to catch invisible killers
                for n in range(10):
                        fl = data.get(f"e{n}_flags", 0)
                        st_raw = data.get(f"e{n}_st", 0)
                        cy = data.get(f"e{n}_coll_y", 0)
                        if fl != 0 or st_raw != 0:
                            from qbert.state import gw_to_pos
                            p = gw_to_pos(data.get(f"e{n}_gw0", 0),
                                          data.get(f"e{n}_gw1", 0))
                            killers += (f"\n    RAW s{n}: fl={fl:#04x}"
                                        f" st={st_raw} pos={p}"
                                        f" cy={cy} a={data.get(f'e{n}_anim',0)}")
                print(f"  DIED at hop {hops} @{state.qbert} cubes={cubes}{killers}")
                # Capture death screenshot
                try:
                    env.request_frame()
                    frame_data = env.step()
                    if "frame" in frame_data:
                        import numpy as np
                        from PIL import Image
                        raw = np.frombuffer(frame_data["frame"], dtype=np.uint8)
                        # MAME snapshot_pixels: rotated Q*bert screen
                        bpp = 3 if len(raw) == 256 * 240 * 3 else 4
                        h, w = 256, 240
                        pixels = raw[:h*w*bpp].reshape(h, w, bpp)
                        if bpp == 4:
                            img = Image.fromarray(pixels[:, :, 2::-1])  # BGRA→RGB
                        else:
                            img = Image.fromarray(pixels)  # already RGB
                        fname = f"death_L{current_level}_h{hops}.png"
                        img.save(fname)
                        print(f"    Screenshot: {fname}")
                except Exception as e:
                    print(f"    Screenshot failed: {e}")
                if state.lives == 0:
                    break
                prev_lives = state.lives
                tracker.reset()
                hops_since_progress = 0
                prev_cube_snapshot = ()
                _recent_positions.clear()
                # Wait for death animation — poll until Q*bert is valid again
                for _ in range(300):
                    data = env.step()
                    state = read_state(data, tracker)
                    if is_valid(state.qbert[0], state.qbert[1]) and data.get("qb_anim", 0) >= 16:
                        break
                state = read_state(data, tracker)
                hops = 0
                continue
            prev_lives = state.lives

            pos = state.qbert
            if not is_valid(pos[0], pos[1]):
                data = env.step()
                continue

            # Filter out used discs (state.discs is read fresh from $0ECC each frame,
            # but game auto-zeroes entries on use, so this is belt-and-suspenders)
            state.discs = [d for d in state.discs if d.side not in used_discs]

            # ── Decide ──
            action = decide(state, hops_since_progress, data)

            if action is None:
                for _ in range(6):
                    data = env.step()
                continue

            # ── Check for disc ride ──
            disc_match = None
            for disc in state.discs:
                if pos == disc.jump_from and action == disc.direction:
                    disc_match = disc
                    break

            if disc_match:
                data, state = execute_disc(env, action, tracker, used_discs, disc_match)
                if data is None:
                    data = env.step()
                    state = read_state(data, tracker)
                    continue
                hops += 1
                hops_since_progress = 0
                cubes = NUM_CUBES - state.remaining_cubes
                print(f"  #{hops:3d} DISC! {pos}→{state.qbert} "
                      f"({disc_match.side} r{disc_match.row})")
                continue

            # ── Normal hop ──
            dr, dc = MOVE_DELTAS[action]
            nr, nc = pos[0] + dr, pos[1] + dc
            if not is_valid(nr, nc):
                data = env.step()
                continue

            pos_before = pos
            # Save pre-hop enemy state for death diagnosis
            pre_hop_enemies = [(e.slot, e.etype, e.pos, e.prev_pos, e.flags, e.anim, e.coll_y)
                               for e in state.enemies if not e.harmless]
            data = execute_hop(env, action, tracker)
            if data is None:
                # Real-time check aborted — enemy at destination
                data = env.step()
                continue
            state = read_state(data, tracker)

            if state.qbert == pos_before:
                continue  # hop didn't register
            hops += 1

            # Check for imminent death — capture enemies before death clears them
            if state.lives < prev_lives:
                killers = ""
                for e in state.enemies:
                    if e.harmless:
                        continue
                    killers += (f"\n    POST s{e.slot}:{e.etype}@{e.pos}"
                                f" fl={e.flags:#x} a={e.anim} prev={e.prev_pos}")
                pre_info = ""
                for s, et, p, pp, fl, an, cy in pre_hop_enemies:
                    pre_info += (f"\n    PRE  s{s}:{et}@{p}"
                                 f" fl={fl:#x} a={an} prev={pp} cy={cy}")
                print(f"  KILLED at hop {hops} @{state.qbert} ({pos_before}→{state.qbert})"
                      f"{pre_info}{killers}")

            # ── Track progress ──
            # Track any cube state change (not just reaching target_color)
            # to handle multi-visit levels where cubes change incrementally
            new_cubes = NUM_CUBES - state.remaining_cubes
            cube_snapshot = tuple(state.cube_states)
            if new_cubes > last_cubes or cube_snapshot != prev_cube_snapshot:
                hops_since_progress = 0
                last_cubes = new_cubes
                prev_cube_snapshot = cube_snapshot
            else:
                hops_since_progress += 1

            # Dump all active enemy slots on level 3 to find Ugg flags
            if current_level >= 3:
                from qbert.state import gw_to_pos
                ugg_slots = []
                for n in range(10):
                    fl = data.get(f"e{n}_flags", 0)
                    st_raw = data.get(f"e{n}_st", 0)
                    if fl == 0 and st_raw == 0:
                        continue
                    p = gw_to_pos(data.get(f"e{n}_gw0", 0), data.get(f"e{n}_gw1", 0))
                    # Off-grid = potential Ugg/Wrongway
                    if not is_valid(p[0], p[1]) and p != (-1, -1):
                        ugg_slots.append(f"s{n}:fl={fl:#04x}@{p} a={data.get(f'e{n}_anim',0)}")
                if ugg_slots:
                    print(f"    OFF-GRID: {' | '.join(ugg_slots)}")

            # ── Log ──
            if hops % 10 == 0 or current_level >= 3:
                coily = find_coily(state)
                cd = grid_dist(pos[0], pos[1], coily[0], coily[1]) if coily else 99
                tgt = pick_target(state)
                enemies_str = " ".join(
                    f"{e.etype[0]}{e.pos}{'!' if e.pos==state.qbert else ''}"
                    for e in state.enemies
                    if not e.harmless
                )
                print(f"  hop {hops}: {state.qbert} cubes={new_cubes}/{NUM_CUBES} "
                      f"→{tgt} cd={cd} lives={state.lives} "
                      f"stuck={hops_since_progress}"
                      f"{' ['+enemies_str+']' if enemies_str else ''}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        print(f"\nTotal hops: {hops}")
        env.close()


if __name__ == "__main__":
    run()
