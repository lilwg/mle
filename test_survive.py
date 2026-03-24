"""Q*bert agent: 3-hop safe lookahead with frame-accurate simulation."""

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
DEAD_END_CORNERS = {(6, 0), (6, 6)}

# Disc positions now read directly from RAM $0ECC table by state.py.
# No hardcoded positions needed. The table is auto-consumed when discs are used.


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


# ── Enemy simulation ───────────────────────────────────────────────

def _build_sim_enemies(state):
    """Convert game state enemies into sim format:
    [pos, prev, anim, etype, direction_bits, flags]"""
    qpos, qprev = state.qbert, state.qbert_prev
    enemies = []

    # Ugg/Wrongway: off-grid positions → map to adjacent on-grid cubes.
    # ROM $B870: uses direction_bits like balls. bit0=1→DOWN(row+1), bit0=0→UP(row-1).
    # Block current adjacent cube AND predicted next adjacent cube.
    for e in state.enemies:
        if e.harmless:
            continue
        r, c = e.pos
        if c < 0:  # left face → dangerous to col=0 cubes
            anim = max(e.anim, 1) if e.anim > 0 else 8
            edge_pos = (r, 0)
            if is_valid(r, 0):
                enemies.append([edge_pos, edge_pos, anim, "ugg", e.direction_bits, e.flags])
                # Also block predicted next row
                nr = r + 1 if (e.direction_bits & 1) else r - 1
                if is_valid(nr, 0):
                    enemies.append([(nr, 0), edge_pos, anim, "ugg", e.direction_bits >> 1, e.flags])
        elif r >= 0 and c > r:  # right face → dangerous to col=row cubes
            anim = max(e.anim, 1) if e.anim > 0 else 8
            edge_pos = (r, r)
            if is_valid(r, r):
                enemies.append([edge_pos, edge_pos, anim, "ugg", e.direction_bits, e.flags])
                nr = r + 1 if (e.direction_bits & 1) else r - 1
                if is_valid(nr, nr):
                    enemies.append([(nr, nr), edge_pos, anim, "ugg", e.direction_bits >> 1, e.flags])

    for e in state.enemies:
        if e.harmless or not is_valid(e.pos[0], e.pos[1]):
            continue

        pos_e, prev_e, etype = e.pos, e.prev_pos, e.etype
        anim = e.anim

        if anim == 0 or (1 <= anim <= 5):
            # In-flight: gw is already the DESTINATION but entity hasn't landed.
            # Measured: flight takes ~30 frames from gw change to landing.
            # Block both current gw (destination) AND predicted next position.
            anim = 30  # frames until landing
            next_pos = _predict_next(pos_e, etype, e.direction_bits, qpos, qprev)
            if next_pos and is_valid(next_pos[0], next_pos[1]):
                enemies.append([next_pos, pos_e, anim, etype, e.direction_bits, e.flags])
        else:
            # Wait phase: anim counts reload→16, then triggers hop + 30f flight.
            # Pre-hatch balls have very high anim (70+) from initial spawn fall.
            if anim > 40:
                # Pre-hatch/spawn animation — estimate conservatively
                anim = anim  # raw countdown, entity won't hop until anim=16
            else:
                # Normal wait: (anim - 16) ticks wait + 30 flight
                anim = max(anim - 16, 0) + 30

        # Purple ball at bottom with Coily flags → about to hatch
        if etype == "ball" and e.flags in (0x60, 0x68) and pos_e[0] >= 6:
            etype = "coily"
            anim = 1

        enemies.append([pos_e, prev_e, anim, etype, e.direction_bits, e.flags])

    return enemies


def _predict_next(pos, etype, direction_bits, qpos, qprev):
    """Predict an enemy's next position after one hop."""
    if etype == "coily":
        target = qprev if pos != qprev else qpos
        return predict_coily(pos[0], pos[1], target[0], target[1])
    elif etype == "ball":
        if direction_bits & 1:
            return (pos[0] + 1, pos[1] + 1)
        else:
            return (pos[0] + 1, pos[1])
    return None


def _step_enemy(en, qpos, qprev):
    """Advance one enemy by one hop in the simulation."""
    epos, etype = en[0], en[3]

    if etype == "ugg":
        # Ugg moves on cube face: bit0=1→DOWN(row+1), bit0=0→UP(row-1)
        # Left side: danger at col=0 (stays 0 regardless of row)
        # Right side: danger at col=row (col tracks row)
        dbits = en[4]
        right_side = epos[1] > 0  # col=0 → left, col=row → right
        if dbits & 1:
            nr = epos[0] + 1
            new_pos = (nr, nr) if right_side else (nr, 0)
        else:
            nr = epos[0] - 1
            new_pos = (nr, nr) if right_side else (nr, 0)
        en[4] = dbits >> 1
        en[2] = BALL_RELOAD  # same timing as balls
        if is_valid(new_pos[0], new_pos[1]):
            en[1] = epos
            en[0] = new_pos
        else:
            en[0] = (-1, -1)
        return

    if etype == "coily":
        target = qprev if epos != qprev else qpos
        new_pos = predict_coily(epos[0], epos[1], target[0], target[1])
        en[2] = COILY_RELOAD
    else:
        dbits = en[4]
        if dbits & 1:
            new_pos = (epos[0] + 1, epos[1] + 1)
        else:
            new_pos = (epos[0] + 1, epos[1])
        en[4] = dbits >> 1
        en[2] = BALL_RELOAD

    if is_valid(new_pos[0], new_pos[1]):
        en[1] = epos
        en[0] = new_pos
    elif en[5] in (0x60, 0x68) and etype != "coily":
        # Purple ball falls off bottom → hatches into Coily
        # First hop is immediate (30f flight), not full COILY_RELOAD
        en[3] = "coily"
        en[2] = 30  # immediate first hop (flight time only)
    else:
        en[0] = (-1, -1)  # deactivated


def is_sequence_safe(state, actions):
    """Frame-by-frame simulation: returns True if Q*bert survives
    all hops in the action sequence."""
    qpos, qprev = state.qbert, state.qbert_prev
    enemies = _build_sim_enemies(state)

    # Check first-hop destination isn't occupied or cross-matched
    if actions:
        dr, dc = MOVE_DELTAS[actions[0]]
        dest = (qpos[0] + dr, qpos[1] + dc)
        for en in enemies:
            if not is_valid(en[0][0], en[0][1]):
                continue
            # Same position
            if en[0] == dest:
                return False
            # Cross-match: landing on enemy's prev while enemy is on our prev
            if en[1] == dest and en[0] == qpos:
                return False

    for action in actions:
        dr, dc = MOVE_DELTAS[action]
        nr, nc = qpos[0] + dr, qpos[1] + dc
        if not is_valid(nr, nc):
            return False

        # Phase 1: hop animation (HOP_FRAMES). Enemies tick, Q*bert in air.
        for _ in range(HOP_FRAMES):
            for en in enemies:
                if not is_valid(en[0][0], en[0][1]):
                    continue
                en[2] -= 1
                if en[2] <= 0:
                    _step_enemy(en, qpos, qprev)

        # Q*bert lands at destination
        qprev = qpos
        qpos = (nr, nc)

        # Phase 2: grounded at destination (10 frames). Check collision
        # as enemies arrive. Q*bert can't move yet (waiting for qb_anim).
        # ROM $BD4C-$BD66 collision: current==current OR cross-match OR overlap
        for _ in range(10):
            for en in enemies:
                if not is_valid(en[0][0], en[0][1]):
                    continue
                en[2] -= 1
                if en[2] <= 0:
                    _step_enemy(en, qpos, qprev)
                # ROM collision checks:
                # 1. Q*bert pos == enemy pos (same cube)
                if qpos == en[0]:
                    return False
                # 2. Cross-match: Q*bert pos == enemy prev AND
                #    Q*bert prev == enemy pos (swapped during hop)
                if qpos == en[1] and qprev == en[0]:
                    return False

    return True


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


def pick_target(state, coily_dead=False):
    """Pick next cube to color.
    - (6,6) first (safe diagonal from start)
    - When Coily is dead: target remaining dead-end corners immediately
    - Otherwise: nearest uncolored
    """
    qr, qc = state.qbert
    idx = pos_to_cube_index(6, 6)
    if idx is not None and state.cube_states[idx] != state.target_color:
        return (6, 6)
    # When Coily dead, hit corners while it's safe
    if coily_dead:
        for corner in [(6, 0), (6, 6)]:
            idx = pos_to_cube_index(corner[0], corner[1])
            if idx is not None and state.cube_states[idx] != state.target_color:
                return corner
    best_d, best = 999, None
    for row in range(MAX_ROW + 1):
        for col in range(row + 1):
            idx = pos_to_cube_index(row, col)
            if idx is not None and state.cube_states[idx] != state.target_color:
                d = grid_dist(qr, qc, row, col)
                if d < best_d:
                    best_d, best = d, (row, col)
    return best


def nearest_disc(state, pos):
    """Return (disc, distance) for the nearest available disc."""
    best_d, best = 999, None
    for disc in state.discs:
        dd = grid_dist(pos[0], pos[1], disc.jump_from[0], disc.jump_from[1])
        if dd < best_d:
            best_d, best = dd, disc
    return best, best_d


def decide(state, hops_since_progress):
    """Pick the best action. Returns action int or None (wait)."""
    row, col = state.qbert
    if not is_valid(row, col):
        return DOWN

    valid = neighbors(row, col)
    if not valid:
        return DOWN

    coily = find_coily(state)
    coily_d = grid_dist(row, col, coily[0], coily[1]) if coily else 99

    # ── Disc: take when at launch point and Coily will follow to its death ──
    # Coily chases Q*bert's previous position. If Q*bert takes a disc from
    # the edge, Coily will hop toward that position and fall off.
    # Take the disc when: (1) at launch point, (2) Coily is close enough
    # that its next hop(s) will reach the disc position and fall off.
    if coily and state.discs and coily_d <= 2:
        for disc in state.discs:
            if (row, col) == disc.jump_from:
                # Predict: will Coily reach this position and fall?
                # Coily targets qbert_prev. After disc ride, Q*bert's prev
                # is the disc launch point. Coily will path toward it.
                # At distance 1, Coily's next hop goes to our position.
                # At distance 2, Coily needs 2 hops but will still follow.
                if coily_d <= 1:
                    return disc.direction  # guaranteed kill
                # Distance 2: take disc as last resort if stuck
                if hops_since_progress >= DISC_STALL_THRESHOLD:
                    return disc.direction

    # ── Pick routing target ──
    # No Coily → target corners while safe
    # Coily + stuck → route toward disc
    target = pick_target(state, coily_dead=(coily is None))
    # Route to disc when stuck OR when Coily is dangerously close
    need_disc = (coily and state.discs and
                 (hops_since_progress >= DISC_STALL_THRESHOLD or
                  (coily_d <= 2 and hops_since_progress >= 2)))
    if need_disc:
        disc, dd = nearest_disc(state, (row, col))
        if disc:
            for a, r, c in valid:
                if (r, c) == disc.jump_from:
                    return a  # 1 hop from disc, go there
            target = disc.jump_from

    # ── Positions to avoid (dead-end corners when Coily active) ──
    # Exception: don't avoid a corner if it's the last cube (completes the level)
    avoid = set()
    if coily:
        for corner in DEAD_END_CORNERS:
            idx = pos_to_cube_index(corner[0], corner[1])
            if idx is not None and state.cube_states[idx] == state.target_color:
                avoid.add(corner)  # already colored, dangerous dead-end
            elif state.remaining_cubes <= 1:
                pass  # last cube — go for it even if it's a corner
            else:
                avoid.add(corner)

    # ── Search: 3-hop safe sequences toward target ──
    safe_moves = {}
    for a1, r1, c1 in valid:
        if (r1, c1) in avoid:
            continue
        for a2, r2, c2 in neighbors(r1, c1):
            for a3, r3, c3 in neighbors(r2, c2):
                if is_sequence_safe(state, [a1, a2, a3]):
                    d = grid_dist(r1, c1, target[0], target[1]) if target else 0
                    if a1 not in safe_moves or d < safe_moves[a1]:
                        safe_moves[a1] = d

    if safe_moves:
        return min(safe_moves, key=safe_moves.get)

    # ── Fallback: any 1-hop safe move ──
    for a, r, c in valid:
        if (r, c) not in avoid and is_sequence_safe(state, [a]):
            return a
    for a, r, c in valid:
        if is_sequence_safe(state, [a]):
            return a

    # ── Last resort: take disc if at launch point (all moves are death) ──
    if state.discs:
        for disc in state.discs:
            if (row, col) == disc.jump_from:
                return disc.direction
        # Or force hop to disc if 1 away
        for disc in state.discs:
            for a, r, c in valid:
                if (r, c) == disc.jump_from:
                    return a

    # Never wait — standing still is death. Pick the move furthest from enemies.
    if valid:
        best_a, best_d = valid[0][0], -1
        for a, r, c in valid:
            min_enemy_d = 99
            for e in state.enemies:
                if e.harmless or not is_valid(e.pos[0], e.pos[1]):
                    continue
                ed = grid_dist(r, c, e.pos[0], e.pos[1])
                if ed < min_enemy_d:
                    min_enemy_d = ed
            if min_enemy_d > best_d:
                best_d = min_enemy_d
                best_a = a
        return best_a
    return DOWN  # truly no moves


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
    env.step_n(port, field, BUTTON_HOLD)
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
    import random
    env.wait(600 + random.randint(0, 200))
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
                # Raw slot dump — catch enemies invisible to state parser
                if hops % 10 == 0 or current_level >= 3:
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
            action = decide(state, hops_since_progress)

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
                    killers += (f"\n    s{e.slot}:{e.etype}@{e.pos}"
                                f" fl={e.flags:#x} a={e.anim} prev={e.prev_pos}")
                print(f"  KILLED at hop {hops} @{state.qbert} ({pos_before}→{state.qbert})"
                      f"{killers}")

            # ── Track progress ──
            new_cubes = NUM_CUBES - state.remaining_cubes
            if new_cubes > last_cubes:
                hops_since_progress = 0
                last_cubes = new_cubes
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
                      f"{' ['+enemies_str+']' if current_level >= 3 and enemies_str else ''}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        print(f"\nTotal hops: {hops}")
        env.close()


if __name__ == "__main__":
    run()
