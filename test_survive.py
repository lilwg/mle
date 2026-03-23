"""Q*bert agent: 3-hop safe lookahead with frame-accurate simulation."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mle import MameEnv
from qbert.state import (
    QBERT_RAM, read_state, is_valid, EnemyTracker,
    NUM_CUBES, MAX_ROW, pos_to_cube_index,
)
from qbert.sim import MOVE_DELTAS, UP, DOWN, LEFT, RIGHT
from qbert.predict import predict_coily
from qbert.planner import neighbors, MOVE_BUTTONS, COIN_BUTTON, START_BUTTON

ROMS_PATH = "/Users/pat/mame/roms"

# ── Timing constants (measured from ROM) ────────────────────────────
# All enemies hop when their animation counter reaches 16, not 0.
# Effective frames until hop = raw_anim - 16.

BUTTON_HOLD = 6         # frames to hold direction button
HOP_FRAMES = 18         # frames per Q*bert hop cycle
COILY_RELOAD = 31       # Coily effective hop cycle (47 total - 16 trigger)
BALL_RELOAD = 27         # Ball effective hop cycle (43 total - 16 trigger)
DISC_STALL_THRESHOLD = 5 # hops without progress before routing to disc
DEAD_END_CORNERS = {(6, 0), (6, 6)}


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
    """Wait for a new level to become playable. Probe with a button press
    to detect when Q*bert can move."""
    for _ in range(3000):
        data = env.step()
        state = read_state(data, tracker)
        pos = state.qbert
        if (is_valid(pos[0], pos[1]) and pos[0] <= 1
                and state.lives > 0
                and data.get("qb_anim", 0) >= 16):
            gw_before = (data.get("qb_gw0", 0), data.get("qb_gw1", 0))
            env.step_n(":IN4", "P1 Right (Down-Right)", BUTTON_HOLD)
            for _ in range(20):
                data = env.step()
                if (data.get("qb_gw0", 0), data.get("qb_gw1", 0)) != gw_before:
                    break
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
            anim = max(e.anim - 16, 1) if e.anim > 0 else BALL_RELOAD
            edge_pos = (r, 0)
            if is_valid(r, 0):
                enemies.append([edge_pos, edge_pos, anim, "ugg", e.direction_bits, e.flags])
                # Also block predicted next row
                nr = r + 1 if (e.direction_bits & 1) else r - 1
                if is_valid(nr, 0):
                    enemies.append([(nr, 0), edge_pos, anim, "ugg", e.direction_bits >> 1, e.flags])
        elif r >= 0 and c > r:  # right face → dangerous to col=row cubes
            anim = max(e.anim - 16, 1) if e.anim > 0 else BALL_RELOAD
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

        if anim == 0:
            # Mid-hop animation: block current AND predicted next position
            anim = COILY_RELOAD if etype == "coily" else BALL_RELOAD
            next_pos = _predict_next(pos_e, etype, e.direction_bits, qpos, qprev)
            if next_pos and is_valid(next_pos[0], next_pos[1]):
                enemies.append([next_pos, pos_e, anim, etype, e.direction_bits, e.flags])
        else:
            anim = max(anim - 16, 1)  # hop triggers at anim=16

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
        # Grid word: DOWN adds (1,1), UP subtracts gw0 by 1
        # We track adjacent on-grid cubes, so just update the row
        dbits = en[4]
        if dbits & 1:
            new_pos = (epos[0] + 1, epos[1])  # row+1, same edge col
        else:
            new_pos = (epos[0] - 1, epos[1])  # row-1, same edge col
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
        en[3] = "coily"
        en[2] = COILY_RELOAD
    else:
        en[0] = (-1, -1)  # deactivated


def is_sequence_safe(state, actions):
    """Frame-by-frame simulation: returns True if Q*bert survives
    all hops in the action sequence."""
    qpos, qprev = state.qbert, state.qbert_prev
    enemies = _build_sim_enemies(state)

    # Check first-hop destination isn't occupied
    if actions:
        dr, dc = MOVE_DELTAS[actions[0]]
        dest = (qpos[0] + dr, qpos[1] + dc)
        for en in enemies:
            if en[0] == dest and is_valid(dest[0], dest[1]):
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
        for _ in range(10):
            for en in enemies:
                if not is_valid(en[0][0], en[0][1]):
                    continue
                en[2] -= 1
                if en[2] <= 0:
                    _step_enemy(en, qpos, qprev)
                if qpos == en[0]:
                    return False

    return True


# ── Decision logic ──────────────────────────────────────────────────

def find_coily(state):
    """Find Coily. The state reader classifies via tracker (fl=0x68 seen,
    upward movement, or fl=0x60 at row>=6). Trust that classification,
    but exclude pre-hatch balls still bouncing down at row <= 4."""
    for e in state.enemies:
        if e.etype != "coily" or e.harmless or not is_valid(e.pos[0], e.pos[1]):
            continue
        # Exclude pre-hatch: fl=0x60 at row <= 4, bouncing down
        if e.flags == 0x60 and e.pos[0] <= 4 and not e.going_up:
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

    # ── Disc: take when at launch point with Coily close ──
    if coily and state.discs and coily_d <= 2:
        for disc in state.discs:
            if (row, col) == disc.jump_from:
                return disc.direction

    # ── Pick routing target ──
    # No Coily → target corners while safe
    # Coily + stuck → route toward disc
    target = pick_target(state, coily_dead=(coily is None))
    if coily and state.discs and hops_since_progress >= DISC_STALL_THRESHOLD:
        disc, dd = nearest_disc(state, (row, col))
        if disc:
            for a, r, c in valid:
                if (r, c) == disc.jump_from:
                    return a  # 1 hop from disc, go there
            target = disc.jump_from

    # ── Positions to avoid (dead-end corners when Coily active) ──
    avoid = DEAD_END_CORNERS if coily else set()

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

    return None  # wait


# ── Game loop ───────────────────────────────────────────────────────

def execute_hop(env, action):
    """Press direction button and wait for Q*bert to land."""
    port, field = MOVE_BUTTONS[action]
    env.step_n(port, field, BUTTON_HOLD)
    return wait_until_landed(env)


def execute_disc(env, action, tracker, used_discs, disc):
    """Take a disc ride: wait until ready, press, wait for arrival at (0,0)."""
    port, field = MOVE_BUTTONS[action]
    wait_until_ready(env)
    env.step_n(port, field, BUTTON_HOLD)
    used_discs.add(disc.side)
    data = env.wait(300)
    # Wait until Q*bert arrives at the top
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

    # Start game
    env.wait(700)
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

    try:
        for _ in range(50000):
            state = read_state(data, tracker)
            cubes = NUM_CUBES - state.remaining_cubes

            # ── Level tracking ──
            if state.remaining_cubes > 0:
                level_active = True

            if level_active and state.remaining_cubes == 0:
                print(f"\n  === LEVEL COMPLETE at hop {hops}! ===\n")
                level_active = False
                used_discs = set()  # discs reset each level
                hops_since_progress = 0
                last_cubes = 0
                tracker.reset()
                env.wait(300)
                data, state = wait_for_level_start(env, tracker)
                prev_lives = state.lives
                hops = 1
                d0a = data.get("disc0_avail", 0)
                d0r = data.get("disc0_row", 0)
                d1a = data.get("disc1_avail", 0)
                d1r = data.get("disc1_row", 0)
                print(f"  New level: lives={state.lives} Q*bert={state.qbert} "
                      f"discs: d0={d0a}@r{d0r} d1={d1a}@r{d1r}")
                continue

            # ── Death handling ──
            if state.lives < prev_lives:
                killers = ""
                for e in state.enemies:
                    if e.harmless:
                        continue
                    killers += (f"\n    s{e.slot}:{e.etype}@{e.pos}"
                                f" fl={e.flags:#x} a={e.anim}"
                                f" prev={e.prev_pos} st={e.state}")
                print(f"  DIED at hop {hops} @{state.qbert} cubes={cubes}{killers}")
                if state.lives == 0:
                    break
                prev_lives = state.lives
                tracker.reset()
                hops_since_progress = 0
                data = env.wait(300)
                state = read_state(data, tracker)
                hops = 0
                continue
            prev_lives = state.lives

            pos = state.qbert
            if not is_valid(pos[0], pos[1]):
                data = env.step()
                continue

            # Filter discs we've already used this level
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
                d0r = data.get("disc0_row", 0)
                d1r = data.get("disc1_row", 0)
                d0a = data.get("disc0_avail", 0)
                d1a = data.get("disc1_avail", 0)
                data, state = execute_disc(env, action, tracker, used_discs, disc_match)
                hops += 1
                hops_since_progress = 0
                cubes = NUM_CUBES - state.remaining_cubes
                print(f"  #{hops:3d} DISC! {pos}→{state.qbert} "
                      f"d0={d0a}@r{d0r} d1={d1a}@r{d1r}")
                continue

            # ── Normal hop ──
            dr, dc = MOVE_DELTAS[action]
            nr, nc = pos[0] + dr, pos[1] + dc
            if not is_valid(nr, nc):
                data = env.step()
                continue

            pos_before = pos
            data = execute_hop(env, action)
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

            # ── Log ──
            if hops % 10 == 0:
                coily = find_coily(state)
                cd = grid_dist(pos[0], pos[1], coily[0], coily[1]) if coily else 99
                tgt = pick_target(state)
                print(f"  hop {hops}: {state.qbert} cubes={new_cubes}/{NUM_CUBES} "
                      f"→{tgt} cd={cd} lives={state.lives} "
                      f"stuck={hops_since_progress}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        print(f"\nTotal hops: {hops}")
        env.close()


if __name__ == "__main__":
    run()
