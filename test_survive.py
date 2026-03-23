"""Q*bert agent: 3-hop safe lookahead + nearest-cube routing + disc usage."""

import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mle import MameEnv
from qbert.state import (QBERT_RAM, read_state, is_valid, EnemyTracker,
                          NUM_CUBES, MAX_ROW, pos_to_cube_index)
from qbert.sim import MOVE_DELTAS, UP, DOWN, LEFT, RIGHT
from qbert.predict import predict_coily
from qbert.planner import neighbors, MOVE_BUTTONS, COIN_BUTTON, START_BUTTON

ROMS_PATH = "/Users/pat/mame/roms"
BUTTON_HOLD = 6
HOP_FRAMES = 18
COILY_RELOAD = 31
BALL_RELOAD = 27


def grid_dist(r1, c1, r2, c2):
    dr = r2 - r1
    dc = c2 - c1
    if dr >= 0 and dc >= 0: return max(dr, dc)
    if dr <= 0 and dc <= 0: return max(-dr, -dc)
    return abs(dr) + abs(dc)


# ── Simulation: is a sequence of Q*bert hops safe? ──────────────────

def _is_sequence_safe(state, actions):
    """Frame-by-frame sim. Returns True if Q*bert survives all hops."""
    qpos = state.qbert
    qprev = state.qbert_prev

    # Build enemy list: [pos, prev, anim, etype, dir_bits, flags]
    enemies = []

    # Ugg/Wrongway: off-grid positions → block adjacent edge cubes
    for e in state.enemies:
        if e.harmless:
            continue
        r, c = e.pos
        if c < 0:
            for dr in (-1, 0, 1):
                if is_valid(r + dr, 0):
                    enemies.append([(r + dr, 0), (r + dr, 0), 999, "ugg", 0, 0])
        elif r >= 0 and c > r:
            for dr in (-1, 0, 1):
                if is_valid(r + dr, r + dr):
                    enemies.append([(r + dr, r + dr), (r + dr, r + dr), 999, "ugg", 0, 0])

    for e in state.enemies:
        if e.harmless or not is_valid(e.pos[0], e.pos[1]):
            continue

        pos_e = e.pos
        prev_e = e.prev_pos
        etype = e.etype
        anim = e.anim

        if anim == 0:
            # Mid-hop animation: entity transitioning to next position.
            # Block both current AND predicted next position.
            anim = COILY_RELOAD if etype == "coily" else BALL_RELOAD
            if etype == "coily":
                target = qprev if pos_e != qprev else qpos
                nr, nc = predict_coily(pos_e[0], pos_e[1], target[0], target[1])
                if is_valid(nr, nc):
                    enemies.append([(nr, nc), pos_e, anim, "coily", e.direction_bits, e.flags])
            elif etype == "ball":
                if e.direction_bits & 1:
                    nr, nc = pos_e[0] + 1, pos_e[1] + 1
                else:
                    nr, nc = pos_e[0] + 1, pos_e[1]
                if is_valid(nr, nc):
                    enemies.append([(nr, nc), pos_e, anim, "ball", e.direction_bits >> 1, e.flags])
        else:
            # All enemies hop when anim reaches 16, not 0.
            anim = max(anim - 16, 1)

        # Purple ball at row 6 with Coily flags → about to hatch
        if etype == "ball" and e.flags in (0x60, 0x68) and pos_e[0] >= 6:
            etype = "coily"
            anim = 1

        enemies.append([pos_e, prev_e, anim, etype, e.direction_bits, e.flags])

    # Check first-hop destination against all enemies
    if actions:
        dr, dc = MOVE_DELTAS[actions[0]]
        first_dest = (qpos[0] + dr, qpos[1] + dc)
        for en in enemies:
            if en[0] == first_dest and is_valid(en[0][0], en[0][1]):
                return False

    # Simulate each hop
    for action in actions:
        dr, dc = MOVE_DELTAS[action]
        nr, nc = qpos[0] + dr, qpos[1] + dc
        if not is_valid(nr, nc):
            return False

        # Tick frames
        for _ in range(HOP_FRAMES):
            for en in enemies:
                epos = en[0]
                if not is_valid(epos[0], epos[1]):
                    continue
                en[2] -= 1
                if en[2] <= 0:
                    old = epos
                    etype = en[3]
                    if etype == "ugg":
                        en[2] = 999
                        continue
                    elif etype == "coily":
                        target = qprev if epos != qprev else qpos
                        epos = predict_coily(epos[0], epos[1], target[0], target[1])
                        en[2] = COILY_RELOAD
                    else:
                        dbits = en[4]
                        if dbits & 1:
                            epos = (epos[0] + 1, epos[1] + 1)
                        else:
                            epos = (epos[0] + 1, epos[1])
                        en[4] = dbits >> 1
                        en[2] = BALL_RELOAD
                    if is_valid(epos[0], epos[1]):
                        en[0] = epos
                        en[1] = old
                    elif en[5] in (0x60, 0x68) and etype != "coily":
                        en[3] = "coily"
                        en[2] = COILY_RELOAD
                    else:
                        en[0] = (-1, -1)

        # Q*bert lands
        qprev = qpos
        qpos = (nr, nc)
        for en in enemies:
            if qpos == en[0] and is_valid(en[0][0], en[0][1]):
                return False

    return True


# ── Decision function ───────────────────────────────────────────────

def find_coily(state):
    """Find confirmed Coily (etype='coily' from tracker or fl=0x68)."""
    for e in state.enemies:
        if e.etype == "coily" and not e.harmless and is_valid(e.pos[0], e.pos[1]):
            return e.pos
    return None


def pick_target(state):
    """Pick next cube to color. Prioritize (6,6), then nearest uncolored."""
    qr, qc = state.qbert
    # (6,6) first — safe diagonal from start
    idx = pos_to_cube_index(6, 6)
    if idx is not None and state.cube_states[idx] != state.target_color:
        return (6, 6)
    # Nearest uncolored
    best_d, best = 999, None
    for row in range(MAX_ROW + 1):
        for col in range(row + 1):
            idx = pos_to_cube_index(row, col)
            if idx is not None and state.cube_states[idx] != state.target_color:
                d = grid_dist(qr, qc, row, col)
                if d < best_d:
                    best_d = d
                    best = (row, col)
    return best


def decide(state, hops_since_progress):
    """Pick the best action. Returns action int or None (wait)."""
    row, col = state.qbert
    if not is_valid(row, col):
        return DOWN

    valid = neighbors(row, col)
    if not valid:
        return DOWN

    coily = find_coily(state)

    # Disc: take it if at launch point and Coily active
    if coily and state.discs:
        for disc in state.discs:
            if (row, col) == disc.jump_from:
                return disc.direction

    # Pick target: cube collection by default, disc when stuck
    target = pick_target(state)
    if coily and state.discs and hops_since_progress >= 5:
        best_dd = 999
        for disc in state.discs:
            dd = grid_dist(row, col, disc.jump_from[0], disc.jump_from[1])
            if dd < best_dd:
                best_dd = dd
                target = disc.jump_from

    # Don't enter dead-end corners when Coily is active
    avoid = set()
    if coily:
        avoid = {(6, 0), (6, 6)}

    # 3-hop safe sequences toward target
    safe_moves = {}
    for a1, r1, c1 in valid:
        if (r1, c1) in avoid:
            continue
        for a2, r2, c2 in neighbors(r1, c1):
            for a3, r3, c3 in neighbors(r2, c2):
                if _is_sequence_safe(state, [a1, a2, a3]):
                    d = grid_dist(r1, c1, target[0], target[1]) if target else 0
                    if a1 not in safe_moves or d < safe_moves[a1]:
                        safe_moves[a1] = d

    if safe_moves:
        return min(safe_moves, key=safe_moves.get)

    # Fallback: any 1-hop safe move
    for a, r, c in valid:
        if (r, c) not in avoid and _is_sequence_safe(state, [a]):
            return a
    for a, r, c in valid:
        if _is_sequence_safe(state, [a]):
            return a

    return None


# ── Game loop ───────────────────────────────────────────────────────

def run():
    env = MameEnv(ROMS_PATH, "qbert", QBERT_RAM, render=True, sound=False,
                  throttle=True)
    tracker = EnemyTracker()

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

            # Level active tracking
            if state.remaining_cubes > 0:
                level_active = True

            # Level complete
            if level_active and state.remaining_cubes == 0:
                print(f"\n  === LEVEL COMPLETE at hop {hops}! ===\n")
                level_active = False
                used_discs = set()
                hops_since_progress = 0
                last_cubes = 0
                tracker.reset()
                env.wait(300)
                for _ in range(3000):
                    data = env.step()
                    state = read_state(data, tracker)
                    pos = state.qbert
                    if (is_valid(pos[0], pos[1]) and pos[0] <= 1
                            and state.lives > 0
                            and data.get("qb_anim", 0) >= 16):
                        gw_before = (data.get("qb_gw0", 0), data.get("qb_gw1", 0))
                        env.step_n(":IN4", "P1 Right (Down-Right)", 6)
                        for _ in range(20):
                            data = env.step()
                            if (data.get("qb_gw0", 0), data.get("qb_gw1", 0)) != gw_before:
                                break
                        if (data.get("qb_gw0", 0), data.get("qb_gw1", 0)) != gw_before:
                            for _ in range(12):
                                data = env.step()
                            state = read_state(data, tracker)
                            break
                prev_lives = state.lives
                hops = 1
                print(f"  New level: lives={state.lives} Q*bert={state.qbert}")
                continue

            # Death
            if state.lives < prev_lives:
                print(f"  DIED at hop {hops}! lives={state.lives} cubes={cubes}")
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

            # Filter used discs
            state.discs = [d for d in state.discs if d.side not in used_discs]

            # Decide
            action = decide(state, hops_since_progress)

            if action is None:
                for _ in range(6):
                    data = env.step()
                continue

            # Log nearby enemies at decision time for death tracing
            dr2, dc2 = MOVE_DELTAS[action]
            dest = (pos[0] + dr2, pos[1] + dc2)
            for e in state.enemies:
                if not e.harmless and is_valid(e.pos[0], e.pos[1]):
                    d = grid_dist(pos[0], pos[1], e.pos[0], e.pos[1])
                    if d <= 2:
                        print(f"  pre{hops}: {pos}→{dest} "
                              f"s{e.slot}:{e.etype}@{e.pos} "
                              f"fl={e.flags:#x} anim={e.anim} d={d}")

            # Disc ride?
            is_disc = False
            for disc in state.discs:
                if pos == disc.jump_from and action == disc.direction:
                    is_disc = True
                    break

            if is_disc:
                port, field = MOVE_BUTTONS[action]
                # Wait until Q*bert is ready to hop
                for _ in range(35):
                    data = env.step()
                    if data.get("qb_anim", 0) >= 16:
                        break
                env.step_n(port, field, BUTTON_HOLD)
                data = env.wait(300)
                tracker.reset()
                used_discs.add(disc.side)
                for _ in range(300):
                    data = env.step()
                    state = read_state(data, tracker)
                    if is_valid(state.qbert[0], state.qbert[1]) and state.qbert[0] <= 1:
                        break
                for _ in range(30):
                    data = env.step()
                state = read_state(data, tracker)
                hops += 1
                hops_since_progress = 0
                print(f"  #{hops:3d} DISC! → {state.qbert}  cubes={cubes}/{NUM_CUBES}")
                continue

            # Normal hop
            dr, dc = MOVE_DELTAS[action]
            nr, nc = pos[0] + dr, pos[1] + dc
            if not is_valid(nr, nc):
                data = env.step()
                continue

            port, field = MOVE_BUTTONS[action]
            pos_before = pos
            env.step_n(port, field, BUTTON_HOLD)
            for _ in range(35):
                data = env.step()
                if data.get("qb_anim", 0) >= 16:
                    break

            state = read_state(data, tracker)
            if state.qbert == pos_before:
                continue
            hops += 1

            # Track progress
            new_cubes = NUM_CUBES - state.remaining_cubes
            if new_cubes > last_cubes:
                hops_since_progress = 0
                last_cubes = new_cubes
            else:
                hops_since_progress += 1

            # Log every 10 hops
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
