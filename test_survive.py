"""Pure survival test: Q*bert dodges everything indefinitely.
No cube collection, no disc usage. Just prove the simulator works."""

import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mle import MameEnv
from qbert.state import QBERT_RAM, read_state, is_valid, EnemyTracker, NUM_CUBES
from qbert.sim import make_sim_state, clone, simulate, MOVE_DELTAS, UP, DOWN, LEFT, RIGHT
from qbert.planner import neighbors, MOVE_BUTTONS, COIN_BUTTON, START_BUTTON
from qbert.strategy import grid_dist

ROMS_PATH = "/Users/pat/mame/roms"
BUTTON_HOLD = 6


def score_survival(result):
    """Pure survival scoring: stay alive, maximize escape routes + coily distance."""
    if not result.alive:
        return result.steps_survived * 100
    base = 1_000_000
    # Use predicted Coily position AND current positions of ALL coilys
    cp = result.coily_final_pos
    if cp and cp != (-1, -1):
        from qbert.strategy import grid_dist
        coily_d = grid_dist(result.final_pos[0], result.final_pos[1], cp[0], cp[1])
    else:
        coily_d = 99
    # Extremely heavy Coily avoidance — distance is everything
    return base + coily_d * 10000 + result.escape_routes * 100


def generate_seqs(r, c, depth, max_depth, path):
    if depth >= max_depth:
        yield list(path)
        return
    for action, nr, nc in neighbors(r, c):
        path.append(action)
        yield from generate_seqs(nr, nc, depth + 1, max_depth, path)
        path.pop()


def _sim_check_collision(qpos, epos):
    """Collision check: current position match."""
    return qpos == epos


def _is_sequence_safe(state, actions):
    """Exact frame-by-frame simulation: returns True if Q*bert survives
    all hops in the action sequence.

    Uses actual anim counters from RAM for exact first-hop timing.
    Ball paths from direction_bits. Coily chase from predict_coily.
    """
    from qbert.predict import predict_coily

    HOP_FRAMES = 18  # Q*bert can hop every ~16-18 frames
    BALL_RELOAD = 43
    COILY_RELOAD = 47  # measured: Coily hops every 47 frames

    qpos = state.qbert
    qprev = state.qbert_prev

    # Ugg/Wrongway: crawl on cube SIDES (off-grid positions).
    # ROM: they move up/down using direction_bits (like balls but vertical).
    # Their grid words have col < 0 (left face) or col > row (right face).
    # Dangerous to Q*bert on adjacent edge cubes.
    ugg_danger = set()
    for e in state.enemies:
        r, c = e.pos
        if e.harmless:
            continue
        if c < 0:
            # Left face: dangerous to edge cubes (row-1..row+1, 0)
            for dr in (-1, 0, 1):
                if is_valid(r + dr, 0):
                    ugg_danger.add((r + dr, 0))
        elif r >= 0 and c > r:
            # Right face: dangerous to edge cubes (row-1..row+1, rightmost)
            for dr in (-1, 0, 1):
                rr = r + dr
                if is_valid(rr, rr):
                    ugg_danger.add((rr, rr))

    # Build enemy list: [pos, prev, anim_counter, etype, direction_bits, flags]
    enemies = []
    # Add Ugg dangers as stationary blockers
    for dp in ugg_danger:
        enemies.append([dp, dp, 999, "ugg", 0, 0])

    for e in state.enemies:
        if e.harmless or not is_valid(e.pos[0], e.pos[1]):
            continue
        etype = e.etype
        pos_e = e.pos
        prev_e = e.prev_pos
        anim = e.anim

        # Ugg/Wrongway detection is handled separately below (off-grid positions)

        # anim=0 means enemy is in HOP ANIMATION (lasts ~17 frames).
        # The entity is between its current and next position. Block BOTH:
        # current position (might still be there) and destination (arriving).
        if anim == 0:
            anim = COILY_RELOAD if etype == "coily" else BALL_RELOAD
            # Predict destination and add as a SECOND blocker
            if etype == "coily":
                qr, qc = state.qbert
                qprev = state.qbert_prev
                target = qprev if pos_e != qprev else (qr, qc)
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
            anim = max(anim, 1)

        # Purple ball at row 6 = about to hatch into Coily immediately
        if etype == "ball" and e.flags in (0x60, 0x68) and pos_e[0] >= 6:
            etype = "coily"
            anim = 1

        enemies.append([
            pos_e, prev_e, anim, etype,
            e.direction_bits, e.flags,
        ])

    # No initial collision check — if Q*bert is alive in the game,
    # trust that. Collision Y may have saved it even if grid words match.

    for action in actions:
        dr, dc = MOVE_DELTAS[action]
        nr, nc = qpos[0] + dr, qpos[1] + dc
        if not is_valid(nr, nc):
            return False  # invalid move = can't use this sequence

        # Tick HOP_FRAMES frames — advance enemy positions
        for _ in range(HOP_FRAMES):
            for en in enemies:
                epos, eprev, anim, etype, dbits, eflags = en
                if not is_valid(epos[0], epos[1]):
                    continue
                anim -= 1
                en[2] = anim
                if anim <= 0:
                    # Enemy hops
                    old = epos
                    if etype == "ugg":
                        # Lateral mover — we don't predict, keep blocking
                        en[2] = 999  # stay put
                        continue
                    elif etype == "coily":
                        target = qprev if epos != qprev else qpos
                        epos = predict_coily(epos[0], epos[1],
                                             target[0], target[1])
                        en[2] = COILY_RELOAD
                    else:
                        if dbits & 1:
                            epos = (epos[0] + 1, epos[1] + 1)
                        else:
                            epos = (epos[0] + 1, epos[1])
                        en[4] = dbits >> 1
                        en[2] = BALL_RELOAD
                    if is_valid(epos[0], epos[1]):
                        en[0] = epos
                        en[1] = old
                    elif eflags in (0x60, 0x68) and etype != "coily":
                        en[3] = "coily"
                        en[2] = COILY_RELOAD
                    else:
                        en[0] = (-1, -1)
                        continue

        # Q*bert lands — check collision ONLY at landing (matching ROM
        # behavior: collision Y check means mid-hop entities don't collide)
        qprev = qpos
        qpos = (nr, nc)

        for en in enemies:
            if not is_valid(en[0][0], en[0][1]):
                continue
            if _sim_check_collision(qpos, en[0]):
                return False

    return True


def _find_coily(state):
    for e in state.enemies:
        if e.etype == "coily" and not e.harmless and is_valid(e.pos[0], e.pos[1]):
            return e.pos
    return None


def decide_survive(state):
    """3-hop lookahead: find all 3-move sequences that survive exact
    frame-by-frame simulation, then pick the one closest to target."""
    row, col = state.qbert
    if not is_valid(row, col):
        return DOWN

    valid = neighbors(row, col)
    if not valid:
        return DOWN

    coily = _find_coily(state)

    # Try all 3-hop sequences toward the TARGET CUBE first
    safe_first_moves = {}
    for a1, r1, c1 in valid:
        for a2, r2, c2 in neighbors(r1, c1):
            for a3, r3, c3 in neighbors(r2, c2):
                if _is_sequence_safe(state, [a1, a2, a3]):
                    d = grid_dist(r1, c1, _target[0], _target[1]) if _target else 0
                    if a1 not in safe_first_moves or d < safe_first_moves[a1]:
                        safe_first_moves[a1] = d

    if safe_first_moves:
        return min(safe_first_moves, key=safe_first_moves.get)

    # No safe cube-collecting moves. If Coily active and discs available,
    # route toward nearest disc as escape plan.
    if coily and state.discs:
        # If already at a disc, take it
        for disc in state.discs:
            if (row, col) == disc.jump_from:
                return disc.direction

        # Route toward nearest disc
        best_dd = 999
        disc_target = None
        for disc in state.discs:
            dd = grid_dist(row, col, disc.jump_from[0], disc.jump_from[1])
            if dd < best_dd:
                best_dd = dd
                disc_target = disc.jump_from

        # Find safe moves toward the disc
        disc_moves = {}
        for a1, r1, c1 in valid:
            if _is_sequence_safe(state, [a1]):
                d = grid_dist(r1, c1, disc_target[0], disc_target[1])
                if a1 not in disc_moves or d < disc_moves[a1]:
                    disc_moves[a1] = d
        if disc_moves:
            return min(disc_moves, key=disc_moves.get)

    # No safe moves toward cubes or disc — just pick any safe move
    # to keep running (Q*bert is faster than Coily)
    for a, r, c in valid:
        if _is_sequence_safe(state, [a]):
            return a

    # Truly stuck — wait
    return None


# Current target cube
_target = None
_level_active = False


def _pick_target(state):
    """Pick nearest uncolored cube as target."""
    global _target
    from qbert.state import pos_to_cube_index, MAX_ROW
    qr, qc = state.qbert

    # If current target is already colored, clear it
    if _target:
        idx = pos_to_cube_index(_target[0], _target[1])
        if idx is not None and state.cube_states[idx] == state.target_color:
            _target = None

    if _target is None:
        # Prioritize dead-end corners — hit them before Coily hatches (~hop 35).
        # (6,6) first since it's a straight diagonal from (0,0).
        for corner in [(6, 6), (6, 0)]:
            idx = pos_to_cube_index(corner[0], corner[1])
            if idx is not None and state.cube_states[idx] != state.target_color:
                _target = corner
                return

        # Otherwise find nearest uncolored cube
        best_d = 999
        for row in range(MAX_ROW + 1):
            for col in range(row + 1):
                idx = pos_to_cube_index(row, col)
                if idx is not None and state.cube_states[idx] != state.target_color:
                    d = grid_dist(qr, qc, row, col)
                    if d < best_d:
                        best_d = d
                        _target = (row, col)


def run():
    env = MameEnv(ROMS_PATH, "qbert", QBERT_RAM, render=True, sound=False, throttle=True)
    tracker = EnemyTracker()
    print("Pure survival test — dodge everything forever")

    env.wait(700)
    env.step_n(*COIN_BUTTON, 15)
    env.wait(180)
    env.step_n(*START_BUTTON, 5)

    # Wait for game start
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
    global _level_active
    _level_active = False

    for _ in range(5000):
        state = read_state(data, tracker)

        cubes = NUM_CUBES - state.remaining_cubes

        # Track if remaining has been initialized (goes above 0)
        if state.remaining_cubes > 0:
            _level_active = True

        # Level complete: remaining drops to 0 after being active
        if _level_active and state.remaining_cubes == 0:
            print(f"\n  === LEVEL COMPLETE at hop {hops}! ===\n")
            global _target
            _target = None
            _level_active = False
            used_discs = set()
            tracker.reset()
            # Wait for transition, then probe until Q*bert can move
            env.wait(300)
            for _ in range(3000):
                data = env.step()
                state = read_state(data, tracker)
                pos = state.qbert
                if (is_valid(pos[0], pos[1]) and pos[0] <= 1
                        and state.lives > 0
                        and data.get("qb_anim", 0) >= 16):
                    # Try a button press and see if grid word changes
                    gw_before = (data.get("qb_gw0", 0), data.get("qb_gw1", 0))
                    env.step_n(":IN4", "P1 Right (Down-Right)", 6)
                    for _ in range(20):
                        data = env.step()
                        if (data.get("qb_gw0", 0), data.get("qb_gw1", 0)) != gw_before:
                            break
                    if (data.get("qb_gw0", 0), data.get("qb_gw1", 0)) != gw_before:
                        # Q*bert moved — new level is active!
                        for _ in range(12):
                            data = env.step()
                        state = read_state(data, tracker)
                        break
            prev_lives = state.lives
            hops = 1  # already made one move
            print(f"  New level: lives={state.lives} remaining={state.remaining_cubes} Q*bert={state.qbert}")
            continue

        # Death check
        if state.lives < prev_lives:
            enemy_detail = ""
            for e in state.enemies:
                if is_valid(e.pos[0], e.pos[1]):
                    enemy_detail += (f"\n    s{e.slot}:{e.etype} @{e.pos} "
                                     f"prev={e.prev_pos} st={e.state} "
                                     f"fl={e.flags:#x} anim={e.anim}")
            print(f"  DIED at hop {hops}! lives={state.lives} cubes={cubes} "
                  f"pos={state.qbert}{enemy_detail}")
            if state.lives == 0:
                break
            prev_lives = state.lives
            tracker.reset()
            _target = None
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

        # Pick target cube and decide move
        _pick_target(state)
        action = decide_survive(state)

        if action is None:
            # Debug: log state when stuck
            enemies_near = [(e.slot, e.etype, e.pos, e.flags, e.anim,
                            grid_dist(pos[0], pos[1], e.pos[0], e.pos[1]))
                           for e in state.enemies
                           if not e.harmless and is_valid(e.pos[0], e.pos[1])]
            at_disc = any(pos == d.jump_from for d in state.discs)
            if enemies_near:
                print(f"  STUCK hop {hops}: Q@{pos} at_disc={at_disc}")
                for s, et, ep, ef, ea, d in enemies_near:
                    print(f"    s{s}:{et} @{ep} fl={ef:#x} anim={ea} d={d}")
            for _ in range(6):
                data = env.step()
            continue

        # Debug: at hops near typical death, show what sim thinks
        if 45 <= hops <= 60 or (hops > 0 and hops % 50 == 0):
            from qbert.sim import MOVE_DELTAS as MD
            dr, dc = MD[action]
            dest = (pos[0]+dr, pos[1]+dc)
            safe3 = _is_sequence_safe(state, [action, action, action])
            enemy_info = [(e.slot, e.etype, e.pos, e.flags, e.anim)
                          for e in state.enemies if is_valid(e.pos[0], e.pos[1])]
            print(f"  DBG hop {hops}: {pos}→{dest} action={action} safe3={safe3}")
            for s, et, ep, ef, ea in enemy_info:
                cy = data.get(f"e{s}_coll_y", 0)
                print(f"      s{s}:{et} @{ep} fl={ef:#x} anim={ea} cy={cy}")

        # Check if this is a disc ride
        is_disc = False
        for disc in state.discs:
            if pos == disc.jump_from and action == disc.direction:
                is_disc = True
                break

        if is_disc:
            # Verify Q*bert is actually at the disc position
            actual_pos = read_state(data, tracker).qbert
            if actual_pos != disc.jump_from:
                data = env.step()
                continue
            print(f"  DISC: pos={pos} side={disc.side}")
            port, field = MOVE_BUTTONS[action]
            env.step_n(port, field, 6)
            # Wait for disc ride to complete — Q*bert should end at (0,0)
            data = env.wait(300)
            tracker.reset()
            used_discs.add(disc.side)
            # Wait until Q*bert is at a valid position (disc ride done)
            for _ in range(300):
                data = env.step()
                state = read_state(data, tracker)
                if is_valid(state.qbert[0], state.qbert[1]) and state.qbert[0] <= 1:
                    break
            for _ in range(30):
                data = env.step()
            state = read_state(data, tracker)
            hops += 1
            print(f"  #{hops:3d} DISC! → {state.qbert}  cubes={cubes}/{NUM_CUBES}")
            continue

        dr, dc = MOVE_DELTAS[action]
        nr, nc = pos[0] + dr, pos[1] + dc
        if not is_valid(nr, nc):
            data = env.step()
            continue

        # Execute hop: hold button, then wait for Q*bert to be fully landed
        # (qb_anim >= 16 = ready for next input). This ensures the grid word
        # has stabilized and the position we read is correct.
        port, field = MOVE_BUTTONS[action]
        pos_before = pos
        env.step_n(port, field, 6)
        for _ in range(35):
            data = env.step()
            if data.get("qb_anim", 0) >= 16:
                break
        # Verify hop registered
        state = read_state(data, tracker)
        if state.qbert == pos_before:
            # Hop didn't register — retry on next iteration
            continue
        hops += 1

        # Post-hop: check if we're now on an enemy (death imminent)
        post_state = read_state(data, tracker)
        for e in post_state.enemies:
            if not e.harmless and is_valid(e.pos[0], e.pos[1]):
                if e.pos == post_state.qbert or e.prev_pos == post_state.qbert:
                    print(f"  !! POST-HOP DANGER hop {hops}: Q@{post_state.qbert} "
                          f"s{e.slot}:{e.etype}@{e.pos} prev={e.prev_pos} "
                          f"fl={e.flags:#x} anim={e.anim}")

        state = read_state(data, tracker)

        # Log when enemies are close OR every 10 hops
        from qbert.strategy import grid_dist as gd
        any_close = any(
            not e.harmless and is_valid(e.pos[0], e.pos[1]) and
            gd(pos[0], pos[1], e.pos[0], e.pos[1]) <= 2
            for e in state.enemies
        )
        if hops % 10 == 0 or any_close:
            qb_cy = data.get("qb_coll_y", 0)
            enemy_str = ""
            for e in state.enemies:
                if not is_valid(e.pos[0], e.pos[1]):
                    continue
                from qbert.strategy import grid_dist
                d = grid_dist(pos[0], pos[1], e.pos[0], e.pos[1])
                e_cy = data.get(f"e{e.slot}_coll_y", 0)
                cy_diff = abs(qb_cy - e_cy)
                enemy_str += (f"\n    s{e.slot}:{e.etype} @{e.pos} "
                              f"st={e.state} fl={e.flags:#x} d={d} "
                              f"cy={e_cy} qcy={qb_cy} diff={cy_diff}")
            tgt = f" →{_target}" if _target else ""
            print(f"  hop {hops}: {state.qbert} cubes={cubes}/{NUM_CUBES}{tgt} lives={state.lives}{enemy_str}")

    print(f"\nSurvived {hops} hops total")
    env.close()


if __name__ == "__main__":
    run()
