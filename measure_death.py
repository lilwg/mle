"""Measure frame-by-frame entity state around a death to find exact killer timing."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mle import MameEnv
from qbert.state import QBERT_RAM, read_state, is_valid, gw_to_pos, EnemyTracker
from qbert.planner import COIN_BUTTON, START_BUTTON, MOVE_BUTTONS
from test_survive import decide, execute_hop, execute_disc, find_coily, grid_dist, wait_for_level_start
from qbert.sim import MOVE_DELTAS

ROMS_PATH = "/Users/pat/mame/roms"

env = MameEnv(ROMS_PATH, "qbert", QBERT_RAM, render=True, sound=False, throttle=True)
tracker = EnemyTracker()
import random
env.wait(600 + random.randint(0, 200))
env.step_n(*COIN_BUTTON, 15)
env.wait(180)
env.step_n(*START_BUTTON, 5)
for _ in range(900):
    data = env.step()
    s = read_state(data, tracker)
    if 0 < s.lives <= 5 and is_valid(s.qbert[0], s.qbert[1]):
        break

state = read_state(data, tracker)
prev_lives = state.lives
hops = 0
used_discs = set()
hsp = 0
lc = 0
current_level = 1

# Play through levels 1-2, then measure L3 deaths frame by frame
for _ in range(50000):
    data = env.step()
    state = read_state(data, tracker)

    if state.remaining_cubes == 0 and state.lives > 0:
        current_level += 1
        used_discs = set()
        tracker.reset()
        hsp = 0; lc = 0
        data, state = wait_for_level_start(env, tracker)
        prev_lives = state.lives
        hops = 1
        print(f"\n=== Level {current_level} ===")
        continue

    if state.lives < prev_lives:
        prev_lives = state.lives
        tracker.reset()
        hsp = 0
        for _ in range(300):
            data = env.step()
            state = read_state(data, tracker)
            if is_valid(state.qbert[0], state.qbert[1]) and data.get("qb_anim", 0) >= 16:
                break
        hops = 0
        if state.lives == 0:
            break
        continue
    prev_lives = state.lives

    if not is_valid(state.qbert[0], state.qbert[1]):
        continue

    state.discs = [d for d in state.discs if d.side not in used_discs]
    action = decide(state, hsp)
    if action is None:
        for _ in range(6): data = env.step()
        continue

    dr, dc = MOVE_DELTAS[action]
    nr, nc = state.qbert[0]+dr, state.qbert[1]+dc
    if not is_valid(nr, nc):
        disc_match = None
        for disc in state.discs:
            if state.qbert == disc.jump_from and action == disc.direction:
                disc_match = disc; break
        if disc_match:
            data, state = execute_disc(env, action, tracker, used_discs, disc_match)
            if data is None: data = env.step(); continue
            hops += 1; hsp = 0; continue
        data = env.step(); continue

    # On level 3, do frame-by-frame logging during hops
    if current_level >= 3 and hops >= 8:
        print(f"\n--- Hop {hops+1}: QB {state.qbert}→({nr},{nc}) ---")
        # Log all enemies BEFORE hop
        for e in state.enemies:
            if e.harmless: continue
            print(f"  PRE: s{e.slot} {e.etype} @{e.pos} a={e.anim} cy={e.coll_y} fl={e.flags:#x}")

        # Execute hop manually, reading EVERY frame
        port, field = MOVE_BUTTONS[action]
        env.step_n(port, field, 6)  # button hold
        for f in range(25):
            data = env.step()
            s2 = read_state(data, tracker)
            # Check if any enemy is at destination
            for e2 in s2.enemies:
                if e2.harmless: continue
                if e2.pos == (nr, nc) or (is_valid(e2.pos[0], e2.pos[1]) and grid_dist(e2.pos[0], e2.pos[1], nr, nc) == 0):
                    print(f"  F{f:2d}: DANGER s{e2.slot} {e2.etype} @{e2.pos} a={e2.anim} cy={e2.coll_y}")
            if s2.lives < prev_lives:
                print(f"  F{f:2d}: DEATH @{s2.qbert} lives={s2.lives}")
                for e2 in s2.enemies:
                    print(f"    s{e2.slot} {e2.etype} @{e2.pos} a={e2.anim} cy={e2.coll_y} fl={e2.flags:#x}")
                # Also raw
                for n in range(10):
                    fl = data.get(f"e{n}_flags", 0)
                    st_raw = data.get(f"e{n}_st", 0)
                    cy = data.get(f"e{n}_coll_y", 0)
                    if fl != 0 or st_raw != 0:
                        p = gw_to_pos(data.get(f"e{n}_gw0",0), data.get(f"e{n}_gw1",0))
                        print(f"    RAW s{n}: fl={fl:#x} st={st_raw} pos={p} cy={cy} a={data.get(f'e{n}_anim',0)}")
                break

        state = read_state(data, tracker)
    else:
        data = execute_hop(env, action, tracker)
        if data is None:
            data = env.step()
            continue
        state = read_state(data, tracker)

    hops += 1
    c = 28 - state.remaining_cubes
    if c > lc: hsp = 0; lc = c
    else: hsp += 1

    if state.lives <= 0:
        break

env.close()
