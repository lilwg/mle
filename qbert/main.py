"""Q*bert agent — main game loop using MLE."""

import argparse
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mle import MameEnv
from qbert.state import QBERT_RAM, read_state, is_valid, NUM_CUBES, MAX_ROW, EnemyTracker, cube_index_to_pos
from qbert.planner import (
    decide, neighbors, grid_dist, MOVE_BUTTONS, MOVE_NAMES, MOVE_DELTAS,
    COIN_BUTTON, START_BUTTON, DOWN,
)

ROMS_PATH = "/Users/pat/mame/roms"

# Timing constants (in frames, 1 frame per step)
# Old MAMEToolkit used frame_ratio=3: 4 steps * 3 = 12 frames hold, 10 * 3 = 30 wait
# Measured: grid word updates immediately. Q*bert needs ~30 frames for full
# hop cycle. Button must be held long enough to register reliably.
BUTTON_HOLD = 6    # reliable input registration
HOP_WAIT = 24      # total ~30 frames per hop (faster than Coily's ~29)


def run(overlay=False):
    if overlay:
        import cv2
        from qbert.overlay import draw_overlay

    env = MameEnv(ROMS_PATH, "qbert", QBERT_RAM, render=True, sound=False,
                  throttle=True)
    print("Q*bert Agent (MLE)")
    tracker = EnemyTracker()

    # Initial wait for game to boot (randomize to vary MAME seed)
    env.wait(600 + random.randint(0, 120))

    episode = 0
    try:
        while True:
            episode += 1
            if episode > 1:
                # Wait through game over / attract mode
                env.wait(900)

            # First episode: throwaway to vary MAME seed
            if episode == 1:
                print("  Throwaway game for seed variation...")
                env.step_n(*COIN_BUTTON, 15)
                env.wait(60)
                env.step_n(*START_BUTTON, 5)
                env.wait(300)
                env.wait(600)
                print("  Done, starting real game.")
                episode += 1
            for attempt in range(3):
                env.step_n(*COIN_BUTTON, 15)
                env.wait(180)
                env.step_n(*START_BUTTON, 5)

                # Wait until game started
                data = env.step()
                state = read_state(data, tracker)
                started = False
                for _ in range(900):
                    if 0 < state.lives <= 5 and is_valid(state.qbert[0], state.qbert[1]):
                        started = True
                        break
                    data = env.step()
                    state = read_state(data, tracker)
                if started:
                    break
                print(f"  Start attempt {attempt + 1} failed, retrying...")

            tracker.reset()
            used_discs = set()
            prev_lives = state.lives
            jumps = 0
            level = 1
            visited = {}
            stuck_count = 0
            prev_pos = None
            qbert_prev_known = None
            pos = state.qbert

            print(f"\n--- Episode {episode}, Level {level} ---")
            print(f"  Lives={state.lives} Q*bert at {pos}")

            for step_num in range(3000):
                state = read_state(data, tracker)
                lives = state.lives

                # Use remaining_cubes from RAM for cube count (works all levels)
                cubes = NUM_CUBES - state.remaining_cubes

                # Death check: lives decreased OR Q*bert is on same square as Coily
                # (game may kill Q*bert before the lives byte updates in RAM)
                coily_on_qbert = any(
                    e.etype == "coily" and e.pos == state.qbert
                    for e in state.enemies
                    if is_valid(e.pos[0], e.pos[1])
                )
                if coily_on_qbert and lives > 0:
                    # Coily caught us — wait for the death to register
                    for _ in range(60):
                        data = env.step()
                        s = read_state(data, tracker)
                        if s.lives < lives:
                            lives = s.lives
                            break
                    state = read_state(data, tracker)

                if lives < prev_lives and prev_lives > 0:
                    pos = state.qbert
                    enemy_info = ""
                    for e in state.enemies:
                        enemy_info += (
                            f"\n    s{e.slot}: {e.etype} "
                            f"@{e.pos} prev={e.prev_pos} fl={e.flags:#x}"
                        )
                    print(f"  DIED at {pos} lives={lives} cubes={cubes} "
                          f"cubes={cubes}{enemy_info}")
                    if lives == 0:
                        break
                    prev_lives = lives
                    tracker.reset()
                    qbert_prev_known = None
                    data = env.wait(300)
                    state = read_state(data, tracker)
                    pos = state.qbert
                    if is_valid(pos[0], pos[1]):
                        visited[pos] = True
                    jumps = 0
                    stuck_count = 0
                    prev_pos = None
                    continue
                prev_lives = lives

                # Position
                pos = state.qbert
                if not is_valid(pos[0], pos[1]):
                    data = env.step()
                    continue

                # Stuck detection
                if pos == prev_pos:
                    stuck_count += 1
                else:
                    stuck_count = 0
                prev_pos = pos

                if stuck_count >= 4:
                    valid = neighbors(pos[0], pos[1])
                    if valid:
                        act = random.choice(valid)[0]
                        port, field = MOVE_BUTTONS[act]
                        qbert_prev_known = pos
                        env.step_n(port, field, BUTTON_HOLD)
                        data = env.wait(HOP_WAIT)
                        stuck_count = 0
                    continue

                # Filter out used discs before deciding
                state.discs = [d for d in state.discs if d.side not in used_discs]

                # Decide action
                from qbert.planner import _danger_set, _search_routes
                dangers_1 = _danger_set(state, 1)
                routes = _search_routes(state, visited)
                action = decide(state, visited, qbert_prev_known)
                dr2, dc2 = MOVE_DELTAS[action]
                dest = (pos[0]+dr2, pos[1]+dc2)
                print(f"    decide: Q@{pos} → {MOVE_NAMES[action]}→{dest} "
                      f"enemies={len(state.enemies)} dangers={len(dangers_1)} "
                      f"routes={len(routes)} valid={is_valid(dest[0],dest[1])}")
                dr, dc = MOVE_DELTAS[action]
                nr, nc = pos[0] + dr, pos[1] + dc

                # Check if this is a disc jump (intentionally off pyramid)
                is_disc_jump = False
                disc_used = None
                if not is_valid(nr, nc):
                    print(f"    INVALID dest {(nr,nc)} — checking discs: "
                          f"discs={[(d.side, d.jump_from, d.direction) for d in state.discs]} "
                          f"used={used_discs} pos={pos} action={action}")
                for disc in state.discs:
                    if pos == disc.jump_from and action == disc.direction:
                        is_disc_jump = True
                        disc_used = disc
                        break

                if is_disc_jump:
                    port, field = MOVE_BUTTONS[action]
                    qbert_prev_known = pos
                    env.step_n(port, field, BUTTON_HOLD)
                    data = env.wait(300)
                    tracker.reset()
                    qbert_prev_known = None
                    used_discs.add(disc_used.side)
                    state = read_state(data, tracker)
                    new_pos = state.qbert
                    if new_pos == (0, 0) or new_pos == (-1, -1):
                        # Disc ride succeeded
                        pos = new_pos if is_valid(new_pos[0], new_pos[1]) else (0, 0)
                        visited[pos] = True
                        jumps += 1
                        print(f"  #{jumps:3d} DISC! → {pos}  cubes={cubes}/{NUM_CUBES}")
                    else:
                        # Disc was already used — Q*bert jumped off pyramid
                        # This is a death, handle in next iteration
                        pos = new_pos
                        jumps += 1
                    prev_pos = None
                    stuck_count = 0
                    continue

                if not is_valid(nr, nc):
                    data = env.step()
                    continue

                # Track Q*bert's position before hopping
                qbert_prev_known = pos

                # Execute jump. Press direction, wait for gw to change
                # (hop started), then read more frames for enemy positions.
                port, field = MOVE_BUTTONS[action]
                gw_before = (data.get("qb_gw0", 0), data.get("qb_gw1", 0))
                env.step_n(port, field, BUTTON_HOLD)
                # Wait for gw change (hop registered)
                hop_frame = 0
                for _f in range(20):
                    data = env.step()
                    hop_frame += 1
                    if (data.get("qb_gw0", 0), data.get("qb_gw1", 0)) != gw_before:
                        break
                # Read remaining frames up to 18 total for enemy data
                while hop_frame < 18 - BUTTON_HOLD:
                    data = env.step()
                    hop_frame += 1
                jumps += 1

                # Update position
                state = read_state(data, tracker)
                new_pos = state.qbert
                if is_valid(new_pos[0], new_pos[1]):
                    pos = new_pos
                was_new = not visited.get(pos, False)
                visited[pos] = True

                # Merge RAM cube states into visited (ground-truth supplement)

                # Log significant events
                enemy_str = ""
                for e in state.enemies:
                    if not is_valid(e.pos[0], e.pos[1]):
                        continue
                    if e.etype == "coily":
                        d = grid_dist(pos[0], pos[1], e.pos[0], e.pos[1])
                        enemy_str += f" C{e.pos}d={d}"
                    elif e.etype == "sam":
                        enemy_str += f" S{e.pos}"
                    elif not e.harmless:
                        enemy_str += f" B{e.pos}"

                if was_new or enemy_str:
                    new_mark = " NEW" if was_new else ""
                    print(f"  #{jumps:3d} {MOVE_NAMES[action]:>5s} {pos}"
                          f"  cubes={cubes}/{NUM_CUBES}  cubes={cubes}"
                          f"{enemy_str}{new_mark}")

                # Overlay (every 3rd jump to save overhead)
                if overlay and (jumps % 3 == 0 or was_new):
                    env.request_frame()
                    frame_data = env.step()
                    if "frame" in frame_data:
                        img = draw_overlay(frame_data["frame"], state, visited)
                        cv2.imshow("Q*bert Agent (MLE)", img)
                        cv2.waitKey(1)
                    data = frame_data

                # Level complete (score-based OR RAM remaining_cubes == 0)
                if cubes >= NUM_CUBES or state.remaining_cubes == 0:
                    print(f"\n  === LEVEL {level} COMPLETE! cubes={cubes} ===\n")
                    level += 1
                    cubes = 0
                    visited = {}
                    jumps = 0
                    stuck_count = 0
                    used_discs = set()
                    # Wait for level transition animation
                    env.wait(600)
                    # Reset tracker and read frames until Q*bert is valid
                    tracker.reset()
                    for _ in range(600):
                        data = env.step()
                        state = read_state(data, tracker)
                        pos = state.qbert
                        if is_valid(pos[0], pos[1]) and pos[0] <= 1:
                            break
                    # Flush a few more frames so tracker votes stabilize
                    for _ in range(30):
                        data = env.step()
                        state = read_state(data, tracker)
                    pos = state.qbert
                    if is_valid(pos[0], pos[1]):
                        visited[pos] = True
                    prev_score = state.score_byte
                    prev_lives = state.lives
                    print(f"--- Level {level} ---")

            missing = [(r, c) for r in range(MAX_ROW + 1) for c in range(r + 1)
                        if not visited.get((r, c), False)]
            print(f"Game over! Cubes: {cubes}/28, Level {level}")
            if missing:
                print(f"  Missing cubes: {missing}\n")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        env.close()
        if overlay:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q*bert heuristic agent")
    parser.add_argument("--overlay", action="store_true", help="Show debug overlay window")
    args = parser.parse_args()
    run(overlay=args.overlay)
