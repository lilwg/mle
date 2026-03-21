"""Q*bert agent — main game loop using MLE."""

import argparse
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mle import MameEnv
from qbert.state import QBERT_RAM, read_state, is_valid, NUM_CUBES, MAX_ROW, EnemyTracker
from qbert.planner import (
    decide, neighbors, grid_dist, MOVE_BUTTONS, MOVE_NAMES, MOVE_DELTAS,
    COIN_BUTTON, START_BUTTON, DOWN,
)

ROMS_PATH = "/Users/pat/mame/roms"

# Timing constants (in frames, 1 frame per step)
# Old MAMEToolkit used frame_ratio=3: 4 steps * 3 = 12 frames hold, 10 * 3 = 30 wait
BUTTON_HOLD = 12   # frames to hold a direction button
HOP_WAIT = 40      # frames to wait after releasing for hop to complete


def run(overlay=False):
    if overlay:
        import cv2
        from qbert.overlay import draw_overlay

    env = MameEnv(ROMS_PATH, "qbert", QBERT_RAM, render=True, sound=False,
                  throttle=True)
    print("Q*bert Agent (MLE)")
    tracker = EnemyTracker()

    # Initial wait for game to boot
    env.wait(600)

    episode = 0
    try:
        while True:
            episode += 1
            if episode > 1:
                # Wait through game over / attract mode
                env.wait(900)

            # Insert coin and start game, retry if needed
            for attempt in range(3):
                env.step_n(*COIN_BUTTON, 15)
                env.wait(180)
                env.step_n(*START_BUTTON, 5)

                # Wait until lives > 0 (game actually started)
                data = env.step()
                state = read_state(data, tracker)
                started = False
                for _ in range(900):
                    if state.lives > 0 and is_valid(state.qbert[0], state.qbert[1]):
                        started = True
                        break
                    data = env.step()
                    state = read_state(data, tracker)
                if started:
                    break
                print(f"  Start attempt {attempt + 1} failed, retrying...")

            tracker.reset()
            prev_lives = state.lives
            prev_score = state.score_byte
            total_score = 0
            cubes = 0
            jumps = 0
            visited = {}
            level = 1
            stuck_count = 0
            prev_pos = None
            qbert_prev_known = None  # Python-tracked Q*bert previous position

            pos = state.qbert
            if is_valid(pos[0], pos[1]):
                visited[pos] = True

            print(f"\n--- Episode {episode}, Level {level} ---")
            print(f"  Lives={state.lives} Q*bert at {pos}")

            for step_num in range(3000):
                state = read_state(data, tracker)
                lives = state.lives
                score = state.score_byte

                # Score tracking (score byte wraps at 256)
                sd = score - prev_score
                if sd < 0:
                    sd += 256
                if sd >= 25:
                    total_score += sd
                    cubes += sd // 25
                prev_score = score

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
                          f"score={total_score}{enemy_info}")
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

                # Decide action
                action = decide(state, visited, qbert_prev_known)
                dr, dc = MOVE_DELTAS[action]
                nr, nc = pos[0] + dr, pos[1] + dc
                if not is_valid(nr, nc):
                    data = env.step()
                    continue

                # Safety override: if destination is on or adjacent to Coily, pick a safer move
                coily_enemy = None
                for e in state.enemies:
                    if e.etype == "coily" and is_valid(e.pos[0], e.pos[1]):
                        coily_enemy = e.pos
                        break
                if coily_enemy:
                    # Build danger zone: Coily + all positions reachable in 2 hops
                    czr, czc = coily_enemy
                    c_zone = {coily_enemy}
                    coily_moves = [(-1, -1), (-1, 0), (1, 0), (1, 1)]
                    hop1 = []
                    for cdr, cdc in coily_moves:
                        cp = (czr + cdr, czc + cdc)
                        if is_valid(cp[0], cp[1]):
                            c_zone.add(cp)
                            hop1.append(cp)
                    for r1, c1 in hop1:
                        for cdr, cdc in coily_moves:
                            cp = (r1 + cdr, c1 + cdc)
                            if is_valid(cp[0], cp[1]):
                                c_zone.add(cp)
                    if (nr, nc) in c_zone:
                        # Planner picked a dangerous move — override with safest alternative
                        best_alt = None
                        best_d = -1
                        for alt_a, alt_r, alt_c in neighbors(pos[0], pos[1]):
                            if (alt_r, alt_c) not in c_zone:
                                d = grid_dist(alt_r, alt_c, czr, czc)
                                if d > best_d:
                                    best_d = d
                                    best_alt = alt_a
                                    nr, nc = alt_r, alt_c
                        if best_alt is not None:
                            action = best_alt
                        # else: all moves are in the zone, no override possible

                # Track Q*bert's position before hopping
                qbert_prev_known = pos

                # Execute jump
                port, field = MOVE_BUTTONS[action]
                env.step_n(port, field, BUTTON_HOLD)
                data = env.wait(HOP_WAIT)
                jumps += 1

                # Update position
                state = read_state(data, tracker)
                new_pos = state.qbert
                if is_valid(new_pos[0], new_pos[1]):
                    pos = new_pos
                was_new = not visited.get(pos, False)
                visited[pos] = True

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
                          f"  cubes={cubes}/{NUM_CUBES}  score={total_score}"
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

                # Level complete
                if cubes >= NUM_CUBES:
                    print(f"\n  === LEVEL {level} COMPLETE! score={total_score} ===\n")
                    level += 1
                    cubes = 0
                    visited = {}
                    jumps = 0
                    stuck_count = 0
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
            print(f"Game over! Score: {total_score}, Level {level}")
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
