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


def _merge_ram_cubes_into_visited(state, visited):
    """Mark cubes as visited if their RAM color matches the target color.

    This supplements Python-tracked visited positions with ground-truth
    from RAM, ensuring the planner doesn't revisit already-completed cubes.
    """
    if state.target_color == 0:
        return  # target not set yet
    for i, color in enumerate(state.cube_states):
        if color == state.target_color:
            pos = cube_index_to_pos(i)
            if pos is not None:
                visited[pos] = True

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
            used_discs = set()  # track which discs we've used (by side)
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

                # Filter out used discs before deciding
                state.discs = [d for d in state.discs if d.side not in used_discs]

                # Decide action
                action = decide(state, visited, qbert_prev_known)
                dr, dc = MOVE_DELTAS[action]
                nr, nc = pos[0] + dr, pos[1] + dc

                # Check if this is a disc jump (intentionally off pyramid)
                is_disc_jump = False
                disc_used = None
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

                # Safety override: if destination is on or adjacent to Coily, pick a safer move
                coily_enemy = None
                for e in state.enemies:
                    if e.etype == "coily" and is_valid(e.pos[0], e.pos[1]):
                        coily_enemy = e.pos
                        break
                if coily_enemy:
                    # Build danger zone: Coily + all its 1-hop moves
                    czr, czc = coily_enemy
                    c_zone = {coily_enemy}
                    for cdr, cdc in [(-1, -1), (-1, 0), (1, 0), (1, 1)]:
                        cp = (czr + cdr, czc + cdc)
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

                # Execute jump, then wait for animation cycle to complete
                port, field = MOVE_BUTTONS[action]
                env.step_n(port, field, BUTTON_HOLD)
                # Phase 1: wait for hop animation to START (anim drops below 16)
                for _ in range(20):
                    data = env.step()
                    if data.get("qb_anim", 16) < 16:
                        break
                # Phase 2: wait for hop animation to COMPLETE (anim returns to >= 16)
                for _ in range(40):
                    data = env.step()
                    if data.get("qb_anim", 0) >= 16:
                        break
                jumps += 1

                # Update position
                state = read_state(data, tracker)
                new_pos = state.qbert
                if is_valid(new_pos[0], new_pos[1]):
                    pos = new_pos
                was_new = not visited.get(pos, False)
                visited[pos] = True

                # Merge RAM cube states into visited (ground-truth supplement)
                _merge_ram_cubes_into_visited(state, visited)

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

                # Level complete (score-based OR RAM remaining_cubes == 0)
                if cubes >= NUM_CUBES or state.remaining_cubes == 0:
                    print(f"\n  === LEVEL {level} COMPLETE! score={total_score} ===\n")
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
