"""Debug overlay — draws grid, Q*bert, enemies, and predicted paths on frames."""

import cv2
import numpy as np

from qbert.state import is_valid, MAX_ROW, GameState
from qbert.predict import predict_coily, predict_ball_path


def grid_to_pixel(r, c):
    """Convert grid position to pixel coordinates (verified against sprites)."""
    return int(110 - r * 16 + c * 32), int(61 + r * 24)


def draw_overlay(frame, state: GameState, visited: dict):
    """Draw debug overlay on frame. Returns BGR image for cv2.imshow."""
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (720, 768), interpolation=cv2.INTER_NEAREST)
    s = 3  # scale factor

    # Grid circles
    for r in range(MAX_ROW + 1):
        for c in range(r + 1):
            px, py = grid_to_pixel(r, c)
            color = (0, 200, 200) if visited.get((r, c), False) else (60, 60, 60)
            cv2.circle(img, (px * s, py * s), 3 * s, color, 1)

    # Q*bert
    qr, qc = state.qbert
    if is_valid(qr, qc):
        px, py = grid_to_pixel(qr, qc)
        cv2.circle(img, (px * s, py * s), 6 * s, (0, 255, 0), 2)

    # Enemies
    for enemy in state.enemies:
        pos = enemy.pos
        if not is_valid(pos[0], pos[1]):
            continue
        px, py = grid_to_pixel(pos[0], pos[1])

        if enemy.etype == "coily":
            cv2.circle(img, (px * s, py * s), 5 * s, (0, 0, 255), 2)
            cv2.putText(img, "C", (px * s - 3 * s, py * s - 7 * s),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (0, 0, 255), s)
            target = state.qbert_prev if state.qbert_prev != pos else state.qbert
            cr, cc = pos
            for i in range(3):
                nr, nc = predict_coily(cr, cc, target[0], target[1])
                if not is_valid(nr, nc):
                    break
                npx, npy = grid_to_pixel(nr, nc)
                cv2.circle(img, (npx * s, npy * s), 3 * s, (0, 0, 200), 1)
                cr, cc = nr, nc
                target = state.qbert

        elif enemy.harmless:
            cv2.circle(img, (px * s, py * s), 4 * s, (0, 255, 0), 1)
            cv2.putText(img, "S", (px * s - 3 * s, py * s - 7 * s),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (0, 255, 0), s)
        else:
            cv2.circle(img, (px * s, py * s), 4 * s, (255, 100, 0), 2)
            path = predict_ball_path(pos[0], pos[1], enemy.direction_bits)
            for fp in path:
                if not is_valid(fp[0], fp[1]):
                    break
                fpx, fpy = grid_to_pixel(fp[0], fp[1])
                cv2.circle(img, (fpx * s, fpy * s), 2 * s, (200, 80, 0), 1)

    return img
