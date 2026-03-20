"""Enemy prediction — all algorithms ROM-verified. See GAME_STATE_MAP.md."""

from qbert.state import is_valid


def predict_coily(coily_row, coily_col, target_row, target_col):
    """Predict Coily's next position. ROM $B6EA.

    Coily chases Q*bert's PREVIOUS position using grid word comparison.
    If already at Q*bert's previous position, chases current instead.
    """
    c_gw1 = coily_row - coily_col + 1
    t_gw1 = target_row - target_col + 1

    if target_row > coily_row:  # target below → go DOWN
        if t_gw1 > c_gw1:
            return (coily_row + 1, coily_col)      # DOWN-LEFT
        else:
            return (coily_row + 1, coily_col + 1)  # DOWN-RIGHT
    else:  # target above or same → go UP
        if t_gw1 < c_gw1:
            return (coily_row - 1, coily_col)      # UP-RIGHT
        else:
            return (coily_row - 1, coily_col - 1)  # UP-LEFT


def predict_coily_n(coily_pos, target_pos, n):
    """Predict Coily's position n hops ahead."""
    r, c = coily_pos
    for _ in range(n):
        r, c = predict_coily(r, c, target_pos[0], target_pos[1])
        if not is_valid(r, c):
            return (r, c)
    return (r, c)


def predict_ball_path(row, col, direction_bits):
    """Predict a ball's entire remaining path from direction bits. ROM $B506.

    Each hop consumes bit 0: 1=RIGHT(+1,+1), 0=LEFT(+1,0).
    7 bits = 7 hops = full pyramid traversal.
    Returns list of future positions (not including current).
    """
    path = []
    r, c = row, col
    bits = direction_bits
    for _ in range(7):
        if bits & 1:
            r, c = r + 1, c + 1  # RIGHT
        else:
            r, c = r + 1, c      # LEFT
        bits >>= 1
        path.append((r, c))
        if r > 7:  # off pyramid
            break
    return path


def predict_ball_next(row, col, direction_bits):
    """Predict ball's next position from direction bits."""
    if direction_bits & 1:
        return (row + 1, col + 1)
    else:
        return (row + 1, col)
