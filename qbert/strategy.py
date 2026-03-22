"""Q*bert strategy — mode selection and mode-specific scoring."""

from qbert.state import is_valid, GameState, NUM_CUBES, MAX_ROW, pos_to_cube_index
from qbert.sim import SimResult

# Strategy modes
COLLECT = "collect"     # Color cubes efficiently
LURE = "lure"           # Lead Coily toward a disc
EVADE = "evade"         # Survive imminent danger
RUSH = "rush"           # Coily active, no discs — finish level fast
INTERCEPT = "intercept" # Chase Sam/Slick


def grid_dist(r1, c1, r2, c2):
    dr = r2 - r1
    dc = c2 - c1
    if dr >= 0 and dc >= 0:
        return max(dr, dc)
    if dr <= 0 and dc <= 0:
        return max(-dr, -dc)
    return abs(dr) + abs(dc)


def _find_coily(state):
    for e in state.enemies:
        if e.etype == "coily" and not e.harmless and is_valid(e.pos[0], e.pos[1]):
            return e.pos
    return None


def _find_sam(state):
    for e in state.enemies:
        if e.etype == "sam" and is_valid(e.pos[0], e.pos[1]):
            return e.pos
    return None


def _has_enemies(state):
    """Any harmful enemies on the board?"""
    return any(not e.harmless and is_valid(e.pos[0], e.pos[1])
               for e in state.enemies)


def _nearest_disc_launch(state):
    """Return the nearest disc launch position and distance, or (None, 999)."""
    if not state.discs:
        return None, 999
    qr, qc = state.qbert
    best = None
    best_d = 999
    for disc in state.discs:
        d = grid_dist(qr, qc, disc.jump_from[0], disc.jump_from[1])
        if d < best_d:
            best_d = d
            best = disc.jump_from
    return best, best_d


def _uncolored_cubes(state):
    """Return list of (row, col) positions of cubes not yet at target color."""
    cubes = []
    idx = 0
    for row in range(MAX_ROW + 1):
        for col in range(row, -1, -1):  # right-to-left per row (matches RAM layout)
            if state.cube_states[idx] != state.target_color:
                cubes.append((row, col))
            idx += 1
    return cubes


def _nearest_uncolored_dist(pos, uncolored):
    """Distance from pos to nearest uncolored cube."""
    if not uncolored:
        return 0
    return min(grid_dist(pos[0], pos[1], r, c) for r, c in uncolored)


def select_mode(state: GameState) -> str:
    """Pick the current strategy mode based on game state."""
    coily = _find_coily(state)
    qr, qc = state.qbert

    if coily:
        coily_d = grid_dist(qr, qc, coily[0], coily[1])

        # Disc lure is the dominant strategy when Coily is active
        if state.discs:
            return LURE

        # No discs — RUSH to finish, or EVADE if very close
        if coily_d <= 2:
            return EVADE
        return RUSH
    else:
        coily_d = 99

    # Chase Sam/Slick if safe
    sam = _find_sam(state)
    if sam and coily_d > 5:
        return INTERCEPT

    return COLLECT


def score_result(result: SimResult, mode: str, state: GameState,
                 recent_positions=None, route_target=None) -> float:
    """Score a simulation result based on the current strategy mode."""
    # Dead sequences: prefer dying later, tiebreak with cubes colored
    if not result.alive:
        return result.steps_survived * 100 + result.cubes_colored * 10

    base = 1_000_000  # alive always beats dead

    # Oscillation penalty
    osc_penalty = 0
    if recent_positions and result.final_pos in recent_positions:
        try:
            recency = len(recent_positions) - recent_positions.index(result.final_pos)
            osc_penalty = recency * 30
        except ValueError:
            pass

    # Route target bonus: strong pull toward the next planned cube
    route_bonus = 0
    if route_target:
        dist_to_target = grid_dist(result.final_pos[0], result.final_pos[1],
                                   route_target[0], route_target[1])
        route_bonus = -dist_to_target * 300  # very strong pull

    # Fallback: pull toward nearest uncolored cube
    uncolored = _uncolored_cubes(state)
    nearest_unc = _nearest_uncolored_dist(result.final_pos, uncolored)

    enemies_present = _has_enemies(state)

    if mode == COLLECT:
        escape_weight = 20 if enemies_present else 2
        return (base
                + result.cubes_colored * 1000
                + route_bonus             # follow the planned route
                - nearest_unc * 100       # fallback pull toward uncolored
                + result.escape_routes * escape_weight
                - osc_penalty)

    if mode == LURE:
        disc_launch, disc_dist = _nearest_disc_launch(state)
        if disc_launch:
            final_disc_dist = grid_dist(result.final_pos[0], result.final_pos[1],
                                        disc_launch[0], disc_launch[1])
            return (base
                    - final_disc_dist * 300   # very strong pull toward disc
                    + result.cubes_colored * 50
                    + result.escape_routes * 5
                    - osc_penalty)
        # No disc reachable, fall back to collect-style
        return (base
                + result.cubes_colored * 1000
                - nearest_unc * 200
                + result.escape_routes * 10
                - osc_penalty)

    if mode == EVADE:
        # Use PREDICTED Coily position
        cp = result.coily_final_pos
        if cp and cp != (-1, -1):
            coily_d = grid_dist(result.final_pos[0], result.final_pos[1],
                                cp[0], cp[1])
        else:
            coily_d = 99
        return (base
                + coily_d * 200
                + result.escape_routes * 50
                + result.cubes_colored * 80
                - osc_penalty)

    if mode == RUSH:
        # Coily active, no discs — collect aggressively, follow route
        cp = result.coily_final_pos
        if cp and cp != (-1, -1):
            coily_d = grid_dist(result.final_pos[0], result.final_pos[1],
                                cp[0], cp[1])
        else:
            coily_d = 99
        return (base
                + result.cubes_colored * 500
                + route_bonus             # follow planned route even under pressure
                - nearest_unc * 100
                + coily_d * 50
                + result.escape_routes * 20
                - osc_penalty)

    if mode == INTERCEPT:
        sam = _find_sam(state)
        if sam:
            sam_d = grid_dist(result.final_pos[0], result.final_pos[1],
                              sam[0], sam[1])
            return (base - sam_d * 100 + result.cubes_colored * 50
                    - osc_penalty)
        return (base + result.cubes_colored * 1000 - osc_penalty)

    return base
