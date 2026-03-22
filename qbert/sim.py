"""Frame-accurate Q*bert game simulator for lookahead search.

Ticks frame-by-frame with per-entity animation counters read from RAM.
Uses predict.py for deterministic enemy movement.
"""

from dataclasses import dataclass, field
from copy import deepcopy

from qbert.state import is_valid, MAX_ROW, NUM_CUBES, GameState, pos_to_cube_index
from qbert.predict import predict_coily

# Hop interval constants (frames between grid-word updates).
# First hop uses actual anim_counter from RAM; subsequent hops reload these.
QBERT_HOP_INTERVAL = 18
COILY_HOP_INTERVAL = 47   # measured: Coily hops every 47 frames
BALL_HOP_INTERVAL = 43    # 0x20 (32) wait + ~11 overhead

# Action constants (match planner)
UP = 0     # (-1, 0)
DOWN = 1   # (+1, 0)
LEFT = 2   # (-1, -1)
RIGHT = 3  # (+1, +1)

MOVE_DELTAS = {UP: (-1, 0), DOWN: (+1, 0), LEFT: (-1, -1), RIGHT: (+1, +1)}


@dataclass
class SimEntity:
    pos: tuple
    prev_pos: tuple
    etype: str            # "coily", "ball", "sam", "slick"
    anim_counter: int     # frames until next hop
    hop_interval: int     # reload value after hop
    direction_bits: int   # for balls/sam/slick
    harmful: bool


@dataclass
class SimResult:
    alive: bool
    steps_survived: int      # how many Q*bert hops completed before death
    cubes_colored: int       # new cubes colored during simulation
    remaining_cubes: int     # cubes still needing coloring after sim
    final_pos: tuple         # Q*bert's final position
    final_prev: tuple        # Q*bert's final previous position
    coily_alive: bool        # is Coily still on the board
    coily_final_pos: tuple   # predicted Coily position at end of sim
    escape_routes: int       # valid moves from final position


@dataclass
class SimState:
    qbert_pos: tuple
    qbert_prev: tuple
    qbert_counter: int       # frames until Q*bert's next hop fires
    entities: list
    cubes: list              # 28 cube color values
    target_color: int
    remaining: int           # cubes left to color
    spawn_countdown: int = 0 # frames until next enemy spawns
    alive: bool = True


def _hop_interval_for(etype):
    if etype == "coily":
        return COILY_HOP_INTERVAL
    return BALL_HOP_INTERVAL


def make_sim_state(state: GameState) -> SimState:
    """Convert a RAM GameState into a SimState for simulation."""
    entities = []
    for e in state.enemies:
        if e.harmless:
            continue  # skip Sam/Slick in danger sim (they don't kill)

        pos = e.pos
        prev = e.prev_pos
        pos_valid = is_valid(pos[0], pos[1])
        prev_valid = is_valid(prev[0], prev[1])

        if pos_valid:
            entities.append(SimEntity(
                pos=pos, prev_pos=prev, etype=e.etype,
                anim_counter=max(e.anim, 1),
                hop_interval=_hop_interval_for(e.etype),
                direction_bits=e.direction_bits,
                harmful=True,
            ))
        elif prev_valid and prev[0] < MAX_ROW:
            # Mid-transition: current pos invalid but prev is valid and
            # not on bottom row (bottom row = ball fell off, not dangerous).
            # Enemy is mid-hop from prev — place it at prev with small
            # anim_counter since it's about to land.
            entities.append(SimEntity(
                pos=prev,
                prev_pos=prev,
                etype=e.etype,
                anim_counter=max(e.anim, 1),
                hop_interval=_hop_interval_for(e.etype),
                direction_bits=e.direction_bits,
                harmful=True,
            ))
    return SimState(
        qbert_pos=state.qbert,
        qbert_prev=state.qbert_prev,
        qbert_counter=1,  # Q*bert can hop immediately (agent controls timing)
        entities=list(entities),
        cubes=list(state.cube_states),
        target_color=state.target_color,
        remaining=state.remaining_cubes,
        spawn_countdown=state.spawn_countdown,
    )


def clone(state: SimState) -> SimState:
    """Deep copy a SimState for branching search."""
    return SimState(
        qbert_pos=state.qbert_pos,
        qbert_prev=state.qbert_prev,
        qbert_counter=state.qbert_counter,
        entities=[SimEntity(
            pos=e.pos, prev_pos=e.prev_pos, etype=e.etype,
            anim_counter=e.anim_counter, hop_interval=e.hop_interval,
            direction_bits=e.direction_bits, harmful=e.harmful,
        ) for e in state.entities],
        cubes=list(state.cubes),
        target_color=state.target_color,
        remaining=state.remaining,
        spawn_countdown=state.spawn_countdown,
        alive=state.alive,
    )


def _check_collision(state: SimState, entity: SimEntity) -> bool:
    """Check ROM-style collision: current/prev grid word cross-check."""
    if not entity.harmful:
        return False
    qp = state.qbert_pos
    qprev = state.qbert_prev
    ep = entity.pos
    eprev = entity.prev_pos
    # Q*bert current vs Enemy current
    if qp == ep:
        return True
    # Q*bert current vs Enemy previous
    if is_valid(eprev[0], eprev[1]) and qp == eprev:
        return True
    # Q*bert previous vs Enemy current
    if is_valid(qprev[0], qprev[1]) and qprev == ep:
        return True
    return False


def _move_entity(entity: SimEntity, qbert_prev: tuple, qbert_pos: tuple):
    """Advance an entity one hop."""
    old_pos = entity.pos
    r, c = old_pos

    if entity.etype == "coily":
        target = qbert_prev if old_pos != qbert_prev else qbert_pos
        nr, nc = predict_coily(r, c, target[0], target[1])
        if is_valid(nr, nc):
            entity.prev_pos = old_pos
            entity.pos = (nr, nc)

    elif entity.etype in ("ball", "sam", "slick"):
        # Consume LSB of direction_bits
        if entity.direction_bits & 1:
            nr, nc = r + 1, c + 1  # down-right
        else:
            nr, nc = r + 1, c      # down-left
        entity.direction_bits >>= 1
        if is_valid(nr, nc):
            entity.prev_pos = old_pos
            entity.pos = (nr, nc)
            entity.harmful = True  # activate on first move (phantom spawns start dormant)
        else:
            # Ball fell off pyramid — deactivate
            entity.harmful = False
            entity.pos = (-1, -1)


def _color_cube(state: SimState, pos: tuple) -> bool:
    """Color the cube at pos. Returns True if cube reached target color.

    For multi-hit levels, each visit increments the color by 1.
    The cube is "done" when it reaches target_color.
    """
    idx = pos_to_cube_index(pos[0], pos[1])
    if idx is None:
        return False
    if state.cubes[idx] != state.target_color:
        state.cubes[idx] += 1
        if state.cubes[idx] == state.target_color:
            state.remaining -= 1
            return True
    return False


def _count_escape_routes(pos):
    """Count valid moves from a position."""
    r, c = pos
    count = 0
    for dr, dc in MOVE_DELTAS.values():
        if is_valid(r + dr, c + dc):
            count += 1
    return count


def _find_coily_pos(entities):
    """Return Coily's current position, or (-1,-1) if not present."""
    for e in entities:
        if e.etype == "coily" and e.pos != (-1, -1):
            return e.pos
    return (-1, -1)


# Spawn point: enemies appear here and bounce down.
SPAWN_POS = (0, 0)
# After spawning, the ball's first hop delay
SPAWN_ANIM_DELAY = 80  # 0x50 from GAME_STATE_MAP spawn delay


def _spawn_enemy(state: SimState):
    """Inject two phantom balls at the spawn point covering both directions.

    We don't know the random direction_bits, so we pessimistically cover
    both left and right paths for the first several hops.
    Phantoms start non-harmful (dormant spawn animation) and become
    harmful when they make their first hop.
    """
    # Ball going all-left: 0b0000000
    state.entities.append(SimEntity(
        pos=SPAWN_POS, prev_pos=(-1, -1), etype="ball",
        anim_counter=SPAWN_ANIM_DELAY,
        hop_interval=BALL_HOP_INTERVAL,
        direction_bits=0b0000000,  # all left
        harmful=False,  # dormant until first move
    ))
    # Ball going all-right: 0b1111111
    state.entities.append(SimEntity(
        pos=SPAWN_POS, prev_pos=(-1, -1), etype="ball",
        anim_counter=SPAWN_ANIM_DELAY,
        hop_interval=BALL_HOP_INTERVAL,
        direction_bits=0b1111111,  # all right
        harmful=False,  # dormant until first move
    ))


def simulate(state: SimState, actions: list) -> SimResult:
    """Simulate a sequence of Q*bert hops through the game.

    Each action is applied when Q*bert's counter fires. Between Q*bert hops,
    enemy counters tick down and enemies move when their counters fire.
    """
    cubes_colored = 0
    steps = 0

    for action in actions:
        dr, dc = MOVE_DELTAS[action]
        nr, nc = state.qbert_pos[0] + dr, state.qbert_pos[1] + dc

        # Invalid move — skip this action
        if not is_valid(nr, nc):
            continue

        # Tick frames until Q*bert's next hop
        for _ in range(QBERT_HOP_INTERVAL):
            # Spawn countdown
            if state.spawn_countdown > 0:
                state.spawn_countdown -= 1
                if state.spawn_countdown <= 0:
                    _spawn_enemy(state)

            # Tick each entity
            for e in state.entities:
                if e.pos == (-1, -1):
                    continue
                e.anim_counter -= 1
                if e.anim_counter <= 0:
                    _move_entity(e, state.qbert_prev, state.qbert_pos)
                    e.anim_counter = e.hop_interval
                # Check collision after entity moves
                if _check_collision(state, e):
                    state.alive = False
                    return SimResult(
                        alive=False, steps_survived=steps,
                        cubes_colored=cubes_colored,
                        remaining_cubes=state.remaining,
                        final_pos=state.qbert_pos,
                        final_prev=state.qbert_prev,
                        coily_alive=any(e.etype == "coily" and e.pos != (-1, -1)
                                        for e in state.entities),
                        coily_final_pos=_find_coily_pos(state.entities),
                        escape_routes=0,
                    )

        # Apply Q*bert's hop
        state.qbert_prev = state.qbert_pos
        state.qbert_pos = (nr, nc)
        steps += 1

        # Color the cube
        if _color_cube(state, state.qbert_pos):
            cubes_colored += 1

        # Check collision at new position
        for e in state.entities:
            if _check_collision(state, e):
                state.alive = False
                return SimResult(
                    alive=False, steps_survived=steps,
                    cubes_colored=cubes_colored,
                    remaining_cubes=state.remaining,
                    final_pos=state.qbert_pos,
                    final_prev=state.qbert_prev,
                    coily_alive=any(e.etype == "coily" and e.pos != (-1, -1)
                                    for e in state.entities),
                    coily_final_pos=_find_coily_pos(state.entities),
                    escape_routes=0,
                )

    return SimResult(
        alive=True,
        steps_survived=steps,
        cubes_colored=cubes_colored,
        remaining_cubes=state.remaining,
        final_pos=state.qbert_pos,
        final_prev=state.qbert_prev,
        coily_alive=any(e.etype == "coily" and e.pos != (-1, -1)
                        for e in state.entities),
        coily_final_pos=_find_coily_pos(state.entities),
        escape_routes=_count_escape_routes(state.qbert_pos),
    )
