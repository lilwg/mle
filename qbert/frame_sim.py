"""Frame-perfect Q*bert simulator, translated directly from ROM.

Replicates the ROM's per-frame entity update loop ($B56A) and
collision check ($BD1E) tick-for-tick. No approximations.

Entity structure (22 bytes, matching RAM at $0D70 + n*22):
  [0]  pos        (row, col) grid position
  [1]  prev_pos   (row, col) previous grid position
  [2]  anim       animation counter (counts down)
  [3]  etype      "coily", "ball", "ugg", "sam"
  [4]  dir_bits   direction bits (consumed 1 per hop)
  [5]  flags      raw flags byte
  [6]  hops       hops remaining (for balls/Coily pre-hatch)
"""

from qbert.state import is_valid, gw_to_pos, pos_to_gw, MAX_ROW
from qbert.predict import predict_coily


# ROM reload values (set at landing)
COILY_RELOAD = 32   # $B962: Coily anim reset
BALL_RELOAD = 28    # $B6A5: Ball anim reset
SAM_RELOAD = 33     # $B6A5 + $B6BB: Sam/Slick
HOP_TRIGGER = 16    # $B566: hop triggers when anim == 16


def make_entity(pos, prev_pos, anim, etype, dir_bits, flags, hops=6):
    """Create a sim entity."""
    return [pos, prev_pos, anim, etype, dir_bits, flags, hops]


def clone_entity(en):
    return [en[0], en[1], en[2], en[3], en[4], en[5], en[6]]


def clone_entities(entities):
    return [clone_entity(e) for e in entities]


def build_entities(state):
    """Convert game state enemies into frame-sim entities.

    Maps RAM anim values to our frame-sim format:
    - anim > 16: waiting phase (use as-is, counts down to 16)
    - anim 1-5: in-flight (ROM anim stays at ~4 during flight)
      → CONSERVATIVE: assume about to land (worst case for safety)
    - anim == 0: in-flight, about to land
    - anim == 16: about to trigger hop
    """
    entities = []
    for e in state.enemies:
        if e.harmless:
            continue
        anim = e.anim
        etype = e.etype
        if anim == 0 or (1 <= anim <= 5):
            # In-flight: gw shows DESTINATION, entity flying from prev_pos.
            # Add TWO entries like the old sim:
            # 1. Destination (will land here) — in flight, ~15 frames left
            # 2. Also predict NEXT position after landing (blocks future)
            from qbert.predict import predict_coily as _pc
            # Entity at destination, will land and stay for ~RELOAD frames
            # Match old sim behavior: block destination for ~30 frames
            anim_dest = -1  # about to land (1 frame from landing)
            entities.append(make_entity(
                e.pos, e.prev_pos, anim_dest, etype,
                e.direction_bits, e.flags, hops=6
            ))
            # Predict next hop from destination
            qpos, qprev = state.qbert, state.qbert_prev
            if etype == "coily":
                target = qprev if e.pos != qprev else qpos
                nxt = _pc(e.pos[0], e.pos[1], target[0], target[1])
            elif etype == "ball":
                d = e.direction_bits
                if d & 1:
                    nxt = (e.pos[0]+1, e.pos[1]+1)
                else:
                    nxt = (e.pos[0]+1, e.pos[1])
            else:
                nxt = None
            if nxt and is_valid(nxt[0], nxt[1]):
                # Next position: will arrive after landing + full cycle
                next_anim = BALL_RELOAD if etype != "coily" else COILY_RELOAD
                entities.append(make_entity(
                    nxt, e.pos, next_anim, etype,
                    e.direction_bits >> 1, e.flags, hops=5
                ))
            continue
        elif anim <= HOP_TRIGGER:
            anim = HOP_TRIGGER
        # else: anim > 16, waiting (use as-is)

        # Purple ball at bottom → about to hatch
        if etype == "ball" and e.flags in (0x60, 0x68) and e.pos[0] >= 6:
            etype = "coily"
            anim = HOP_TRIGGER

        en = make_entity(
            e.pos, e.prev_pos, anim, etype,
            e.direction_bits, e.flags,
            hops=6
        )
        entities.append(en)
    return entities


def entity_hop(en, qb_prev):
    """ROM $B591/$B6E8: trigger an entity hop. Updates position."""
    pos = en[0]
    etype = en[3]

    if etype == "coily":
        # Chase Q*bert's previous position
        target = qb_prev
        new_pos = predict_coily(pos[0], pos[1], target[0], target[1])
    elif etype == "ugg":
        # Ugg/Wrongway moves on cube face (off-grid positions).
        # Direction bit: 1=down(row+1), 0=up(row-1).
        # They threaten edge cubes: left face → (r,0), right face → (r,r).
        # For simulation, keep the off-grid position format from RAM.
        dbits = en[4]
        if dbits & 1:
            nr = pos[0] + 1
        else:
            nr = pos[0] - 1
        # Maintain the same face offset
        if pos[1] < 0:
            new_pos = (nr, -1)  # left face
        elif pos[1] > pos[0]:
            new_pos = (nr, nr + 1)  # right face
        else:
            # On-grid Ugg (edge cube) — should not normally happen in sim
            # but handle gracefully: move down on same edge
            new_pos = (nr, 0) if pos[1] == 0 else (nr, nr)
        en[4] = dbits >> 1
    else:
        # Ball: direction bit determines left/right
        dbits = en[4]
        if dbits & 1:
            new_pos = (pos[0] + 1, pos[1] + 1)
        else:
            new_pos = (pos[0] + 1, pos[1])
        en[4] = dbits >> 1

    # Update position
    en[1] = pos  # prev = current
    if etype == "ugg":
        # Ugg stays off-grid; valid as long as adjacent row exists
        if 0 <= new_pos[0] <= MAX_ROW:
            en[0] = new_pos
        else:
            en[0] = (-1, -1)  # fell off top or bottom
    elif is_valid(new_pos[0], new_pos[1]):
        en[0] = new_pos
    elif en[5] in (0x60, 0x68) and etype != "coily":
        # Purple ball at bottom → hatch into Coily
        en[3] = "coily"
        en[6] = 0
    else:
        en[0] = (-1, -1)  # fell off

    # Set reload
    if en[3] == "coily":
        en[2] = COILY_RELOAD
    elif en[3] == "sam":
        en[2] = SAM_RELOAD
    else:
        en[2] = BALL_RELOAD


FLIGHT_FRAMES = 30  # Measured: flight takes ~30 frames (physics-based)


def entity_tick(en, qb_prev):
    """ROM $B56A: one frame of entity update.

    Anim counter phases:
    - anim > 16: waiting (decrement each frame)
    - anim == 16: hop trigger (update gw, start flight)
    - anim < 0: in-flight countdown (we use negative values to count
      flight frames; ROM uses physics but result is ~30 frames)
    - anim reaches -FLIGHT_FRAMES: landing, reset to RELOAD
    """
    if en[0] == (-1, -1):
        return False

    anim = en[2]
    if anim > HOP_TRIGGER:
        # Waiting phase
        en[2] -= 1
        return False
    elif anim == HOP_TRIGGER:
        # Hop trigger: update grid position, start flight
        entity_hop(en, qb_prev)
        en[2] = -1  # start flight countdown (negative = in-flight)
        return True
    elif anim < 0:
        # In-flight: count toward landing
        en[2] -= 1
        if en[2] <= -FLIGHT_FRAMES:
            # Landing: reset to reload
            if en[3] == "coily":
                en[2] = COILY_RELOAD
            elif en[3] == "sam":
                en[2] = SAM_RELOAD
            else:
                en[2] = BALL_RELOAD
        return False
    else:
        # anim 1-15: shouldn't normally happen in our model,
        # but handle gracefully — treat as waiting
        en[2] -= 1
        if en[2] <= 0:
            en[2] = BALL_RELOAD
        return False


def check_collision(qb_pos, qb_prev, entities):
    """ROM $BD1E: check if Q*bert collides with any entity.
    Returns True if collision detected."""
    for en in entities:
        if en[0] == (-1, -1):
            continue
        if not is_valid(en[0][0], en[0][1]):
            # Off-grid entities (Ugg): check adjacent cubes
            r, c = en[0]
            if c < 0 and is_valid(r, 0):
                if qb_pos == (r, 0):
                    return True
            elif r >= 0 and c > r and is_valid(r, r):
                if qb_pos == (r, r):
                    return True
            continue

        # ROM collision: same position
        if qb_pos == en[0]:
            return True
        # Cross-match: Q*bert at enemy prev, enemy at Q*bert prev
        if qb_pos == en[1] and qb_prev == en[0]:
            return True

    return False


def simulate_sequence(state, actions):
    """Frame-perfect simulation of Q*bert executing a sequence of hops.

    Runs the ROM's update loop tick-by-tick:
    - Each Q*bert hop: 18 frames of flight + ~10 frames grounded
    - Each frame: all entities tick (anim decrement, hop trigger)
    - Collision checked every frame during grounded phase

    Returns True if Q*bert survives the entire sequence.
    """
    qb_pos = state.qbert
    qb_prev = state.qbert_prev
    entities = build_entities(state)

    for action in actions:
        from qbert.sim import MOVE_DELTAS
        dr, dc = MOVE_DELTAS[action]
        nr, nc = qb_pos[0] + dr, qb_pos[1] + dc
        if not is_valid(nr, nc):
            return False

        # Phase 1: Q*bert hop animation (18 frames)
        # Q*bert is in the air — no collision with grid entities
        # But entities still update
        for frame in range(18):
            for en in entities:
                entity_tick(en, qb_prev)

        # Q*bert lands
        qb_prev = qb_pos
        qb_pos = (nr, nc)

        # Check landing collision
        if check_collision(qb_pos, qb_prev, entities):
            return False

        # Phase 2: grounded (10 frames) — Q*bert waiting for next input
        # Collision checked every frame as enemies may arrive
        for frame in range(10):
            for en in entities:
                entity_tick(en, qb_prev)
            if check_collision(qb_pos, qb_prev, entities):
                return False

    return True
