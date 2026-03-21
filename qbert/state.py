"""Q*bert game state reader. All RAM addresses ROM-verified (see GAME_STATE_MAP.md)."""

from collections import Counter
from dataclasses import dataclass, field

MAX_ROW = 6
NUM_CUBES = 28

# RAM addresses — simple dict, no Address class needed
QBERT_RAM = {
    "lives": 0x0D00,
    "score": 0x00BE,
    "qb_gw0": 0x0D66,
    "qb_gw1": 0x0D67,
    "qb_prev0": 0x0D68,
    "qb_prev1": 0x0D69,
}

# Enemy slots: 10 slots at $0D70 + n*22, various offsets
for _n in range(10):
    _base = 0x0D70 + _n * 22
    QBERT_RAM[f"e{_n}_st"] = _base          # +0: state byte
    QBERT_RAM[f"e{_n}_flags"] = _base + 10   # +10: type flags
    QBERT_RAM[f"e{_n}_anim"] = _base + 11    # +11: animation counter
    QBERT_RAM[f"e{_n}_dir"] = _base + 14     # +14: direction bits
    QBERT_RAM[f"e{_n}_gw0"] = _base + 18     # +18: current grid word byte 0
    QBERT_RAM[f"e{_n}_gw1"] = _base + 19     # +19: current grid word byte 1
    QBERT_RAM[f"e{_n}_pw0"] = _base + 20     # +20: previous grid word byte 0
    QBERT_RAM[f"e{_n}_pw1"] = _base + 21     # +21: previous grid word byte 1


@dataclass
class Enemy:
    slot: int
    pos: tuple  # (row, col)
    prev_pos: tuple
    direction_bits: int
    anim: int
    flags: int
    state: int
    going_up: bool  # True = Coily behavior, False = ball behavior
    harmless: bool   # True = Sam/Slick
    etype: str       # "coily", "ball", "sam", "ugg", "unknown"


@dataclass
class GameState:
    qbert: tuple          # (row, col) current position
    qbert_prev: tuple     # (row, col) previous position (Coily chases this)
    enemies: list = field(default_factory=list)
    lives: int = 0
    score_byte: int = 0


class EnemyTracker:
    """Tracks per-slot flags across frames to determine reliable enemy types.

    The flags byte (offset+10) can read incorrectly on any single frame due to
    mid-update reads. By accumulating values across frames and voting, we get
    the true value that was set at spawn time.

    Known flags from ROM:
        0x60 = Coily (deadly, chases Q*bert)
        0x22 = Red ball (deadly, bounces down)
        0x58 = Coily variant
    Collision code at $BD1E: flags & 0x06 >= 4 → harmless (Sam/Slick)
    """

    def __init__(self):
        self._was_active = {}   # slot -> bool
        self._flags_votes = {}  # slot -> Counter

    def update(self, slot, active, flags):
        """Call once per frame per slot with current state."""
        was = self._was_active.get(slot, False)
        if active and not was:
            # New spawn — reset vote counter
            self._flags_votes[slot] = Counter()
        if active:
            self._flags_votes[slot][flags] += 1
        self._was_active[slot] = active

    def get_type(self, slot):
        """Return (etype, harmless) based on accumulated flags votes."""
        votes = self._flags_votes.get(slot)
        if not votes:
            return "unknown", False

        flags = votes.most_common(1)[0][0]

        # ROM collision code: flags & 0x06 >= 4 → harmless handler
        if flags & 0x06 >= 4:
            return "sam", True

        # Known deadly types
        # 0x60 = Coily, 0x58 = Coily variant, 0x68 = Coily alternate phase
        if flags in (0x60, 0x58, 0x68):
            return "coily", False
        if flags == 0x22:
            return "ball", False

        # Unknown but deadly (bits 1-2 < 4)
        return "deadly", False

    def reset(self):
        """Call on death/level change to clear stale slot data."""
        self._was_active.clear()
        self._flags_votes.clear()


def gw_to_pos(gw0, gw1):
    """Convert grid word to (row, col). ROM-verified."""
    return (gw0 - 1, gw0 - gw1)


def pos_to_gw(row, col):
    """Convert (row, col) to grid word."""
    return (row + 1, row - col + 1)


def is_valid(r, c):
    return 0 <= r <= MAX_ROW and 0 <= c <= r


def read_state(data, tracker=None):
    """Parse RAM dict into a GameState. All addresses ROM-verified.

    If tracker is provided, uses multi-frame flag voting for type detection.
    Otherwise falls back to behavior-only detection.
    """
    qb_gw0 = data.get("qb_gw0", 0)
    qb_gw1 = data.get("qb_gw1", 0)
    qb_pos = gw_to_pos(qb_gw0, qb_gw1)

    qb_pw0 = data.get("qb_prev0", 0)
    qb_pw1 = data.get("qb_prev1", 0)
    qb_prev = gw_to_pos(qb_pw0, qb_pw1)

    enemies = []
    for n in range(10):
        st = data.get(f"e{n}_st", 0)
        flags = data.get(f"e{n}_flags", 0)
        active = st != 0

        # Update tracker every frame, even for inactive slots
        if tracker:
            tracker.update(n, active, flags)

        if not active:
            continue

        gw0 = data.get(f"e{n}_gw0", 0)
        gw1 = data.get(f"e{n}_gw1", 0)
        pw0 = data.get(f"e{n}_pw0", 0)
        pw1 = data.get(f"e{n}_pw1", 0)
        pos = gw_to_pos(gw0, gw1)
        prev = gw_to_pos(pw0, pw1)

        going_up = pos[0] < prev[0] if is_valid(prev[0], prev[1]) else False

        if tracker:
            etype, harmless = tracker.get_type(n)
            # Flags tell us the entity TYPE, behavior tells us the PHASE.
            # Purple ball (flags=0x60) bounces down before hatching into Coily.
            # Only treat as active Coily when actually going up (chasing).
            if etype == "sam":
                going_up = False  # Sam goes up too, but is harmless
        else:
            harmless = False
            etype = "coily" if going_up else "ball"

        enemies.append(Enemy(
            slot=n, pos=pos, prev_pos=prev,
            direction_bits=data.get(f"e{n}_dir", 0),
            anim=data.get(f"e{n}_anim", 0),
            flags=flags, state=st,
            going_up=going_up, harmless=harmless,
            etype=etype,
        ))

    return GameState(
        qbert=qb_pos, qbert_prev=qb_prev,
        enemies=enemies,
        lives=data.get("lives", 0),
        score_byte=data.get("score", 0),
    )
