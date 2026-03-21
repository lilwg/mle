"""Q*bert game state reader. All RAM addresses ROM-verified (see GAME_STATE_MAP.md)."""

from dataclasses import dataclass, field

MAX_ROW = 6
NUM_CUBES = 28

QBERT_RAM = {
    "lives": 0x0D00,
    "score": 0x00BE,
    "qb_gw0": 0x0D66,
    "qb_gw1": 0x0D67,
    "qb_prev0": 0x0D68,
    "qb_prev1": 0x0D69,
}

for _n in range(10):
    _base = 0x0D70 + _n * 22
    QBERT_RAM[f"e{_n}_st"] = _base
    QBERT_RAM[f"e{_n}_flags"] = _base + 10
    QBERT_RAM[f"e{_n}_anim"] = _base + 11
    QBERT_RAM[f"e{_n}_dir"] = _base + 14
    QBERT_RAM[f"e{_n}_gw0"] = _base + 18
    QBERT_RAM[f"e{_n}_gw1"] = _base + 19
    QBERT_RAM[f"e{_n}_pw0"] = _base + 20
    QBERT_RAM[f"e{_n}_pw1"] = _base + 21


@dataclass
class Enemy:
    slot: int
    pos: tuple
    prev_pos: tuple
    direction_bits: int
    anim: int
    flags: int
    state: int
    going_up: bool
    harmless: bool
    etype: str  # "coily", "ball", "unknown"


@dataclass
class GameState:
    qbert: tuple
    qbert_prev: tuple
    enemies: list = field(default_factory=list)
    lives: int = 0
    score_byte: int = 0


class EnemyTracker:
    """Tracks whether each slot has ever gone up (= Coily).

    Purple ball bounces down for 7 hops then hatches into Coily which goes up.
    Once we see a slot go up, it's Coily until the slot goes inactive.
    No flags byte needed — pure behavior.
    """

    def __init__(self):
        self._was_active = {}
        self._is_coily = {}

    def update(self, slot, active, pos, prev_pos):
        if active and not self._was_active.get(slot, False):
            self._is_coily[slot] = False
        if active and is_valid(pos[0], pos[1]) and is_valid(prev_pos[0], prev_pos[1]):
            if pos[0] < prev_pos[0]:
                self._is_coily[slot] = True
        self._was_active[slot] = active

    def is_coily(self, slot):
        return self._is_coily.get(slot, False)

    def reset(self):
        self._was_active.clear()
        self._is_coily.clear()


def gw_to_pos(gw0, gw1):
    return (gw0 - 1, gw0 - gw1)


def pos_to_gw(row, col):
    return (row + 1, row - col + 1)


def is_valid(r, c):
    return 0 <= r <= MAX_ROW and 0 <= c <= r


def read_state(data, tracker=None):
    qb_gw0 = data.get("qb_gw0", 0)
    qb_gw1 = data.get("qb_gw1", 0)
    qb_pos = gw_to_pos(qb_gw0, qb_gw1)

    qb_pw0 = data.get("qb_prev0", 0)
    qb_pw1 = data.get("qb_prev1", 0)
    qb_prev = gw_to_pos(qb_pw0, qb_pw1)

    enemies = []
    for n in range(10):
        st = data.get(f"e{n}_st", 0)
        if st == 0:
            gw0 = data.get(f"e{n}_gw0", 0)
            gw1 = data.get(f"e{n}_gw1", 0)
            pw0 = data.get(f"e{n}_pw0", 0)
            pw1 = data.get(f"e{n}_pw1", 0)
            if tracker:
                tracker.update(n, False, gw_to_pos(gw0, gw1), gw_to_pos(pw0, pw1))
            continue

        flags = data.get(f"e{n}_flags", 0)
        gw0 = data.get(f"e{n}_gw0", 0)
        gw1 = data.get(f"e{n}_gw1", 0)
        pw0 = data.get(f"e{n}_pw0", 0)
        pw1 = data.get(f"e{n}_pw1", 0)
        pos = gw_to_pos(gw0, gw1)
        prev = gw_to_pos(pw0, pw1)

        going_up = pos[0] < prev[0] if is_valid(prev[0], prev[1]) else False

        if tracker:
            tracker.update(n, True, pos, prev)
            is_coily = tracker.is_coily(n)
        else:
            is_coily = going_up

        if is_coily:
            etype = "coily"
            going_up = True  # Coily is always treated as chasing
        else:
            etype = "ball"

        harmless = False

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
