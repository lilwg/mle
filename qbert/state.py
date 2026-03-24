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
    # Disc table at $0ECC: 14 entries (7 right + 7 left), 4 bytes each.
    # Non-zero word = disc present. Read as 16-bit: low byte at addr, high at addr+1.
    # Right discs: $0ECC + gw0*4, Left discs: $0ECC + gw0*4 + 0x1C
    **{f"disc_r{r}": 0x0ECC + r * 4 for r in range(7)},       # right side
    **{f"disc_r{r}_hi": 0x0ECC + r * 4 + 1 for r in range(7)},
    **{f"disc_l{r}": 0x0ECC + r * 4 + 0x1C for r in range(7)}, # left side
    **{f"disc_l{r}_hi": 0x0ECC + r * 4 + 0x1C + 1 for r in range(7)},
    # Spawn timer: $0085 is 8-bit countdown, spawn happens at 0, reloads from $0D17
    "spawn_countdown": 0x0085,
    # Q*bert animation counter: 0 = mid-hop, >= 16 = ready for next hop
    "qb_anim": 0x0D5F,
    # Q*bert collision Y (used in distance check before grid word comparison)
    "qb_coll_y": 0x0D61,
    # Cube tracking: 28 cube color states, target color, remaining count
    "target_color": 0x0D1E,
    "remaining_cubes": 0x0D23,
}

# Add 28 cube state addresses ($0D28-$0D43)
for _c in range(NUM_CUBES):
    QBERT_RAM[f"cube{_c}"] = 0x0D28 + _c

for _n in range(10):
    _base = 0x0D70 + _n * 22
    QBERT_RAM[f"e{_n}_st"] = _base
    QBERT_RAM[f"e{_n}_flags"] = _base + 10
    QBERT_RAM[f"e{_n}_anim"] = _base + 11
    QBERT_RAM[f"e{_n}_coll_y"] = _base + 13
    QBERT_RAM[f"e{_n}_dir"] = _base + 14
    QBERT_RAM[f"e{_n}_gw0"] = _base + 18
    QBERT_RAM[f"e{_n}_gw1"] = _base + 19
    QBERT_RAM[f"e{_n}_pw0"] = _base + 20
    QBERT_RAM[f"e{_n}_pw1"] = _base + 21


@dataclass
class Disc:
    row: int       # disc is between this row and row+1 on the edge
    side: str      # "left" or "right"
    jump_from: tuple  # (row, col) Q*bert must be at
    direction: int    # action index to jump to disc


@dataclass
class Enemy:
    slot: int
    pos: tuple
    prev_pos: tuple
    direction_bits: int
    anim: int
    flags: int
    state: int
    coll_y: int
    going_up: bool
    harmless: bool
    etype: str


@dataclass
class GameState:
    qbert: tuple
    qbert_prev: tuple
    enemies: list = field(default_factory=list)
    discs: list = field(default_factory=list)
    lives: int = 0
    score_byte: int = 0
    spawn_countdown: int = 0  # frames until next enemy spawns at (1,0)
    remaining_cubes: int = NUM_CUBES  # cubes left to color-change (from $0D23)
    target_color: int = 0             # target cube color for this level ($0D1E)
    cube_states: list = field(default_factory=list)  # 28 cube color values from RAM


class EnemyTracker:
    """Tracks whether each slot has ever moved upward (= Coily).

    Once a slot shows upward movement, it's marked as Coily until reset().
    No active/inactive tracking — the state byte flickers and is unreliable
    for determining slot activity.
    """

    def __init__(self):
        self._is_coily = {}

    def update(self, slot, pos, prev_pos, flags=0):
        if is_valid(pos[0], pos[1]) and is_valid(prev_pos[0], prev_pos[1]):
            # Single-hop upward movement = Coily chasing
            dr = abs(pos[0] - prev_pos[0])
            dc = abs(pos[1] - prev_pos[1])
            if pos[0] < prev_pos[0] and dr == 1 and dc <= 1:
                self._is_coily[slot] = True
        # fl=0x68 is the definitive hatched Coily flag.
        # fl alternates between 0x60/0x68 after hatch — if we ever see 0x68
        # for this slot, it's permanently Coily.
        # DON'T mark fl=0x60 at row>=6 — that's a pre-hatch purple ball
        # that hasn't confirmed as Coily yet. Wait for 0x68 or upward movement.
        # 0x68 definitive. 0x62/0x6a also Coily if going up.
        # 0x60 going up from row 6+ = just hatched, also Coily.
        if flags == 0x68:
            self._is_coily[slot] = True
        elif flags in (0x60, 0x62, 0x6a) and is_valid(pos[0], pos[1]) and is_valid(prev_pos[0], prev_pos[1]):
            if pos[0] < prev_pos[0]:  # going up = chasing
                self._is_coily[slot] = True

    def is_coily(self, slot):
        return self._is_coily.get(slot, False)

    def reset(self):
        self._is_coily.clear()


def gw_to_pos(gw0, gw1):
    return (gw0 - 1, gw0 - gw1)


def pos_to_gw(row, col):
    return (row + 1, row - col + 1)


def is_valid(r, c):
    return 0 <= r <= MAX_ROW and 0 <= c <= r


# Cube index to grid position mapping.
# RAM stores 28 cubes at $0D28-$0D43. Layout is right-to-left per row:
#   cube0=(0,0)
#   cube1=(1,1), cube2=(1,0)
#   cube3=(2,2), cube4=(2,1), cube5=(2,0)
#   etc.
_CUBE_INDEX_TO_POS = []
for _row in range(MAX_ROW + 1):
    for _col in range(_row, -1, -1):  # right-to-left
        _CUBE_INDEX_TO_POS.append((_row, _col))

# Reverse mapping: (row, col) -> cube index
_POS_TO_CUBE_INDEX = {pos: idx for idx, pos in enumerate(_CUBE_INDEX_TO_POS)}


def cube_index_to_pos(idx):
    """Convert cube index (0-27) to grid position (row, col)."""
    if 0 <= idx < len(_CUBE_INDEX_TO_POS):
        return _CUBE_INDEX_TO_POS[idx]
    return None


def pos_to_cube_index(row, col):
    """Convert grid position (row, col) to cube index (0-27), or None."""
    return _POS_TO_CUBE_INDEX.get((row, col))


LEFT = 2   # Up-Left action
UP = 0     # Up-Right action


def _parse_discs(data):
    """Parse disc availability from the disc table at RAM $0ECC.

    ROM-verified: SI=$0ECC is hardcoded. 14 entries (7 right + 7 left),
    4 bytes each. Non-zero 16-bit word = disc present.
    Right: $0ECC + row*4, Left: $0ECC + row*4 + $1C.
    """
    discs = []
    for gw0 in range(7):
        # Right side disc: Q*bert at grid (gw0, gw0) jumps UP-RIGHT
        lo = data.get(f"disc_r{gw0}", 0)
        hi = data.get(f"disc_r{gw0}_hi", 0)
        if lo | hi:
            discs.append(Disc(
                row=gw0, side="right",
                jump_from=(gw0, gw0),
                direction=UP,
            ))
        # Left side disc: Q*bert at grid (gw0, 0) jumps UP-LEFT
        lo = data.get(f"disc_l{gw0}", 0)
        hi = data.get(f"disc_l{gw0}_hi", 0)
        if lo | hi:
            discs.append(Disc(
                row=gw0, side="left",
                jump_from=(gw0, 0),
                direction=LEFT,
            ))
    return discs


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
        flags = data.get(f"e{n}_flags", 0)

        # Primary check: st != 0 means active.
        # Secondary: st == 0 can mean flickering (active enemy caught mid-update)
        # or truly inactive (stale data). Distinguish by: if flags != 0 AND
        # anim counter > 0 AND position is valid, it's flickering (active).
        gw0 = data.get(f"e{n}_gw0", 0)
        gw1 = data.get(f"e{n}_gw1", 0)
        pw0 = data.get(f"e{n}_pw0", 0)
        pw1 = data.get(f"e{n}_pw1", 0)
        anim = data.get(f"e{n}_anim", 0)
        pos_check = gw_to_pos(gw0, gw1)

        if st == 0:
            on_face_check = (not is_valid(pos_check[0], pos_check[1])
                             and pos_check[0] >= 1
                             and (pos_check[1] < 0 or pos_check[1] > pos_check[0]))
            if on_face_check and flags != 0:
                # Ugg/Wrongway: on pyramid face, st==0 but flags non-zero
                # Keep regardless of anim (anim=0 = mid-hop, still active)
                pass
            elif flags != 0 and anim > 0 and is_valid(pos_check[0], pos_check[1]):
                # Flickering active enemy — include it
                pass  # fall through to normal processing
            else:
                # Truly inactive
                if tracker:
                    tracker.update(n, pos_check, gw_to_pos(pw0, pw1))
                continue
        pos = gw_to_pos(gw0, gw1)
        prev = gw_to_pos(pw0, pw1)

        going_up = pos[0] < prev[0] if is_valid(prev[0], prev[1]) else False

        if tracker:
            tracker.update(n, pos, prev, flags)
            is_coily = tracker.is_coily(n)
        else:
            is_coily = going_up

        # Classify enemy type using flags byte and tracker.
        # ROM collision $BD41: flag_type >= 4 → Sam/Slick handler (harmless).
        # This applies regardless of position — Sam on a pyramid face is still harmless.
        # Actual Ugg flags: 0x25 (right), 0x27 (left) — these have flag_type=4/6
        # but are deadly. However they also have fl & 0xE0 = 0x20, same as balls.
        # The ROM at $BD72 checks [di+0xF] (hops_remaining) for Sam collision.
        flag_type = flags & 0x06
        off_grid = not is_valid(pos[0], pos[1])
        on_face = (off_grid and 1 <= pos[0] <= 7
                   and (pos[1] < 0 or pos[1] > pos[0]))
        if flag_type >= 4:
            if flags in (0x25, 0x27):
                # Ugg (0x25) / Wrongway (0x27): deadly, on pyramid faces
                etype = "ugg"
                harmless = False
            else:
                # Sam/Slick: harmless (fl=0x44, 0x46, 0x4A, etc.)
                etype = "sam"
                harmless = True
        elif flags == 0x68 or is_coily or (going_up and flag_type == 0):
            etype = "coily"
            harmless = False
        else:
            etype = "ball"
            harmless = False

        coll_y = data.get(f"e{n}_coll_y", 0)
        # ROM $B576/$B6C6: entity is INACTIVE when [bp+0xD] (coll_y) == 0.
        # This is the ROM's own check — same one the renderer uses.
        if coll_y == 0:
            continue

        enemies.append(Enemy(
            slot=n, pos=pos, prev_pos=prev,
            direction_bits=data.get(f"e{n}_dir", 0),
            anim=anim,
            flags=flags, state=st,
            coll_y=coll_y,
            going_up=going_up, harmless=harmless,
            etype=etype,
        ))

    discs = _parse_discs(data)

    # Read cube states from RAM
    cube_states = [data.get(f"cube{i}", 0) for i in range(NUM_CUBES)]

    return GameState(
        qbert=qb_pos, qbert_prev=qb_prev,
        enemies=enemies, discs=discs,
        lives=data.get("lives", 0),
        score_byte=data.get("score", 0),
        spawn_countdown=data.get("spawn_countdown", 0),
        remaining_cubes=data.get("remaining_cubes", NUM_CUBES),
        target_color=data.get("target_color", 0),
        cube_states=cube_states,
    )
