"""ROM-accurate collision detection for Q*bert.

Translated directly from ROM disassembly at $BD1E-$BDA0.
Entity structure offsets (22 bytes per slot):
  +0x00: screen Y velocity (word)
  +0x02: screen X velocity (word)
  +0x04: screen Y position (word)
  +0x06: screen X position (word)
  +0x08: unknown (word)
  +0x0A: flags byte
  +0x0B: anim counter
  +0x0C: sprite tile
  +0x0D: coll_y (active check)
  +0x0E: direction_bits
  +0x0F: hops_remaining
  +0x10: screen position accumulator (word)
  +0x12: gw0 (grid word 0)
  +0x13: gw1 (grid word 1)
  +0x14: prev_gw0
  +0x15: prev_gw1

Q*bert base: $0D54 (same structure)
Entity slots: $0D70 + n*22 for n=0..9
"""


def rom_distance(qb_screen_y, qb_screen_x, en_screen_y, en_screen_x):
    """ROM $BCF3: approximate distance between two entities.
    Uses 16-bit screen coordinates. Returns (ch, distance).
    ch is the high byte of the approximate distance."""
    # cx = |en_y - qb_y|
    cx = abs(en_screen_y - qb_screen_y) & 0xFFFF
    # dx = |en_x - qb_x|
    dx = abs(en_screen_x - qb_screen_x) & 0xFFFF
    # ensure cx.h >= dx.h (cx has the larger component)
    if (cx >> 8) <= (dx >> 8):
        cx, dx = dx, cx
    # distance ≈ max + min/2
    dx = (dx >> 1) & 0xFFFF
    dist = (cx + dx) & 0xFFFF
    if dist < cx:  # overflow
        ch = 0xFF
    else:
        ch = (dist >> 8) & 0xFF
    return ch


def rom_collision_check(data):
    """ROM $BD1E: check all 10 entity slots for collision with Q*bert.

    Returns (collided, slot, reason) or (False, -1, None).
    Uses the EXACT same logic as the ROM.

    data: dict of RAM values from env.step()
    """
    # Q*bert data (bp = $0D54)
    qb_coll_y = data.get("qb_coll_y", 0)  # [bp+0xD]
    qb_screen_y = data.get("qb_sy_lo", 0) | (data.get("qb_sy_hi", 0) << 8)  # [bp+4]
    qb_screen_x = data.get("qb_sx_lo", 0) | (data.get("qb_sx_hi", 0) << 8)  # [bp+6]
    qb_gw = (data.get("qb_gw0", 0), data.get("qb_gw1", 0))  # [0xD66]
    qb_prev_gw = (data.get("qb_prev0", 0), data.get("qb_prev1", 0))  # [0xD68]

    for n in range(10):
        base = 0x0D70 + n * 22

        # $BD2D: cmp [di+0xD], 0 — skip inactive
        en_coll_y = data.get(f"e{n}_coll_y", 0)
        if en_coll_y == 0:
            continue

        # $BD33: call distance function
        # Need entity screen positions (offsets +4, +6)
        en_sy = data.get(f"e{n}_sy_lo", 0) | (data.get(f"e{n}_sy_hi", 0) << 8)
        en_sx = data.get(f"e{n}_sx_lo", 0) | (data.get(f"e{n}_sx_hi", 0) << 8)
        ch = rom_distance(qb_screen_y, qb_screen_x, en_sy, en_sx)

        # $BD36: cmp ch, 4 — too far
        if ch >= 4:
            continue

        # $BD3B: flags & 6
        flags = data.get(f"e{n}_flags", 0)
        flag_type = flags & 6

        # $BD41: flag_type >= 4 → Sam/Slick handler
        if flag_type >= 4:
            # $BD72: Sam handler
            hops_remaining = data.get(f"e{n}_hops", 0)  # [di+0xF]
            if hops_remaining != 0:
                return (True, n, f"sam_catch fl={flags:#x}")
            # $BD78: check if entity past pyramid
            # threshold_y = data.get("threshold_y", 0)  # [0xE9C]
            # if entity_y > threshold: skip
            continue

        # $BD46: test flags & 0x20
        if not (flags & 0x20):
            # Bit 5 clear → immediate death (non-special entity)
            return (True, n, f"non_bit5 fl={flags:#x}")

        # $BD4C-$BD61: grid word comparison (bit 5 set)
        en_gw0 = data.get(f"e{n}_gw0", 0)
        en_gw1 = data.get(f"e{n}_gw1", 0)
        en_gw = (en_gw0, en_gw1)
        en_pw0 = data.get(f"e{n}_pw0", 0)
        en_pw1 = data.get(f"e{n}_pw1", 0)
        en_prev_gw = (en_pw0, en_pw1)

        # Q*bert current gw == enemy current gw
        if qb_gw == en_gw:
            return (True, n, f"gw_match fl={flags:#x}")

        # Q*bert current gw == enemy prev gw
        if qb_gw == en_prev_gw:
            # AND Q*bert prev gw == enemy current gw (cross-match)
            if qb_prev_gw == en_gw:
                return (True, n, f"cross_match fl={flags:#x}")

        # $BD63: very close (ch < 1)
        if ch < 1:
            return (True, n, f"proximity fl={flags:#x} ch={ch}")

    return (False, -1, None)


def classify_entity_rom(flags, flag_type, di_value):
    """Classify entity type based on ROM logic.

    The ROM uses two classification systems:
    1. Collision at $BD3B: flag_type (flags & 6)
       - >= 4: Sam/Slick (special handler)
       - < 4, bit5 clear: immediate kill on proximity
       - < 4, bit5 set: grid word comparison needed

    2. Update at $B591: upper flags (flags & 0xE0) + di (flags & 6)
       - di=0 (flag_type=0): top-entry, moves down-left
       - di=2 (flag_type=2): top-entry, moves down-right
       - di=4 (flag_type=4): Ugg (right side face)
       - di=6 (flag_type=6): Wrongway (left side face)
       - flags & 0xE0 == 0x60: Coily/purple ball
       - flags & 0xE0 == 0x40: Sam/Slick
       - flags & 0xE0 == 0x20: red ball / top-entry
       - flags & 0xE0 == 0x00: also ball variant

    Returns (etype, harmless, bit5)
    """
    upper = flags & 0xE0
    bit5 = bool(flags & 0x20)

    if flag_type >= 4:
        # Sam/Slick (di=4/6) OR Ugg/Wrongway (fl=0x25/0x27)
        if flags in (0x25, 0x27):
            return ("ugg", False, bit5)
        else:
            return ("sam", True, bit5)

    # flag_type < 4
    if upper == 0x60:
        # Coily family: 0x60 (pre-hatch), 0x62, 0x68 (hatched), 0x6A
        return ("coily", False, bit5)
    elif upper == 0x40:
        # Sam/Slick variant
        return ("sam", True, bit5)
    elif upper == 0x20:
        # Red ball / top-entry ball: 0x20, 0x22
        return ("ball", False, bit5)
    elif upper == 0x00:
        # Ball variant
        return ("ball", False, bit5)
    else:
        return ("ball", False, bit5)
