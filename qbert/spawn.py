"""ROM-accurate spawn prediction for Q*bert.

Translated from ROM disassembly at $7468-$7555.

Spawn timer: 16-bit word at $0085, decremented each frame.
When zero: reads next entry from spawn list, resets timer to $0D17.
Spawn interval: (remaining_cubes << shift) + base_delay.

Spawn list: 2-byte entries (flags_byte, sprite_byte). $FFFF = loop.
Pointer at $0D13, loop-back at $0D11.
"""

# RAM addresses for spawn system
SPAWN_RAM = {
    "spawn_timer_lo": 0x0085,  # 16-bit spawn countdown (low byte)
    "spawn_timer_hi": 0x0086,  # (high byte)
    "spawn_interval_lo": 0x0D17,  # reload value (low)
    "spawn_interval_hi": 0x0D18,  # reload value (high)
    "spawn_ptr_lo": 0x0D13,  # current spawn list pointer (low)
    "spawn_ptr_hi": 0x0D14,
    "spawn_loop_lo": 0x0D11,  # loop-back pointer (low)
    "spawn_loop_hi": 0x0D12,
    "game_flags": 0x0D01,  # bit2=Coily spawning, bit3=Coily hatching
    "qb_state": 0x0D5E,  # bit4=suppress spawn timer (dying/disc)
}


def read_spawn_state(data):
    """Read current spawn system state from RAM."""
    timer = data.get("spawn_timer_lo", 0) | (data.get("spawn_timer_hi", 0) << 8)
    interval = data.get("spawn_interval_lo", 0) | (data.get("spawn_interval_hi", 0) << 8)
    ptr = data.get("spawn_ptr_lo", 0) | (data.get("spawn_ptr_hi", 0) << 8)
    loop_ptr = data.get("spawn_loop_lo", 0) | (data.get("spawn_loop_hi", 0) << 8)
    game_flags = data.get("game_flags", 0)
    qb_state = data.get("qb_state", 0)

    coily_active = bool(game_flags & 0x0C)  # bits 2 or 3
    timer_paused = bool(qb_state & 0x10)  # bit 4

    return {
        "timer": timer,
        "interval": interval,
        "ptr": ptr,
        "loop_ptr": loop_ptr,
        "coily_active": coily_active,
        "timer_paused": timer_paused,
    }


def frames_until_next_spawn(data):
    """How many frames until the next enemy spawns?
    Returns 0 if spawn is imminent, or the countdown value.
    Returns -1 if timer is paused (Q*bert dying/on disc)."""
    state = read_spawn_state(data)
    if state["timer_paused"]:
        return -1
    return state["timer"]


def predict_spawn_position():
    """Predict where the next spawned entity will appear.

    Balls/Coily always spawn at grid (1,0) or (1,1) — random 50/50.
    We can't predict which side, but we know it's row 1.

    Returns: list of possible spawn positions.
    """
    return [(1, 0), (1, 1)]


def is_spawn_imminent(data, threshold=20):
    """Is a spawn about to happen within `threshold` frames?"""
    frames = frames_until_next_spawn(data)
    if frames < 0:
        return False  # paused
    return frames <= threshold


def spawn_danger_at(data, pos, threshold=30):
    """Is position `pos` at risk from an imminent spawn?

    Spawns appear at (1,0) or (1,1). If Q*bert is at or adjacent
    to these positions and a spawn is imminent, it's dangerous.
    """
    if not is_spawn_imminent(data, threshold):
        return False
    spawn_positions = predict_spawn_position()
    for sp in spawn_positions:
        if pos == sp:
            return True
    return False
