# Q*bert Game State - Complete RAM Map

All information verified from ROM disassembly and empirical testing.

## Grid Word Encoding

Every position on the pyramid is encoded as a 2-byte "grid word":
- **byte0 = row + 1** (1-7 for rows 0-6, 8+ = off pyramid = death)
- **byte1 = row - col + 1** (1 = rightmost column, higher = more left)

Inverse: **row = byte0 - 1, col = byte0 - byte1**

Verified visually: overlay markers land exactly on sprites.

## Object Table Structure

### Q*bert: base $0D54

| Offset | Address | Description |
|--------|---------|-------------|
| +0,+1 | $0D54-55 | Internal position (animation) |
| +2,+3 | $0D56-57 | Internal position (animation) |
| +5 | $0D59 | Internal Y subpixel (changes smoothly during hop) |
| +7 | $0D5B | Internal X subpixel |
| +11 | $0D5F | Animation counter (counts down each frame) |
| +12 | $0D60 | Sprite character code |
| +13 | $0D61 | Collision Y value (used for distance check) |
| +18,+19 | $0D66-67 | **CURRENT grid word** ← THE position value |
| +20,+21 | $0D68-69 | **PREVIOUS grid word** (last cube before current hop) |

### Enemy Table: 10 slots starting at $0D70, stride 22 bytes

| Slot | Base | Grid Word | Prev GW | Notes |
|------|------|-----------|---------|-------|
| 0 | $0D70 | $0D82-83 | $0D84-85 | Often unused in Level 1 |
| 1 | $0D86 | $0D98-99 | $0D9A-9B | First enemy (purple ball → Coily) |
| 2 | $0D9C | $0DAE-AF | $0DB0-B1 | Second enemy (red ball) |
| 3 | $0DB2 | $0DC4-C5 | $0DC6-C7 | Third enemy |
| ... | +22 each | +18,+19 | +20,+21 | Up to slot 9 |

### Enemy Slot Byte Map (22 bytes per slot)

| Offset | Description | Notes |
|--------|-------------|-------|
| +0 | **State byte** | 0=inactive, see state values below |
| +1 | Animation phase | 0 or 255 alternating |
| +2,+3 | Subpixel position | Changes smoothly during hop animation |
| +5 | Internal Y | Subpixel Y coordinate |
| +7 | Internal X | Subpixel X coordinate |
| +8,+9 | Subpixel backup | Copy of +2,+3 |
| +10 | **Type/flags byte** | Identifies enemy type (see below) |
| +11 | **Animation counter** | Counts down each frame. 0 = start next hop |
| +12 | Sprite character | Which sprite to draw |
| +13 | **Collision Y** | Used for quick distance check before full collision |
| +14 | **Direction bits** | Encodes bounce direction for next hops |
| +15 | **Hop timer** | Per-hop countdown |
| +16,+17 | Pixel position | Screen pixel coordinates |
| +18,+19 | **CURRENT grid word** | Where enemy IS now |
| +20,+21 | **PREVIOUS grid word** | Where enemy WAS before current hop |

## State Byte Values

With frame_ratio=3, the state alternates between two values each frame:

| State | Alt State | Meaning |
|-------|-----------|---------|
| 0 | - | Inactive/empty slot |
| 112 | - | Dormant (pre-spawn) |
| 115 | 141 | **Red ball** bouncing |
| 96 | 160 | **Coily/purple ball** (frame_ratio=1) |
| 94 | 162 | **Coily/purple ball** (frame_ratio=3) |

Note: 115 + 141 = 256, 94 + 162 = 256. The alternating values are byte complements.

## Enemy Type Identification (offset+10, flags byte)

| Value | Bits 1-2 | Bit 5 | Type |
|-------|----------|-------|------|
| 0x60 | 0 | 1 | **Coily** (snake chasing Q*bert) |
| 0x22 | 1 | 1 | **Red ball** (bouncing down) |
| 0x58 | 0 | 0 | **Coily variant** (different collision path) |

The collision code at $BD1E checks:
- bits 1-2 (mask 0x06): if >= 4, different handler (Sam/Slick = harmless?)
- bit 5 (0x20): affects collision path

## Collision Detection (ROM $BD1E)

```
Loop 10 slots from $0D70, stride 22:
1. Quick distance check: compare collision Y values (offset+13)
   If distance >= 4: skip (too far)
2. Grid word comparison:
   Q*bert current ($0D66-67) vs Enemy current (offset+18-19): MATCH = DEATH
   Q*bert current ($0D66-67) vs Enemy previous (offset+20-21): check further
   Q*bert previous ($0D68-69) vs Enemy current: MATCH = DEATH
3. If distance < 1: always collision regardless of grid word
```

## Coily Chase Algorithm (ROM $B6EA)

Coily chases Q*bert's **PREVIOUS** position ($0D68), not current ($0D66).
Exception: if Coily IS at Q*bert's previous position, chase CURRENT instead.

Direction decision based on grid word comparison:
```
if qbert_gw0 > coily_gw0:     (Q*bert below → Coily goes DOWN)
    if qbert_gw1 > coily_gw1: → DOWN-LEFT  (row+1, col same)
    else:                      → DOWN-RIGHT (row+1, col+1)
else:                          (Q*bert above → Coily goes UP)
    if qbert_gw1 < coily_gw1: → UP-RIGHT   (row-1, col same)
    else:                      → UP-LEFT    (row-1, col-1)
```

In (row,col) terms with gw1 = row - col + 1:
- Compare rows: go toward Q*bert's row
- Compare (row-col): go toward Q*bert's column

## Red Ball Bounce Direction (ROM $B506)

Ball direction is set at spawn by a 7-bit random value stored at offset+14.
Each hop, one bit is consumed (SHR at $B607):
- Bit 0 = 1: bounce RIGHT (+1,+1)
- Bit 0 = 0: bounce LEFT  (+1,0)

After shifting, the value is stored back. 7 bits = 7 hops = exactly enough to traverse
the pyramid from top (row 0) to bottom (row 6). The ball's entire path is determined
at spawn time.

## Hop Timing

Animation counter (offset+11) counts down each frame:
- **0x20 (32)**: Standard wait between hops → ~37-43 frames per full hop cycle
- **0x10 (16)**: Faster movement
- **0x05 (5)**: Hop animation duration
- **0x50 (80)**: Spawn delay
- **0x18 (24)**: Coily movement speed
- **0x1C (28)**: Medium delay

Full hop cycle ≈ wait_frames + animation_frames ≈ 32 + 5 + overhead ≈ 40-43 frames

## Enemy Types (from spawn table byte 1)

| Type | Enemy | Behavior | Deadly? |
|------|-------|----------|---------|
| 0 | **Red ball** | Bounces down randomly | Yes |
| 1 | **Purple ball → Coily** | Bounces down, hatches into chasing snake | Yes |
| 2 | **Slick** | Like Sam (level 2+) | No (catch for bonus) |
| 3 | **Sam** | Bounces around, changes cubes back | No (catch for bonus) |
| 5 | **Ugg** | Crawls on cube sides | Yes |
| 7 | **Wrongway** | Like Ugg, opposite direction (level 2+) | Yes |

## Spawn System

Level configuration table at ROM $A003, 6-byte records per spawn event.

**Level 1 spawns** (7 events):
| Timer | Type | Enemy |
|-------|------|-------|
| 15 | 0 | Red ball |
| 25 | 0 | Red ball |
| 30 | 0 | Red ball |
| 50 | 0 | Red ball |
| 100 | 1 | Purple ball (→ Coily) |
| 300 | 3 | Sam |
| 500 | 5 | Ugg |

**Level 2+ spawns** (4 events each):
Types 0 (red ball), 2 (Slick), 5 (Ugg), 7 (Wrongway)

Enemies always spawn at top of pyramid. Maximum 2-3 enemies active simultaneously.

## Key Addresses for Agent

| Address | What | How to use |
|---------|------|------------|
| $0D00 | Lives | Check for death |
| $00BE | Score low byte | Track score changes |
| $0D66-67 | Q*bert grid word | row = byte0-1, col = byte0-byte1 |
| $0D68-69 | Q*bert prev grid word | Used by Coily chase |
| $0D86 (slot 1 +0) | Enemy 1 state | 0=inactive, 115/141=ball, 94/162=Coily |
| $0D94 (slot 1 +10) | Enemy 1 type flags | 0x60=Coily, 0x22=ball |
| $0D98-99 (slot 1 +18-19) | Enemy 1 grid word | Same encoding as Q*bert |
| $0D9C (slot 2 +0) | Enemy 2 state | Same as above |
| $0DB0 (slot 2 +10) | Enemy 2 type flags | |
| $0DAE-AF (slot 2 +18-19) | Enemy 2 grid word | |
