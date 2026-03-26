"""MLE (MAME Learning Environment) — thin Python wrapper around MAME.

Usage:
    env = MameEnv("/path/to/roms", "qbert", {"lives": 0x0D00, "score": 0x00BE})
    data = env.step()                    # advance 1 frame, returns {"lives": 3, ...}
    data = env.step(":IN1", "Coin 1")    # press button for 1 frame
    data = env.step_n(":IN4", "P1 Up (Up-Right)", 18)  # hold for 18 frames
    env.request_frame()                  # next step() will include pixel data
    env.close()

Protocol per frame (synchronous handshake):
    Lua: release old buttons -> write RAM line (+ optional pixels) -> flush -> read action line
    Python: read RAM line (+ optional pixels) -> write action line
"""

import atexit
import os
import select
import threading

import numpy as np

from mle.console import MameConsole


class MameEnv:
    def __init__(self, roms_path, game_id, ram_addresses, render=True, sound=False,
                 throttle=True):
        self.ram_addresses = ram_addresses
        self._want_frame = False
        self._frame_sent = False
        self._frame_dims = None
        self._closed = False

        # Start MAME
        self.console = MameConsole(roms_path, game_id, render=render, sound=sound,
                                   throttle=throttle)
        self._wait_for_resources()

        # Get screen dimensions
        result = self.console.writeln_expect(
            'local w,h = manager.machine.video:snapshot_size(); print(w..","..h)'
        )
        w, h = result.split(",")
        self._frame_dims = (int(h), int(w))
        self._frame_size = int(w) * int(h) * 4

        # Create Lua variables
        self.console.writeln('iop = manager.machine.ioport')
        self.console.writeln('s = manager.machine.screens[":screen"]')
        self.console.writeln('mem = manager.machine.devices[":maincpu"].spaces["program"]')
        self.console.writeln('video = manager.machine.video')
        self.console.writeln('releaseQueue = {}')
        self.console.writeln('want_frame = false')

        # Open pipes one at a time (same pattern as old MAMEToolkit)
        # Action pipe: Lua reads, Python writes
        # Open Lua side first (blocks until Python side opens), Python side in thread
        self.console.writeln(
            f'actionPipe = assert(io.open("{self.console.action_path}", "r"))'
        )
        self._action_file = None
        t = threading.Thread(target=self._open_action)
        t.start()
        t.join(timeout=10)
        if self._action_file is None:
            raise IOError("Failed to open action pipe")

        # Data pipe: Lua writes, Python reads
        self.console.writeln(
            f'dataPipe = assert(io.open("{self.console.data_path}", "w"))'
        )
        self._data_file = None
        t = threading.Thread(target=self._open_data)
        t.start()
        t.join(timeout=10)
        if self._data_file is None:
            raise IOError("Failed to open data pipe")

        # Build and register the frame callback
        self._setup_frame_callback(ram_addresses)

        # Bootstrap: read first data from Lua, then send empty action
        self._bootstrap_data = self._read_data_line()
        self._write_action_line(None, False)

        atexit.register(self.close)

    def _open_action(self):
        self._action_file = open(self.console.action_path, "w")

    def _open_data(self):
        self._data_file = open(self.console.data_path, "rb")

    def _wait_for_resources(self, attempts=10):
        for i in range(attempts):
            r = self.console.writeln_expect(
                'print(manager.machine.screens[":screen"])'
            )
            if r and r != "nil":
                r2 = self.console.writeln_expect(
                    'print(manager.machine.devices[":maincpu"].spaces["program"])'
                )
                if r2 and r2 != "nil":
                    return
        raise EnvironmentError("MAME resources not available after startup")

    def _setup_frame_callback(self, ram_addresses):
        """Build and register the Lua frame callback.

        Uses address tables + loop to avoid Lua's expression depth limit
        when reading many RAM addresses.
        """
        # Build Lua table literals for addresses and names
        addr_items = list(ram_addresses.items())
        addrs_lua = ",".join(f"{addr:#x}" for _, addr in addr_items)
        names_lua = ",".join(f'"{name}"' for name, _ in addr_items)
        n = len(addr_items)

        # Set up tables as globals (avoids creating them every frame)
        self.console.writeln(f"mle_addrs = {{{addrs_lua}}}")
        self.console.writeln(f"mle_names = {{{names_lua}}}")
        self.console.writeln(f"mle_n = {n}")

        lua = (
            'function mle_frame() '
            'for i=1,#releaseQueue do '
            'releaseQueue[i]:set_value(0); '
            'releaseQueue[i]=nil; '
            'end; '
            # Build data line using table + loop
            'local t = {}; '
            'for i=1,mle_n do '
            't[i] = mle_names[i]..":"..mem:read_u8(mle_addrs[i]); '
            'end; '
            'dataPipe:write(table.concat(t,",").."\\n"); '
            'if want_frame then '
            'dataPipe:write(video:snapshot_pixels()); '
            'want_frame = false; '
            'end; '
            'dataPipe:flush(); '
            'local line = actionPipe:read("*l"); '
            'if line == nil then return end; '
            'for token in string.gmatch(line, "[^+]+") do '
            'if token == "F" then '
            'want_frame = true; '
            'else '
            'local port, field = string.match(token, "([^\\t]+)\\t([^\\t]+)"); '
            'if port and field then '
            'local p = iop.ports[port]; '
            'if p then '
            'local f = p.fields[field]; '
            'if f then '
            'f:set_value(1); '
            'releaseQueue[#releaseQueue+1] = f; '
            'end; end; end; end; end; '
            'end'
        )
        self.console.writeln(lua)
        self.console.writeln('emu.register_frame(mle_frame)')

    def _write_action_line(self, buttons, want_frame):
        """Write a single action line. Format: "F+port\\tfield+port\\tfield\\n" """
        tokens = []
        if want_frame:
            tokens.append("F")
        if buttons:
            for port, field in buttons:
                tokens.append(f"{port}\t{field}")
        self._action_file.write("+".join(tokens) + "\n")
        self._action_file.flush()

    def _read_data_line(self):
        """Read one line of RAM data from the data pipe."""
        readable, _, _ = select.select([self._data_file], [], [], 10)
        if not readable:
            raise TimeoutError("Timeout reading from MAME data pipe")
        line = self._data_file.readline()
        if not line:
            raise IOError("MAME data pipe closed")
        line = line.rstrip(b"\n")

        data = {}
        if line:
            for pair in line.decode("utf-8").split(","):
                if ":" in pair:
                    name, val = pair.split(":", 1)
                    data[name] = int(val)
        return data

    def _read_frame_data(self):
        """Read raw pixel data from the data pipe."""
        frame_bytes = b""
        while len(frame_bytes) < self._frame_size:
            readable, _, _ = select.select([self._data_file], [], [], 10)
            if not readable:
                raise TimeoutError("Timeout reading frame data")
            chunk = self._data_file.read(self._frame_size - len(frame_bytes))
            if not chunk:
                raise IOError("Data pipe closed reading frame")
            frame_bytes += chunk

        raw = np.frombuffer(frame_bytes, dtype="<u4").reshape(self._frame_dims)
        frame = raw.view(np.uint8).reshape(
            self._frame_dims[0], self._frame_dims[1], 4
        )
        return frame[:, :, 2::-1].copy()  # BGRA -> RGB

    def _do_step(self, buttons, want_frame_next):
        """Execute one frame: read data from Lua, then write action."""
        data = self._read_data_line()

        if self._frame_sent:
            data["frame"] = self._read_frame_data()
            self._frame_sent = False

        self._write_action_line(buttons, want_frame_next)
        if want_frame_next:
            self._frame_sent = True

        return data

    def step(self, port=None, field=None):
        """Advance 1 MAME frame. Optionally press a button.

        If request_frame() was called, advances 2 frames internally:
        one to send the flag, one to read the pixels back.
        """
        buttons = [(port, field)] if port and field else None
        if self._want_frame:
            self._want_frame = False
            # Frame 1: send F flag along with buttons
            self._do_step(buttons, want_frame_next=True)
            # Frame 2: read back RAM + pixels
            return self._do_step(None, want_frame_next=False)
        return self._do_step(buttons, want_frame_next=False)

    def step_n(self, port, field, n):
        """Hold a button for n frames. Returns final frame's data."""
        buttons = [(port, field)]
        data = None
        for i in range(n):
            if i == n - 1 and self._want_frame:
                self._want_frame = False
                self._do_step(buttons, want_frame_next=True)
                data = self._do_step(None, want_frame_next=False)
            else:
                data = self._do_step(buttons, want_frame_next=False)
        return data

    def wait(self, n):
        """Advance n frames with no input. Returns final frame's data."""
        data = None
        for i in range(n):
            if i == n - 1 and self._want_frame:
                self._want_frame = False
                self._do_step(None, want_frame_next=True)
                data = self._do_step(None, want_frame_next=False)
            else:
                data = self._do_step(None, want_frame_next=False)
        return data

    def request_frame(self):
        """Request pixel data on the next step()/wait() call."""
        self._want_frame = True

    def read_ram(self, start, count):
        """Bulk-read `count` bytes starting at `start` via Lua console.

        This runs between frames and does NOT use the pipe protocol.
        Call step() once before and after to keep MAME in sync.

        Returns list of int values, or empty list on failure.
        """
        results = []
        for chunk_start in range(start, start + count, 256):
            chunk_count = min(256, start + count - chunk_start)
            r = self.console.writeln_expect(
                f'local t={{}};for a={chunk_start},{chunk_start + chunk_count - 1} do '
                f't[#t+1]=mem:read_u8(a) end;print(table.concat(t,","))')
            if r:
                try:
                    results.extend(int(v.strip()) for v in r.split(','))
                except (ValueError, IndexError):
                    results.extend([0] * chunk_count)
            else:
                results.extend([0] * chunk_count)
        return results

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._action_file:
                self._action_file.close()
        except Exception:
            pass
        try:
            if self._data_file:
                self._data_file.close()
        except Exception:
            pass
        self.console.close()
