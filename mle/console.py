"""MAME console process manager. Starts MAME with the console plugin and FIFO pipes."""

import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from subprocess import Popen, PIPE


def find_mame():
    """Find MAME binary and data directory."""
    binary = shutil.which("mame")
    if not binary:
        for p in ["/opt/homebrew/bin/mame", "/usr/local/bin/mame"]:
            if os.path.isfile(p) and os.access(p, os.X_OK):
                binary = p
                break
    if not binary:
        raise FileNotFoundError("MAME not found. Install via: brew install mame")

    bin_dir = os.path.dirname(os.path.abspath(binary))
    for candidate in [
        os.path.join(bin_dir, "..", "share", "mame"),
        os.path.join(bin_dir, "..", "share", "games", "mame"),
        bin_dir,
    ]:
        candidate = os.path.abspath(candidate)
        if os.path.isdir(os.path.join(candidate, "plugins")):
            return binary, candidate
    return binary, None


def suppress_warnings(cfg_dir, game_id, binary):
    """Create cfg file to suppress MAME's imperfect emulation warning dialog."""
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_file = os.path.join(cfg_dir, f"{game_id}.cfg")
    now_ts = str(int(time.time()))

    if os.path.exists(cfg_file):
        try:
            root = ET.parse(cfg_file).getroot()
            for uw in root.iter("ui_warnings"):
                if abs(int(time.time()) - int(uw.get("warned", "0"))) < 86400:
                    return cfg_dir
        except Exception:
            pass

    features_xml = ""
    try:
        result = subprocess.run(
            [binary, game_id, "-listxml"],
            capture_output=True, text=True, timeout=30,
        )
        for m in re.finditer(
            r'<feature\s+type="(\w+)"\s+status="(unemulated|imperfect)"\s*/?>',
            result.stdout,
        ):
            features_xml += (
                f'            <feature device="{game_id}" '
                f'type="{m.group(1)}" status="{m.group(2)}" />\n'
            )
    except Exception:
        pass
    if not features_xml:
        features_xml = (
            f'            <feature device="{game_id}" '
            f'type="sound" status="imperfect" />\n'
        )

    with open(cfg_file, "w") as f:
        f.write(f'<?xml version="1.0"?>\n<mameconfig version="10">\n')
        f.write(f'    <system name="{game_id}">\n')
        f.write(f'        <ui_warnings launched="{now_ts}" warned="{now_ts}">\n')
        f.write(features_xml)
        f.write(f"        </ui_warnings>\n    </system>\n</mameconfig>\n")
    return cfg_dir


class _StdoutReader(threading.Thread):
    """Background thread that reads MAME stdout lines into a queue."""

    def __init__(self, pipe):
        super().__init__(daemon=True)
        self.pipe = pipe
        self.q = queue.Queue()

    def run(self):
        for line in iter(self.pipe.readline, b""):
            self.q.put(line[:-1] if line.endswith(b"\n") else line)

    def wait_for_cursor(self):
        """Wait for MAME console startup (3 blank lines)."""
        newlines = 0
        while newlines < 3:
            line = self.pipe.readline()
            if line == b"\n":
                newlines += 1

    def read_all(self, timeout=0.5):
        """Read all available lines, waiting up to timeout for the first."""
        lines = []
        while True:
            try:
                line = self.q.get(timeout=timeout if not lines else 0.05)
                lines.append(line)
            except queue.Empty:
                break
        return lines


class MameConsole:
    """Manages the MAME subprocess with console plugin and FIFO pipes."""

    def __init__(self, roms_path, game_id, render=True, sound=False, throttle=True):
        self.binary, self.data_dir = find_mame()
        self.game_id = game_id

        # Create temp directory for pipes
        self.pipes_dir = tempfile.mkdtemp(prefix="mle_pipes_")
        self.action_path = os.path.join(self.pipes_dir, "action.pipe")
        self.data_path = os.path.join(self.pipes_dir, "data.pipe")
        os.mkfifo(self.action_path)
        os.mkfifo(self.data_path)

        # Suppress warning dialogs
        suppress_warnings("cfg", game_id, self.binary)

        # Build command as string (shell=True, same as working MAMEToolkit)
        roms_abs = str(Path(roms_path).absolute())
        cmd = f"exec {self.binary} -rompath '{roms_abs}'"
        cmd += " -skip_gameinfo -console"
        if render:
            cmd += " -window -nomaximize"
        else:
            cmd += " -window -nomaximize -video none"
            cmd += " -resolution 1x1"  # minimize window footprint
        if self.data_dir:
            cmd += f" -pluginspath '{os.path.join(self.data_dir, 'plugins')}'"
            cmd += f" -hashpath '{os.path.join(self.data_dir, 'hash')}'"
        if not sound:
            cmd += " -sound none"
        cmd += " -throttle" if throttle else " -nothrottle"
        cmd += f" {game_id}"

        self.process = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE)

        # Background stdout reader (same pattern as StreamGobbler)
        self._reader = _StdoutReader(self.process.stdout)
        self._reader.wait_for_cursor()
        self._reader.start()

    def writeln(self, command, timeout=0.5):
        """Send a Lua command and wait for MAME to process it."""
        self.process.stdin.write(command.encode("utf-8") + b"\n")
        self.process.stdin.flush()
        self._reader.read_all(timeout=timeout)

    def writeln_expect(self, command, timeout=3):
        """Send a Lua command and return the first non-echo output line."""
        self.process.stdin.write(command.encode("utf-8") + b"\n")
        self.process.stdin.flush()

        cmd_stripped = command.strip()
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = max(0.05, deadline - time.time())
            try:
                raw = self._reader.q.get(timeout=remaining)
            except queue.Empty:
                return None
            text = self._strip_prompt(raw)
            if not text:
                continue
            if text == cmd_stripped or text.endswith(cmd_stripped):
                continue
            return text
        return None

    @staticmethod
    def _strip_prompt(data):
        """Strip ANSI escapes and [MAME]> prompt from raw bytes."""
        text = data.decode("utf-8", errors="replace")
        text = re.sub(r"\x1b\[[0-9;]*m", "", text)
        text = text.replace("[MAME]> ", "")
        return text.strip()

    def close(self):
        """Kill MAME and clean up pipes."""
        try:
            self.process.kill()
            self.process.wait(timeout=3)
        except Exception:
            pass
        shutil.rmtree(self.pipes_dir, ignore_errors=True)
