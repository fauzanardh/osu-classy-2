import lzma
import struct
import datetime
from pathlib import Path
from enum import Enum, IntFlag
from dataclasses import dataclass

import numpy as np


class GameMode(Enum):
    STD = 0
    TAIKO = 1
    CTB = 2
    MANIA = 3


class Mod(IntFlag):
    NoMod = 0
    NoFail = 1 << 0
    Easy = 1 << 1
    TouchDevice = 1 << 2
    Hidden = 1 << 3
    HardRock = 1 << 4
    SuddenDeath = 1 << 5
    DoubleTime = 1 << 6
    Relax = 1 << 7
    HalfTime = 1 << 8
    Nightcore = 1 << 9
    Flashlight = 1 << 10
    Autoplay = 1 << 11
    SpunOut = 1 << 12
    Autopilot = 1 << 13
    Perfect = 1 << 14
    Key4 = 1 << 15
    Key5 = 1 << 16
    Key6 = 1 << 17
    Key7 = 1 << 18
    Key8 = 1 << 19
    FadeIn = 1 << 20
    Random = 1 << 21
    Cinema = 1 << 22
    Target = 1 << 23
    Key9 = 1 << 24
    KeyCoop = 1 << 25
    Key1 = 1 << 26
    Key3 = 1 << 27
    Key2 = 1 << 28
    ScoreV2 = 1 << 29
    Mirror = 1 << 30


class Key(IntFlag):
    M1 = 1 << 0
    M2 = 1 << 1
    K1 = 1 << 2
    K2 = 1 << 3
    SMOKE = 1 << 4


@dataclass
class ReplayEventOsu:
    time_delta: int
    x: float
    y: float
    keys: Key


@dataclass
class LifeBarState:
    time: int
    life: float


class _unpacker:
    def __init__(self, replay_data: bytes):
        self.replay_data = replay_data
        self.offset = 0

    def string_length(self) -> int:
        result = 0
        shift = 0
        while True:
            b = self.replay_data[self.offset]
            self.offset += 1
            result |= (b & 0x7F) << shift
            if not b & 0x80:
                break
            shift += 7
        return result

    def unpack_string(self):
        if self.replay_data[self.offset] == 0x0:
            self.offset += 1
        elif self.replay_data[self.offset] == 0xB:
            self.offset += 1
            length = self.string_length()
            result = self.replay_data[self.offset : self.offset + length].decode(
                "utf-8"
            )
            self.offset += length
            return result
        else:
            raise ValueError("Invalid string")

    def unpack_once(self, fmt: str):
        specifier = f"<{fmt}"
        unpacked = struct.unpack_from(specifier, self.replay_data, self.offset)
        self.offset += struct.calcsize(specifier)
        return unpacked[0]

    def unpack_timestamp(self):
        ticks = self.unpack_once("q")
        timestamp = datetime.datetime.min + datetime.timedelta(microseconds=ticks / 10)
        timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)
        return timestamp

    @staticmethod
    def parse_replay_data(replay_data_str: str):
        replay_data_str = replay_data_str.rstrip(",")
        events = [event.split("|") for event in replay_data_str.split(",")]

        rng_seed = None
        play_data = []
        for event in events:
            time_delta = int(event[0])
            x = float(event[1])
            y = float(event[2])
            keys = int(event[3])

            if time_delta == -12345 and event == events[-1]:
                rng_seed = keys
                continue

            play_data.append(ReplayEventOsu(time_delta, x, y, Key(keys)))
        return rng_seed, play_data

    def unpack_replay_data(self):
        length = self.unpack_once("i")
        data = self.replay_data[self.offset : self.offset + length]
        data = lzma.decompress(data, format=lzma.FORMAT_AUTO)
        data = data.decode("ascii")
        self.offset += length
        return self.parse_replay_data(data)

    def unpack_replay_id(self):
        try:
            replay_id = self.unpack_once("q")
        except struct.error:
            replay_id = self.unpack_once("l")
        return replay_id

    def unpack_life_bar(self):
        lifebar = self.unpack_string()
        if not lifebar:
            return None

        lifebar = lifebar.rstrip(",")
        states = [state.split("|") for state in lifebar.split(",")]

        return [LifeBarState(int(state[0]), float(state[1])) for state in states]


class Replay:
    def __init__(self, replay_path: str):
        self._unpacker = _unpacker(Path(replay_path).read_bytes())

        self._mode = GameMode(self._unpacker.unpack_once("b"))
        if self._mode != GameMode.STD:
            raise ValueError("Only std replays are supported")

        self._game_version = self._unpacker.unpack_once("i")
        self._beatmap_hash = self._unpacker.unpack_string()
        self._username = self._unpacker.unpack_string()
        self._replay_hash = self._unpacker.unpack_string()
        self._count_300 = self._unpacker.unpack_once("h")
        self._count_100 = self._unpacker.unpack_once("h")
        self._count_50 = self._unpacker.unpack_once("h")
        self._count_geki = self._unpacker.unpack_once("h")
        self._count_katu = self._unpacker.unpack_once("h")
        self._count_miss = self._unpacker.unpack_once("h")
        self._score = self._unpacker.unpack_once("i")
        self._max_combo = self._unpacker.unpack_once("h")
        self._perfect = self._unpacker.unpack_once("?")
        self._mods = Mod(self._unpacker.unpack_once("i"))
        self._life_bar = self._unpacker.unpack_life_bar()
        self._timestamp = self._unpacker.unpack_timestamp()
        self._rng_seed, self._replay_data = self._unpacker.unpack_replay_data()
        self._replay_data_np = None
        self._replay_id = self._unpacker.unpack_replay_id()

        self.replay_data_to_np()

    def replay_data_to_np(self):
        t = 0
        # ignoring keys for now
        arr = np.zeros((len(self._replay_data), 3), dtype=np.float32)
        for i, event in enumerate(self._replay_data):
            t += event.time_delta
            arr[i, 0] = float(t)
            arr[i, 1] = event.x
            arr[i, 2] = event.y
        # sort by time
        self._replay_data_np = arr[arr[:, 0].argsort()]

    def cursor(self, t):
        """
        interpolates linearly between events
        return cursor position + time since last click at time t (ms)
        """

        # find closest event before t
        idx = np.searchsorted(self._replay_data_np[:, 0], t, side="right") - 1
        if idx < 0:
            raise ValueError("t is before first event")

        # if t is after last event, return last event
        if idx == len(self._replay_data_np) - 1:
            return (self._replay_data_np[idx, 1], self._replay_data_np[idx, 2]), 0

        # interpolate between events
        t0, x0, y0 = self._replay_data_np[idx]
        t1, x1, y1 = self._replay_data_np[idx + 1]
        alpha = (t - t0) / (t1 - t0)
        return (x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0)), t1 - t
