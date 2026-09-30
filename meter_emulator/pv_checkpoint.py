"""Read the existing agent's checksummed PVS1 checkpoint for emulator evidence."""
import struct
from pathlib import Path


def inspect(path):
    data = Path(path).read_bytes()
    if not 20 <= len(data) <= 1048576 or data[:4] != b"PVS1":
        raise ValueError("Invalid checkpoint size/version")
    checksum = 14695981039346656037
    for byte in data[20:]:
        checksum = ((checksum ^ byte) * 1099511628211) & ((1 << 64) - 1)
    if data[4:20] != f"{checksum:016x}".encode():
        raise ValueError("Checkpoint checksum mismatch")
    offset = 20

    def unpack(fmt):
        nonlocal offset
        result = struct.unpack_from("<" + fmt, data, offset)
        offset += struct.calcsize("<" + fmt)
        return result

    def string():
        nonlocal offset
        length, = unpack("I")
        if length > 4096 or offset + length > len(data):
            raise ValueError("Invalid checkpoint string")
        result = data[offset:offset + length].decode()
        offset += length
        return result

    def configuration():
        result = {}
        while True:
            key = string()
            if not key:
                return result
            result[key] = string()

    def row():
        keys = ["start", "p_count", "q_count", "p_sum", "q_sum", "import_kwh", "export_kwh", "flags"]
        return dict(zip(keys, unpack("qHHddddI")))

    algorithm = string()
    config, pending_config = configuration(), configuration()
    pending, active, last = unpack("BBq")
    current = row()
    unassigned = unpack("dd")
    count, = unpack("I")
    if count > 5760:
        raise ValueError("Oversized history")
    history = [row() for _ in range(count)]
    state, candidate, streak, valid_days = unpack("BBBB")
    totals = {key: sum(item[key] for item in history + [current])
              for key in ["p_count", "q_count", "import_kwh", "export_kwh"]}
    return {"algorithm": algorithm, "config": config, "pending_config": pending_config,
            "pending": bool(pending), "active": bool(active), "last_sample": last,
            "current": current, "history": history, "totals": totals,
            "unassigned_kwh": unassigned, "state": state, "candidate": candidate,
            "streak": streak, "valid_days": valid_days, "checksum": data[4:20].decode()}
