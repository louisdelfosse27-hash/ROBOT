import csv
import time
import serial

PORT = "COM7"    
BAUD = 115200
N_TOURS = 3
OUT = "data/scan_01.csv"


def parse(line):
    """Renvoie (index, angle, dist, intensite) ou None si ligne invalide."""
    parts = line.strip().split(",")
    if len(parts) != 6:
        return None
    try:
        v = [float(p) for p in parts]
    except ValueError:
        return None
    return int(v[0]), v[1], v[2], v[3]


rows = []
scan_id = -1
last_idx = None
t0 = time.perf_counter()

with serial.Serial(PORT, BAUD, timeout=1) as ser:
    ser.reset_input_buffer()
    while scan_id < N_TOURS:
        pt = parse(ser.readline().decode(errors="ignore"))
        if pt is None:
            continue
        idx, angle, dist, inten = pt
        if last_idx is not None and idx < last_idx:   # nouveau tour
            scan_id += 1
            if scan_id >= N_TOURS:
                break
        last_idx = idx
        if scan_id >= 0:                               # on jette le 1er tour partiel
            t_ms = (time.perf_counter() - t0) * 1000
            rows.append((scan_id, round(t_ms, 1), angle, dist, inten))

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scan_id", "t_ms", "angle_deg", "dist_mm", "intensity"])
    w.writerows(rows)

print(f"{len(rows)} points, {N_TOURS} tours -> {OUT}")