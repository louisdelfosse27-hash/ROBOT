import csv
import sys
import time
from collections import Counter
 
import serial
 
from tours import DetecteurTour
 
PORT = "COM7"
BAUD = 115200
N_TOURS = 3
OUT = sys.argv[1] if len(sys.argv) > 1 else "data/scan_01.csv"
 
 
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
det = DetecteurTour()
t0 = time.perf_counter()
 
with serial.Serial(PORT, BAUD, timeout=1) as ser:
    ser.reset_input_buffer()
    while scan_id < N_TOURS:
        pt = parse(ser.readline().decode(errors="ignore"))
        if pt is None:
            continue
        idx, angle, dist, inten = pt
 
        # Nouveau tour : l'indice repart plus bas, OU l'angle a déjà fait plus d'un tour
        # (cas où deux révolutions sont regroupées sans retour à zéro de l'indice)
        if last_idx is not None and idx < last_idx:
            nouveau = True
            det.reset()
            det.nouveau_tour(angle)
        else:
            nouveau = det.nouveau_tour(angle)
        last_idx = idx
 
        if nouveau:
            scan_id += 1
            if scan_id >= N_TOURS:
                break
        if scan_id >= 0:                               # on jette le 1er tour partiel
            t_ms = (time.perf_counter() - t0) * 1000
            rows.append((scan_id, round(t_ms, 1), angle, dist, inten))
 
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scan_id", "t_ms", "angle_deg", "dist_mm", "intensity"])
    w.writerows(rows)
 
par_tour = Counter(r[0] for r in rows)
print(f"{len(rows)} points, {N_TOURS} tours -> {OUT}")
print("points par tour :", dict(sorted(par_tour.items())))