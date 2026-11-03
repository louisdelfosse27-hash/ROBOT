import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

FICHIER = sys.argv[1] if len(sys.argv) > 1 else "data/scan_01.csv"
df = pd.read_csv(FICHIER)
ids = sorted(df.scan_id.unique())

# Polaire -> cartésien (mm), même convention que l'Arduino : y = -d*sin(theta)
scans = {}
for i in ids:
    s = df[df.scan_id == i]
    a = np.radians(s.angle_deg.values)
    d = s.dist_mm.values
    scans[i] = np.column_stack([d * np.cos(a), -d * np.sin(a)])

poses = {i: [0.0, 0.0, 0.0] for i in ids}   # x (mm), y (mm), theta (deg)


def transforme(pts, pose):
    x, y, th = pose
    c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([x, y])


fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(left=0.25, bottom=0.30)
ax.set_aspect("equal")
ax.grid(True)
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")

couleurs = plt.cm.tab10.colors
nuages = {}
for k, i in enumerate(ids):
    p = transforme(scans[i], poses[i])
    nuages[i] = ax.scatter(p[:, 0], p[:, 1], s=4, color=couleurs[k % 10],
                           label=f"scan {i}")
ax.legend()

actif = [ids[0]]
lim = 1.1 * max(np.abs(s).max() for s in scans.values())

sl = {
    "x": Slider(plt.axes([0.25, 0.20, 0.65, 0.03]), "x (mm)", -lim, lim, valinit=0),
    "y": Slider(plt.axes([0.25, 0.15, 0.65, 0.03]), "y (mm)", -lim, lim, valinit=0),
    "th": Slider(plt.axes([0.25, 0.10, 0.65, 0.03]), "θ (°)", -180, 180, valinit=0),
}
radio = RadioButtons(plt.axes([0.02, 0.4, 0.15, 0.2]), [f"scan {i}" for i in ids])


def maj(_=None):
    i = actif[0]
    poses[i] = [sl["x"].val, sl["y"].val, sl["th"].val]
    p = transforme(scans[i], poses[i])
    nuages[i].set_offsets(p)
    fig.canvas.draw_idle()


def change_scan(label):
    actif[0] = int(label.split()[1])
    for nom, v in zip(["x", "y", "th"], poses[actif[0]]):
        sl[nom].eventson = False       # évite de déclencher maj()
        sl[nom].set_val(v)
        sl[nom].eventson = True


for s in sl.values():
    s.on_changed(maj)
radio.on_clicked(change_scan)

ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
plt.show()