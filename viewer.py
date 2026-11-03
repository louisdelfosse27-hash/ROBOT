import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

from gridmap import scan_local   # même filtrage que pour la grid map

fichiers = sys.argv[1:] or ["data/scan_01.csv"]
SORTIE = "data/poses_ajustees.csv"

# Un fichier = un scan : on prend le premier tour de chaque CSV (points en mètres)
scans = {}
for f in fichiers:
    df = pd.read_csv(f)
    scans[os.path.basename(f)] = scan_local(df[df.scan_id == df.scan_id.min()])
noms = list(scans)
poses = {n: [0.0, 0.0, 0.0] for n in noms}      # x (m), y (m), theta (deg)


def transforme(pts, pose):
    x, y, th = pose
    c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
    return pts @ np.array([[c, -s], [s, c]]).T + np.array([x, y])


fig, ax = plt.subplots(figsize=(9, 9))
plt.subplots_adjust(left=0.25, bottom=0.30)
ax.set_aspect("equal")
ax.grid(True)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

couleurs = plt.cm.tab10.colors
nuages = {}
for k, n in enumerate(noms):
    p = transforme(scans[n], poses[n])
    nuages[n] = ax.scatter(p[:, 0], p[:, 1], s=4, color=couleurs[k % 10], label=n)
ax.legend(loc="upper right")

lim = 1.5 * max(np.abs(s).max() for s in scans.values())
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)

sl = {
    "x": Slider(plt.axes([0.25, 0.20, 0.65, 0.03]), "x (m)", -lim, lim, valinit=0, valstep=0.01),
    "y": Slider(plt.axes([0.25, 0.15, 0.65, 0.03]), "y (m)", -lim, lim, valinit=0, valstep=0.01),
    "th": Slider(plt.axes([0.25, 0.10, 0.65, 0.03]), "θ (°)", -180, 180, valinit=0, valstep=0.5),
}
radio = RadioButtons(plt.axes([0.02, 0.35, 0.2, 0.05 * len(noms) + 0.05]), noms)
bouton = Button(plt.axes([0.02, 0.15, 0.2, 0.06]), "Sauver poses")
actif = [noms[0]]


def maj(_=None):
    n = actif[0]
    poses[n] = [sl["x"].val, sl["y"].val, sl["th"].val]
    nuages[n].set_offsets(transforme(scans[n], poses[n]))
    fig.canvas.draw_idle()


def change_scan(label):
    actif[0] = label
    for nom, v in zip(["x", "y", "th"], poses[label]):
        sl[nom].eventson = False        # évite de déclencher maj()
        sl[nom].set_val(v)
        sl[nom].eventson = True


def sauver(_):
    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    lignes = pd.DataFrame([[n, *poses[n]] for n in noms],
                          columns=["fichier", "x_m", "y_m", "theta_deg"])
    lignes.to_csv(SORTIE, index=False)
    print(f"Poses sauvées dans {SORTIE}")


for s in sl.values():
    s.on_changed(maj)
radio.on_clicked(change_scan)
bouton.on_clicked(sauver)
plt.show()