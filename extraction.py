import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from segments import Segment, longueur, sont_proches, fusionner
from gridmap import scan_local      # même filtrage que pour la grid map

# ---- Paramètres (à justifier dans ton rendu) ----
SEUIL_RUPTURE = 0.15     # m : trou entre deux points consécutifs => nouveau groupe
SEUIL_DEVIATION = 0.03   # m : écart max à la droite avant de couper (bruit du lidar ~ 1-3 cm)
MIN_POINTS = 6           # une chaîne avec moins de points est ignorée
MIN_LONGUEUR = 0.20      # m : segments plus courts jetés


def decouper_en_groupes(pts):
    groupes, courant = [], [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - courant[-1]) > SEUIL_RUPTURE:
            groupes.append(np.array(courant))
            courant = [p]
        else:
            courant.append(p)
    groupes.append(np.array(courant))
    return groupes


def split(pts):
    """Renvoie la liste des chaînes de points quasi-droites."""
    if len(pts) < MIN_POINTS:
        return []
    a, b = pts[0], pts[-1]
    ab = b - a
    L = np.linalg.norm(ab)
    if L < 1e-9:
        return []
    n = np.array([-ab[1], ab[0]]) / L              # normale à la droite a-b
    dev = np.abs((pts - a) @ n)                    # distance de chaque point à la droite
    k = int(np.argmax(dev))
    if dev[k] <= SEUIL_DEVIATION:
        return [pts]
    return split(pts[:k + 1]) + split(pts[k:])


def ajuster(pts):
    """Droite par moindres carrés (SVD), bornée par les projections extrêmes des points."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    u = vt[0]
    if u @ (pts[-1] - pts[0]) < 0:                 # garde le sens de parcours du scan
        u = -u
    t = (pts - c) @ u
    return Segment(tuple(c + t.min() * u), tuple(c + t.max() * u))


def fusionner_proches(segs):
    segs = list(segs)
    change = True
    while change:
        change = False
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                if sont_proches(segs[i], segs[j]):
                    segs[i] = fusionner(segs[i], segs[j])
                    del segs[j]
                    change = True
                    break
            if change:
                break
    return segs


def extraire_segments(pts):
    """Nuage de points (N,2) en mètres, dans l'ordre du scan -> liste de Segment."""
    if len(pts) < MIN_POINTS:
        return []
    segs = []
    for groupe in decouper_en_groupes(pts):
        for chaine in split(groupe):
            s = ajuster(chaine)
            if longueur(s) >= MIN_LONGUEUR:
                segs.append(s)
    return fusionner_proches(segs)


if __name__ == "__main__":
    fichier = sys.argv[1] if len(sys.argv) > 1 else "data/scan_01.csv"
    sid = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    df = pd.read_csv(fichier)
    pts = scan_local(df[df.scan_id == sid])
    segs = extraire_segments(pts)
    print(f"{len(pts)} points -> {len(segs)} segments")

    plt.scatter(pts[:, 0], pts[:, 1], s=4, color="lightgray", label="points")
    for k, s in enumerate(segs):
        plt.plot([s.p1[0], s.p2[0]], [s.p1[1], s.p2[1]], "r-", lw=2,
                 label="segments" if k == 0 else None)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()