import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Paramètres (à justifier dans ton rendu) ----
TAILLE_M = 24.0     # carte de 24 m x 24 m, robot au centre au départ
RES_M = 0.05         # 5 cm par cellule -> 240 x 240
INCONNU = 127
DELTA_LIBRE = 10     # décrément des cellules traversées
DELTA_OCC = 40       # incrément de la cellule d'impact
D_MIN, D_MAX = 100, 8000   # mm : on jette les points hors plage
I_MIN = 50           # intensité minimale (à ajuster)


def nouvelle_grille():
    n = int(TAILLE_M / RES_M)
    return np.full((n, n), INCONNU, dtype=np.uint8)   # grille[j, i] : ligne = y, colonne = x


def scan_local(s):
    """DataFrame d'un tour -> points (x, y) en mètres dans le repère du robot, après filtrage."""
    s = s[(s.dist_mm >= D_MIN) & (s.dist_mm <= D_MAX) & (s.intensity >= I_MIN)]
    a = np.radians(s.angle_deg.values)
    d = s.dist_mm.values / 1000.0
    return np.column_stack([d * np.cos(a), -d * np.sin(a)])


def local_vers_global(pts, pose):
    x, y, th = pose                       # mètres, mètres, degrés
    c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([x, y])


def monde_vers_indices(p, n):
    i = int(np.floor(p[0] / RES_M)) + n // 2
    j = int(np.floor(p[1] / RES_M)) + n // 2
    return i, j


def bresenham(i0, j0, i1, j1):
    """Cellules de (i0,j0) à (i1,j1), extrémités incluses."""
    cells = []
    di, dj = abs(i1 - i0), -abs(j1 - j0)
    si = 1 if i0 < i1 else -1
    sj = 1 if j0 < j1 else -1
    err = di + dj
    while True:
        cells.append((i0, j0))
        if i0 == i1 and j0 == j1:
            return cells
        e2 = 2 * err
        if e2 >= dj:
            err += dj
            i0 += si
        if e2 <= di:
            err += di
            j0 += sj


def mettre_a_jour(grille, pts_locaux, pose):
    """Scan + pose -> met à jour la grille (traversées = libres, impact = occupé)."""
    n = grille.shape[0]
    ir, jr = monde_vers_indices(pose[:2], n)
    for p in local_vers_global(pts_locaux, pose):
        i, j = monde_vers_indices(p, n)
        if not (0 <= i < n and 0 <= j < n and 0 <= ir < n and 0 <= jr < n):
            continue
        for (ci, cj) in bresenham(ir, jr, i, j)[:-1]:          # sans la cellule d'impact
            if 0 <= ci < n and 0 <= cj < n:
                grille[cj, ci] = max(0, int(grille[cj, ci]) - DELTA_LIBRE)
        grille[j, i] = min(255, int(grille[j, i]) + DELTA_OCC)
    return grille


if __name__ == "__main__":
    fichier = sys.argv[1] if len(sys.argv) > 1 else "data/scan_01.csv"
    df = pd.read_csv(fichier)

    grille = nouvelle_grille()
    for sid in sorted(df.scan_id.unique()):
        mettre_a_jour(grille, scan_local(df[df.scan_id == sid]), (0.0, 0.0, 0.0))

    h = TAILLE_M / 2
    plt.imshow(grille, cmap="gray_r", vmin=0, vmax=255, origin="lower",
               extent=[-h, h, -h, h])
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Grid map (blanc = libre, noir = occupé, gris = inconnu)")
    plt.show()