import os
import time
import numpy as np
import pandas as pd

from gridmap import nouvelle_grille, scan_local, mettre_a_jour, RES_M

# ---- Paramètres de la recherche (à justifier dans ton rendu) ----
FENETRE_XY = 0.50     # m   : on explore pose_init ± 0.50 m
PAS_XY = 0.05         # m   : = la résolution de la grille
FENETRE_TH = 15.0     # deg : pose_init ± 15°
PAS_TH = 1.0          # deg
SEUIL_POS = 0.05      # m   : succès si erreur de position < 5 cm ...
SEUIL_ANG = 2.0       # deg : ... et erreur angulaire < 2°

TESTS = ["pos_B.csv", "pos_E.csv", "pos_G.csv"]   # les scans à localiser

# Décalages (dx m, dy m, dtheta deg) appliqués à la référence pour fabriquer les poses initiales
DECALAGES = [(0, 0, 0), (0.2, 0, 0), (0, 0.2, 0), (0.3, 0.3, 5),
             (-0.3, 0.2, -8), (0.4, -0.4, 10), (0.8, 0, 0)]   # le dernier sort de la fenêtre


def scores_translation(grille, pts, xs, ys):
    """Score de toutes les translations (xs, ys) pour des points pts (N,2) déjà tournés."""
    n = grille.shape[0]
    X = pts[None, None, :, 0] + xs[:, None, None]
    Y = pts[None, None, :, 1] + ys[None, :, None]
    I = np.floor(X / RES_M).astype(int) + n // 2
    J = np.floor(Y / RES_M).astype(int) + n // 2
    ok = (I >= 0) & (I < n) & (J >= 0) & (J < n)
    vals = grille[np.clip(J, 0, n - 1), np.clip(I, 0, n - 1)].astype(np.float32)
    return np.where(ok, vals, 0.0).sum(axis=2)          # (len(xs), len(ys))


def localiser_brute_force(grille, pts_locaux, pose_init):
    """Renvoie (pose trouvée, score)."""
    x0, y0, th0 = pose_init
    xs = x0 + np.arange(-FENETRE_XY, FENETRE_XY + 1e-9, PAS_XY)
    ys = y0 + np.arange(-FENETRE_XY, FENETRE_XY + 1e-9, PAS_XY)
    meilleur = (-np.inf, None)
    for dth in np.arange(-FENETRE_TH, FENETRE_TH + 1e-9, PAS_TH):
        th = th0 + dth
        c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
        pts = pts_locaux @ np.array([[c, -s], [s, c]]).T      # rotation seule
        sc = scores_translation(grille, pts, xs, ys)
        a, b = np.unravel_index(np.argmax(sc), sc.shape)
        if sc[a, b] > meilleur[0]:
            meilleur = (sc[a, b], (xs[a], ys[b], th))
    return meilleur[1], meilleur[0]


def ecart_angle(a, b):
    return (a - b + 180) % 360 - 180


def construire_grille(poses, exclu):
    """Grid map avec tous les scans sauf `exclu`."""
    grille = nouvelle_grille()
    for nom, r in poses.iterrows():
        if nom != exclu:
            df = pd.read_csv(f"data/{nom}")
            mettre_a_jour(grille, scan_local(df[df.scan_id == df.scan_id.min()]),
                          (r.x_m, r.y_m, r.theta_deg))
    return grille


if __name__ == "__main__":
    poses = pd.read_csv("data/poses_ajustees.csv").set_index("fichier")
    # Si tu as mesuré certaines poses au mètre ruban, elles servent de référence
    mesurees = (pd.read_csv("data/poses_mesurees.csv").set_index("fichier")
                if os.path.exists("data/poses_mesurees.csv") else None)

    lignes = []
    for test in TESTS:
        grille = construire_grille(poses, exclu=test)
        df = pd.read_csv(f"data/{test}")
        pts = scan_local(df[df.scan_id == df.scan_id.min()])
        source = mesurees if (mesurees is not None and test in mesurees.index) else poses
        ref = tuple(source.loc[test, ["x_m", "y_m", "theta_deg"]])
        print(f"\n{test} : référence = {'mesurée' if source is mesurees else 'ajustée'}")

        for dx, dy, dth in DECALAGES:
            init = (ref[0] + dx, ref[1] + dy, ref[2] + dth)
            t0 = time.perf_counter()
            pose, score = localiser_brute_force(grille, pts, init)
            dt = time.perf_counter() - t0
            e_pos = float(np.hypot(pose[0] - ref[0], pose[1] - ref[1]))
            e_ang = abs(ecart_angle(pose[2], ref[2]))
            lignes.append(dict(fichier=test, dx=dx, dy=dy, dth=dth,
                               err_pos_m=round(e_pos, 3), err_ang_deg=round(e_ang, 2),
                               temps_s=round(dt, 2),
                               succes=e_pos < SEUIL_POS and e_ang < SEUIL_ANG))
            print(lignes[-1])

    res = pd.DataFrame(lignes)
    print("\nTaux de réussite par décalage initial :")
    print(res.groupby(["dx", "dy", "dth"]).succes.mean().mul(100).round(0))
    print(f"\nTaux de réussite global : {res.succes.mean() * 100:.0f} %")
    res.to_csv("data/resultats_brute_force_grid.csv", index=False)