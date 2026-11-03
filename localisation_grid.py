import os
import time
import numpy as np
import pandas as pd

from gridmap import nouvelle_grille, scan_local, mettre_a_jour, RES_M

# ---- Passe grossière (recherche exhaustive) ----
FENETRE_XY = 0.50     # m   : on explore pose_init ± 0.50 m
PAS_XY = 0.05         # m   : = la résolution de la grille
FENETRE_TH = 15.0     # deg : pose_init ± 15°
PAS_TH = 1.0          # deg

# ---- Passe fine (multi-résolution) ----
FINE_XY = 0.08        # m   : on affine autour du meilleur résultat grossier
FINE_PAS_XY = 0.01    # m   : 5 fois plus fin que la grille
FINE_TH = 1.0         # deg
FINE_PAS_TH = 0.25    # deg

# ---- Critère de succès ----
SEUIL_POS = 0.10      # m
SEUIL_ANG = 2.0       # deg

TESTS = ["pos_B.csv", "pos_E.csv", "pos_G.csv"]   # les scans à localiser

# Décalages (dx m, dy m, dtheta deg) appliqués à la référence pour fabriquer les poses initiales
DECALAGES = [(0, 0, 0), (0.2, 0, 0), (0, 0.2, 0), (0.3, 0.3, 5),
             (-0.3, 0.2, -8), (0.4, -0.4, 10), (0.8, 0, 0)]   # le dernier sort de la fenêtre


def scores_translation(grille, pts, xs, ys):
    """Score (cellule la plus proche) de toutes les translations (xs, ys), pts (N,2) déjà tournés."""
    n = grille.shape[0]
    X = pts[None, None, :, 0] + xs[:, None, None]
    Y = pts[None, None, :, 1] + ys[None, :, None]
    I = np.floor(X / RES_M).astype(int) + n // 2
    J = np.floor(Y / RES_M).astype(int) + n // 2
    ok = (I >= 0) & (I < n) & (J >= 0) & (J < n)
    vals = grille[np.clip(J, 0, n - 1), np.clip(I, 0, n - 1)].astype(np.float32)
    return np.where(ok, vals, 0.0).sum(axis=2)          # (len(xs), len(ys))


def scores_translation_interp(grille, pts, xs, ys):
    """Comme scores_translation, mais avec interpolation bilinéaire : le score varie
    de façon continue, ce qui permet de chercher plus finement que la taille d'une cellule."""
    n = grille.shape[0]
    u = (pts[None, None, :, 0] + xs[:, None, None]) / RES_M + n // 2 - 0.5   # indices continus
    v = (pts[None, None, :, 1] + ys[None, :, None]) / RES_M + n // 2 - 0.5   # (centres de cellules)
    i0 = np.floor(u).astype(int)
    j0 = np.floor(v).astype(int)
    fu, fv = u - i0, v - j0
    ok = (i0 >= 0) & (i0 < n - 1) & (j0 >= 0) & (j0 < n - 1)
    i0 = np.clip(i0, 0, n - 2)
    j0 = np.clip(j0, 0, n - 2)
    g = grille.astype(np.float32)
    val = (g[j0, i0] * (1 - fu) * (1 - fv) + g[j0, i0 + 1] * fu * (1 - fv)
           + g[j0 + 1, i0] * (1 - fu) * fv + g[j0 + 1, i0 + 1] * fu * fv)
    return np.where(ok, val, 0.0).sum(axis=2)


def _recherche(grille, pts_locaux, centre, fen_xy, pas_xy, fen_th, pas_th, fonction_score):
    x0, y0, th0 = centre
    xs = x0 + np.arange(-fen_xy, fen_xy + 1e-9, pas_xy)
    ys = y0 + np.arange(-fen_xy, fen_xy + 1e-9, pas_xy)
    meilleur = (-np.inf, None)
    for dth in np.arange(-fen_th, fen_th + 1e-9, pas_th):
        th = th0 + dth
        c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
        pts = pts_locaux @ np.array([[c, -s], [s, c]]).T      # rotation seule
        sc = fonction_score(grille, pts, xs, ys)
        a, b = np.unravel_index(np.argmax(sc), sc.shape)
        if sc[a, b] > meilleur[0]:
            meilleur = (sc[a, b], (xs[a], ys[b], th))
    return meilleur[1], meilleur[0]


def localiser_brute_force(grille, pts_locaux, pose_init):
    """Recherche exhaustive à la résolution de la grille. Renvoie (pose trouvée, score)."""
    return _recherche(grille, pts_locaux, pose_init,
                      FENETRE_XY, PAS_XY, FENETRE_TH, PAS_TH, scores_translation)


def localiser_multires(grille, pts_locaux, pose_init):
    """Passe grossière, puis passe fine (pas 1 cm / 0.25°) avec score interpolé."""
    pose_grossiere, _ = localiser_brute_force(grille, pts_locaux, pose_init)
    return _recherche(grille, pts_locaux, pose_grossiere,
                      FINE_XY, FINE_PAS_XY, FINE_TH, FINE_PAS_TH, scores_translation_interp)


METHODES = {"brute_force": localiser_brute_force, "multires": localiser_multires}


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
            for nom_methode, methode in METHODES.items():
                t0 = time.perf_counter()
                pose, score = methode(grille, pts, init)
                dt = time.perf_counter() - t0
                e_pos = float(np.hypot(pose[0] - ref[0], pose[1] - ref[1]))
                e_ang = float(abs(ecart_angle(pose[2], ref[2])))
                lignes.append(dict(methode=nom_methode, fichier=test, dx=dx, dy=dy, dth=dth,
                                   err_pos_m=round(e_pos, 3), err_ang_deg=round(e_ang, 2),
                                   temps_s=round(dt, 2),
                                   succes=e_pos < SEUIL_POS and e_ang < SEUIL_ANG))
                print(lignes[-1])

    res = pd.DataFrame(lignes)
    print("\nTaux de réussite (%) par décalage initial :")
    print(res.pivot_table(index=["dx", "dy", "dth"], columns="methode",
                          values="succes", aggfunc="mean").mul(100).round(0))
    print("\nMoyennes par méthode :")
    print(res.groupby("methode")[["err_pos_m", "err_ang_deg", "temps_s"]].mean().round(3))
    print("\nTaux de réussite global (%) :")
    print(res.groupby("methode").succes.mean().mul(100).round(0))
    res.to_csv("data/resultats_grid.csv", index=False)