import sys
import time
import numpy as np
import pandas as pd

from gridmap import scan_local
from extraction import extraire_segments
from carte_vectorielle import retirer_absences, ajouter, nettoyer, vers_global
from localisation_grid import (FENETRE_XY, PAS_XY, FENETRE_TH, PAS_TH,
                               FINE_XY, FINE_PAS_XY, FINE_TH, FINE_PAS_TH,
                               DECALAGES, TESTS, SEUIL_POS, SEUIL_ANG, ecart_angle)

D_MAX = 0.25        # m : au-delà, un point est considéré aberrant (son coût est plafonné)
N_MAX_PTS = 250     # on sous-échantillonne le scan pour garder un calcul rapide


# ---------- Construction de la carte vectorielle (mêmes étapes que carte_vectorielle.py) ----------
def construire_carte(poses, exclu=None):
    """Liste de Segment, avec tous les scans sauf `exclu`."""
    carte = []
    for num, (nom, r) in enumerate(poses.iterrows()):
        if nom == exclu:
            continue
        pose = (r.x_m, r.y_m, r.theta_deg)
        df = pd.read_csv(f"data/{nom}")
        pts = scan_local(df[df.scan_id == df.scan_id.min()])
        retirer_absences(carte, pts, pose)
        for s in extraire_segments(pts):
            ajouter(carte, vers_global(s, pose), num)
    return [f.seg for f in nettoyer(carte)]


# ---------- Score : distance des points du scan aux segments de la carte ----------
def _cout_colonne(seg_arr, pts, x, ys):
    """Coût (plus petit = meilleur) pour une translation x et toutes les translations ys."""
    Ax, Ay, ABx, ABy, L2 = seg_arr
    px = (pts[:, 0] + x)[None, :, None] - Ax[None, None, :]              # (1, N, S)
    py = (pts[:, 1][None, :] + ys[:, None])[:, :, None] - Ay[None, None, :]   # (ny, N, S)
    t = np.clip((px * ABx + py * ABy) / L2, 0.0, 1.0)
    dx = px - t * ABx
    dy = py - t * ABy
    d2 = (dx * dx + dy * dy).min(axis=2)                                  # segment le plus proche
    return np.minimum(d2, D_MAX ** 2).sum(axis=1)                         # (ny,)


def _recherche(carte, pts_locaux, centre, fen_xy, pas_xy, fen_th, pas_th):
    A = np.array([s.p1 for s in carte])
    B = np.array([s.p2 for s in carte])
    AB = B - A
    seg_arr = (A[:, 0], A[:, 1], AB[:, 0], AB[:, 1], (AB ** 2).sum(axis=1))

    x0, y0, th0 = centre
    xs = x0 + np.arange(-fen_xy, fen_xy + 1e-9, pas_xy)
    ys = y0 + np.arange(-fen_xy, fen_xy + 1e-9, pas_xy)
    meilleur = (np.inf, None)
    for dth in np.arange(-fen_th, fen_th + 1e-9, pas_th):
        th = th0 + dth
        c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
        pts = pts_locaux @ np.array([[c, -s], [s, c]]).T                  # rotation seule
        for x in xs:
            cout = _cout_colonne(seg_arr, pts, x, ys)
            k = int(np.argmin(cout))
            if cout[k] < meilleur[0]:
                meilleur = (cout[k], (x, ys[k], th))
    return meilleur[1], -meilleur[0]


def _sous_echantillon(pts):
    return pts[::max(1, len(pts) // N_MAX_PTS)]


def localiser_brute_force(carte, pts_locaux, pose_init):
    return _recherche(carte, _sous_echantillon(pts_locaux), pose_init,
                      FENETRE_XY, PAS_XY, FENETRE_TH, PAS_TH)


def localiser_multires(carte, pts_locaux, pose_init):
    pts = _sous_echantillon(pts_locaux)
    pose_grossiere, _ = _recherche(carte, pts, pose_init, FENETRE_XY, PAS_XY, FENETRE_TH, PAS_TH)
    return _recherche(carte, pts, pose_grossiere, FINE_XY, FINE_PAS_XY, FINE_TH, FINE_PAS_TH)


METHODES = {"brute_force": localiser_brute_force, "multires": localiser_multires}


def resume(res, colonnes_index):
    print("\nMoyennes par méthode :")
    print(res.groupby("methode")[["err_pos_m", "err_ang_deg"] + (["temps_s"] if "temps_s" in res else [])]
          .agg(["mean", "max"]).round(3))
    if "succes" in res:
        print("\nTaux de réussite (%) par décalage initial :")
        print(res.pivot_table(index=colonnes_index, columns="methode",
                              values="succes", aggfunc="mean").mul(100).round(0))
        print("\nTaux de réussite global (%) :")
        print(res.groupby("methode").succes.mean().mul(100).round(0))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "eval"
    poses = pd.read_csv("data/poses_ajustees.csv").set_index("fichier")
    lignes = []

    if mode == "verite":
        # Test de cohérence : tous les scans sont dans la carte, décalages aléatoires
        rng = np.random.default_rng(0)
        carte = construire_carte(poses)
        print(f"Carte : {len(carte)} segments")
        for test in TESTS:
            df = pd.read_csv(f"data/{test}")
            pts = scan_local(df[df.scan_id == df.scan_id.min()])
            ref = tuple(poses.loc[test, ["x_m", "y_m", "theta_deg"]])
            for _ in range(10):
                dx, dy = rng.uniform(-0.3, 0.3, 2)
                dth = rng.uniform(-8, 8)
                init = (ref[0] + dx, ref[1] + dy, ref[2] + dth)
                for nom_m, m in METHODES.items():
                    pose, _ = m(carte, pts, init)
                    lignes.append(dict(methode=nom_m, fichier=test,
                                       err_pos_m=float(np.hypot(pose[0] - ref[0], pose[1] - ref[1])),
                                       err_ang_deg=float(abs(ecart_angle(pose[2], ref[2])))))
        res = pd.DataFrame(lignes)
        resume(res, None)
        res.to_csv("data/resultats_vecto_verite.csv", index=False)
    else:
        # Évaluation : carte sans le scan testé, mêmes poses initiales que pour la grid map
        import os
        mesurees = (pd.read_csv("data/poses_mesurees.csv").set_index("fichier")
                    if os.path.exists("data/poses_mesurees.csv") else None)
        for test in TESTS:
            carte = construire_carte(poses, exclu=test)
            df = pd.read_csv(f"data/{test}")
            pts = scan_local(df[df.scan_id == df.scan_id.min()])
            source = mesurees if (mesurees is not None and test in mesurees.index) else poses
            ref = tuple(source.loc[test, ["x_m", "y_m", "theta_deg"]])
            print(f"\n{test} : référence = {'mesurée' if source is mesurees else 'ajustée'}, "
                  f"carte de {len(carte)} segments")
            for dx, dy, dth in DECALAGES:
                init = (ref[0] + dx, ref[1] + dy, ref[2] + dth)
                for nom_m, m in METHODES.items():
                    t0 = time.perf_counter()
                    pose, _ = m(carte, pts, init)
                    dt = time.perf_counter() - t0
                    e_pos = float(np.hypot(pose[0] - ref[0], pose[1] - ref[1]))
                    e_ang = float(abs(ecart_angle(pose[2], ref[2])))
                    lignes.append(dict(methode=nom_m, fichier=test, dx=dx, dy=dy, dth=dth,
                                       err_pos_m=round(e_pos, 3), err_ang_deg=round(e_ang, 2),
                                       temps_s=round(dt, 2),
                                       succes=e_pos < SEUIL_POS and e_ang < SEUIL_ANG))
                    print(lignes[-1])
        res = pd.DataFrame(lignes)
        resume(res, ["dx", "dy", "dth"])
        res.to_csv("data/resultats_vecto.csv", index=False)