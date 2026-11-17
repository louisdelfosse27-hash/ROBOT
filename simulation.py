"""Simulation à pose connue : un robot virtuel parcourt une salle simulée, un lidar virtuel
produit des scans dont les poses sont EXACTES par construction (vérité terrain).

La salle simulée reprend les dimensions mesurées de la vraie salle (contour d'environ 13 x 10 m avec
le décrochement au sud-est) et ajoute deux obstacles. Ce n'est pas la vraie salle : c'est un monde
dont on connaît la géométrie, pour mesurer le SLAM contre une vérité sans incertitude.

Usage :  python simulation.py [n_tirages]        expérience principale (défaut 5)
         python simulation.py sens [n_tirages]   sensibilité à la qualité de l'odométrie (défaut 5)
"""
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gridmap
import slam
from slam import lancer, resume, DOSSIER_FIGURES

# ---------------------------------------------------------------- monde simulé
CONTOUR = [(-10.05, -6.15), (-4.7, -6.15), (-4.7, -4.0), (3.0, -4.0), (3.0, 3.85), (-10.05, 3.85)]
OBSTACLES = [[(-6.8, -0.8), (-5.8, -0.8), (-5.8, 0.2), (-6.8, 0.2)],      # une table
             [(-1.2, 1.8), (-0.4, 1.8), (-0.4, 2.6), (-1.2, 2.6)]]        # une autre
CHEMIN = [(-9.0, -5.0), (-9.0, 3.0), (-4.0, 3.0), (-4.0, -2.5), (2.0, -2.5), (2.0, 3.0), (-2.0, 0.5)]
PAS_M = 0.8                  # distance entre deux poses consécutives (m)

N_FAISCEAUX = 450            # comme le LD06 dans la salle (~454 points par tour)
SIGMA_LIDAR = 0.015          # m : bruit gaussien sur la distance
P_PERTE = 0.03               # proportion de faisceaux sans retour
D_MAX_SIM = 12.0             # m


def murs_du_monde():
    """Tableau (S, 4) de segments x1,y1,x2,y2."""
    murs = []
    for poly in [CONTOUR] + OBSTACLES:
        for a, b in zip(poly, poly[1:] + poly[:1]):
            murs.append((*a, *b))
    return np.array(murs, dtype=float)


def trajectoire_vraie():
    """Poses (x, y, θ) tous les PAS_M le long de CHEMIN ; θ = direction de marche, lissée."""
    pts = [np.array(CHEMIN[0], float)]
    for a, b in zip(CHEMIN[:-1], CHEMIN[1:]):
        a, b = np.array(a, float), np.array(b, float)
        n = int(np.ceil(np.linalg.norm(b - a) / PAS_M))
        pts += [a + (b - a) * i / n for i in range(1, n + 1)]
    pts = np.array(pts)
    v = np.gradient(pts, axis=0)
    cap = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    cap = np.convolve(np.pad(cap, 3, mode="edge"), np.ones(7) / 7, mode="valid")   # lissage des virages
    return [(p[0], p[1], float(np.degrees(c))) for p, c in zip(pts, cap)]


def scan_simule(pose, murs, rng):
    """Nuage de points (N,2) dans le repère du robot, avec les conventions du LD06
    (angles décroissants, x = d cos a, y = -d sin a)."""
    a = np.radians(359.3 - np.arange(N_FAISCEAUX) * 360 / N_FAISCEAUX)
    dir_loc = np.column_stack([np.cos(a), -np.sin(a)])
    th = np.radians(pose[2])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    d = dir_loc @ R.T                                   # directions dans le repère de la salle
    O = np.array(pose[:2])
    A, E = murs[:, :2], murs[:, 2:] - murs[:, :2]
    AO = A - O                                          # (S,2)
    denom = d[:, None, 0] * E[None, :, 1] - d[:, None, 1] * E[None, :, 0]          # (N,S)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (AO[None, :, 0] * E[None, :, 1] - AO[None, :, 1] * E[None, :, 0]) / denom
        u = (AO[None, :, 0] * d[:, None, 1] - AO[None, :, 1] * d[:, None, 0]) / denom
    t = np.where((np.abs(denom) > 1e-12) & (t > 1e-6) & (u >= 0) & (u <= 1), t, np.inf)
    dist = t.min(axis=1)
    ok = np.isfinite(dist) & (dist < D_MAX_SIM) & (rng.random(N_FAISCEAUX) > P_PERTE)
    dist = dist[ok] + rng.normal(0, SIGMA_LIDAR, ok.sum())
    return dir_loc[ok] * dist[:, None]


def generer(seed=123):
    rng = np.random.default_rng(seed)
    murs = murs_du_monde()
    refs = trajectoire_vraie()
    return [scan_simule(p, murs, rng) for p in refs], refs, murs


# ---------------------------------------------------------------- figures
def figure_monde(murs, refs, resultats, odom, chemin):
    fig, ax = plt.subplots(figsize=(11, 7.5))
    h = gridmap.TAILLE_M / 2
    poses, grille, _ = resultats["grid"]
    ax.imshow(grille, cmap="gray_r", vmin=0, vmax=255, origin="lower", extent=[-h, h, -h, h])
    for m in murs:
        ax.plot([m[0], m[2]], [m[1], m[3]], "-", color="tab:blue", lw=1, alpha=0.8,
                label="murs réels (simulation)" if m is murs[0] else None)
    for pts, style, lab in [(refs, "k.-", "vérité terrain"), (odom, "g--", "odométrie seule"),
                            (poses, "r.-", "SLAM (grid map)"),
                            (resultats["vecto"][0], "m.:", "SLAM (carte vectorielle)")]:
        a = np.array(pts)
        ax.plot(a[:, 0], a[:, 1], style, ms=5, label=lab)
    ax.set_xlim(-11.5, 4.5)
    ax.set_ylim(-7.5, 5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(chemin, dpi=110)
    plt.close(fig)


def figure_derive(res, chemin):
    """Erreur de position selon l'avancement, moyenne sur les tirages."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for meth, style in [("odometrie", "g--"), ("grid_sans_prior", "y-"), ("grid", "r-"), ("vecto", "m:")]:
        g = res[res.methode == meth].groupby("k").err_pos_m.mean() * 100
        ax.plot(g.index, g.values, style, lw=2, label={"odometrie": "odométrie seule", "grid_sans_prior": "SLAM grid map sans a priori",
                                                       "grid": "SLAM grid map avec a priori",
                                                       "vecto": "SLAM carte vectorielle"}[meth])
    ax.set_xlabel("numéro de pose")
    ax.set_ylabel("erreur de position (cm)")
    ax.set_title("Dérive : odométrie seule vs SLAM (moyenne sur les tirages)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(chemin, dpi=110)
    plt.close(fig)


def experience_principale(n_tirages, scans, refs, murs):
    # odométrie de robot mecanum : glissement ~5 % de la distance, dérive de cap ~0,8°/pose
    bruit = dict(sig_fixe=0.01, sig_par_m=0.05, sig_th=0.8)
    res, resultats, odom = lancer(scans, refs, n_tirages, **bruit)
    res.to_csv("data/resultats_slam_simu.csv", index=False)
    print(f"\nSLAM simulé, {n_tirages} tirages d'odométrie bruitée (vérité terrain exacte) :")
    print(resume(res).to_string())
    figure_monde(murs, refs, resultats, odom, f"{DOSSIER_FIGURES}/slam_simulation.png")
    figure_derive(res, f"{DOSSIER_FIGURES}/slam_derive.png")
    print(f"\nFigures : {DOSSIER_FIGURES}/slam_simulation.png, slam_derive.png")


def experience_sensibilite(n_tirages, scans, refs):
    """Taux de réussite des deux variantes de SLAM sur grid map selon la qualité de l'odométrie."""
    lignes = []
    for glis in [0.03, 0.05, 0.10, 0.15, 0.20]:
        r, _, _ = lancer(scans, refs, n_tirages, modes=["grid_sans_prior", "grid"],
                         sig_fixe=0.01, sig_par_m=glis, sig_th=0.8)
        # une trajectoire est "réussie" si TOUTES ses poses restent sous les seuils (pas de perte)
        for meth, g in r.groupby("methode"):
            if meth == "odometrie":
                continue
            par_tirage = g[g.k >= 1].groupby("tirage").apply(
                lambda x: bool(((x.err_pos_m < 0.10) & (x.err_ang_deg < 2)).all()), include_groups=False)
            lignes.append(dict(glissement_pct=int(round(glis * 100)), methode=meth,
                               trajectoires_ok_pct=round(par_tirage.mean() * 100),
                               err_pos_moy_cm=round(g[g.k >= 1].err_pos_m.mean() * 100, 1)))
    sens = pd.DataFrame(lignes)
    sens.to_csv("data/sensibilite_odometrie.csv", index=False)
    print(f"\nSensibilité à l'odométrie ({n_tirages} tirages par niveau ; σ position = 1 cm + glissement × distance) :")
    print("Trajectoires entièrement réussies (%) :")
    print(sens.pivot(index="glissement_pct", columns="methode", values="trajectoires_ok_pct").to_string())
    print("Erreur de position moyenne (cm) :")
    print(sens.pivot(index="glissement_pct", columns="methode", values="err_pos_moy_cm").to_string())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].isdigit() else "principale"
    n = int([a for a in sys.argv[1:] if a.isdigit()][0]) if any(a.isdigit() for a in sys.argv[1:]) else 5
    scans, refs, murs = generer()
    print(f"Simulation : {len(refs)} poses, pas de {PAS_M} m, {np.mean([len(s) for s in scans]):.0f} points/scan, "
          f"bruit lidar {SIGMA_LIDAR * 100:.1f} cm")
    if mode == "sens":
        experience_sensibilite(n, scans, refs)
    else:
        experience_principale(n, scans, refs, murs)
