"""Jour 3 : SLAM (localisation + cartographie simultanées) sur les 7 scans réels.

Principe, pour chaque nouveau scan k :
  1. prédiction : pose_init = pose estimée de k-1 composée avec le déplacement relatif (odométrie) ;
  2. recalage  : le scan est localisé sur la carte construite jusqu'ici (scan matching, mêmes
                 algorithmes qu'au jour 2) ;
  3. mise à jour : le scan, à la pose recalée, est ajouté à la carte.
Deux cartes possibles : grid map ("grid") ou carte vectorielle ("vecto").

L'odométrie n'existe pas pour les 7 positions réelles (le robot a été déplacé à la main) :
elle est SIMULÉE = déplacement relatif des poses ajustées + bruit. Voir `odometrie_bruitee`.
La première pose est fixée (c'est l'origine du repère) ; les autres sont estimées.

Usage :  python slam.py            (7 scans réels, 20 tirages de bruit d'odométrie)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gridmap
from gridmap import nouvelle_grille, scan_local, mettre_a_jour, local_vers_global
import localisation_grid as lg
import localisation_vecto as lv
from segments import longueur
from extraction import extraire_segments
from carte_vectorielle import Feature, retirer_absences, ajouter, nettoyer, vers_global, MIN_LONGUEUR_CARTE

DOSSIER_FIGURES = "illustation_avancement"
DEPART = "pos_E.csv"            # premier scan du SLAM (définit l'origine du repère)


# ---------------------------------------------------------------- géométrie des poses
def ecart_angle(a, b):
    return (a - b + 180) % 360 - 180


def composer(pose, d):
    """Pose obtenue en appliquant le déplacement relatif d = (dx, dy, dθ) exprimé dans le repère du robot."""
    x, y, th = pose
    c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
    return (x + c * d[0] - s * d[1], y + s * d[0] + c * d[1], th + d[2])


def relatif(p, q):
    """Déplacement de la pose p à la pose q, exprimé dans le repère du robot en p."""
    c, s = np.cos(np.radians(p[2])), np.sin(np.radians(p[2]))
    dx, dy = q[0] - p[0], q[1] - p[1]
    return (c * dx + s * dy, -s * dx + c * dy, ecart_angle(q[2], p[2]))


def odometrie_bruitee(poses_vraies, rng, sig_fixe=0.02, sig_par_m=0.03, sig_th=1.0):
    """Déplacements relatifs entre poses consécutives, avec un bruit gaussien :
         σ_position = sig_fixe + sig_par_m × distance parcourue (m)   (par axe)
         σ_angle    = sig_th (deg)
       Renvoie une liste odo où odo[k] relie la pose k-1 à la pose k (odo[0] = None)."""
    odo = [None]
    for p, q in zip(poses_vraies[:-1], poses_vraies[1:]):
        d = relatif(p, q)
        sig = sig_fixe + sig_par_m * np.hypot(d[0], d[1])
        odo.append((d[0] + rng.normal(0, sig), d[1] + rng.normal(0, sig), d[2] + rng.normal(0, sig_th)))
    return odo


def odometrie_seule(pose0, odo):
    """Estimation sans recalage (dead reckoning) : on empile l'odométrie, sans carte."""
    poses = [tuple(pose0)]
    for d in odo[1:]:
        poses.append(composer(poses[-1], d))
    return poses


def erreurs(est, ref):
    """Erreurs (position m, angle deg) de chaque pose estimée par rapport à la référence."""
    return [(float(np.hypot(e[0] - r[0], e[1] - r[1])), float(abs(ecart_angle(e[2], r[2]))))
            for e, r in zip(est, ref)]


# ---------------------------------------------------------------- SLAM
KAPPA_PRIOR = 0.07      # poids de l'a priori (points de score par point du scan, en unités de σ²)
SIG_ODO_FIXE = 0.02     # m  : modèle d'incertitude de l'odométrie supposé par le SLAM
SIG_ODO_PAR_M = 0.05    # m par m parcouru ; σ_pas = SIG_ODO_FIXE + SIG_ODO_PAR_M × distance du pas


def sigma_pas(d):
    """Incertitude supposée (m) sur la position prédite après le déplacement relatif d."""
    return SIG_ODO_FIXE + SIG_ODO_PAR_M * np.hypot(d[0], d[1])


def localiser_multires_prior(grille, pts_locaux, pose_init, sigma, kappa=KAPPA_PRIOR):
    """localisation_grid.localiser_multires, avec un score pénalisé par la distance de Mahalanobis (en
    position) à la pose prédite par l'odométrie :  score - kappa × N × (écart / sigma)².
    Le long d'un mur droit, le score seul est presque plat et son maximum peut tomber n'importe où dans
    la fenêtre de recherche (problème d'ouverture) ; l'a priori choisit alors la solution la plus proche
    de l'odométrie. Il s'affaiblit quand l'odométrie est incertaine (grand pas), pour ne pas la croire
    à tort."""
    pen = kappa * len(pts_locaux) / sigma ** 2
    x0, y0 = pose_init[0], pose_init[1]

    def avec_prior(f):
        def score(grille, pts, xs, ys):
            return f(grille, pts, xs, ys) - pen * ((xs[:, None] - x0) ** 2 + (ys[None, :] - y0) ** 2)
        return score

    grossiere, _ = lg._recherche(grille, pts_locaux, pose_init, lg.FENETRE_XY, lg.PAS_XY,
                                 lg.FENETRE_TH, lg.PAS_TH, avec_prior(lg.scores_translation))
    return lg._recherche(grille, pts_locaux, grossiere, lg.FINE_XY, lg.FINE_PAS_XY,
                         lg.FINE_TH, lg.FINE_PAS_TH, avec_prior(lg.scores_translation_interp))


def slam_grid(scans, odo, pose0, prior=True):
    """SLAM sur grid map. scans : liste de nuages locaux (N,2) en m. Renvoie (poses, grille, temps).
    prior=False : algorithme du jour 2 tel quel (sans a priori d'odométrie)."""
    grille = nouvelle_grille()
    mettre_a_jour(grille, scans[0], pose0)
    poses, temps = [tuple(pose0)], [0.0]
    for k in range(1, len(scans)):
        init = composer(poses[-1], odo[k])
        t0 = time.perf_counter()
        if prior:
            pose, _ = localiser_multires_prior(grille, scans[k], init, sigma_pas(odo[k]))
        else:
            pose, _ = lg.localiser_multires(grille, scans[k], init)
        temps.append(time.perf_counter() - t0)
        pose = tuple(float(v) for v in pose)
        mettre_a_jour(grille, scans[k], pose)
        poses.append(pose)
    return poses, grille, temps


def slam_vecto(scans, odo, pose0):
    """SLAM sur carte vectorielle. Renvoie (poses, liste de Segment, temps)."""
    carte = []

    def integrer(pts, pose, num):
        retirer_absences(carte, pts, pose)
        for s in extraire_segments(pts):
            ajouter(carte, vers_global(s, pose), num)

    integrer(scans[0], pose0, 0)
    poses, temps = [tuple(pose0)], [0.0]
    for k in range(1, len(scans)):
        init = composer(poses[-1], odo[k])
        segs = [f.seg for f in carte if longueur(f.seg) >= MIN_LONGUEUR_CARTE]   # sans modifier la carte
        t0 = time.perf_counter()
        pose, _ = lv.localiser_multires(segs, scans[k], init)
        temps.append(time.perf_counter() - t0)
        pose = tuple(float(v) for v in pose)
        integrer(scans[k], pose, k)
        poses.append(pose)
    return poses, [f.seg for f in nettoyer(carte)], temps


SLAMS = {"grid": slam_grid,
         "grid_sans_prior": lambda scans, odo, pose0: slam_grid(scans, odo, pose0, prior=False),
         "vecto": slam_vecto}


# ---------------------------------------------------------------- données réelles
def ordre_plus_proche(poses, depart):
    """Ordre de visite : on part de `depart`, puis toujours la position non visitée la plus proche
    (pour que deux scans consécutifs aient des murs en commun)."""
    reste = [n for n in poses.index if n != depart]
    ordre = [depart]
    while reste:
        p = poses.loc[ordre[-1]]
        k = min(reste, key=lambda n: np.hypot(poses.loc[n, "x_m"] - p.x_m, poses.loc[n, "y_m"] - p.y_m))
        ordre.append(k)
        reste.remove(k)
    return ordre


def charger_reel():
    poses = pd.read_csv("data/poses_ajustees.csv").set_index("fichier")
    ordre = ordre_plus_proche(poses, DEPART)
    scans, refs = [], []
    for nom in ordre:
        df = pd.read_csv(f"data/{nom}")
        scans.append(scan_local(df[df.scan_id == df.scan_id.min()]))
        refs.append(tuple(poses.loc[nom, ["x_m", "y_m", "theta_deg"]]))
    return ordre, scans, refs


# ---------------------------------------------------------------- figures
def tracer_trajectoires(ax, ref, est, odom, titre):
    for pts, style, lab in [(ref, "k.-", "référence (poses ajustées)"),
                            (odom, "g--", "odométrie seule"),
                            (est, "r.-", "SLAM")]:
        a = np.array(pts)
        ax.plot(a[:, 0], a[:, 1], style, ms=9, label=lab)
    ax.set_title(titre)


def figure_slam(ordre, ref, resultats, odom, chemin):
    """resultats : dict mode -> (poses, carte, temps)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    h = gridmap.TAILLE_M / 2
    poses, grille, _ = resultats["grid"]
    axes[0].imshow(grille, cmap="gray_r", vmin=0, vmax=255, origin="lower", extent=[-h, h, -h, h])
    tracer_trajectoires(axes[0], ref, poses, odom, "SLAM sur grid map")
    poses, segs, _ = resultats["vecto"]
    for k, s in enumerate(segs):
        axes[1].plot([s.p1[0], s.p2[0]], [s.p1[1], s.p2[1]], "-", color="tab:blue", lw=2,
                     label="carte vectorielle" if k == 0 else None)
    tracer_trajectoires(axes[1], ref, poses, odom, "SLAM sur carte vectorielle")
    for ax in axes:
        ax.set_xlim(-11.5, 4.5)
        ax.set_ylim(-7.5, 5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(loc="lower right", fontsize=8)
    for i, nom in enumerate(ordre):
        axes[0].annotate(nom[4], (ref[i][0], ref[i][1]), xytext=(5, 5), textcoords="offset points")
    fig.tight_layout()
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    fig.savefig(chemin, dpi=110)
    plt.close(fig)


def sauver_carte_vecto(segs, chemin):
    with open(chemin, "w") as f:
        f.write("# Carte vectorielle de la salle (SLAM) : x1,y1,x2,y2\n")
        for s in segs:
            f.write(f"{s.p1[0]:.3f},{s.p1[1]:.3f},{s.p2[0]:.3f},{s.p2[1]:.3f}\n")


def lancer(scans, refs, n_tirages, seed0=0, modes=None, **bruit):
    """Fait tourner les SLAM (par défaut les deux) pour n_tirages bruits d'odométrie différents.
    Renvoie (DataFrame des erreurs par pose, résultats du tirage 0, odométrie seule du tirage 0)."""
    modes = modes or list(SLAMS)
    lignes, premier = [], None
    for t in range(n_tirages):
        rng = np.random.default_rng(seed0 + t)
        odo = odometrie_bruitee(refs, rng, **bruit)
        odom = odometrie_seule(refs[0], odo)
        res = {}
        for mode in modes:
            res[mode] = SLAMS[mode](scans, odo, refs[0])
        if premier is None:
            premier = (res, odom)
        for nom_est, est, tps in [("odometrie", odom, [0] * len(refs))] + \
                                 [(m, res[m][0], res[m][2]) for m in modes]:
            for k, (ep, ea) in enumerate(erreurs(est, refs)):
                lignes.append(dict(tirage=t, methode=nom_est, k=k, err_pos_m=ep, err_ang_deg=ea,
                                   temps_s=tps[k]))
    return pd.DataFrame(lignes), premier[0], premier[1]


def resume(res, seuil_pos=0.10, seuil_ang=2.0):
    """Tableau par méthode, sur les poses estimées (k >= 1) ; la dernière pose est celle qui a le plus dérivé."""
    r = res[res.k >= 1]
    ok = (r.err_pos_m < seuil_pos) & (r.err_ang_deg < seuil_ang)
    derniere = res[res.k == res.k.max()]
    t = pd.DataFrame({
        "err_pos_moy_cm": (r.groupby("methode").err_pos_m.mean() * 100).round(1),
        "err_ang_moy_deg": r.groupby("methode").err_ang_deg.mean().round(2),
        "err_pos_derniere_cm": (derniere.groupby("methode").err_pos_m.mean() * 100).round(1),
        "taux_ok_%": (ok.groupby(r.methode).mean() * 100).round(0),
        "temps_moy_s": r.groupby("methode").temps_s.mean().round(2),
    })
    return t.loc[[m for m in ["odometrie", "grid_sans_prior", "grid", "vecto"] if m in t.index]]


if __name__ == "__main__":
    n_tirages = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    ordre, scans, refs = charger_reel()
    print("Ordre des scans :", [o[4] for o in ordre], "| odométrie simulée, bruit par défaut")
    res, resultats, odom = lancer(scans, refs, n_tirages)
    res.to_csv("data/resultats_slam.csv", index=False)
    print(f"\nSLAM sur les 7 scans réels, {n_tirages} tirages d'odométrie bruitée "
          f"(référence = poses ajustées à la main) :")
    print(resume(res).to_string())

    figure_slam(ordre, refs, resultats, odom, f"{DOSSIER_FIGURES}/slam_reel.png")
    pd.DataFrame([dict(fichier=n, x_m=p[0], y_m=p[1], theta_deg=p[2])
                  for n, p in zip(ordre, resultats["vecto"][0])]).to_csv("data/slam_poses.csv", index=False)
    sauver_carte_vectorielle = sauver_carte_vecto(resultats["vecto"][1], "data/carte_salle_slam.txt")
    print(f"\nFigure : {DOSSIER_FIGURES}/slam_reel.png | poses : data/slam_poses.csv | "
          f"carte : data/carte_salle_slam.txt")
