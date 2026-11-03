import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from segments import Segment, longueur, couper, sont_proches, fusionner
from extraction import extraire_segments
from gridmap import scan_local, local_vers_global

# ---- Paramètres (à justifier dans ton rendu) ----
MARGE_RAYON = 0.10         # m : le rayon s'arrête 10 cm avant l'impact (sinon il couperait son propre mur)
MIN_LONGUEUR_CARTE = 0.20  # m : features plus courtes supprimées au nettoyage
SORTIE = "data/carte_vectorielle.txt"


@dataclass
class Feature:
    seg: Segment
    confiance: int = 1     # nombre de scans qui ont vu cette feature
    derniere_obs: int = 0  # numéro du dernier scan qui l'a vue


# ---------- Étape 2/3 : scan -> segments dans le repère global ----------
def vers_global(seg, pose):
    p = local_vers_global(np.array([seg.p1, seg.p2]), pose)
    return Segment(tuple(p[0]), tuple(p[1]))


# ---------- Étape 5 : absence de détection ----------
def retirer_absences(carte, pts_locaux, pose):
    """Un rayon robot -> point traverse une feature : celle-ci est coupée (on garde le plus long côté)."""
    robot = np.array(pose[:2])
    for p in local_vers_global(pts_locaux, pose):
        v = p - robot
        d = np.linalg.norm(v)
        if d <= MARGE_RAYON:
            continue
        fin = robot + v * (d - MARGE_RAYON) / d
        rayon = Segment(tuple(robot), tuple(fin))
        for f in carte:
            f.seg = couper(f.seg, rayon)


# ---------- Étape 4 : ajout / fusion ----------
def ajouter(carte, seg, num_scan):
    for f in carte:
        if sont_proches(f.seg, seg):
            f.seg = fusionner(f.seg, seg)
            f.confiance += 1
            f.derniere_obs = num_scan
            return
    carte.append(Feature(seg, 1, num_scan))


# ---------- Étape 6 : nettoyage ----------
def nettoyer(carte):
    carte = [f for f in carte if longueur(f.seg) >= MIN_LONGUEUR_CARTE]
    fusion_faite = True
    while fusion_faite:
        fusion_faite = False
        for i in range(len(carte)):
            for j in range(i + 1, len(carte)):
                if sont_proches(carte[i].seg, carte[j].seg):
                    carte[i].seg = fusionner(carte[i].seg, carte[j].seg)
                    carte[i].confiance += carte[j].confiance
                    del carte[j]
                    fusion_faite = True
                    break
            if fusion_faite:
                break
    return carte


def sauver(carte, chemin):
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    with open(chemin, "w") as f:
        f.write("# Carte vectorielle (segments) : x1,y1,x2,y2\n")
        for ft in carte:
            f.write(f"{ft.seg.p1[0]:.3f},{ft.seg.p1[1]:.3f},{ft.seg.p2[0]:.3f},{ft.seg.p2[1]:.3f}\n")


if __name__ == "__main__":
    poses = pd.read_csv("data/poses_ajustees.csv")
    carte, tous_les_points = [], []

    for num, r in poses.iterrows():
        pose = (r.x_m, r.y_m, r.theta_deg)
        df = pd.read_csv(f"data/{r.fichier}")
        pts = scan_local(df[df.scan_id == df.scan_id.min()])        # 1er tour, comme dans viewer.py
        tous_les_points.append(local_vers_global(pts, pose))

        retirer_absences(carte, pts, pose)                           # étape 5
        nouveaux = [vers_global(s, pose) for s in extraire_segments(pts)]   # étapes 2-3
        for s in nouveaux:
            ajouter(carte, s, num)                                   # étape 4
        print(f"{r.fichier} : {len(nouveaux)} segments extraits -> carte à {len(carte)} features")

    carte = nettoyer(carte)                                          # étape 6
    print(f"Carte finale : {len(carte)} features")
    sauver(carte, SORTIE)

    pts_all = np.vstack(tous_les_points)
    plt.scatter(pts_all[:, 0], pts_all[:, 1], s=2, color="lightgray", label="points des scans")
    for k, ft in enumerate(carte):
        plt.plot([ft.seg.p1[0], ft.seg.p2[0]], [ft.seg.p1[1], ft.seg.p2[1]], "r-", lw=2,
                 label="carte vectorielle" if k == 0 else None)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Carte vectorielle")
    plt.show()