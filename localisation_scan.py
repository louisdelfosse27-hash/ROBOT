import time
import numpy as np
import pandas as pd

from gridmap import nouvelle_grille, scan_local, mettre_a_jour
from localisation_grid import (METHODES, ecart_angle, DECALAGES, SEUIL_POS, SEUIL_ANG)

# (scan de référence, scan à localiser) : deux positions PROCHES, avec murs en commun.
PAIRES = [("pos_G.csv", "pos_B.csv"),
          ("pos_D.csv", "pos_F.csv"),
          ("pos_A.csv", "pos_G.csv"),
          ("pos_B.csv", "pos_D.csv")]


def grille_du_scan(nom, pose):
    """Mini grid map construite avec un seul fichier (tous ses tours), posé à `pose`."""
    grille = nouvelle_grille()
    df = pd.read_csv(f"data/{nom}")
    for sid in sorted(df.scan_id.unique()):
        mettre_a_jour(grille, scan_local(df[df.scan_id == sid]), pose)
    return grille


if __name__ == "__main__":
    poses = pd.read_csv("data/poses_ajustees.csv").set_index("fichier")

    lignes = []
    for ref_nom, test_nom in PAIRES:
        pose_ref = tuple(poses.loc[ref_nom, ["x_m", "y_m", "theta_deg"]])
        pose_test = tuple(poses.loc[test_nom, ["x_m", "y_m", "theta_deg"]])
        grille = grille_du_scan(ref_nom, pose_ref)

        df = pd.read_csv(f"data/{test_nom}")
        pts = scan_local(df[df.scan_id == df.scan_id.min()])
        print(f"\n{test_nom} localisé sur {ref_nom}")

        for dx, dy, dth in DECALAGES:
            init = (pose_test[0] + dx, pose_test[1] + dy, pose_test[2] + dth)
            for nom_methode, methode in METHODES.items():
                t0 = time.perf_counter()
                pose, score = methode(grille, pts, init)
                dt = time.perf_counter() - t0
                e_pos = float(np.hypot(pose[0] - pose_test[0], pose[1] - pose_test[1]))
                e_ang = float(abs(ecart_angle(pose[2], pose_test[2])))
                lignes.append(dict(methode=nom_methode, ref=ref_nom, test=test_nom,
                                   dx=dx, dy=dy, dth=dth,
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
    res.to_csv("data/resultats_scan_scan.csv", index=False)