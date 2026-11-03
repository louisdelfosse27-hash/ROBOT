import numpy as np
import pandas as pd

from gridmap import scan_local
from localisation_grid import METHODES, construire_grille, ecart_angle, TESTS

rng = np.random.default_rng(0)       # graine fixe : résultats reproductibles
poses = pd.read_csv("data/poses_ajustees.csv").set_index("fichier")

# TOUS les scans sont dans la carte : la pose de chaque scan est donc cohérente avec
# la carte par construction. On connaît la "vérité" exacte, sans erreur d'ajustement manuel.
grille = construire_grille(poses, exclu=None)

lignes = []
for test in TESTS:
    df = pd.read_csv(f"data/{test}")
    pts = scan_local(df[df.scan_id == df.scan_id.min()])
    ref = tuple(poses.loc[test, ["x_m", "y_m", "theta_deg"]])
    for _ in range(10):
        # décalages aléatoires : la vérité n'est plus sur la grille de recherche
        dx, dy = rng.uniform(-0.3, 0.3, 2)
        dth = rng.uniform(-8, 8)
        init = (ref[0] + dx, ref[1] + dy, ref[2] + dth)
        for nom, methode in METHODES.items():
            pose, _ = methode(grille, pts, init)
            lignes.append(dict(methode=nom, fichier=test,
                               err_pos_m=float(np.hypot(pose[0] - ref[0], pose[1] - ref[1])),
                               err_ang_deg=float(abs(ecart_angle(pose[2], ref[2])))))

res = pd.DataFrame(lignes)
print(res.groupby("methode")[["err_pos_m", "err_ang_deg"]].agg(["mean", "max"]).round(3))
res.to_csv("data/resultats_verite_terrain.csv", index=False)