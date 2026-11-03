import sys
import pandas as pd
import matplotlib.pyplot as plt

import gridmap
from gridmap import scan_local, local_vers_global
from localisation_grid import construire_grille, localiser_brute_force

test = sys.argv[1] if len(sys.argv) > 1 else "pos_B.csv"
poses = pd.read_csv("data/poses_ajustees.csv").set_index("fichier")
grille = construire_grille(poses, exclu=test)

df = pd.read_csv(f"data/{test}")
pts = scan_local(df[df.scan_id == df.scan_id.min()])
ref = tuple(poses.loc[test, ["x_m", "y_m", "theta_deg"]])
trouvee, _ = localiser_brute_force(grille, pts, ref)
print("référence :", ref)
print("trouvée   :", trouvee)

h = gridmap.TAILLE_M / 2
plt.imshow(grille, cmap="gray_r", vmin=0, vmax=255, origin="lower", extent=[-h, h, -h, h])
for pose, couleur, label in [(ref, "tab:blue", "référence (ajustée)"),
                             (trouvee, "tab:red", "pose trouvée")]:
    p = local_vers_global(pts, pose)
    plt.scatter(p[:, 0], p[:, 1], s=3, color=couleur, label=label)
plt.legend()
plt.axis("equal")
plt.show()