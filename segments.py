import math
from typing import NamedTuple, Optional

# ---- Tolérances numériques (à expliquer dans ton rendu) ----
EPS_PAR = 1e-9          # deux segments sont parallèles si |sin(angle)| <= EPS_PAR
EPS_T = 1e-9            # tolérance sur la position le long d'un segment (0 = début, 1 = fin)
TOL_LONGUEUR = 1e-6     # deux longueurs sont "égales" si elles diffèrent de moins que ça (en m)


class Segment(NamedTuple):
    p1: tuple   # (x, y)
    p2: tuple   # (x, y)


def longueur(s: Segment) -> float:
    return math.dist(s.p1, s.p2)


def _cross(a, b):
    """Produit vectoriel 2D (un nombre) : a.x*b.y - a.y*b.x"""
    return a[0] * b[1] - a[1] * b[0]


def intersection(s1: Segment, s2: Segment) -> Optional[tuple]:
    """
    Intersection de deux segments.
    Renvoie (point, t, u) ou None.
      point = (x, y)
      t = position sur s1 : point = s1.p1 + t * (s1.p2 - s1.p1), avec t dans [0, 1]
      u = idem sur s2
    None si parallèles (colinéaires inclus), de longueur nulle, ou si les segments ne se touchent pas.
    """
    r = (s1.p2[0] - s1.p1[0], s1.p2[1] - s1.p1[1])
    q = (s2.p2[0] - s2.p1[0], s2.p2[1] - s2.p1[1])
    denom = _cross(r, q)

    # Parallèles ? (comparaison relative aux longueurs, donc indépendante des unités)
    if abs(denom) <= EPS_PAR * math.hypot(*r) * math.hypot(*q):
        return None

    w = (s2.p1[0] - s1.p1[0], s2.p1[1] - s1.p1[1])
    t = _cross(w, q) / denom
    u = _cross(w, r) / denom

    if not (-EPS_T <= t <= 1 + EPS_T and -EPS_T <= u <= 1 + EPS_T):
        return None

    point = (s1.p1[0] + t * r[0], s1.p1[1] + t * r[1])
    return point, t, u


def couper(a: Segment, b: Segment) -> Segment:
    """
    Coupe le segment a avec le segment b et garde le plus long côté.
    Si aucune coupe valable, renvoie a inchangé.
    """
    res = intersection(a, b)
    if res is None:                       # pas d'intersection, ou colinéaires
        return a
    point, t, _ = res

    if not (EPS_T < t < 1 - EPS_T):       # intersection à une extrémité de a : rien ne change
        return a

    L = longueur(a)
    l_debut = t * L                       # côté contenant a.p1
    l_fin = (1 - t) * L                   # côté contenant a.p2

    # Égalité à la tolérance près -> on garde le côté du premier point
    if abs(l_debut - l_fin) <= TOL_LONGUEUR or l_debut > l_fin:
        return Segment(a.p1, point)
    return Segment(point, a.p2)

# ---- Seuils de proximité (en m et en degrés, à ajuster avec tes vrais scans) ----
SEUIL_ANGLE_DEG = 10.0   # écart d'orientation maximal
SEUIL_DIST = 0.05        # distance perpendiculaire maximale entre les deux segments
SEUIL_GAP = 0.10         # trou maximal le long du segment entre les deux


def _dir_unitaire(s: Segment):
    L = longueur(s)
    return ((s.p2[0] - s.p1[0]) / L, (s.p2[1] - s.p1[1]) / L)


def angle_entre(s1: Segment, s2: Segment) -> float:
    """
    Angle entre deux segments, en degrés, dans [0, 90].
    Le sens de parcours est ignoré : un mur scanné de gauche à droite ou de droite à gauche
    donne la même orientation.
    """
    d1 = (s1.p2[0] - s1.p1[0], s1.p2[1] - s1.p1[1])
    d2 = (s2.p2[0] - s2.p1[0], s2.p2[1] - s2.p1[1])
    return math.degrees(math.atan2(abs(_cross(d1, d2)), abs(d1[0] * d2[0] + d1[1] * d2[1])))


def sont_proches(s1: Segment, s2: Segment) -> bool:
    """Proches = angle faible ET proches en position (décalage perpendiculaire et trou longitudinal)."""
    if angle_entre(s1, s2) > SEUIL_ANGLE_DEG:
        return False

    u = _dir_unitaire(s1)
    n = (-u[1], u[0])                              # normale à s1
    L1 = longueur(s1)

    # Position des extrémités de s2 dans le repère de s1 : (le long, perpendiculaire)
    proj, perp = [], []
    for p in (s2.p1, s2.p2):
        v = (p[0] - s1.p1[0], p[1] - s1.p1[1])
        proj.append(v[0] * u[0] + v[1] * u[1])
        perp.append(abs(v[0] * n[0] + v[1] * n[1]))

    if max(perp) > SEUIL_DIST:
        return False

    gap = max(0.0, min(proj)) - min(L1, max(proj))   # <= 0 : ils se chevauchent
    return gap <= SEUIL_GAP


def fusionner(s1: Segment, s2: Segment) -> Segment:
    """
    Fusionne deux segments proches en un seul.
    Axe = direction moyenne pondérée par les longueurs, passant par le centre pondéré.
    Les 4 extrémités sont projetées sur cet axe ; on garde les deux plus éloignées.
    Le résultat est orienté dans le même sens que s1.
    """
    L1, L2 = longueur(s1), longueur(s2)
    d1, d2 = _dir_unitaire(s1), _dir_unitaire(s2)
    if d1[0] * d2[0] + d1[1] * d2[1] < 0:           # même sens avant de moyenner
        d2 = (-d2[0], -d2[1])

    ax = (L1 * d1[0] + L2 * d2[0], L1 * d1[1] + L2 * d2[1])
    norme = math.hypot(*ax)
    ax = (ax[0] / norme, ax[1] / norme)

    m1 = ((s1.p1[0] + s1.p2[0]) / 2, (s1.p1[1] + s1.p2[1]) / 2)
    m2 = ((s2.p1[0] + s2.p2[0]) / 2, (s2.p1[1] + s2.p2[1]) / 2)
    c = ((L1 * m1[0] + L2 * m2[0]) / (L1 + L2), (L1 * m1[1] + L2 * m2[1]) / (L1 + L2))

    ts = [(p[0] - c[0]) * ax[0] + (p[1] - c[1]) * ax[1] for p in (s1.p1, s1.p2, s2.p1, s2.p2)]
    tmin, tmax = min(ts), max(ts)
    return Segment((c[0] + tmin * ax[0], c[1] + tmin * ax[1]),
                   (c[0] + tmax * ax[0], c[1] + tmax * ax[1]))