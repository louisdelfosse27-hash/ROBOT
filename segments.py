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