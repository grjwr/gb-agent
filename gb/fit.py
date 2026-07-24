"""
Granular Ball fitting — GB-FFN paper (Gurjwar et al., Applied Soft Computing 2026), Sec 3.4.
Recursive 2-means splitting until purity >= theta or a stopping condition fires.
"""
import numpy as np
from sklearn.cluster import KMeans
from dataclasses import dataclass, field


@dataclass
class GBConfig:
    theta: float = 0.95          # purity threshold (paper: 0.95)
    min_sample_size: int = None  # None -> adaptive max(40, N/100), paper Eq.17
    max_depth: int = 50
    max_balls: int = 20000
    seed: int = 42


@dataclass
class GranularBall:
    ball_id: int
    centroid: np.ndarray
    label: int
    purity: float
    size: int
    member_idx: np.ndarray = field(repr=False)
    depth: int = 0


def _purity(y):
    """Eq.5: fraction of the majority class."""
    if len(y) == 0:
        return 0.0, 0
    counts = np.bincount(y, minlength=2)
    return counts.max() / len(y), int(counts.argmax())


def fit_granular_balls(X, y, cfg: GBConfig = GBConfig()):
    """X: (N,d) standardized embeddings. y: (N,) binary. Returns list[GranularBall]."""
    N = len(X)
    min_size = cfg.min_sample_size if cfg.min_sample_size else max(40, N // 100)

    balls, stack, next_id = [], [(np.arange(N), 0)], 0
    stop_reason = {"purity": 0, "min_size": 0, "depth": 0, "max_balls": 0, "no_variance": 0}

    while stack:
        idx, depth = stack.pop()
        y_sub = y[idx]
        pur, maj = _purity(y_sub)

        reason = None
        if pur >= cfg.theta:
            reason = "purity"
        elif len(idx) < min_size:
            reason = "min_size"
        elif depth >= cfg.max_depth:
            reason = "depth"
        elif len(balls) + len(stack) + 1 >= cfg.max_balls:
            reason = "max_balls"

        if reason is None:
            X_sub = X[idx]
            if np.allclose(X_sub.std(axis=0).sum(), 0):
                reason = "no_variance"
            else:
                km = KMeans(n_clusters=2, init="k-means++",
                            n_init=10, random_state=cfg.seed).fit(X_sub)
                a, b = idx[km.labels_ == 0], idx[km.labels_ == 1]
                if len(a) == 0 or len(b) == 0:
                    reason = "no_variance"
                else:
                    stack.extend([(a, depth + 1), (b, depth + 1)])
                    continue

        stop_reason[reason] += 1
        balls.append(GranularBall(
            ball_id=next_id, centroid=X[idx].mean(axis=0),   # Eq.8
            label=maj, purity=pur, size=len(idx),            # Eq.9
            member_idx=idx, depth=depth))
        next_id += 1

    return balls, {"min_sample_size": min_size, "n_balls": len(balls),
                   "compression": N / len(balls), "stop_reasons": stop_reason}
