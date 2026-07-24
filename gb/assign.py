"""
GBAssigner — routing brain for the GB-Agent (Day 7).
Loads a fitted GB artifact and, for any input text, returns the nearest
granular ball plus an escalate flag (option 1: distance OR purity).
Grounded in the paper's own quantities: centroid distance and ball purity (Eq. 5).
"""
import pickle, numpy as np
import gb.fit  # noqa: needed to unpickle GranularBall


class GBAssigner:
    def __init__(self, artifact_path, tau, purity_floor=0.80):
        d = pickle.load(open(artifact_path, "rb"))
        self.C = d["centroids"]                    # (M, 768)
        self.labels = d["ball_labels"]
        self.purities = d["ball_purities"]
        self.sizes = np.array([b.size for b in d["balls"]])
        self.mu, self.sigma = d["mu"], d["sigma"]
        self.tau = tau
        self.purity_floor = purity_floor
        self.M = len(self.C)

    def _standardize(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mu) / self.sigma

    def assign(self, embedding):
        """embedding: raw ERNIE [CLS] vector (768,). Returns routing decision."""
        z = self._standardize(embedding).reshape(1, -1)
        dists = np.sqrt(((z - self.C) ** 2).sum(1))
        j = int(dists.argmin())
        dist = float(dists[j])
        purity = float(self.purities[j])

        far = dist > self.tau
        impure = purity < self.purity_floor
        escalate = bool(far or impure)

        return {
            "ball_id": j,
            "distance": round(dist, 3),
            "ball_label": int(self.labels[j]),
            "ball_purity": round(purity, 4),
            "ball_size": int(self.sizes[j]),
            "far_from_prototype": bool(far),
            "low_purity": bool(impure),
            "escalate": escalate,
            "reason": ("far_from_all_balls" if far and impure else
                       "far_from_all_balls" if far else
                       "nearest_ball_ambiguous" if impure else
                       "confident_in_distribution"),
        }

    def assign_batch(self, embeddings):
        return [self.assign(e) for e in np.asarray(embeddings)]
