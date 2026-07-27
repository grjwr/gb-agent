"""
Multi-dataset GB router (Option C, individual sets).
Loads all four dataset ball-sets, each with its own mu/sigma. For a claim,
finds the nearest prototype ACROSS all sets and routes on that. Uses each
dataset's clean individual balls instead of a degraded merged fit.
"""
import pickle, numpy as np
import gb.fit  # noqa: unpickle GranularBall

class MultiGBAssigner:
    def __init__(self, artifact_paths, taus, purity_floor=0.80):
        # artifact_paths: {name: path}; taus: {name: tau}
        self.sets = {}
        for name, path in artifact_paths.items():
            d = pickle.load(open(path, "rb"))
            self.sets[name] = {
                "C": d["centroids"], "labels": d["ball_labels"],
                "purities": d["ball_purities"], "mu": d["mu"], "sigma": d["sigma"],
                "tau": taus[name]}
        self.purity_floor = purity_floor

    def assign(self, embedding):
        emb = np.asarray(embedding, dtype=np.float64)
        best = None
        for name, s in self.sets.items():
            z = (emb - s["mu"]) / s["sigma"]
            dists = np.sqrt(((z - s["C"]) ** 2).sum(1))
            j = int(dists.argmin())
            # normalize distance by this set's tau so cross-set comparison is fair
            d_norm = dists[j] / s["tau"]
            cand = {"dataset": name, "ball_id": j, "distance": float(dists[j]),
                    "dist_norm": float(d_norm), "tau": s["tau"],
                    "ball_label": int(s["labels"][j]),
                    "ball_purity": float(s["purities"][j])}
            if best is None or d_norm < best["dist_norm"]:
                best = cand
        # route: escalate if nearest-across-all is beyond its tau OR impure
        far = best["dist_norm"] > 1.0
        impure = best["ball_purity"] < self.purity_floor
        best["far_from_prototype"] = far
        best["low_purity"] = impure
        best["escalate"] = bool(far or impure)
        best["reason"] = ("far_from_all_balls" if far else
                          "nearest_ball_ambiguous" if impure else
                          "confident_in_distribution")
        return best
