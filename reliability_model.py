"""NumPy-only classifiers (no scipy / scikit-learn), so they run on machines where compiled
scientific libraries are blocked. Used for training (train_reliability.py) and for live inference."""
import json
from pathlib import Path

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class Imputer:
    def fit(self, X):
        with np.errstate(all="ignore"):
            med = np.nanmedian(X, axis=0)
        self.med = np.where(np.isnan(med), 0.0, med)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        r, c = np.where(np.isnan(X))
        X[r, c] = self.med[c]
        return X


class Logistic:
    kind = "logistic"

    def __init__(self, l2=1.0, iters=50):
        self.l2, self.iters = l2, iters

    def fit(self, X, y, w):
        self.imp = Imputer().fit(X)
        X = self.imp.transform(X)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        R = self.l2 * np.eye(Z.shape[1]); R[0, 0] = 0.0
        beta = np.zeros(Z.shape[1])
        for _ in range(self.iters):                      # Newton / IRLS steps
            p = _sigmoid(Z @ beta)
            g = Z.T @ (w * (p - y)) + R @ beta
            H = (Z.T * (w * p * (1 - p))) @ Z + R + 1e-8 * np.eye(Z.shape[1])
            step = np.linalg.solve(H, g)
            beta -= step
            if np.abs(step).max() < 1e-6:
                break
        self.beta = beta
        return self

    def predict_proba(self, X):
        X = self.imp.transform(X)
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        return _sigmoid(Z @ self.beta)

    def to_dict(self):
        return {"kind": self.kind, "med": self.imp.med.tolist(), "mu": self.mu.tolist(),
                "sd": self.sd.tolist(), "beta": self.beta.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls(); m.imp = Imputer(); m.imp.med = np.array(d["med"])
        m.mu, m.sd, m.beta = np.array(d["mu"]), np.array(d["sd"]), np.array(d["beta"])
        return m


class GBM:
    """Gradient-boosted decision trees (Newton boosting, logistic loss, histogram splits)."""
    kind = "gbm"

    def __init__(self, n_trees=120, depth=3, lr=0.1, nbins=64, min_leaf=30, l2=1.0):
        self.n_trees, self.depth, self.lr = n_trees, depth, lr
        self.nbins, self.min_leaf, self.l2 = nbins, min_leaf, l2

    def _bin(self, X):
        return np.column_stack([np.searchsorted(self.edges[j], X[:, j], side="left")
                                for j in range(X.shape[1])]).astype(np.int32)

    def _grow(self, B, g, h):
        N, D = B.shape
        d, NB, l2 = self.depth, self.nbins, self.l2
        n_int = 2 ** d - 1
        feat = np.zeros(n_int, int)
        thr = np.full(n_int, np.inf)        # inf = everything goes left (node not split)
        bthr = np.full(n_int, 10 ** 9)
        node = np.zeros(N, int)
        ar = np.arange(N)
        for level in range(d):
            first, nn = 2 ** level - 1, 2 ** level
            loc = node - first
            Gt, Ht = np.bincount(loc, g, nn), np.bincount(loc, h, nn)
            Ct = np.bincount(loc, minlength=nn)
            parent = Gt ** 2 / (Ht + l2)
            best, bf, bb = np.zeros(nn), np.full(nn, -1), np.zeros(nn, int)
            rows = np.arange(nn)
            for j in range(D):
                idx = loc * NB + B[:, j]
                G = np.bincount(idx, g, nn * NB).reshape(nn, NB)
                H = np.bincount(idx, h, nn * NB).reshape(nn, NB)
                C = np.bincount(idx, minlength=nn * NB).reshape(nn, NB)
                GL, HL, CL = (np.cumsum(a, 1)[:, :-1] for a in (G, H, C))
                GR, HR, CR = Gt[:, None] - GL, Ht[:, None] - HL, Ct[:, None] - CL
                gain = GL ** 2 / (HL + l2) + GR ** 2 / (HR + l2) - parent[:, None]
                gain[(CL < self.min_leaf) | (CR < self.min_leaf)] = -1.0
                k = gain.argmax(1)
                gk = gain[rows, k]
                better = gk > best
                best = np.where(better, gk, best)
                bf = np.where(better, j, bf)
                bb = np.where(better, k, bb)
            for m in range(nn):
                if bf[m] >= 0:
                    i = first + m
                    feat[i], bthr[i], thr[i] = bf[m], bb[m], self.edges[bf[m]][bb[m]]
            node = 2 * node + 1 + (B[ar, feat[node]] > bthr[node])
        leaf = node - n_int
        Gl, Hl = np.bincount(leaf, g, 2 ** d), np.bincount(leaf, h, 2 ** d)
        return feat, thr, -Gl / (Hl + l2), leaf

    def fit(self, X, y, w):
        self.imp = Imputer().fit(X)
        X = self.imp.transform(X)
        qs = np.linspace(0, 1, self.nbins + 1)[1:-1]
        self.edges = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
        B = self._bin(X)
        prior = (w * y).sum() / w.sum()
        self.base = float(np.log(prior / (1 - prior)))
        F = np.full(len(y), self.base)
        self.trees = []
        for _ in range(self.n_trees):
            p = _sigmoid(F)
            feat, thr, val, leaf = self._grow(B, w * (p - y), w * p * (1 - p) + 1e-6)
            F += self.lr * val[leaf]
            self.trees.append((feat, thr, val))
        return self

    def predict_proba(self, X):
        X = self.imp.transform(X)
        N = len(X); ar = np.arange(N)
        F = np.full(N, self.base)
        for feat, thr, val in self.trees:
            node = np.zeros(N, int)
            for _ in range(self.depth):
                node = 2 * node + 1 + (X[ar, feat[node]] > thr[node])
            F += self.lr * val[node - (2 ** self.depth - 1)]
        return _sigmoid(F)

    def to_dict(self):
        big = lambda t: np.where(np.isinf(t), 1e300, t).tolist()
        return {"kind": self.kind, "med": self.imp.med.tolist(), "base": self.base, "lr": self.lr,
                "depth": self.depth,
                "trees": [{"feat": f.tolist(), "thr": big(t), "val": v.tolist()} for f, t, v in self.trees]}

    @classmethod
    def from_dict(cls, d):
        m = cls(depth=d["depth"], lr=d["lr"]); m.imp = Imputer(); m.imp.med = np.array(d["med"])
        m.base = d["base"]
        m.trees = [(np.array(t["feat"], int), np.array(t["thr"], float), np.array(t["val"], float))
                   for t in d["trees"]]
        return m


def save_model(path, model, meta):
    Path(path).write_text(json.dumps({"meta": meta, "model": model.to_dict()}))


def load_model(path):
    d = json.loads(Path(path).read_text())
    cls = {"logistic": Logistic, "gbm": GBM}[d["model"]["kind"]]
    return cls.from_dict(d["model"]), d["meta"]
