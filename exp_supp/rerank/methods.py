"""Re-rankers on a fixed top-K shortlist, in score space (pure numpy; D148).

A `Block` is a set of query rows that share one gallery mask (segment protocol: the segments of one clip; track
protocol: all clips). Every method maps a Block to new scores R [r, Ng]; only shortlist entries of R are used. Scores
are cosines of the trained heads (or their 10-head mean). Gallery-side matrices:
  G [Ng, Ng]  gallery-gallery cosine (segment protocol: windows at their own duration; track: whole files)
  B [Nb, Ng]  bank-gallery scores, bank = the fold's TRAINING queries (never val / test queries)
Methods that need "other queries" (k-reciprocal, mutual kNN, dual softmax) take them from B, so nothing is
normalised by the test set.

Query expansion in score space: with unit vectors, <q + sum_j w_j g_j, g> = S[q, g] + sum_j w_j G[j, g]; the norm of
the expanded query is constant within a row, so ranking is unchanged. DBA: g' = sum of g's k nearest gallery items
(itself included), score <q, g'> / ||g'||, ||g'||^2 from G.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

K_SHORT = 20
NEG = -1e9


@dataclass
class Block:
    S: np.ndarray                 # [r, Ng] base scores (any value on invalid columns)
    valid: np.ndarray             # [Ng] bool, shared by the rows
    G: np.ndarray                 # [Ng, Ng]
    B: np.ndarray                 # [Nb, Ng]
    group: np.ndarray             # [Ng] song group of each column (segment: music file; track: track id)
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------ helpers
def masked(S: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.where(valid[None], S, NEG)


def topk_idx(S: np.ndarray, k: int) -> np.ndarray:
    """[r, k] column indices of the k largest entries per row (descending, stable for ties)."""
    k = min(k, S.shape[1])
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    o = np.argsort(-np.take_along_axis(S, idx, 1), axis=1, kind="stable")
    return np.take_along_axis(idx, o, 1)


def shortlist(S: np.ndarray, valid: np.ndarray, K: int | None = None) -> np.ndarray:
    """[r, Ng] bool: the base top-K valid columns of each row (K = None: module K_SHORT, read at call time)."""
    K = K_SHORT if K is None else K
    Sm = masked(S, valid)
    m = np.zeros(S.shape, bool)
    np.put_along_axis(m, topk_idx(Sm, min(K, int(valid.sum()))), True, 1)
    return m


def col_rank_vs_bank(B: np.ndarray, S: np.ndarray, Bs: np.ndarray | None = None) -> np.ndarray:
    """[r, Ng]: number of bank queries scoring column g strictly above the query (0 = query is g's best).
    Bs: B sorted ascending per column (optional, precomputed)."""
    Bs = np.sort(B, 0) if Bs is None else Bs
    out = np.empty(S.shape, np.int64)
    for g in range(S.shape[1]):
        out[:, g] = len(B) - np.searchsorted(Bs[:, g], S[:, g], side="right")
    return out


def bank_k(k: int, n_bank: int, n_gal: int) -> int:
    """Column-side neighbourhood size: k scaled from a gallery of n_gal to a bank of n_bank queries."""
    return max(1, int(round(k * n_bank / max(n_gal, 1))))


# ------------------------------------------------------------------ A. no training
def aqe(b: Block, k: int) -> np.ndarray:
    """Average query expansion: q' = q + sum of its k best gallery items (equal weights)."""
    Sm = masked(b.S, b.valid)
    nn = topk_idx(Sm, k)
    return b.S + b.G[nn].sum(1)


def alpha_qe(b: Block, k: int, alpha: float) -> np.ndarray:
    """alpha-QE (Radenovic et al. 2018): neighbours weighted by max(sim, 0)^alpha; the query keeps weight 1."""
    Sm = masked(b.S, b.valid)
    nn = topk_idx(Sm, k)
    w = np.maximum(np.take_along_axis(b.S, nn, 1), 0.0) ** alpha       # [r, k]
    return b.S + np.einsum("rk,rkg->rg", w, b.G[nn])


def dba(b: Block, k: int) -> np.ndarray:
    """Database-side augmentation: each gallery item replaced by the sum of its k nearest valid gallery items
    (itself included), re-normalised."""
    Gm = np.where(b.valid[None], b.G, NEG)
    np.fill_diagonal(Gm, np.inf)                                    # self first
    nn = topk_idx(Gm, k)                                            # [Ng, k]
    A = np.zeros_like(b.G)
    np.put_along_axis(A, nn, 1.0, 1)                                # A[g, j] = 1 if j in N(g)
    norm = np.sqrt(np.maximum(((A @ b.G) * A).sum(1), 1e-12))
    return (b.S @ A.T) / norm[None]


def k_reciprocal(b: Block, k1: int, k2: int, lam: float) -> np.ndarray:
    """k-reciprocal re-ranking (Zhong et al. 2017), cross-modal form. Query side: g is a reciprocal neighbour of q if
    g is in q's top-k1 and q is in g's top-k1' among {q} + bank queries (k1' = k1 scaled to the bank size). Gallery
    side: mutual k1-NN in G. Vectors weighted exp(-d), d = 2 - 2 cos; local expansion over k2 neighbours; final
    distance (1 - lam) d_Jaccard + lam d."""
    Ng = b.S.shape[1]
    nv = int(b.valid.sum())
    kk = min(k1, nv)
    Sm = masked(b.S, b.valid)
    inq = shortlist(b.S, b.valid, kk)                                # g in top-k1(q)
    colr = col_rank_vs_bank(b.B, b.S, b.extra.get("Bs"))
    rec_q = inq & (colr < bank_k(k1, len(b.B), nv)) & b.valid[None]
    Gm = np.where(b.valid[None] & b.valid[:, None], b.G, NEG)
    A = shortlist(Gm, b.valid, kk) & b.valid[:, None]                # A[g, g'] g' in top-k1(g)
    M = A & A.T
    Vg = np.where(M, np.exp(-(2 - 2 * b.G)), 0.0)
    Vq = np.where(rec_q, np.exp(-(2 - 2 * b.S)), 0.0)
    Vg /= np.maximum(Vg.sum(1, keepdims=True), 1e-12)
    Vq /= np.maximum(Vq.sum(1, keepdims=True), 1e-12)
    if k2 > 1:
        nnG = topk_idx(np.where(np.eye(Ng, dtype=bool), np.inf, Gm), k2)      # self + k2-1 neighbours
        Vg = Vg[nnG].mean(1)
        nnq = topk_idx(Sm, k2 - 1)
        Vq = (Vq + Vg[nnq].sum(1)) / k2
    sl = shortlist(b.S, b.valid)
    R = np.full(b.S.shape, NEG)
    for i in range(len(b.S)):
        cols = np.where(sl[i])[0]
        mn = np.minimum(Vq[i][None], Vg[cols]).sum(1)
        mx = np.maximum(Vq[i][None], Vg[cols]).sum(1)
        dj = 1 - mn / np.maximum(mx, 1e-12)
        R[i, cols] = -((1 - lam) * dj + lam * (2 - 2 * b.S[i, cols]))
    return R


def mutual_knn(b: Block, k: int) -> np.ndarray:
    """Keep mutual neighbours first: g in q's top-k and q in g's top-k' among {q} + bank (k' scaled); then by score."""
    nv = int(b.valid.sum())
    inq = shortlist(b.S, b.valid, min(k, nv))
    mut = inq & (col_rank_vs_bank(b.B, b.S, b.extra.get("Bs")) < bank_k(k, len(b.B), nv))
    return b.S + 1e3 * mut


def dual_softmax(b: Block, T: float) -> np.ndarray:
    """log of softmax over gallery (row) x softmax over queries (column, = {q} + bank). The row term is constant per
    row, so the ranking score is log(row) + log(col) + const = 2 S/T - logsumexp_col."""
    Z = np.concatenate([b.B[None].repeat(len(b.S), 0), b.S[:, None]], 1) / T       # [r, Nb+1, Ng]
    m = Z.max(1)
    lse = m + np.log(np.exp(Z - m[:, None]).sum(1))
    return 2 * b.S / T - lse


def pool(x: np.ndarray, how: str) -> float:
    if how == "max":
        return float(x.max())
    if how == "mean":
        return float(x.mean())
    return float(np.sort(x)[-3:].mean())                      # top3 (mean of all if fewer than 3)


def track_agg(b: Block, how: str) -> np.ndarray:
    """Track-level aggregation. Track protocol: score of a file = pool over its fixed windows (extra['W'] [r, Nw],
    extra['w_file'] [Nw]). Segment protocol: score of a window = its own score + pool over the valid gallery windows
    of the same music file (its song group)."""
    if "W" in b.extra:
        W, wf = b.extra["W"], b.extra["w_file"]
        R = np.full(b.S.shape, NEG)
        for f in range(b.S.shape[1]):
            sel = wf == f
            if sel.any():
                R[:, f] = [pool(W[i, sel], how) for i in range(len(W))]
        return R
    R = b.S.copy()
    for gid in np.unique(b.group[b.valid]):
        cols = np.where((b.group == gid) & b.valid)[0]
        R[:, cols] += np.array([pool(b.S[i, cols], how) for i in range(len(b.S))])[:, None]
    return R


# ------------------------------------------------------------------ B. information already there
def dedup(b: Block) -> np.ndarray:
    """Song-group de-duplication: inside the shortlist keep the best item of each group; later items of a group
    move below every group leader (their order kept)."""
    sl = shortlist(b.S, b.valid)
    R = b.S.copy()
    for i in range(len(b.S)):
        cols = np.where(sl[i])[0]
        cols = cols[np.argsort(-b.S[i, cols], kind="stable")]
        seen = set()
        for c in cols:
            if b.group[c] in seen:
                R[i, c] -= 1e3
            seen.add(b.group[c])
    return R


def key_dist(pc1, mode1, pc2, mode2) -> np.ndarray:
    """Circle-of-fifths distance in [0, 1] between keys; minor keys mapped to their relative major."""
    a = (np.asarray(pc1) + 3 * (1 - np.asarray(mode1))) % 12
    c = (np.asarray(pc2) + 3 * (1 - np.asarray(mode2))) % 12
    f = (7 * (a - c)) % 12
    return np.minimum(f, 12 - f) / 6.0


def tempo_dist(t1, t2) -> np.ndarray:
    """|log2 tempo ratio|, octave errors folded (x0.5 / x2), clipped to [0, 1]."""
    r = np.abs(np.log2(np.maximum(np.asarray(t1, float), 1) / np.maximum(np.asarray(t2, float), 1)))
    return np.clip(np.minimum(r, np.abs(r - 1)), 0, 1)


def zrows(S: np.ndarray, valid: np.ndarray) -> np.ndarray:
    v = S[:, valid]
    return (S - v.mean(1, keepdims=True)) / (v.std(1, keepdims=True) + 1e-8)


def keytempo(b: Block, w: float) -> np.ndarray:
    """Segment protocol. R = z(S) - w * mean over neighbouring segments n of Δ(candidate, top-1 of n), Δ = mean of
    key and tempo distance of the files. extra: 'nb' list of neighbour row-index arrays (into extra['S_all']),
    'S_all' [Nq, Ng] scores of every segment of the split, 'valid_all', 'col_file', 'key', 'mode', 'tempo' per file."""
    e = b.extra
    Z = zrows(b.S, b.valid)
    pen = np.zeros_like(Z)
    for i, nb in enumerate(e["nb"]):
        if len(nb) == 0:
            continue
        top = [int(np.argmax(np.where(e["valid_all"][j], e["S_all"][j], NEG))) for j in nb]
        fn = e["col_file"][top]
        fc = e["col_file"]
        d = 0.5 * key_dist(e["key"][fc][None], e["mode"][fc][None], e["key"][fn][:, None], e["mode"][fn][:, None]) \
            + 0.5 * tempo_dist(e["tempo"][fc][None], e["tempo"][fn][:, None])
        pen[i] = d.mean(0)
    return Z - w * pen


def seqctx(b: Block, w: float) -> np.ndarray:
    """Segment protocol. R = z(S) + w * mean over neighbouring segments n of z_n(file of the candidate), z_n(f) = the
    best z-scored window of file f for neighbour n (track-level evidence from the same video's other scenes)."""
    e = b.extra
    Z = zrows(b.S, b.valid)
    out = Z.copy()
    fc = e["col_file"]
    nf = int(fc.max()) + 1
    for i, nb in enumerate(e["nb"]):
        if len(nb) == 0:
            continue
        acc = np.zeros(nf)
        cnt = np.zeros(nf)
        for j in nb:
            vj = e["valid_all"][j] & b.valid
            zj = zrows(e["S_all"][j][None], e["valid_all"][j])[0]
            best = np.full(nf, -np.inf)
            np.maximum.at(best, fc[vj], zj[vj])
            ok = np.isfinite(best)
            acc[ok] += best[ok]
            cnt[ok] += 1
        ctx = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
        out[i] = Z[i] + w * ctx[fc]
    return out


# ------------------------------------------------------------------ final ranks on the shortlist
def final_ranks(S: np.ndarray, R: np.ndarray, valid: np.ndarray, hits: np.ndarray, K: int | None = None) -> np.ndarray:
    """Rank of the first hit after re-ranking the base top-K by R (ties in R broken by the base score S); items
    outside the shortlist keep their base order below it, so a hit outside the shortlist keeps its base rank.
    S, R, hits [r, Ng]; valid [r, Ng] or [Ng]. Base rule = strict > (rank_of_first_hit)."""
    K = K_SHORT if K is None else K
    valid = np.broadcast_to(valid, S.shape)
    out = np.empty(len(S), np.int64)
    for i in range(len(S)):
        s, rr, v, h = S[i], R[i], valid[i], hits[i] & valid[i]
        sm = np.where(v, s, -np.inf)
        best = sm[h].max()
        base = 1 + int(((sm > best) & ~h & v).sum())
        k = min(K, int(v.sum()))
        sl = np.zeros(len(s), bool)
        sl[np.argsort(-sm, kind="stable")[:k]] = True
        hs = h & sl
        if not hs.any():
            out[i] = base
            continue
        j = np.where(hs)[0]
        jb = j[np.lexsort((s[j], rr[j]))[-1]]                       # best hit by (R, S)
        above = sl & ~h & ((rr > rr[jb]) | ((rr == rr[jb]) & (s > s[jb])))
        out[i] = 1 + int(above.sum())
    return out


def summary(r: np.ndarray) -> dict:
    r = np.asarray(r)
    return {"R1": float((r <= 1).mean()), "R5": float((r <= 5).mean()), "R10": float((r <= 10).mean()),
            "MRR": float((1 / r).mean()), "MedR": float(np.median(r))}
