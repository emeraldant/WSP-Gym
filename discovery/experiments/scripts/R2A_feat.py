"""R2A: generic candidate space + prompt-visible (weights/spec only) features + labels.

Candidate space (generic; no generator scale factors such as 0.3/0.4/2.5/3.0):
  scale row/col of any W and any bias with factor in {-1, 0.5, 2}
  swap two columns of a hidden-feeding matrix (W2..WL), swap two output units (row swap + bias swap, 2 ops)
  set bias to 0; add bias +-{0.5,1,2} x (mean |nonzero| of the neuron's incoming row)
  set one weight to 0, to -w, or to another distinct value found in the same row
  add +-{0.5,1} x median|W| to every entry of a column of W2..WL
Features use only weights, architecture and the spec text (family, n_in, n_out, kind, input domain).
Labels (training only) use forward passes: recovery >= 0.95 via env-identical accuracy.
"""
import numpy as np
from weightgym.edits import apply_edit

FAMS = ["dnf", "parity", "threshold", "compare", "minmax", "arith", "argmax", "count", "equality", "mux", "interval"]
KINDS = {"binary": 0, "class": 1, "reg": 2}
FACT = (-1.0, 0.5, 2.0)


def domain(task):
    hi = 1.0 if "binary" in task.input_desc else 9.0
    return np.zeros(task.n_in), np.full(task.n_in, hi)


def _nan():
    return np.nan


def _robust_z(v):
    v = np.asarray(v, float)
    if len(v) < 3:
        return np.zeros_like(v)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    return (v - med) / max(mad, 0.15)


def _cos(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-9 else 0.0


class NetInfo:
    """Per-network weight statistics (no forward passes)."""

    def __init__(self, net, task):
        self.net, self.task = net, task
        L = net.n_layers
        self.L = L
        lo, hi = domain(task)
        self.dom_w = float(hi[0] - lo[0]) if len(hi) else 1.0
        # interval arithmetic
        self.pre_lo, self.pre_hi = [], []
        a_lo, a_hi = lo, hi
        for l in range(L):
            W, b = net.Ws[l], net.bs[l]
            Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
            pl = Wp @ a_lo + Wn @ a_hi + b
            ph = Wp @ a_hi + Wn @ a_lo + b
            self.pre_lo.append(pl)
            self.pre_hi.append(ph)
            a_lo, a_hi = np.maximum(pl, 0), np.maximum(ph, 0)
        self.rn = [np.linalg.norm(W, axis=1) + 1e-9 for W in net.Ws]
        self.cn = [np.linalg.norm(W, axis=0) + 1e-9 for W in net.Ws]
        # linearised (relu ignored) input->output map, from weights only
        M = net.Ws[0]
        self.M_prefix = [M]
        for l in range(1, L):
            M = net.Ws[l] @ M
            self.M_prefix.append(M)
        self.M0 = M  # (n_out, n_in)
        # neuron stats for hidden layers l=0..L-2
        self.nf = []  # per layer: array (width, nfeat)
        self.cluster = []
        for l in range(L - 1):
            self.nf.append(self._neuron_feats(l))

    NF_NAMES = ["rn_rel", "rn_rank", "cn_rel", "cn_rank", "gain_z", "bias_rel", "row_sum_sgn", "row_pos", "row_nnz",
                "col_sum_sgn", "col_pos", "rowcol_sgn", "match_cos_row", "match_cos_col", "match_col_ratio",
                "cluster_cons", "iv_lo_rel", "iv_hi_rel", "dead", "always_on", "col_zero", "path_share", "path_sgn_vs_net"]

    def _neuron_feats(self, l):
        net = self.net
        W, b, Wn = net.Ws[l], net.bs[l], net.Ws[l + 1]
        w = W.shape[0]
        rn, cn = self.rn[l], self.cn[l + 1]
        gain = np.log(rn * cn)
        live = cn > 0.05
        gz = np.zeros(w)
        if live.sum() >= 3:
            gz[live] = _robust_z(gain[live])
        rrank = (np.argsort(np.argsort(rn)) / max(w - 1, 1))
        crank = (np.argsort(np.argsort(cn)) / max(w - 1, 1))
        Wun = W / rn[:, None]
        cos_rows = Wun @ Wun.T
        np.fill_diagonal(cos_rows, -2)
        pl, ph = self.pre_lo[l], self.pre_hi[l]
        scale = rn * max(self.dom_w, 1.0)
        # path share: contribution magnitude of neuron to net linearised output
        paths = np.array([np.abs(Wn[:, j]).sum() * np.abs(W[j]).sum() for j in range(w)])
        out = np.zeros((w, len(self.NF_NAMES)))
        tot_sgn = np.sign(self.M0.sum()) if self.M0.size else 0.0
        for j in range(w):
            k = int(np.argmax(cos_rows[j])) if w > 1 else j
            cr = float(cos_rows[j, k]) if w > 1 else 0.0
            cc = _cos(Wn[:, j], Wn[:, k]) if w > 1 else 0.0
            ratio = float(cn[j] / cn[k]) if w > 1 else 1.0
            psgn = np.sign(Wn[:, j].sum()) * np.sign(W[j].sum()) * tot_sgn
            out[j] = [rn[j] / rn.mean(), rrank[j], cn[j] / cn.mean(), crank[j], gz[j], b[j] / rn[j], np.sign(W[j].sum()),
                      (W[j] > 0).mean(), (np.abs(W[j]) > 1e-9).mean(), np.sign(Wn[:, j].sum()), (Wn[:, j] > 0).mean(),
                      np.sign(W[j].sum()) * np.sign(Wn[:, j].sum()), cr, cc, ratio, cr * cc,
                      pl[j] / scale[j], ph[j] / scale[j], float(ph[j] <= 0), float(pl[j] >= 0), float(cn[j] < 0.05),
                      paths[j] / paths.sum(), psgn]
        return out


def spec_feats(task, net):
    fam = [float(task.family == f) for f in FAMS]
    return fam + [float(FAMS.index(task.family)) if task.family in FAMS else -1.0, task.n_in, task.n_out, KINDS[task.kind],
                  net.n_layers, net.widths[1], net.widths[2] if net.n_layers > 2 else 0, net.n_weights, float("binary" in task.input_desc)]


SPEC_NAMES = [f"fam_{f}" for f in FAMS] + ["fam_id", "n_in", "n_out", "kind", "depth", "w1", "w2", "n_weights", "bool_in"]
OP_NAMES = ["op_code", "factor", "layer", "is_last", "axis", "n_ops"]
CAND_NAMES = ["entry_abs", "entry_z", "entry_dup", "row_distinct", "new_rel", "delta_rel", "line_rn_rel", "line_rank", "line_sum_sgn",
              "line_pos", "pair_cos", "pair_gap", "pair_gain_gain", "pair_idx_gap", "pair_eff_diag", "pair_dead_any", "pair_nf_min_gain",
              "pair_nf_max_gain", "pair_cons_min", "pair_zafter", "pair_zbefore", "unit_scale", "m0_diag"]
ALL_NEURON = ["nf_" + n for n in NetInfo.NF_NAMES]
NB = len(ALL_NEURON)
FEAT_NAMES = SPEC_NAMES + OP_NAMES + ["nb_" + n for n in NetInfo.NF_NAMES] + ["nb2_" + n for n in NetInfo.NF_NAMES] + CAND_NAMES


def _blank_c():
    return [np.nan] * len(CAND_NAMES)


def gen_candidates(net, task):
    """Return list of (ops, feature_row). Feature rows: spec | op | neuron A feats | neuron B feats | cand feats."""
    info = NetInfo(net, task)
    L = net.n_layers
    spec = spec_feats(task, net)
    out = []
    nan_n = [np.nan] * NB

    def neuron_of(l, i):
        return list(info.nf[l][i]) if 0 <= l < L - 1 else nan_n

    def add(ops, opc, factor, l, axis, na, nb, c):
        out.append((ops, spec + [opc, factor, l, float(l == L - 1), axis, len(ops)] + na + nb + c))

    for l in range(L):
        W, b = net.Ws[l], net.bs[l]
        rn, cn = info.rn[l], info.cn[l]
        wmed = float(np.median(np.abs(W[np.abs(W) > 1e-9]))) if np.any(np.abs(W) > 1e-9) else 1.0
        # ----- row scale
        for i in range(W.shape[0]):
            na = neuron_of(l, i)
            c = _blank_c()
            c[6] = rn[i] / rn.mean(); c[7] = np.argsort(np.argsort(rn))[i] / max(len(rn) - 1, 1)
            c[8] = np.sign(W[i].sum()); c[9] = (W[i] > 0).mean()
            if l == L - 1 and info.M0.shape[1] == info.M0.shape[0]:
                c[22] = info.M0[i, i] / (np.abs(info.M0).mean() + 1e-9)
            for f in FACT:
                add([{"op": "scale", "tensor": f"W{l+1}", "axis": "row", "index": i, "factor": f}], 0, f, l, 0, na, nan_n, c)
        # ----- col scale
        for j in range(W.shape[1]):
            na = neuron_of(l - 1, j) if l >= 1 else nan_n
            c = _blank_c()
            c[6] = cn[j] / cn.mean(); c[7] = np.argsort(np.argsort(cn))[j] / max(len(cn) - 1, 1)
            c[8] = np.sign(W[:, j].sum()); c[9] = (W[:, j] > 0).mean()
            for f in FACT:
                add([{"op": "scale", "tensor": f"W{l+1}", "axis": "col", "index": j, "factor": f}], 1, f, l, 1, na, nan_n, c)
        # ----- bias scale / set 0 / add
        for i in range(W.shape[0]):
            na = neuron_of(l, i)
            nzr = np.abs(W[i])[np.abs(W[i]) > 1e-9]
            unit = float(nzr.mean()) if len(nzr) else 1.0
            c = _blank_c(); c[21] = unit
            for f in FACT:
                add([{"op": "scale", "tensor": f"b{l+1}", "index": i, "factor": f}], 2, f, l, 2, na, nan_n, c)
            c = _blank_c(); c[21] = unit; c[4] = (0 - b[i]) / unit; c[5] = (0 - b[i]) / unit
            add([{"op": "set", "tensor": f"b{l+1}", "index": i, "value": 0.0}], 3, 0.0, l, 2, na, nan_n, c)
            for m in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
                c = _blank_c(); c[21] = unit; c[4] = (b[i] + m * unit) / unit; c[5] = m
                add([{"op": "add", "tensor": f"b{l+1}", "index": i, "value": round(m * unit, 4)}], 4, m, l, 2, na, nan_n, c)
        # ----- set weight
        for r in range(W.shape[0]):
            na = neuron_of(l, r)
            row = W[r]
            nz = np.abs(row)[np.abs(row) > 1e-9]
            rscale = float(np.median(nz)) if len(nz) else 1.0
            lm = np.log(np.maximum(np.abs(row), 1e-3))
            z = _robust_z(lm) if len(row) >= 3 else np.zeros(len(row))
            distinct = np.unique(np.round(row, 2))
            for cidx in range(W.shape[1]):
                w = float(row[cidx])
                dup = float(np.sum(np.abs(row - w) < 1e-9) - 1)
                base = _blank_c()
                base[0] = abs(w) / rscale; base[1] = z[cidx]; base[2] = dup; base[3] = len(distinct)
                base[6] = rn[r] / rn.mean()
                vals = [(0.0, 5), (-w, 6)] + [(float(v), 7) for v in distinct if abs(v - w) > 0.005 and abs(v) > 1e-9]
                seen = set()
                for v, opc in vals:
                    key = round(v, 3)
                    if key in seen or abs(v - w) < 1e-9:
                        continue
                    seen.add(key)
                    c = list(base); c[4] = v / rscale; c[5] = (v - w) / rscale
                    add([{"op": "set", "tensor": f"W{l+1}", "index": [r, cidx], "value": round(v, 4)}], opc, v, l, 3, na, nan_n, c)
        # ----- add ones*unit to a column of W2..WL
        if l >= 1:
            for j in range(W.shape[1]):
                na = neuron_of(l - 1, j)
                c = _blank_c(); c[21] = wmed
                for m in (-1.0, -0.5, 0.5, 1.0):
                    add([{"op": "add", "tensor": f"W{l+1}", "axis": "col", "index": j, "vector": [round(m * wmed, 4)] * W.shape[0]}],
                        8, m, l, 1, na, nan_n, c)
            # ----- swap columns (hidden units feeding W_l)
            nfl = info.nf[l - 1]
            gain_z = nfl[:, 4]
            for i in range(W.shape[1]):
                for j in range(i + 1, W.shape[1]):
                    a, bb = nfl[i], nfl[j]
                    c = _blank_c()
                    c[10] = _cos(W[:, i], W[:, j]); c[11] = abs(cn[i] - cn[j]) / cn.mean()
                    # gain z after swap: log(rn_i*cn_j), log(rn_j*cn_i)
                    rnl = info.rn[l - 1]
                    g = np.log(rnl * cn)
                    live = cn > 0.05
                    med = np.median(g[live]) if live.sum() else 0.0
                    mad = max(np.median(np.abs(g[live] - med)) * 1.4826, 0.15) if live.sum() else 0.15
                    za = (np.log(rnl[i] * cn[j]) - med) / mad; zb = (np.log(rnl[j] * cn[i]) - med) / mad
                    c[19] = abs(za) + abs(zb); c[20] = abs(gain_z[i]) + abs(gain_z[j])
                    c[12] = c[20] - c[19]
                    c[13] = (j - i) / W.shape[1]
                    c[15] = float(a[18] + bb[18] > 0)
                    c[16] = min(a[4], bb[4]); c[17] = max(a[4], bb[4])
                    c[18] = min(a[15], bb[15])
                    c[11] = abs(cn[i] - cn[j]) / cn.mean()
                    # order the pair so features are symmetric: lower-gain-z neuron first
                    p, q = (a, bb) if a[2] <= bb[2] else (bb, a)
                    add([{"op": "swap", "tensor": f"W{l+1}", "axis": "col", "i": i, "j": j}], 9, 0.0, l, 1, list(p), list(q), c)
    # ----- swap output units (compound: row swap + bias swap)
    n_out = net.Ws[-1].shape[0]
    if n_out >= 2:
        l = L - 1
        W = net.Ws[-1]
        M0 = info.M0
        for i in range(n_out):
            for j in range(i + 1, n_out):
                c = _blank_c()
                c[10] = _cos(W[i], W[j]); c[11] = abs(info.rn[l][i] - info.rn[l][j]) / info.rn[l].mean()
                c[13] = (j - i) / n_out
                if M0.shape[0] == M0.shape[1]:
                    diag = abs(M0[i, i]) + abs(M0[j, j]); off = abs(M0[i, j]) + abs(M0[j, i])
                    c[14] = (diag - off) / (np.abs(M0).mean() + 1e-9)
                    c[22] = (M0[i, i] + M0[j, j] - M0[i, j] - M0[j, i]) / (np.abs(M0).mean() + 1e-9)
                c[21] = abs(net.bs[l][i] - net.bs[l][j])
                add([{"op": "swap", "tensor": f"W{l+1}", "axis": "row", "i": i, "j": j}, {"op": "swap", "tensor": f"b{l+1}", "i": i, "j": j}],
                    10, 0.0, l, 0, nan_n, nan_n, c)
    return out, info


EP_NAMES = SPEC_NAMES + ["max_gain_z", "n_gain_out", "min_cluster_cons", "n_cons_neg", "n_dead", "n_always_on", "max_bias_rel",
                         "max_entry_z", "n_dup_entries", "frac_cn_small", "m0_abs", "n_cand",
                         "top1", "top2", "top3", "gap12", "ent", "sum_p", "max_p_hidden", "top1_pred_clean"]


def episode_static_feats(info, net, task):
    gz, cons, dead, aon, brel, small = [], [], 0, 0, [], []
    for nf in info.nf:
        gz += list(np.abs(nf[:, 4])); cons += list(nf[:, 15]); dead += int(nf[:, 18].sum()); aon += int(nf[:, 19].sum())
        brel += list(np.abs(nf[:, 5])); small += list(nf[:, 20])
    ez, dup = [], 0
    for W in net.Ws:
        for row in W:
            if len(row) >= 3:
                ez.append(np.abs(_robust_z(np.log(np.maximum(np.abs(row), 1e-3)))).max())
            dup += int(len(row) - len(np.unique(np.round(row, 2))))
    gz = np.array(gz) if gz else np.zeros(1); cons = np.array(cons) if cons else np.zeros(1)
    return spec_feats(task, net) + [gz.max(), float((gz > 3).sum()), cons.min(), float((cons < -0.3).sum()), dead, aon,
                                    max(brel) if brel else 0.0, max(ez) if ez else 0.0, dup, float(np.mean(small)) if small else 0.0,
                                    float(np.abs(info.M0).mean())]


def label_candidates(ep, cands):
    """Forward-pass labels: success iff recovery >= 0.95 (faulty); for clean nets success of an edit is 0 by definition."""
    y = np.zeros(len(cands), dtype=np.int8)
    if ep.is_clean:
        return y
    den = ep.acc_clean - ep.acc_faulty
    for k, (ops, _) in enumerate(cands):
        acc = ep.task.accuracy(apply_edit(ep.net, ops))
        y[k] = (acc - ep.acc_faulty) / den >= 0.95
    return y


def episode_rows(ep, with_labels=True):
    cands, info = gen_candidates(ep.net, ep.task)
    X = np.array([c[1] for c in cands], dtype=np.float32) if cands else np.zeros((0, len(FEAT_NAMES)), np.float32)
    y = label_candidates(ep, cands) if with_labels else np.zeros(len(cands), np.int8)
    ops = [c[0] for c in cands]
    return dict(seed=ep.seed, family=ep.family, fault=ep.fault, clean=ep.is_clean, depth=ep.net.n_layers - 1, X=X, y=y, ops=ops,
                ep_static=np.array(episode_static_feats(info, ep.net, ep.task), dtype=np.float32), variant=ep.variant,
                source=ep.source)
