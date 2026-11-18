"""Forward-pass-free template-alignment solver (R1C / H3).
Uses only: shown weights (net) and the spec-derived canonical net. Never evaluates any network on inputs.

Per hidden layer: (1) cluster shown neurons whose (incoming row, bias) agree up to a positive scale (split copies,
cancel pairs); (2) Hungarian-match clusters to canonical neurons (tolerant row cost + outgoing-column direction);
(3) collapse group columns into canonical coordinates; (4) residuals: row anomalies (flip / bias / entry),
column residuals (per canonical group, from the next layer's matched rows), cancel-cluster residuals (sum != 0),
output-row swap; (5) 'stray' test: a neuron whose row deviates (flip/bias/rowscale) from a cluster's template and
whose inclusion cancels the residual is blamed instead; (6) pick strongest anomaly, emit <=3 DSL ops."""
import numpy as np
from scipy.optimize import linear_sum_assignment

P = dict(TOLW=0.07, TOLB=0.09, CAP=4.0, LAM=3.0, ATTACH_RC=2.0, PAIR_RC=2.0, EMIT=0.10, BIAS_MIN=0.3, COLSEL=0.35,
         OUTL=1.5, REVIVE=0.5, STRAY_RC=6.0, STRAY_FRAC=0.4)


def fit_row(u, b, t, bt, P=P, allow_neg=True):
    nz = np.abs(t) > 1e-9
    cands = []
    if nz.any():
        cands += list(u[nz] / t[nz])
        cands.append(float(u @ t) / float(t @ t))
    if abs(bt) > P["BIAS_MIN"] and abs(b / bt) > 0.05:
        cands.append(b / bt)
    best = (1e9, 1.0, None, None)
    seen = set()
    for c in cands:
        if abs(c) < 0.05 or abs(c) > 20 or (c < 0 and not allow_neg):
            continue
        key = round(c, 3)
        if key in seen:
            continue
        seen.add(key)
        s = abs(c)
        r = u - c * t
        rb = b - s * bt
        cost = np.minimum((np.abs(r) / P["TOLW"]) ** 2, P["CAP"]).sum() + min((abs(rb) / P["TOLB"]) ** 2, P["CAP"]) + (0.5 if c < 0 else 0.0)
        if cost < best[0]:
            best = (cost, c, r, rb)
    if best[2] is None:
        return 1e3, 1.0, u.copy(), b
    return best


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(abs(a @ b) / (na * nb))


def _components(M, adj):
    parent = list(range(M))

    def f(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in range(M):
        for j in range(i + 1, M):
            if adj[i, j]:
                parent[f(i)] = f(j)
    cl = {}
    for i in range(M):
        cl.setdefault(f(i), []).append(i)
    return list(cl.values())


def stray_options(j, A, b, tmpl_w, tmpl_b, P):
    """Row hypotheses under which neuron j is a deviant copy of the template row (tmpl_w, tmpl_b).
    Returns list of (kind, scale c, ops_builder_args). c is the scale of j relative to the template."""
    rc, c, r, rb = fit_row(A[j], b[j], tmpl_w, tmpl_b, P)
    opts = []
    if rc > P["STRAY_RC"]:
        return opts, rc
    out = np.flatnonzero(np.abs(r) > P["TOLW"] * P["OUTL"])
    bout = abs(rb) > P["TOLB"] * P["OUTL"]
    if c < 0:
        opts.append(("flip", abs(c)))
    elif bout and len(out) == 0:
        opts.append(("bias", abs(c)))  # scale from weights
        if abs(tmpl_b) > P["BIAS_MIN"] and abs(b[j] / tmpl_b) > 0.05:
            opts.append(("rowscale", abs(b[j] / tmpl_b)))  # scale from bias
    elif len(out) == 1 and not bout:
        opts.append(("entry", abs(c), int(out[0])))
    elif len(out) == 0 and not bout:
        opts.append(("plain", abs(c)))
    return opts, rc


def _row_ops(kind, j, l, c, tmpl_w, tmpl_b, c_w, e=None):
    t = f"W{l+1}"
    if kind == "flip":
        return [{"op": "scale", "tensor": t, "axis": "row", "index": int(j), "factor": -1.0}]
    if kind == "bias":
        return [{"op": "set", "tensor": f"b{l+1}", "index": int(j), "value": float(round(c * tmpl_b, 2))}]
    if kind == "rowscale":  # weights should be scaled by c_b / c_w
        return [{"op": "scale", "tensor": t, "axis": "row", "index": int(j), "factor": float(round(c / c_w, 4))}]
    if kind == "entry":
        return [{"op": "set", "tensor": t, "index": [int(j), int(e)], "value": float(round(c * tmpl_w[e], 2))}]
    return []


def pre_ubs(net, hi):
    """Interval upper bounds of every hidden pre-activation (weights-only range analysis, input box [0,hi])."""
    U = np.full(net.Ws[0].shape[1], float(hi))
    out = []
    for l in range(net.n_layers - 1):
        W = net.Ws[l]
        ub = net.bs[l] + np.clip(W, 0, None) @ U
        out.append(ub)
        U = np.maximum(ub, 0.0)
    return out


def analyze(net, canon, hi=1, P=P):
    UB = pre_ubs(net, hi)
    L = net.n_layers
    anomalies = []
    groups_prev = None
    A = net.Ws[0].copy()
    for l in range(L):
        Wc, bc = canon.Ws[l], canon.bs[l]
        b = net.bs[l]
        m = Wc.shape[0]
        if l > 0:
            Wl = net.Ws[l]
            A = np.zeros((Wl.shape[0], len(groups_prev)))
            for g, mem in enumerate(groups_prev):
                for p, c in mem:
                    A[:, g] += Wl[:, p] * abs(c)
        M = A.shape[0]
        if l == L - 1:
            _output_layer(net, canon, l, A, anomalies, groups_prev, P)
            break
        Wn, Cn = net.Ws[l + 1], canon.Ws[l + 1]
        # ---- clusters
        adj = np.zeros((M, M), dtype=bool)
        rel = np.ones((M, M))
        for p in range(M):
            for q in range(p + 1, M):
                rc, c, r, rb = fit_row(A[p], b[p], A[q], b[q], P, allow_neg=False)
                if rc <= P["PAIR_RC"] and c > 0:
                    adj[p, q] = adj[q, p] = True
                    rel[p, q] = c; rel[q, p] = 1.0 / c
        clusters = _components(M, adj)
        # relative scales to cluster rep (first member): s[p] with row_p ~ s[p]*row_rep
        reps, relscale = [], {}
        for cl in clusters:
            rep = cl[0]
            reps.append(rep)
            for p in cl:
                relscale[p] = 1.0 if p == rep else float(rel[p, rep])
        K = len(clusters)
        cost = np.zeros((K, m)); fitc = {}
        for k, cl in enumerate(clusters):
            rep = reps[k]
            ssum = sum(relscale[p] * Wn[:, p] for p in cl)
            for g in range(m):
                rc, c, r, rb = fit_row(A[rep], b[rep], Wc[g], bc[g], P)
                cc = (1.0 - _cos(ssum, Cn[:, g])) if Wn.shape[0] == Cn.shape[0] else 0.0
                cost[k, g] = rc + P["LAM"] * cc
                fitc[(k, g)] = (c, r, rb, rc)
        rk, cg = linear_sum_assignment(cost)
        assigned = {int(g): int(k) for k, g in zip(rk, cg)}
        used = set(assigned.values())
        groups = []
        row_anoms_candidates = []
        for g in range(m):
            k = assigned[g]
            c_rep, r, rb, rc = fitc[(k, g)]
            mem = []
            for p in clusters[k]:
                mem.append((p, abs(c_rep) * relscale[p]))
            groups.append(mem)
            rep = reps[k]
            s = abs(c_rep)
            if len(clusters[k]) >= 1:
                if c_rep < 0:
                    anomalies.append(dict(strength=2.0, kind="flip_in", layer=l, ops=[{"op": "scale", "tensor": f"W{l+1}", "axis": "row", "index": int(rep), "factor": -1.0}]))
                if abs(rb) > P["TOLB"] * P["OUTL"]:
                    anomalies.append(dict(strength=abs(rb) / s, kind="bias", layer=l, j=rep, g=g, c=s, ops=[{"op": "set", "tensor": f"b{l+1}", "index": int(rep), "value": float(round(s * bc[g], 2))}]))
                out = np.flatnonzero(np.abs(r) > P["TOLW"] * P["OUTL"])
                if l == 0 and 1 <= len(out) <= 2 and c_rep > 0:
                    e = int(out[np.argmax(np.abs(r[out]))])
                    anomalies.append(dict(strength=abs(r[e]) / s, kind="entry_in", layer=l, ops=[{"op": "set", "tensor": f"W{l+1}", "index": [int(rep), e], "value": float(round(s * Wc[g][e], 2))}]))
        strays = [p for k, cl in enumerate(clusters) if k not in used and len(cl) == 1 for p in cl]
        if l == 0:
            for j in strays:
                if UB[0][j] > P["REVIVE"]:
                    anomalies.append(dict(strength=float(UB[0][j]) / 2.0 + 0.5, kind="revived", layer=0,
                                          ops=[{"op": "set", "tensor": "b1", "index": int(j), "value": float(round(b[j] - UB[0][j] - 1.0, 2))}]))
        # ---- D clusters (size>=2): sum must vanish
        for k, cl in enumerate(clusters):
            if k in used or len(cl) < 2:
                continue
            ssum = sum(relscale[p] * Wn[:, p] for p in cl)
            if all(UB[l][p] <= 0.0 for p in cl):
                continue  # certainly dead cluster: harmless whatever its columns
            if (np.abs(ssum) > P["TOLW"] * P["OUTL"]).any():
                st = float(np.linalg.norm(ssum))
                rep = reps[k]
                done = False
                best = None
                for j in strays:
                    opts, rc = stray_options(j, A, b, A[rep], b[rep], P)
                    for o in opts:
                        if o[0] == "plain":
                            continue
                        res = ssum + o[1] * Wn[:, j]
                        if np.linalg.norm(res) < P["STRAY_FRAC"] * st and (best is None or np.linalg.norm(res) < best[0]):
                            c_w = fit_row(A[j], b[j], A[rep], b[rep], P)[1]
                            best = (np.linalg.norm(res), o, j, c_w)
                if best is not None:
                    _, o, j, c_w = best
                    anomalies.append(dict(strength=st, kind="stray", layer=l, ops=_row_ops(o[0], j, l, o[1], A[rep], b[rep], abs(c_w), o[2] if len(o) > 2 else None)))
                else:
                    p0 = cl[0]
                    v = np.where(np.abs(ssum) > 0.02, -ssum / relscale[p0], 0.0)
                    anomalies.append(dict(strength=st, kind="pair", layer=l + 1, ops=[{"op": "add", "tensor": f"W{l+2}", "axis": "col", "index": int(p0), "vector": [float(round(x, 2)) for x in v]}]))
        # ---- column residuals of canonical groups at next layer
        An = np.zeros((Wn.shape[0], len(groups)))
        for g, mem in enumerate(groups):
            for p, c in mem:
                An[:, g] += Wn[:, p] * abs(c)
        if l + 1 < L - 1:
            Wc2, bc2 = canon.Ws[l + 1], canon.bs[l + 1]
            M2, m2 = An.shape[0], Wc2.shape[0]
            cost2 = np.zeros((M2, m2)); inf2 = {}
            Wn2, Cn2 = net.Ws[l + 2], canon.Ws[l + 2]
            for j in range(M2):
                for g in range(m2):
                    rc, c, r, rb = fit_row(An[j], net.bs[l + 1][j], Wc2[g], bc2[g], P)
                    cc = (1.0 - _cos(Wn2[:, j], Cn2[:, g])) if Wn2.shape[0] == Cn2.shape[0] else 0.0
                    cost2[j, g] = rc + P["LAM"] * cc
                    inf2[(j, g)] = (c, r, rb)
            rj2, cg2 = linear_sum_assignment(cost2)
            fits = {int(g): inf2[(int(j), int(g))] for j, g in zip(rj2, cg2)}
            jmap = {int(g): int(j) for j, g in zip(rj2, cg2)}
        else:
            fits, jmap = None, None
        if fits is not None:
            _col_anoms(net, l + 1, fits, jmap, groups, strays, A, b, l, Wc, bc, anomalies, P)
        groups_prev = groups
        if l + 1 == L - 1:
            pass
    ops = _select(anomalies, P)
    return ops, anomalies


def _output_layer(net, canon, l, A, anomalies, groups_prev, P):
    Wc, bc, b = canon.Ws[l], canon.bs[l], net.bs[l]
    m = Wc.shape[0]
    fits = {g: (1.0, A[g] - Wc[g], b[g] - bc[g]) for g in range(m)}
    bad = [g for g in fits if (np.abs(fits[g][1]) > P["TOLW"] * P["OUTL"]).any() or abs(fits[g][2]) > P["TOLB"] * P["OUTL"]]
    swapped = False
    if len(bad) >= 2:
        full = sum(np.abs(fits[g][1]).sum() + abs(fits[g][2]) for g in range(m))
        bestp = None
        for i in range(m):
            for j in range(i + 1, m):
                tot = (np.abs(A[i] - Wc[j]).sum() + abs(b[i] - bc[j]) + np.abs(A[j] - Wc[i]).sum() + abs(b[j] - bc[i]) +
                       sum(np.abs(fits[g][1]).sum() + abs(fits[g][2]) for g in range(m) if g not in (i, j)))
                if bestp is None or tot < bestp[0]:
                    bestp = (tot, i, j)
        if bestp and bestp[0] < 0.35 * full:
            _, i, j = bestp
            anomalies.append(dict(strength=float(full - bestp[0]), kind="label_swap", layer=l,
                                  ops=[{"op": "swap", "tensor": f"W{l+1}", "axis": "row", "i": i, "j": j}, {"op": "swap", "tensor": f"b{l+1}", "i": i, "j": j}]))
            swapped = True
    if not swapped:
        _col_anoms(net, l, fits, {g: g for g in range(m)}, groups_prev, [], None, None, l - 1, None, None, anomalies, P)


def _col_anoms(net, l, fits, jmap, groups, strays, Aprev, bprev, lprev, Wc_prev, bc_prev, anomalies, P):
    """Column residuals of canonical groups of the previous hidden layer, at weight matrix net.Ws[l]."""
    W = net.Ws[l]
    cols = {}
    for g, (c, R, rb) in fits.items():
        for e in np.flatnonzero(np.abs(R) > P["TOLW"] * P["OUTL"]):
            cols.setdefault(int(e), []).append(g)
    for e, glist in cols.items():
        mem = groups[e]
        if not mem:
            continue
        resid = np.array([fits[g][1][e] for g in glist])
        rows = [jmap[g] for g in glist]
        st = float(np.linalg.norm(resid))
        # stray test (only when previous layer info is available)
        blamed = None
        if strays and Aprev is not None:
            best = None
            for j in strays:
                opts, rc = stray_options(j, Aprev, bprev, Wc_prev[e], bc_prev[e], P)
                c_w = fit_row(Aprev[j], bprev[j], Wc_prev[e], bc_prev[e], P)[1]
                for o in opts:
                    if o[0] == "plain":
                        continue
                    # R (shown units, next row scale folded in) vs contribution of j: W[rows, j] * c
                    allrows = np.array([fits[g][1][e] for g in glist])
                    res = allrows + o[1] * W[rows, j]
                    if np.linalg.norm(res) < P["STRAY_FRAC"] * st and (best is None or np.linalg.norm(res) < best[0]):
                        best = (np.linalg.norm(res), o, j, abs(c_w))
            if best is not None:
                _, o, j, c_w = best
                anomalies.append(dict(strength=st, kind="stray", layer=l, ops=_row_ops(o[0], j, lprev, o[1], Wc_prev[e], bc_prev[e], c_w, o[2] if len(o) > 2 else None)))
                continue
        p, cp = max(mem, key=lambda t: abs(t[1]))
        cp = abs(cp)
        if len(glist) == 1:
            j = rows[0]
            old = float(W[j, p])
            ops = [{"op": "set", "tensor": f"W{l+1}", "index": [int(j), int(p)], "value": float(round(old - resid[0] / cp, 2))}]
        else:
            vec = np.zeros(W.shape[0])
            for g2, (c2, R2, rb2) in fits.items():
                vec[jmap[g2]] = -R2[e] / cp
            vec[np.abs(vec) < 0.02] = 0.0
            ops = [{"op": "add", "tensor": f"W{l+1}", "axis": "col", "index": int(p), "vector": [float(round(x, 2)) for x in vec]}]
        anomalies.append(dict(strength=st, kind="col", layer=l, g=e, ops=ops, n=len(glist)))


def _select(anoms, P):
    if not anoms:
        return []
    top = max(anoms, key=lambda a: a["strength"])
    if top["strength"] < P["EMIT"]:
        return []
    if top["kind"] == "col":
        same = sorted([a for a in anoms if a["kind"] == "col" and a["layer"] == top["layer"] and a["strength"] >= P["COLSEL"] * top["strength"]], key=lambda a: -a["strength"])[:3]
        ops = []
        for a in same:
            ops += a["ops"]
        return ops[:3]
    return top["ops"]
