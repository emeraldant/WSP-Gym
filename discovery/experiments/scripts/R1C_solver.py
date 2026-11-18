"""Forward-pass-free template-alignment solver (R1C / H3).
Uses only: shown weights (net), spec-derived canonical net. Never evaluates any network on inputs."""
import numpy as np
from scipy.optimize import linear_sum_assignment

P = dict(TOLW=0.07, TOLB=0.09, CAP=4.0, LAM=3.0, ATTACH_RC=2.0, PAIR_RC=2.0, COLFILL=True, EMIT=0.10, BIAS_MIN=0.3, COLSEL=0.35)


def fit_row(u, b, t, bt, P=P):
    """Fit shown row (u,b) ~ c*(t,bt) with c signed on weights (c<0 = flipped weights, bias keeps |c|).
    Returns cost, c, resid_w, resid_b."""
    nz = np.abs(t) > 1e-9
    cands = []
    if nz.any():
        cands += list(u[nz] / t[nz])
        tt = float(t @ t)
        cands += [float(u @ t) / tt, -float(u @ t) / tt if False else float(u @ t) / tt]
    if abs(bt) > P["BIAS_MIN"] and abs(b / bt) > 0.05:
        cands.append(b / bt)
    best = (1e9, 1.0, None, None)
    seen = set()
    for c in cands:
        if abs(c) < 0.05 or abs(c) > 20:
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


def analyze(net, canon, P=P):
    """Return (ops, anomalies). anomalies: list of dicts(strength, kind, ops, layer)."""
    L = net.n_layers
    anomalies = []
    A = net.Ws[0].copy()
    groups_prev = None  # list over canonical prev neurons: list of (shown idx, c)
    for l in range(L):
        Wc, bc = canon.Ws[l], canon.bs[l]
        b = net.bs[l]
        m = Wc.shape[0]
        if l > 0:  # collapse shown W_l columns onto canonical prev neurons
            Wl = net.Ws[l]
            A = np.zeros((Wl.shape[0], len(groups_prev)))
            for g, mem in enumerate(groups_prev):
                for p, c in mem:
                    A[:, g] += Wl[:, p] * abs(c)
        M = A.shape[0]
        if l == L - 1:
            rows = [(g, g) for g in range(m)]  # output rows fixed
            fits = {}
            for g in range(m):
                R = A[g] - Wc[g]
                fits[g] = (1.0, R, b[g] - bc[g])
            # label-swap test
            Rb = {g: fits[g][2] for g in fits}
            bad = [g for g in fits if (np.abs(fits[g][1]) > P["TOLW"] * 1.5).any() or abs(Rb[g]) > P["TOLB"] * 1.5]
            swap_done = False
            if len(bad) >= 2 and m >= 2:
                bestp = None
                base = sum(np.abs(fits[g][1]).sum() + abs(Rb[g]) for g in bad)
                for i in range(m):
                    for j in range(i + 1, m):
                        sc = (np.abs(A[i] - Wc[j]).sum() + abs(b[i] - bc[j]) + np.abs(A[j] - Wc[i]).sum() + abs(b[j] - bc[i]))
                        others = sum(np.abs(fits[g][1]).sum() + abs(Rb[g]) for g in range(m) if g not in (i, j))
                        tot = sc + others
                        if bestp is None or tot < bestp[0]:
                            bestp = (tot, i, j)
                full = sum(np.abs(fits[g][1]).sum() + abs(Rb[g]) for g in range(m))
                if bestp and bestp[0] < 0.35 * full:
                    _, i, j = bestp
                    anomalies.append(dict(strength=float(np.abs(A[i] - A[j]).sum() * 0 + (full - bestp[0])), kind="label_swap", layer=l,
                                          ops=[{"op": "swap", "tensor": f"W{l+1}", "axis": "row", "i": i, "j": j},
                                               {"op": "swap", "tensor": f"b{l+1}", "i": i, "j": j}]))
                    swap_done = True
            if not swap_done:
                _col_anoms(net, l, fits, {g: g for g in range(m)}, groups_prev, anomalies, P)
            break
        # ---- hidden layer: match rows -> canonical neurons
        Wn = net.Ws[l + 1]
        Cn = canon.Ws[l + 1]
        cost = np.zeros((M, m))
        info = {}
        for j in range(M):
            for g in range(m):
                rc, c, r, rb = fit_row(A[j], b[j], Wc[g], bc[g], P)
                cc = (1.0 - _cos(Wn[:, j], Cn[:, g])) if Wn.shape[0] == Cn.shape[0] else 0.0
                cost[j, g] = rc + P["LAM"] * cc
                info[(j, g)] = (rc, c, r, rb, cc)
        rj, cg = linear_sum_assignment(cost)
        assign = {int(g): int(j) for j, g in zip(rj, cg)}
        members = {g: [assign[g]] for g in range(m)}
        used = set(assign.values())
        for j in range(M):
            if j in used:
                continue
            gbest = min(range(m), key=lambda g: info[(j, g)][0])
            rc, c, r, rb, cc = info[(j, gbest)]
            if rc <= P["ATTACH_RC"]:
                members[gbest].append(j)
        groups = []
        fitsrow = {}
        for g in range(m):
            mem = []
            for j in members[g]:
                rc, c, r, rb, cc = info[(j, g)]
                mem.append((j, c))
                fitsrow[(j, g)] = (c, r, rb)
            groups.append(mem)
        # row-level anomalies
        for (j, g), (c, r, rb) in fitsrow.items():
            s = abs(c)
            if c < 0:
                anomalies.append(dict(strength=2.0, kind="flip_in", layer=l, ops=[{"op": "scale", "tensor": f"W{l+1}", "axis": "row", "index": int(j), "factor": -1.0}]))
            if abs(rb) > P["TOLB"] and l >= 0:
                anomalies.append(dict(strength=abs(rb) / max(s, 1e-6), kind="bias", layer=l, j=j, g=g, c=s, rb=rb,
                                      ops=[{"op": "set", "tensor": f"b{l+1}", "index": int(j), "value": float(round(s * bc[g], 2))}]))
            out = np.flatnonzero(np.abs(r) > P["TOLW"] * 1.5)
            if l == 0 and len(out) >= 1 and c > 0:
                e = int(out[np.argmax(np.abs(r[out]))])
                if len(out) <= 2:
                    anomalies.append(dict(strength=abs(r[e]) / s, kind="entry_in", layer=l,
                                          ops=[{"op": "set", "tensor": f"W{l+1}", "index": [int(j), e], "value": float(round(s * Wc[g][e], 2))}]))
        # cancel-pair detection among unassigned (distractor) neurons
        Dn = [j for j in range(M) if all(j not in [q for q, _ in mem] for mem in groups)]
        for a_i in range(len(Dn)):
            for b_i in range(a_i + 1, len(Dn)):
                p_, q_ = Dn[a_i], Dn[b_i]
                rc, c, r, rb = fit_row(A[p_], b[p_], A[q_], b[q_], P)
                if rc <= P["PAIR_RC"] and c > 0:
                    S_ = c * Wn[:, p_] + Wn[:, q_]
                    if np.linalg.norm(S_[np.abs(S_) > 0]) > P["TOLW"] * 1.5 and (np.abs(S_) > P["TOLW"] * 1.5).any():
                        v = np.where(np.abs(S_) > 0.02, -S_ / c, 0.0)
                        anomalies.append(dict(strength=float(np.linalg.norm(S_)), kind="pair", layer=l + 1,
                                              ops=[{"op": "add", "tensor": f"W{l+2}", "axis": "col", "index": int(p_), "vector": [float(round(x, 2)) for x in v]}]))
        groups_prev = groups
        # stash this layer's groups for next iteration's residual/col analysis
        A_next_groups = groups
        # next iteration will compute A (collapsed) and handle residuals via the next layer match; but col residuals of
        # THIS layer's groups need the next layer matching -> do it in the next iteration after matching (below)
        if l + 1 < L - 1:
            pass
        # pre-compute next-layer match needs next iteration; col analysis is done there via _hidden_next
        # (for hidden next layers we do it at the top of the next iteration after matching)
        pending = (l, groups)
        # hidden-next col analysis is done immediately: collapse and match next rows now
        Wl = net.Ws[l + 1]
        An = np.zeros((Wl.shape[0], len(groups)))
        for g, mem in enumerate(groups):
            for p, c in mem:
                An[:, g] += Wl[:, p] * abs(c)
        if l + 1 < L - 1:
            # match next-layer rows to canonical to obtain residuals
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
            fits2 = {int(g): (inf2[(int(j), int(g))][0], inf2[(int(j), int(g))][1], inf2[(int(j), int(g))][2]) for j, g in zip(rj2, cg2)}
            jmap = {int(g): int(j) for j, g in zip(rj2, cg2)}
            _col_anoms(net, l + 1, fits2, jmap, groups, anomalies, P)
        else:
            pass  # output layer handled at next iteration
        # continue loop: next iteration recomputes A for layer l+1
        if l + 1 < L - 1:
            continue
        if l + 1 == L - 1:
            # output layer residuals computed at top of next iteration; groups_prev set
            continue
    # ---- combine
    return _select(anomalies, P), anomalies


def _col_anoms(net, l, fits, jmap, groups, anomalies, P):
    """fits[g] = (c', Rw (len=#prev groups), Rb) for matched next-layer row of canonical g (shown row jmap[g])."""
    Wn_shown = net.Ws[l]
    ng = len(groups)
    cols = {}
    for g, (c, R, rb) in fits.items():
        for e in np.flatnonzero(np.abs(R) > P["TOLW"] * 1.5):
            cols.setdefault(int(e), []).append((jmap[g], float(R[e])))
    for e, lst in cols.items():
        mem = groups[e]
        if not mem:
            continue
        p, cp = max(mem, key=lambda t: abs(t[1]))
        cp = abs(cp)
        strength = float(np.sqrt(sum(v * v for _, v in lst)))
        if len(lst) == 1:
            j, v = lst[0]
            old = float(Wn_shown[j, p])
            ops = [{"op": "set", "tensor": f"W{l+1}", "index": [int(j), int(p)], "value": float(round(old - v / cp, 2))}]
        else:
            vec = np.zeros(Wn_shown.shape[0])
            for g2, (c2, R2, rb2) in fits.items():
                vec[jmap[g2]] = -R2[e] / cp
            vec[np.abs(vec) < 0.02] = 0.0
            ops = [{"op": "add", "tensor": f"W{l+1}", "axis": "col", "index": int(p), "vector": [float(round(x, 2)) for x in vec]}]
        anomalies.append(dict(strength=strength, kind="col", layer=l, g=e, ops=ops, n=len(lst)))
    # hidden next layer bias / flip of next-layer rows are handled when that layer is the 'current' one


def _select(anoms, P):
    if not anoms:
        return []
    # pair bias+col anomaly in same group => row-scale fault
    anoms = list(anoms)
    cols = [a for a in anoms if a["kind"] == "col"]
    biases = [a for a in anoms if a["kind"] == "bias"]
    top = max(anoms, key=lambda a: a["strength"])
    if top["strength"] < P["EMIT"]:
        return []
    if top["kind"] == "pair":
        return top["ops"]
    if top["kind"] == "col":
        # emit all col anomalies in the same layer that are comparable (<=3 ops)
        same = sorted([a for a in cols if a["layer"] == top["layer"] and a["strength"] >= P["COLSEL"] * top["strength"]], key=lambda a: -a["strength"])[:3]
        ops = []
        for a in same:
            ops += a["ops"]
        return ops[:3]
    return top["ops"]
