"""R2C spec-free reader: weights + architecture + input range ONLY. No spec text, no family name, no build_task,
no forward passes (only interval range analysis).  Structure inferred from the weights:
 per-neuron lattice unit (undo rescaling), clustering of split/cancel copies, dead/always-on interval tests,
 lattice residuals of rows/columns (off-lattice entries), cancel-pair consistency, sign-consistency checks."""
import numpy as np

P = dict(TAU=0.06, LAM=0.4, CAP=9.0, OFF=0.15, CL=0.3, DEAD=0.1, EMIT=0.2, W_ROW=1.0, W_COL=0.8, W_CL=1.0,
         W_DEAD=1.0, W_SIGN=0.0, ONOFF=1.0, ZTOL=0.12, DF_AMB=0.0, SW_MIN=0.25, NONPAR=0.0, NEAR=0.0, HALF_LO=0.4, HALF_W=0.3, SW_FRAC=0.35, W_SW=1.0)


def unit_est(v, P=P, umin=0.25, umax=8.0):
    a = np.abs(np.asarray(v, float))
    big = a[a > 0.25]
    if len(big) == 0:
        return 1.0
    cands = set()
    for e in big:
        for k in (1, 2, 3):
            u = e / k
            if umin <= u <= umax:
                cands.add(round(u, 3))
    best = None
    for u in cands:
        q = np.round(a / u)
        err = ((a - u * q) / P["TAU"]) ** 2
        cost = np.minimum(err, P["CAP"]).sum() + P["LAM"] * q.sum()
        if best is None or cost < best[0]:
            best = (cost, u)
    return best[1] if best else 1.0


def intervals(net, hi):
    """interval [LB,UB] of each hidden pre-activation over input box [0,hi]^n (weights only)."""
    lo = np.zeros(net.Ws[0].shape[1]); up = np.full_like(lo, float(hi))
    out = []
    for l in range(net.n_layers - 1):
        W, b = net.Ws[l], net.bs[l]
        ub = b + np.clip(W, 0, None) @ up + np.clip(W, None, 0) @ lo
        lb = b + np.clip(W, 0, None) @ lo + np.clip(W, None, 0) @ up
        out.append((lb, ub))
        lo, up = np.maximum(lb, 0), np.maximum(ub, 0)
    return out


def _off(x, P=P):
    """residual of normalised value to the integer lattice (nonzero entries cannot round to 0)."""
    r = np.abs(x - np.round(x))
    return r


def IV_prev_up(net, l, hi):
    lo = np.zeros(net.Ws[0].shape[1]); up = np.full_like(lo, float(hi))
    for k in range(l):
        W, b = net.Ws[k], net.bs[k]
        ub = b + np.clip(W, 0, None) @ up + np.clip(W, None, 0) @ lo
        lb = b + np.clip(W, 0, None) @ lo + np.clip(W, None, 0) @ up
        lo, up = np.maximum(lb, 0), np.maximum(ub, 0)
    return up, lo


def analyze(net, hi=1, P=P):
    L = net.n_layers
    IV = intervals(net, hi)
    anoms = []
    A = net.Ws[0].copy()
    groups_prev = None      # list of lists of member indices (raw neuron index in layer l-1)
    ucache_prev = None
    for l in range(L):
        b = net.bs[l]
        M = A.shape[0]
        last = (l == L - 1)
        U = np.ones(M); N = np.zeros((M, A.shape[1] + 1))
        for j in range(M):
            row = np.r_[A[j], b[j]]
            if last:
                U[j] = 1.0
            else:
                U[j] = unit_est(A[j], P) if np.abs(A[j]).max() > 0.25 else unit_est(row, P)
            N[j] = row / U[j]
        # ---- row-level lattice anomalies (this layer's rows in canonical coordinates of previous layer)
        if not last:
            lb, ub = IV[l]
            alive = ub > P["DEAD"]
        else:
            alive = np.ones(M, bool)
        # cluster alive neurons
        if not last:
            adj = np.zeros((M, M), bool)
            for p in range(M):
                for q in range(p + 1, M):
                    if alive[p] and alive[q] and np.abs(N[p] - N[q]).max() < P["CL"]:
                        adj[p, q] = adj[q, p] = True
            seen = set(); clusters = []
            for p in range(M):
                if p in seen or not alive[p]:
                    continue
                comp = [p]; seen.add(p); stack = [p]
                while stack:
                    x = stack.pop()
                    for y in np.flatnonzero(adj[x]):
                        if y not in seen:
                            seen.add(int(y)); comp.append(int(y)); stack.append(int(y))
                clusters.append(sorted(comp))
        else:
            clusters = [[j] for j in range(M)]
        # ---- next-layer columns
        if not last:
            Wn = net.Ws[l + 1]
            V = Wn * U[None, :]                         # normalised-per-neuron columns (raw next-layer rows)
            lam_info = []
            groups = []
            for cl in clusters:
                vs = [V[:, p] for p in cl]
                d = max(vs, key=lambda v: np.linalg.norm(v))
                nd = np.linalg.norm(d)
                if nd < 1e-6 or len(cl) == 1:
                    groups.append(dict(members=cl, col=np.sum(vs, axis=0), cancel=False))
                    continue
                lam = np.array([float(v @ d) / (nd * nd) for v in vs])
                res = np.array([np.linalg.norm(v - lm * d) for v, lm in zip(vs, lam)])
                nonpar = res.max() > 0.12 * max(nd, 1.0)
                tot = np.sum(vs, axis=0)
                cancel = np.linalg.norm(tot) < 0.1 * max(nd, 1.0) + 0.08
                groups.append(dict(members=cl, col=tot, cancel=cancel, lam=lam, nonpar=nonpar))
                # --- cluster consistency anomalies
                if cancel:
                    continue
                if nonpar and P["NONPAR"] > 0:
                    # copies with the same row but non-parallel columns: planted fault on a member column
                    k = int(np.argmax(res))
                    p = cl[k]
                    others = [v for i, v in enumerate(vs) if i != k]
                    # repair: make member k parallel: if cancel-like partner exists use -partner, else set via add
                    target = -np.sum(others, axis=0) if len(cl) == 2 else None
                    anoms.append(dict(strength=P["W_CL"] * 1.0, kind="cl_nonpar", layer=l, cl=cl, k=p, res=res,
                                      tgt=target, Wn_layer=l + 1, U=U))
                elif not nonpar:
                    # parallel; mixed signs but non-zero sum -> scaled/flipped cancel member
                    al = np.abs(lam); rr = al / max(al.min(), 1e-9)
                    stair = bool(np.all(np.abs(rr - np.round(rr)) < 0.1) and rr.max() < 6.5)
                    if (lam > 0).any() and (lam < 0).any() and not stair:
                        k = int(np.argmin(np.abs(lam)) if False else np.argmax(np.abs(lam)))
                        anoms.append(dict(strength=P["W_CL"] * 0.9, kind="cl_mixed", layer=l, cl=cl, lam=lam, d=d, U=U))
                    elif len(cl) == 2 and abs(lam[0] / lam[1] - 1) < 0.06:
                        anoms.append(dict(strength=P["W_CL"] * 0.5, kind="cl_equal", layer=l, cl=cl, lam=lam, U=U))
        else:
            groups = []
        # ---- row anomalies for non-cluster-conflicted alive rows
        if not last:
            for cl in clusters:
                p = cl[0]
                q = N[p]
                ent = q[:-1]; bn = q[-1]
                offe = np.where(np.abs(ent) > P["OFF"], _off(ent), 0.0)
                offe = np.where(np.abs(ent) < P["OFF"], 0.0, offe)
                # entries between 0 and 1 (e.g. 0.4) are off-lattice
                offe = np.array([min(abs(e - round(e)), abs(e)) if abs(e) > P["OFF"] else 0.0 for e in ent])
                offb = abs(bn - round(bn))
                col_anom = None
                if offe.max() > P["OFF"]:
                    e = int(np.argmax(offe))
                    anoms.append(dict(strength=P["W_ROW"] * float(offe[e]) * (1.0 if offe[e] < P["HALF_LO"] else P["HALF_W"]), kind="row_entry", layer=l, j=p, e=e,
                                      N=q, u=U[p], cl=cl, U=U, A=A, b=b))
                elif offb > P["OFF"]:
                    anoms.append(dict(strength=P["W_ROW"] * float(offb), kind="row_bias", layer=l, j=p, N=q, u=U[p], cl=cl,
                                      U=U, A=A, b=b))
            # dead-with-purpose / always-on (interval) tests
            for j in range(M):
                u = U[j]
                ubn = ub[j] / u; lbn = lb[j] / u
                if (not alive[j]) and np.abs(V[:, j]).max() > 0.2:
                    # flipped-row hypothesis: would the sign-reversed row be alive on the box?
                    rowj = net.Ws[l][j]
                    upr = IV_prev_up(net, l, hi)
                    ubf = b[j] + np.clip(-rowj, 0, None) @ upr[0] + np.clip(-rowj, None, 0) @ upr[1]
                    allneg = bool((A[j] <= 0.15 * u).all() and (A[j] < -0.25 * u).any())
                    if ubf > 0.5 * u and allneg:
                        anoms.append(dict(strength=P["W_DEAD"] * (0.8 if ub[j] > -0.3 * u else P["DF_AMB"]), kind="dead_flip", layer=l, j=j))
                    elif ub[j] > -0.7 * u:
                        anoms.append(dict(strength=P["NEAR"], kind="dead_nearmiss", layer=l, j=j, ub=ub[j], u=u))
        # ---- column anomalies at next layer: handled when processing layer l+1 (A built from groups)
        if last:
            # output layer: rows' lattice entries
            for j in range(M):
                pass
            break
        # build A for next layer
        live = [g for g in groups if not g["cancel"]]
        An = np.zeros((net.Ws[l + 1].shape[0], len(live)))
        for gi, g in enumerate(live):
            An[:, gi] = g["col"]
        groups_prev = live
        # column anomaly detection in canonical coordinates needs next-layer units: compute them now
        bn_next = net.bs[l + 1]
        u2v_pre = np.ones(An.shape[0])
        for j2 in range(An.shape[0]):
            if l + 1 < L - 1:
                u2v_pre[j2] = unit_est(An[j2], P) if np.abs(An[j2]).max() > 0.25 else 1.0
        for j2 in range(An.shape[0]):
            row = np.r_[An[j2], bn_next[j2]]
            if l + 1 == L - 1:
                u2 = 1.0                                  # output rows are not rescaled
            else:
                u2 = unit_est(An[j2], P) if np.abs(An[j2]).max() > 0.25 else unit_est(row, P)
            qn = An[j2] / u2
            for gi in range(len(live)):
                x = qn[gi]
                if abs(x) > P["OFF"]:
                    off = min(abs(x - round(x)), abs(x))
                    if off > P["OFF"]:
                        off = off if off < P["HALF_LO"] else off * P["HALF_W"]
                    anoms.append(dict(strength=P["W_COL"] * float(off), kind="col_entry", layer=l + 1, row=j2, g=live[gi], gi=gi,
                                          x=float(x), u2=u2, U=U, A=An, Lprev=l, Ucur=U))
        # ---- sign consistency: a group whose column sign is the odd one out in every row where it is active
        if P["W_SIGN"] > 0 and len(live) >= 3:
            odd = {gi: [0, 0] for gi in range(len(live))}
            for j2 in range(An.shape[0]):
                u2 = u2v_pre[j2]
                qn = An[j2] / u2
                act = [gi for gi in range(len(live)) if abs(qn[gi]) > P["OFF"]]
                unit = [gi for gi in act if abs(abs(qn[gi]) - 1) < 0.2]
                if len(act) >= 3 and len(unit) == len(act):
                    pos = [gi for gi in unit if qn[gi] > 0]; neg = [gi for gi in unit if qn[gi] < 0]
                    for gi in unit:
                        odd[gi][1] += 1
                    if len(pos) == 1 and len(neg) >= 2:
                        odd[pos[0]][0] += 1
                    elif len(neg) == 1 and len(pos) >= 2:
                        odd[neg[0]][0] += 1
            for gi, (o, t) in odd.items():
                if t >= 1 and o == t:
                    g = live[gi]
                    anoms.append(dict(strength=P["W_SIGN"], kind="sign_out", layer=l, g=g, U=U))
        # ---- swap hypothesis among raw neurons (singleton clusters, dead or alive)
        u2v = np.ones(An.shape[0])
        for j2 in range(An.shape[0]):
            if l + 1 < L - 1:
                u2v[j2] = unit_est(An[j2], P) if np.abs(An[j2]).max() > 0.25 else 1.0
        multi = set(p for g in groups for p in g["members"] if len(g["members"]) > 1)
        Vn = (Wn * U[None, :]) / u2v[:, None]
        def cres(v):
            a = np.abs(v)
            return float(np.where(a > P["OFF"], np.minimum(np.abs(a - np.round(a)), a), 0.0).sum())
        cur = {p: cres(Vn[:, p]) for p in range(M)}
        for p in range(M):
            if p in multi:
                continue
            for q in range(p + 1, M):
                if q in multi:
                    continue
                vp = (Wn[:, q] * U[p]) / u2v
                vq = (Wn[:, p] * U[q]) / u2v
                new = cres(vp) + cres(vq)
                old = cur[p] + cur[q]
                if old > P["SW_MIN"] and new < P["SW_FRAC"] * old and max(cur[p], cur[q]) > 0.15:
                    anoms.append(dict(strength=P["W_SW"] * (min(old, 1.0) + 0.3 - new + 0.2 * (min(cur[p], cur[q]) > 0.15)), kind="swap", layer=l, p=p, q=q, old=old, new=new))
        A = An
    return anoms, IV


def _largest_member(g, U, Wn):
    return max(g["members"], key=lambda p: U[p] * float(np.abs(Wn[:, p]).max()))


def to_ops(net, anoms, P=P):
    """pick the strongest anomaly and turn it into <=3 DSL ops."""
    if not anoms:
        return [], None
    top = max(anoms, key=lambda a: a["strength"])
    if top["strength"] < P["EMIT"]:
        return [], top
    return _ops_for(net, top, anoms, P), top


def _ops_for(net, a, anoms, P):
    k = a["kind"]
    l = a["layer"]
    if k == "col_entry":
        # group a['g'] column; collect all off-lattice rows of the same group at this layer
        same = [x for x in anoms if x["kind"] == "col_entry" and x["layer"] == l and x["g"] is a["g"] and x["strength"] > P["OFF"]]
        Wn = net.Ws[l]
        U = a["U"]
        p = _largest_member(a["g"], U, Wn)
        if len(same) == 1 or Wn.shape[0] == 1:
            x = a["x"]
            tgt = np.sign(x) * max(1.0, float(np.round(abs(x))) if abs(x) > 3.5 else 1.0)
            # scale-like: nearest lattice
            return [{"op": "scale", "tensor": f"W{l+1}", "axis": "col", "index": int(p), "factor": float(round(tgt / x, 4))}] if Wn.shape[0] == 1 else \
                   [{"op": "set", "tensor": f"W{l+1}", "index": [int(a["row"]), int(p)],
                     "value": float(round(Wn[a["row"], p] + (tgt - x) * a["u2"] / U[p], 2))}]
        # several rows: scale the column so the entries land on the lattice
        xs = np.array([s["x"] for s in same])
        best = None
        for f in (1 / 3, 0.4, 2.5, 3.0, 0.5, 2.0, 1 / 0.3, 1.5, 0.75, 4 / 3, 0.6):
            r = np.abs(xs * f - np.round(xs * f)).max()
            if best is None or r < best[0]:
                best = (r, f)
        return [{"op": "scale", "tensor": f"W{l+1}", "axis": "col", "index": int(p), "factor": float(round(best[1], 4))}]
    if k == "row_entry":
        q = a["N"]; e = a["e"]; l = a["layer"]
        tgt = np.sign(q[e]) * max(1.0, abs(round(q[e])))
        raw_old = a["A"][a["j"], e]
        if l == 0:
            return [{"op": "set", "tensor": "W1", "index": [int(a["j"]), int(e)], "value": float(round(tgt * a["u"], 2))}]
        return []
    if k == "row_bias":
        q = a["N"]; j = a["j"]; l = a["layer"]
        bn = q[-1]
        for kk in (0, -1, -2, 1, -3, -4, 2):
            if abs(kk - bn) >= 0.9:
                break
        return [{"op": "set", "tensor": f"b{l+1}", "index": int(j), "value": float(round(kk * a["u"], 2))}]
    if k == "sign_out":
        p = _largest_member(a["g"], a["U"], net.Ws[l + 1])
        if len(a["g"]["members"]) == 1:
            return [{"op": "scale", "tensor": f"W{l+2}", "axis": "col", "index": int(p), "factor": -1.0}]
        return []
    if k == "swap":
        return [{"op": "swap", "tensor": f"W{l+2}", "axis": "col", "i": int(a["p"]), "j": int(a["q"])}]
    if k == "dead_flip":
        return [{"op": "scale", "tensor": f"W{l+1}", "axis": "row", "index": int(a["j"]), "factor": -1.0}]
    if k == "dead_nearmiss":
        j = a["j"]; l = a["layer"]
        return [{"op": "set", "tensor": f"b{l+1}", "index": int(j), "value": float(round(net.bs[l][j] - a["ub"] + a["u"], 2))}]
    if k == "cl_equal":
        cl = a["cl"]
        p = cl[0]
        return [{"op": "scale", "tensor": f"W{l+2}", "axis": "col", "index": int(p), "factor": -1.0}]
    if k == "cl_mixed":
        cl = a["cl"]; lam = a["lam"]
        # make columns cancel: scale the member with the smaller |lam| to -lam_other
        order = np.argsort(np.abs(lam))
        i, j = int(order[0]), int(order[-1])
        pos = [x for x in range(len(cl)) if lam[x] > 0]; neg = [x for x in range(len(cl)) if lam[x] < 0]
        if len(cl) == 2:
            f = -lam[j] / lam[i]
            return [{"op": "scale", "tensor": f"W{l+2}", "axis": "col", "index": int(cl[i]), "factor": float(round(f, 4))}]
        return []
    if k == "cl_nonpar":
        p = a["k"]; cl = a["cl"]
        tgt = a["tgt"]
        if tgt is None:
            return []
        U = a["U"]; Wn = net.Ws[l + 1]
        cur = Wn[:, p] * U[p]
        vec = (tgt - cur) / U[p]
        vec = np.where(np.abs(vec) > 0.02, vec, 0.0)
        return [{"op": "add", "tensor": f"W{l+2}", "axis": "col", "index": int(p), "vector": [float(round(x, 2)) for x in vec]}]
    return []
