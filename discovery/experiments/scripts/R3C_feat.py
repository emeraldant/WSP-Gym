"""R3C neuron features: R2A NetInfo neuron features (23) + within-net robust-z/rank versions (scale-free across widths)
+ EXTRA generic interval/fan-in features (flagged 'x_')."""
import numpy as np
import R2A_feat as F

BASE = F.NetInfo.NF_NAMES

def _z(v):
    return F._robust_z(v)

def net_neuron_table(net, task):
    info = F.NetInfo(net, task)
    X, layer, idx = [], [], []
    for l, nf in enumerate(info.nf):
        w = nf.shape[0]
        W, b, Wn = net.Ws[l], net.bs[l], net.Ws[l + 1]
        z = np.stack([_z(nf[:, k]) for k in range(nf.shape[1])], 1)
        mag = np.abs(W)
        strong = mag > 0.25 * np.maximum(mag.max(1, keepdims=True), 1e-9)
        nstrong = strong.sum(1)
        minstrong = np.where(strong, mag, np.inf).min(1)
        minstrong[~np.isfinite(minstrong)] = 1.0
        peak = np.maximum(W, 0).sum(1) + b
        cn = info.cn[l + 1]
        ex = np.stack([nstrong / W.shape[1], nstrong.astype(float), peak / info.rn[l], (peak - minstrong) / minstrong,
                       float(1) * (peak > 0), (peak - minstrong > 0).astype(float), np.log(cn),
                       peak * np.abs(Wn).max(0) ], 1)
        exz = np.stack([_z(ex[:, k]) for k in (0, 2, 3, 6, 7)], 1)
        X.append(np.hstack([nf, z, ex, exz]))
        layer += [l] * w; idx += list(range(w))
    return dict(X=np.vstack(X).astype(np.float32), layer=np.array(layer), idx=np.array(idx))

NAMES = BASE + ["z_" + n for n in BASE] + ["x_fanin_frac", "x_fanin", "x_peak_rel", "x_peak_margin", "x_canfire", "x_allagree",
        "x_logcn", "x_effect", "xz_fanin_frac", "xz_peak_rel", "xz_peak_margin", "xz_logcn", "xz_effect"]
N_BASE = 2 * len(BASE)  # first N_BASE columns = adapted R2A feature set (PRIMARY); rest = extra
