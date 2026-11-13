/* WSP-Gym showcase. Every number drawn here is read from the inlined data (site/data.json).
   Each animation is a pure function of time t, so it can pause off-screen, replay, honour
   prefers-reduced-motion, and be rendered frame by frame for video export (?clip=<name>). */
(function () {
  "use strict";
  const DATA = JSON.parse(document.getElementById("data").textContent);
  const RM = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const params = new URLSearchParams(location.search);
  const CLIP = params.get("clip");
  const NS = "http://www.w3.org/2000/svg";

  // ---------------------------------------------------------------- helpers
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const ease = (x) => { x = clamp(x); return 1 - Math.pow(1 - x, 3); };
  const lerp = (a, b, s) => a + (b - a) * s;
  const seg = (t, a, b) => clamp((t - a) / (b - a));
  const r1 = (x) => (Math.round(x * 1000 + 1e-7) / 10).toFixed(1);
  const pct1 = (x) => r1(x) + "%";
  const pct0 = (x) => Math.round(x * 100 + 1e-7) + "%";
  const gap1 = (x) => (x >= 0 ? "+" : "") + r1(x) + "%";
  const fmtInt = (x) => Math.round(x).toLocaleString("en-US");
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  function S(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function T(parent, x, y, text, attrs) {
    const e = S("text", Object.assign({ x, y }, attrs || {}), parent);
    e.textContent = text;
    return e;
  }
  function svgFor(el, w, h, label) {
    el.innerHTML = "";
    const s = S("svg", { viewBox: `0 0 ${w} ${h}`, width: w, height: h, role: "img", "aria-label": label || "" }, el);
    return s;
  }
  const cw = (el) => (el.clientWidth || 700) / (CLIP ? 1.4 : 1);  // clips draw smaller and scale up, for larger text
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const halo = { "paint-order": "stroke", stroke: "var(--bg)", "stroke-width": 4, "stroke-linejoin": "round" };

  // tooltip
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.setAttribute("role", "tooltip");
  document.body.appendChild(tip);
  document.addEventListener("pointerover", (e) => {
    const el = e.target.closest && e.target.closest("[data-tip]");
    if (!el) { tip.style.opacity = 0; return; }
    tip.textContent = el.getAttribute("data-tip");
    tip.style.opacity = 1;
  });
  document.addEventListener("pointermove", (e) => {
    tip.style.left = Math.min(window.innerWidth - 270, e.clientX + 12) + "px";
    tip.style.top = e.clientY + 14 + "px";
  });

  // theme toggle
  document.querySelectorAll(".theme-toggle").forEach((b) => b.addEventListener("click", () => {
    const dark = document.documentElement.getAttribute("data-theme")
      ? document.documentElement.getAttribute("data-theme") === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.setAttribute("data-theme", dark ? "light" : "dark");
    redrawAll();
  }));
  if (params.get("theme")) document.documentElement.setAttribute("data-theme", params.get("theme"));

  // ---------------------------------------------------------------- animation registry
  const ANIMS = {};
  let running = false, last = null;
  function register(name, dur, setup) {
    const fig = document.querySelector(`[data-anim="${name}"]`);
    if (!fig) return;
    const a = { name, fig, dur, setup, t: 0, state: "idle", visible: false };
    a.draw = setup(fig);
    ANIMS[name] = a;
    const btn = fig.querySelector(".replay");
    if (btn) btn.addEventListener("click", () => { a.t = 0; a.state = "playing"; a.draw(0); label(a); kick(); });
    if (RM || CLIP) { a.t = dur; a.state = "done"; a.draw(dur); }
    else a.draw(0);
    label(a);
    if (!CLIP && "IntersectionObserver" in window) {
      new IntersectionObserver((ents) => {
        for (const en of ents) {
          a.visible = en.isIntersecting;
          if (a.visible && a.state === "idle") a.state = "playing";
          if (a.visible) kick();
        }
      }, { threshold: 0.3 }).observe(fig);
    } else if (!CLIP && !RM) { a.visible = true; a.state = "playing"; kick(); }
  }
  function label(a) {
    const btn = a.fig.querySelector(".replay");
    if (btn) btn.textContent = a.state === "playing" ? "Replay" : RM && a.t >= a.dur && !a.played ? "Play" : "Replay";
    if (a.state === "playing") a.played = true;
  }
  function kick() { if (!running) { running = true; last = null; requestAnimationFrame(frame); } }
  function frame(ts) {
    const dt = last == null ? 0 : Math.min(0.05, (ts - last) / 1000);
    last = ts;
    let any = false;
    for (const k in ANIMS) {
      const a = ANIMS[k];
      if (a.state !== "playing") continue;
      if (a.visible) {
        a.t = Math.min(a.dur, a.t + dt);
        a.draw(a.t);
        if (a.t >= a.dur) { a.state = "done"; label(a); continue; }
        any = true;
      }
    }
    if (any) requestAnimationFrame(frame); else running = false;
  }
  function redrawAll() { for (const k in ANIMS) { const a = ANIMS[k]; a.draw = a.setup(a.fig); a.draw(a.t); } }
  let rz = null, lastW = window.innerWidth;
  window.addEventListener("resize", () => {
    if (window.innerWidth === lastW) return;
    lastW = window.innerWidth;
    clearTimeout(rz); rz = setTimeout(redrawAll, 150);
  });

  // ---------------------------------------------------------------- 1. token-for-token race
  // Beat 1: one network, both models. Beat 2: at the same token count, the trained model keeps answering
  // the next faulty test networks from its log (DATA.race.tiles: logged n_out and success per answer).
  const smooth = (x) => { x = clamp(x); return x * x * (3 - 2 * x); };
  function setupRace(fig) {
    const ex = DATA.example, th = ex.answers.untrained_thinking, R = DATA.race, think = ex.thinking, B = R.budget;
    const ps = fig.querySelector(".pane-think .stream"), grid = fig.querySelector(".tiles-grid");
    const latest = fig.querySelector(".latest"), lid = fig.querySelector(".latest-id"), lans = fig.querySelector(".latest-ans");
    const ct = fig.querySelector('[data-count="think"]'), cr = fig.querySelector('[data-count="rl"]');
    const nr = fig.querySelector('[data-nets="rl"]'), vt = fig.querySelector('[data-verdict="think"]');
    grid.innerHTML = "";
    const tiles = R.tiles.map((x, i) => {
      const d = document.createElement("i");
      d.className = "tl " + (x.success ? "ok" : "miss") + (i === 0 ? " hero" : "");
      d.setAttribute("data-tip", `network ${x.seed} (${x.family}, fault: ${x.fault}) · ${x.op} · ${x.n_out} tokens · ${x.success ? "repaired" : "missed"}`);
      grid.appendChild(d);
      return d;
    });
    const hero = R.tiles[0];
    const H0 = 0.6, H1 = 2.0, H2 = 3.6, E = 16.6;
    const tokAt = (t) => (t < H0 ? 0 : t < H1 ? hero.n_out * seg(t, H0, H1) : t < H2 ? hero.n_out : hero.n_out + (B - hero.n_out) * smooth(seg(t, H2, E)));
    const fin = (a) => `${a.success ? "Repaired" : "Not repaired"}: ${pct0(ex.acc_faulty)} → ${pct0(a.acc_after)}`;
    const WINDOW = 1500;
    return (t) => {
      const tok = tokAt(t), done = t >= E;
      // left: one network
      const n = Math.floor(think.length * Math.min(1, tok / B));
      if (done) ps.innerHTML = esc(think.slice(Math.max(0, think.length - WINDOW + 200))) + '\n\n<span class="final-answer">' + esc(th.answer) + "</span>";
      else ps.textContent = think.slice(Math.max(0, n - WINDOW), n);
      ps.scrollTop = ps.scrollHeight;
      ct.textContent = fmtInt(done ? th.n_out : Math.min(th.n_out, Math.round(tok)));
      vt.textContent = done ? fin(th) : "";
      // right: answers in log order while the token count lasts
      let k = 0;
      while (k < tiles.length && R.tiles[k].cum <= tok + 1e-6) k++;
      let rep = 0;
      tiles.forEach((d, i) => { d.classList.toggle("on", i < k); if (i < k && R.tiles[i].success) rep++; });
      cr.textContent = fmtInt(done ? R.used : Math.min(R.used, Math.round(tok)));
      nr.textContent = k ? `${k} network${k > 1 ? "s" : ""} answered, ${rep} repaired` : "";
      latest.classList.toggle("big", t < H2);
      if (k === 0) {
        lid.textContent = t >= H0 ? `network ${hero.seed}` : "";
        lans.textContent = hero.answer.slice(0, Math.floor(hero.answer.length * seg(t, H0, H1)));
      } else {
        const x = R.tiles[k - 1];
        lid.textContent = k === 1 ? `network ${x.seed} · ${fin(ex.answers.scale_rl)}` : `network ${x.seed} · ${x.fault} · ${x.success ? "repaired" : "missed"}`;
        lans.textContent = x.answer;
        latest.classList.toggle("miss", !x.success);
      }
    };
  }

  // ---------------------------------------------------------------- 1b. nothing to memorise
  function setupDisguise(fig) {
    const V = DATA.disguise.variants;
    const svg = fig.querySelector("svg.dnet"), pre = fig.querySelector(".dweights"), note = fig.querySelector(".dnote");
    const ol = fig.querySelector(".dsteps"), sub = fig.querySelector(".same-sub");
    ol.innerHTML = V.map((v, i) => `<li>${esc(v.title)}</li>`).join("");
    const lis = [...ol.children];
    svg.innerHTML = "";
    const Wd = 380, Hd = 290;
    svg.setAttribute("viewBox", `0 0 ${Wd} ${Hd}`);
    const X = [60, 190, 320], nIn = V[0].W[0][0].length;
    const ids = [];
    V.forEach((v) => v.ids.forEach((id) => { if (!ids.includes(id)) ids.push(id); }));
    const ypos = (n, i) => Hd / 2 + 8 + (i - (n - 1) / 2) * Math.min(66, 210 / Math.max(1, n - 1));
    const yIn = Array.from({ length: nIn }, (_, j) => ypos(nIn, j)), yOut = Hd / 2 + 8;
    const state = (v, id) => {
      const i = v.ids.indexOf(id);
      if (i < 0) return null;
      return { y: ypos(v.ids.length, i), row: v.W[0][i], b: v.b[0][i], out: v.W[1][0][i], name: `h${i}` };
    };
    const gE = S("g", {}, svg), gN = S("g", {}, svg);
    const edgeStyle = (l, w, op) => {
      l.setAttribute("stroke", Math.abs(w) < 0.005 ? "var(--muted)" : w > 0 ? "var(--pos)" : "var(--neg)");
      l.setAttribute("stroke-width", Math.abs(w) < 0.005 ? 1 : 1 + 2.6 * Math.min(2, Math.abs(w)));
      l.setAttribute("stroke-dasharray", Math.abs(w) < 0.005 ? "3 4" : "");
      l.setAttribute("opacity", 0.85 * op);
    };
    const E = ids.map((id) => ({ id, ins: yIn.map(() => S("line", { "stroke-linecap": "round" }, gE)), out: S("line", { "stroke-linecap": "round" }, gE) }));
    const N = ids.map((id) => {
      const g = S("g", {}, gN);
      S("circle", { r: 14, fill: "var(--bg)", stroke: "var(--ink-2)", "stroke-width": 1.5 }, g);
      const lab = T(g, 0, 4.5, "", { "text-anchor": "middle", "font-size": 12, class: "t-ink", "font-family": "var(--mono)" });
      const bl = T(g, 0, 29, "", { "text-anchor": "middle", "font-size": 10.5 });
      return { id, g, lab, bl };
    });
    yIn.forEach((y, j) => {
      S("circle", { cx: X[0], cy: y, r: 14, fill: "var(--bg)", stroke: "var(--ink-2)", "stroke-width": 1.5 }, gN);
      T(gN, X[0], y + 4.5, `x${j}`, { "text-anchor": "middle", "font-size": 12, class: "t-ink", "font-family": "var(--mono)" });
    });
    S("circle", { cx: X[2], cy: yOut, r: 14, fill: "var(--bg)", stroke: "var(--ink-2)", "stroke-width": 1.5 }, gN);
    T(gN, X[2], yOut + 4.5, "y", { "text-anchor": "middle", "font-size": 12, class: "t-ink", "font-family": "var(--mono)" });
    T(gN, X[2], yOut + 30, `b = ${V[0].b[1][0]}`, { "text-anchor": "middle", "font-size": 10.5 });
    const SLOT = 2.5, M = 0.9;
    const maxDiff = Math.max(...V.map((v) => v.max_diff));
    return (t) => {
      const k = Math.min(V.length - 1, Math.floor(t / SLOT));
      const s = k === 0 ? 1 : ease(seg(t, k * SLOT, k * SLOT + M));
      const cur = V[k], prev = V[Math.max(0, k - 1)];
      E.forEach((e, n) => {
        const a = state(prev, e.id), b = state(cur, e.id);
        const node = N[n];
        if (!a && !b) { node.g.setAttribute("opacity", 0); e.ins.forEach((l) => l.setAttribute("opacity", 0)); e.out.setAttribute("opacity", 0); return; }
        const A = a || Object.assign({}, b, { row: b.row.map(() => 0), out: 0, b: b.b }), Bq = b || A;
        const op = a ? 1 : s;
        const y = lerp(A.y, Bq.y, s);
        node.g.setAttribute("transform", `translate(${X[1]},${y})`);
        node.g.setAttribute("opacity", op);
        node.lab.textContent = (s < 0.5 && a ? a : Bq).name;
        node.bl.textContent = `b = ${fmtW(s < 0.5 && a ? a.b : Bq.b)}`;
        e.ins.forEach((l, j) => {
          l.setAttribute("x1", X[0] + 14); l.setAttribute("y1", yIn[j]); l.setAttribute("x2", X[1] - 14); l.setAttribute("y2", y);
          edgeStyle(l, lerp(A.row[j], Bq.row[j], s), op);
        });
        e.out.setAttribute("x1", X[1] + 14); e.out.setAttribute("y1", y); e.out.setAttribute("x2", X[2] - 14); e.out.setAttribute("y2", yOut);
        edgeStyle(e.out, lerp(A.out, Bq.out, s), op);
      });
      pre.textContent = (s < 0.5 ? prev : cur).text;
      note.innerHTML = `<b>${esc(cur.title)}.</b> ${esc(cur.note)} <span class="muted">${cur.n_weights} weights.</span>`;
      lis.forEach((li, i) => { li.classList.toggle("on", i <= k); li.classList.toggle("cur", i === k); });
      sub.textContent = `largest difference from the original on any input: ${maxDiff.toExponential(0)} (floating-point rounding)`;
    };
  }
  function fmtW(v) { let s = Number(v).toFixed(2).replace(/0+$/, "").replace(/\.$/, ""); return s === "-0" || s === "" ? "0" : s; }

  // ---------------------------------------------------------------- summary card minis
  function setupCard(fig) {
    const H = DATA.headline, tr = DATA.training, im = DATA.improve.series;
    const draws = [];
    // (a) tokens per network
    {
      const el = fig.querySelector('[data-mini="tokens"]');
      const W = Math.max(200, cw(el)), Hh = CLIP ? 165 : 92;
      const svg = svgFor(el, W, Hh, "");
      const rows = [
        { lab: "Trained, no reasoning", tok: H.trained.tokens, rate: H.trained.rate, color: "var(--trained)" },
        { lab: "Untrained, reasoning", tok: H.reasoning.tokens, rate: H.reasoning.rate, color: "var(--untrained)" },
      ];
      const max = H.reasoning.tokens;
      const bars = rows.map((r, i) => {
        const y = 8 + i * Math.round(Hh / 2.1);
        T(svg, 0, y + 10, `${r.lab}: ~${r.tok >= 1000 ? fmtInt(Math.round(r.tok / 100) * 100) : Math.round(r.tok)} tokens, ${pct1(r.rate)} repaired`, { "font-size": 11.5, class: "t-ink" });
        const b = S("rect", { x: 0, y: y + 16, height: 12, rx: 3, fill: r.color }, svg);
        return { b, w: Math.max(3, (W - 4) * r.tok / max) };
      });
      draws.push((t) => bars.forEach((x, i) => x.b.setAttribute("width", Math.max(2, x.w * ease(seg(t, 0.2 + 0.3 * i, 1.4 + 0.3 * i))))));
    }
    // (b) validation curve
    {
      const el = fig.querySelector('[data-mini="curve"]');
      const W = Math.max(200, cw(el)), Hh = CLIP ? 165 : 92, s0 = tr.sft_steps, s1 = s0 + tr.rl_steps;
      const svg = svgFor(el, W, Hh, "");
      const xs = (v) => 4 + (W - 8) * (v <= s0 ? 0.72 * v / s0 : 0.72 + 0.28 * (v - s0) / (s1 - s0)), ys = (v) => Hh - 6 - v * (Hh - 14);
      S("line", { x1: xs(s0), x2: xs(s0), y1: 4, y2: Hh - 4, stroke: "var(--rule)", "stroke-dasharray": "3 3" }, svg);
      const clipId = "m" + Math.random().toString(36).slice(2);
      const cr = S("rect", { x: 0, y: 0, width: 0, height: Hh }, S("clipPath", { id: clipId }, S("defs", {}, svg)));
      const g = S("g", { "clip-path": `url(#${clipId})` }, svg);
      for (const [key, col] of [["val", "var(--trained)"], ["val_trained", "var(--aqua)"]]) {
        const pts = tr.sft.map((p) => [p.step, p[key]]).concat(tr.rl.map((p) => [p.step, p[key]]));
        S("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${xs(p[0])},${ys(p[1])}`).join(""), fill: "none", stroke: col, "stroke-width": 2 }, g);
      }
      const sel = tr.rl.find((p) => p.chunk === tr.selected_chunk);
      const dot = S("circle", { cx: xs(sel.step), cy: ys(sel.val), r: 4.5, fill: "none", stroke: "var(--ink)", "stroke-width": 1.5, opacity: 0 }, svg);
      T(svg, xs(s0) - 4, Hh - 2, "SFT", { "font-size": 9.5, "text-anchor": "end" });
      T(svg, xs(s0) + 4, Hh - 2, "RL", { "font-size": 9.5 });
      draws.push((t) => { cr.setAttribute("width", W * ease(seg(t, 0.2, 1.8))); dot.setAttribute("opacity", seg(t, 1.8, 2.1)); });
    }
    // (c) errors removed with 8 proposals
    {
      const el = fig.querySelector('[data-mini="improve"]');
      const W = Math.max(200, cw(el)), Hh = CLIP ? 165 : 92;
      const svg = svgFor(el, W, Hh, "");
      const rows = [["RL-trained", im.trained[1].gap_closed, "var(--trained)"], ["Random edits", im.random[1].gap_closed, "var(--gray-series)"], ["Untrained", im.untrained[1].gap_closed, "var(--untrained)"]];
      const L = 86, max = 0.15;
      const bars = rows.map((r, i) => {
        const y = 6 + i * Math.round(Hh / 3.3);
        T(svg, 0, y + 12, r[0], { "font-size": 11, class: "t-ink" });
        const b = S("rect", { x: L, y: y + 2, height: 14, rx: 3, fill: r[2] }, svg);
        const v = T(svg, L, y + 13, gap1(r[1]), { "font-size": 11, opacity: 0 });
        return { b, v, w: (W - L - 48) * r[1] / max };
      });
      draws.push((t) => bars.forEach((x, i) => {
        const g = ease(seg(t, 0.3 + 0.2 * i, 1.4 + 0.2 * i));
        x.b.setAttribute("width", Math.max(2, x.w * g)); x.v.setAttribute("x", L + x.w * g + 6); x.v.setAttribute("opacity", seg(t, 1.2 + 0.2 * i, 1.6 + 0.2 * i));
      }));
    }
    return (t) => draws.forEach((d) => d(t));
  }

  // ---------------------------------------------------------------- 2. the repair
  function setupRepair(fig) {
    const ex = DATA.example, rl = ex.answers.scale_rl;
    const W1 = ex.W[0], W2 = ex.W[1];
    const [a, b] = ex.fault_info.neurons;  // the two hidden neurons whose outgoing columns were swapped
    // weights text, exactly as in the prompt
    const block = ex.prompt.split("## Weights\n")[1].split("\n\n")[0];
    const pre = fig.querySelector(".weights");
    const lines = block.split("\n");
    const w2i = lines.findIndex((l) => l.startsWith("  W2[0] = ["));
    const m = lines[w2i].match(/^(  W2\[0\] = \[)([^\]]*)(\].*)$/);
    const vals = m[2].split(", ");
    const head = lines.slice(0, w2i).map(esc).join("\n"), tail = lines.slice(w2i + 1).map(esc).join("\n");
    pre.innerHTML = head + "\n" + esc(m[1]) + vals.map((v, j) => `<span class="v" data-j="${j}">${esc(v)}</span>`).join(", ") + esc(m[3]) + (tail ? "\n" + tail : "");
    const spans = [...pre.querySelectorAll(".v")];
    const shortWeights = lines.map((l) => l.length).reduce((x, y) => Math.max(x, y), 0);
    pre.style.fontSize = "";

    // node-link diagram
    const svg = fig.querySelector("svg.net");
    svg.innerHTML = "";
    const Wd = 380, Hd = 270;
    svg.setAttribute("viewBox", `0 0 ${Wd} ${Hd}`);
    const nIn = W1[0].length, nH = W1.length, nOut = W2.length;
    const X = [60, 190, 320];
    const ys = (n) => Array.from({ length: n }, (_, i) => Hd / 2 + (i - (n - 1) / 2) * 78);
    const yIn = ys(nIn), yH = ys(nH), yOut = ys(nOut);
    const gE = S("g", {}, svg), gHL = S("g", {}, svg), gN = S("g", {}, svg);
    svg.insertBefore(gHL, gE);
    const edge = (x1, y1, x2, y2, w, tipText) => {
      const l = S("line", { x1, y1, x2, y2, "stroke-linecap": "round" }, gE);
      l.setAttribute("data-tip", tipText);
      const set = (wv) => {
        l.setAttribute("stroke", wv === 0 ? "var(--muted)" : wv > 0 ? "var(--pos)" : "var(--neg)");
        l.setAttribute("stroke-width", wv === 0 ? 1 : 1 + 2.6 * Math.abs(wv));
        l.setAttribute("stroke-dasharray", wv === 0 ? "3 4" : "");
        l.setAttribute("opacity", 0.85);
      };
      set(w);
      return { l, set };
    };
    for (let i = 0; i < nH; i++) for (let j = 0; j < nIn; j++) edge(X[0] + 14, yIn[j], X[1] - 14, yH[i], W1[i][j], `W1[${i}][${j}] = ${W1[i][j]}`);
    const out = [];
    for (let j = 0; j < nH; j++) out.push(edge(X[1] + 14, yH[j], X[2] - 14, yOut[0], W2[0][j], `W2[0][${j}]`));
    const hl = [a, b].map((j) => S("line", { x1: X[1] + 14, y1: yH[j], x2: X[2] - 14, y2: yOut[0], stroke: "var(--hl)", "stroke-width": 14, "stroke-linecap": "round", opacity: 0 }, gHL));
    const node = (x, y, lab, sub) => {
      S("circle", { cx: x, cy: y, r: 14, fill: "var(--bg)", stroke: "var(--ink-2)", "stroke-width": 1.5 }, gN);
      T(gN, x, y + 4.5, lab, { "text-anchor": "middle", "font-size": 12, class: "t-ink", "font-family": "var(--mono)" });
      if (sub) T(gN, x, y + 30, sub, { "text-anchor": "middle", "font-size": 11 });
    };
    yIn.forEach((y, j) => node(X[0], y, `x${j}`));
    yH.forEach((y, i) => node(X[1], y, `h${i}`, `b = ${ex.b[0][i]}`));
    node(X[2], yOut[0], "y", `b = ${ex.b[1][0]}`);
    T(gN, X[0], 22, "inputs", { "text-anchor": "middle", "font-size": 12 });
    T(gN, X[1], 22, "hidden (ReLU)", { "text-anchor": "middle", "font-size": 12 });
    T(gN, X[2], 22, "output", { "text-anchor": "middle", "font-size": 12 });

    const steps = [...fig.querySelectorAll(".steps li")];
    const fill = fig.querySelector(".accbar .acc-fill"), accv = fig.querySelector(".accbar .acc-val");
    const S1 = 1.6, S2 = 4.0, SW0 = 4.6, SW1 = 5.8, S3 = 6.4, A1 = 8.0;
    const v0 = W2[0][a], v1 = W2[0][b];
    return (t) => {
      const step = t < S1 ? 0 : t < S2 ? 1 : t < S3 ? 2 : 3;
      steps.forEach((li, i) => li.classList.toggle("on", i <= step));
      const pulse = step === 1 ? 0.35 + 0.25 * Math.sin((t - S1) * 5) : step >= 2 ? 0.0 : 0;
      hl.forEach((h) => h.setAttribute("opacity", step === 1 ? pulse : step === 2 ? 0.25 * (1 - seg(t, SW0, SW1)) : 0));
      const s = ease(seg(t, SW0, SW1));
      out[a].set(lerp(v0, v1, s));
      out[b].set(lerp(v1, v0, s));
      const swapped = s >= 0.5;
      spans.forEach((sp, j) => {
        let txt = vals[j];
        if (swapped && j === a) txt = vals[b];
        if (swapped && j === b) txt = vals[a];
        sp.textContent = txt;
        sp.classList.toggle("hl", (j === a || j === b) && step >= 1 && !swapped);
        sp.classList.toggle("fixed", (j === a || j === b) && swapped);
      });
      const acc = lerp(ex.acc_faulty, rl.acc_after, ease(seg(t, S3 + 0.2, A1)));
      fill.style.width = 100 * acc + "%";
      accv.textContent = t >= S3 ? `${pct0(ex.acc_faulty)} → ${pct0(acc)}` : `accuracy ${pct0(ex.acc_faulty)}`;
      fill.style.background = t >= S3 ? "var(--trained)" : "var(--gray-series)";
    };
  }

  // ---------------------------------------------------------------- 3. results across six splits
  function setupSplits(fig) {
    const el = fig.querySelector('[data-chart="splits"]');
    const rows = DATA.splits;
    const W = Math.max(300, cw(el));
    const narrow = W < 600;
    const L = narrow ? 8 : 200, R = 64, top = 6;
    const rowH = narrow ? 70 : 52;
    const H = top + rows.length * rowH + 30;
    const svg = svgFor(el, W, H, "Bar chart of repair rate on six test sets, trained against untrained, with the heuristic marked. " +
      rows.map((r) => `${r.label}: untrained ${pct1(r.untrained)}, trained ${pct1(r.trained)}, heuristic ${pct1(r.outlier)}`).join("; "));
    const x = (v) => L + v * (W - L - R);
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      S("line", { x1: x(v), x2: x(v), y1: top, y2: H - 24, class: "grid" }, svg);
      T(svg, x(v), H - 8, `${v * 100}%`, { "text-anchor": "middle", "font-size": 11 });
    }
    const items = rows.map((r, i) => {
      const y0 = top + i * rowH + (narrow ? 22 : 8);
      T(svg, narrow ? L : L - 12, narrow ? y0 - 7 : y0 + 14, r.short, { "text-anchor": narrow ? "start" : "end", "font-size": 13, class: "t-ink" });
      if (!narrow) T(svg, L - 12, y0 + 30, `${Math.round(r.n_faulty)} faulty networks`, { "text-anchor": "end", "font-size": 10.5 });
      const tb = S("rect", { x: L, y: y0, height: 18, rx: 4, fill: "var(--trained)", "data-tip": `${r.label}: trained ${pct1(r.trained)} [${r1(r.trained_lo)}, ${r1(r.trained_hi)}]` }, svg);
      const ub = S("rect", { x: L, y: y0 + 22, height: 7, rx: 3, fill: "var(--untrained)", "data-tip": `${r.label}: untrained ${pct1(r.untrained)}` }, svg);
      const ci = S("g", { opacity: 0 }, svg);
      S("line", { x1: x(r.trained_lo), x2: x(r.trained_hi), y1: y0 + 9, y2: y0 + 9, stroke: "var(--ink)", "stroke-width": 1.5 }, ci);
      for (const v of [r.trained_lo, r.trained_hi]) S("line", { x1: x(v), x2: x(v), y1: y0 + 4, y2: y0 + 14, stroke: "var(--ink)", "stroke-width": 1.5 }, ci);
      const tick = S("line", { x1: x(r.outlier), x2: x(r.outlier), y1: y0 - 3, y2: y0 + 32, stroke: "var(--ink)", "stroke-width": 2.5, "stroke-linecap": "round", opacity: 0, "data-tip": `Heuristic: ${pct1(r.outlier)}` }, svg);
      const lab = T(svg, x(r.trained_hi) + 6, y0 + 14, pct1(r.trained), Object.assign({ "font-size": 13, class: "t-strong", opacity: 0 }, halo));
      const ulab = T(svg, x(Math.max(r.untrained, r.outlier)) + 6, y0 + 30, pct1(r.untrained), Object.assign({ "font-size": 10.5, opacity: 0 }, halo));
      return { r, tb, ub, ci, tick, lab, ulab };
    });
    return (t) => {
      items.forEach((it, i) => {
        const g = ease(seg(t, 0.2 + 0.18 * i, 1.3 + 0.18 * i));
        it.tb.setAttribute("width", Math.max(0, x(it.r.trained * g) - L));
        it.ub.setAttribute("width", Math.max(2, x(it.r.untrained * g) - L));
        const o = seg(t, 1.2 + 0.18 * i, 1.6 + 0.18 * i);
        [it.ci, it.lab, it.ulab].forEach((e) => e.setAttribute("opacity", o));
        it.tick.setAttribute("opacity", seg(t, 0.1, 0.5));
      });
    };
  }

  // ---------------------------------------------------------------- generic line chart
  function lineChart(el, opt) {
    // opt: {xs: fn value->[0,1], xTicks:[{v,label}], yMax, yTicks, series:[{pts:[{x,y,lo,hi}], color, dash, label, ring}], hlines:[{y,label,pos}], label, height}
    const W = Math.max(300, cw(el));
    const H = opt.height || Math.round(Math.min(380, Math.max(260, W * 0.5)));
    const L = 46, R = opt.right == null ? 64 : opt.right, top = 14, B = 40;
    const svg = svgFor(el, W, H, opt.label);
    const x = (v) => L + opt.xs(v) * (W - L - R);
    const y = (v) => top + (1 - v / opt.yMax) * (H - top - B);
    for (const v of opt.yTicks) {
      S("line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "grid" }, svg);
      T(svg, L - 8, y(v) + 4, `${Math.round(v * 100)}%`, { "text-anchor": "end", "font-size": 11 });
    }
    for (const tk of opt.xTicks) T(svg, x(tk.v), H - B + 18, tk.label, { "text-anchor": "middle", "font-size": 11 });
    if (opt.xLabel) T(svg, L + (W - L - R) / 2, H - 4, opt.xLabel, { "text-anchor": "middle", "font-size": 11.5 });
    const clipId = "c" + Math.random().toString(36).slice(2);
    const clip = S("clipPath", { id: clipId }, S("defs", {}, svg));
    const cr = S("rect", { x: 0, y: 0, width: 0, height: H }, clip);
    const gD = S("g", { "clip-path": `url(#${clipId})` }, svg);
    const fades = [];
    for (const h of opt.hlines || []) {
      const g = S("g", { opacity: 0 }, svg);
      S("line", { x1: L, x2: W - R, y1: y(h.y), y2: y(h.y), stroke: "var(--ink-2)", "stroke-width": 1.3, "stroke-dasharray": "5 4" }, g);
      const right = h.anchor === "end";
      T(g, right ? W - R : L + 6, y(h.y) + (h.below ? 14 : -6), h.label, Object.assign({ "font-size": 11.5, "text-anchor": right ? "end" : "start" }, halo));
      fades.push(g);
    }
    for (const s of opt.series) {
      const pts = s.pts;
      if (s.band && pts.length > 1) {
        const d = pts.map((p, i) => `${i ? "L" : "M"}${x(p.x)},${y(p.hi)}`).join("") + pts.slice().reverse().map((p) => `L${x(p.x)},${y(p.lo)}`).join("") + "Z";
        S("path", { d, fill: s.color, opacity: 0.13 }, gD);
      }
      if (pts.length > 1 && !s.ring) S("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${x(p.x)},${y(p.y)}`).join(""), fill: "none", stroke: s.color, "stroke-width": s.width || 2.2, "stroke-dasharray": s.dash || "", "stroke-linejoin": "round" }, gD);
      for (const p of pts) {
        if (s.whisker && p.lo != null) S("line", { x1: x(p.x), x2: x(p.x), y1: y(p.lo), y2: y(p.hi), stroke: s.color, "stroke-width": 1.5 }, gD);
        S("circle", { cx: x(p.x), cy: y(p.y), r: s.r || 4.5, fill: s.ring ? "var(--bg)" : s.color, stroke: s.ring ? s.color : "var(--bg)", "stroke-width": s.ring ? 2.2 : 1.5, "data-tip": `${s.label}: ${p.tip || pct1(p.y)}` }, gD);
      }
      if (s.endLabel) {
        const p = pts[pts.length - 1];
        const g = S("g", { opacity: 0 }, svg);
        T(g, x(p.x) + 9, y(p.y) + 4 + (s.endDy || 0), s.endLabel, Object.assign({ "font-size": 12.5, class: "t-strong" }, halo));
        fades.push(g);
      }
    }
    return { svg, x, y, W, H, L, R, top, B, reveal: (f) => cr.setAttribute("width", L + f * (W - L) + 12), fade: (o) => fades.forEach((g) => g.setAttribute("opacity", o)) };
  }

  // ---------------------------------------------------------------- 4. training curve
  function setupTrain(fig) {
    const el = fig.querySelector('[data-chart="train"]');
    const tr = DATA.training;
    const split = 0.7, s0 = tr.sft_steps, s1 = s0 + tr.rl_steps;
    const xs = (v) => (v <= s0 ? (v / s0) * split : split + ((v - s0) / (s1 - s0)) * (1 - split));
    const pts = (key) => tr.sft.map((p) => ({ x: p.step, y: p[key] })).concat(tr.rl.map((p) => ({ x: p.step, y: p[key], tip: `RL chunk ${p.chunk}: ${pct1(p[key])}` })));
    const sel = tr.rl.find((p) => p.chunk === tr.selected_chunk);
    const c = lineChart(el, {
      xs, yMax: 1, yTicks: [0, 0.25, 0.5, 0.75, 1], right: 20,
      xTicks: [0, 300, 600, 900, s1].map((v) => ({ v, label: String(v) })), xLabel: "training step",
      label: `Validation repair rate during training. Supervised stage from ${pct0(tr.sft[0].val)} to ${pct1(tr.sft[tr.sft.length - 1].val)} on hand-built and ${pct1(tr.sft[tr.sft.length - 1].val_trained)} on gradient-made networks; selected RL checkpoint chunk ${tr.selected_chunk}: ${pct1(sel.val)} and ${pct1(sel.val_trained)}.`,
      series: [
        { pts: pts("val"), color: "var(--trained)", label: "Hand-built validation" },
        { pts: pts("val_trained"), color: "var(--aqua)", label: "Gradient-made validation" },
      ],
    });
    const { svg, x, y, H, B, top } = c;
    const xd = x(s0);
    S("line", { x1: xd, x2: xd, y1: top, y2: H - B, stroke: "var(--ink-2)", "stroke-width": 1, "stroke-dasharray": "3 3" }, svg);
    T(svg, xd - 8, top + 12, "supervised", { "text-anchor": "end", "font-size": 11.5 });
    T(svg, xd + 8, top + 12, "RL", { "font-size": 11.5 });
    const g = S("g", { opacity: 0 }, svg);
    const xsSel = x(sel.step);
    S("line", { x1: xsSel, x2: xsSel, y1: y(sel.val) - 8, y2: H - B, stroke: "var(--ink)", "stroke-width": 1.2 }, g);
    S("circle", { cx: xsSel, cy: y(sel.val), r: 8, fill: "none", stroke: "var(--ink)", "stroke-width": 1.6 }, g);
    S("circle", { cx: xsSel, cy: y(sel.val_trained), r: 8, fill: "none", stroke: "var(--ink)", "stroke-width": 1.6 }, g);
    const lx = xsSel - 10;
    T(g, lx, y(0.3), `selected: chunk ${tr.selected_chunk}`, Object.assign({ "text-anchor": "end", "font-size": 12, class: "t-strong" }, halo));
    T(g, lx, y(0.3) + 16, `${pct1(sel.val)} / ${pct1(sel.val_trained)}`, Object.assign({ "text-anchor": "end", "font-size": 12 }, halo));
    return (t) => { c.reveal(ease(seg(t, 0.2, 3.0))); g.setAttribute("opacity", seg(t, 3.0, 3.5)); };
  }

  // ---------------------------------------------------------------- 5. tokens vs accuracy
  function setupTokens(fig) {
    const el = fig.querySelector('[data-chart="tokens"]');
    const comp = DATA.comparison, base = DATA.baselines_id;
    const by = Object.fromEntries(comp.map((c) => [c.key, c]));
    const lo = Math.log10(10), hi = Math.log10(30000);
    const xs = (v) => (Math.log10(v) - lo) / (hi - lo);
    const narrow = cw(el) < 560;
    const c = lineChart(el, {
      xs, yMax: 1, yTicks: [0, 0.25, 0.5, 0.75, 1], right: 16, height: narrow ? 360 : 400,
      xTicks: [10, 100, 1000, 10000].map((v) => ({ v, label: v.toLocaleString("en-US") })), xLabel: "tokens generated per network (log scale)",
      label: "Scatter of repair rate against tokens per network. " + comp.map((c) => `${c.label}: ${pct1(c.rate)} at about ${fmtInt(c.tokens)} tokens`).join("; ") +
        `. Methods without tokens: forward-pass search ${pct1(base.search)}, gradient descent ${pct1(base.gradient)}, heuristic ${pct1(base.outlier)}.`,
      hlines: [
        { y: base.search, label: `${narrow ? "Search" : "Forward-pass search"} ${pct1(base.search)}` },
        { y: base.gradient, label: `${narrow ? "Gradient" : "Gradient descent"} ${pct1(base.gradient)}`, below: true },
        { y: base.outlier, label: `${narrow ? "Heuristic" : "Statistics-only heuristic"} ${pct1(base.outlier)}`, anchor: narrow ? "end" : "start" },
      ],
      series: [],
    });
    const { svg, x, y } = c;
    const g = S("g", {}, svg);
    const pts = [];
    const dot = (cc, color, ring) => {
      const gg = S("g", { opacity: 0 }, g);
      if (!cc.offshelf) S("line", { x1: x(cc.tokens), x2: x(cc.tokens), y1: y(cc.lo), y2: y(cc.hi), stroke: color, "stroke-width": 1.6 }, gg);
      S("circle", { cx: x(Math.min(cc.tokens, 30000)), cy: y(cc.rate), r: ring ? (cc.open ? 4.5 : 5.5) : 7, fill: ring ? "var(--bg)" : color, stroke: ring ? color : "var(--bg)", "stroke-width": ring && cc.open ? 1.6 : 2,
        opacity: cc.offshelf && cc.cut >= 0.25 ? 0.38 : 1,
        "data-tip": `${cc.label}: ${pct1(cc.rate)}, ~${fmtInt(cc.tokens)} tokens${cc.offshelf ? ` (first ${Math.round(cc.n_total)} networks${cc.cut >= 0.005 ? `, ${Math.round(100 * cc.cut)}% of answers cut off` : ""})` : ""}` }, gg);
      pts.push(gg);
      return gg;
    };
    const off = comp.filter((c) => c.offshelf);
    const fr = off.filter((c) => /^frontier_(sol|astra)_/.test(c.key));
    const gOff = S("g", { opacity: 0 }, g);
    off.filter((c) => c.open).concat(off.filter((c) => !c.open)).forEach((f) => { const d = dot(f, f.open ? "var(--untrained)" : "var(--ink-2)", true); pts.pop(); d.setAttribute("opacity", 1); gOff.appendChild(d); });
    pts.push(gOff);
    const keyEl = fig.querySelector('[data-key="tokens"]'), key = [];
    const lab = (gg, n, px, py, text, attrs, kx, ky) => {
      if (!narrow) { T(gg, px, py, text, Object.assign({ "font-size": 12 }, attrs, halo)); return; }
      const b = S("g", {}, gg);
      S("circle", { cx: kx, cy: ky, r: 8, fill: "var(--ink)" }, b);
      T(b, kx, ky + 3.8, String(n), { "text-anchor": "middle", "font-size": 10.5, style: "fill: var(--bg); font-weight: 600" });
      key.push(`<li><span class="kn">${n}</span>${esc(text)}</li>`);
    };
    const gfl = S("g", { opacity: 0 }, g);
    const fx = Math.max(...fr.map((f) => x(f.tokens)));
    lab(gfl, 1, fx + 10, y(0.985), "GPT-6 Astra / GPT-6.1 Sol", { "font-size": 11.5 }, fx + 16, y(1));
    pts.push(gfl);
    const dT = dot(by.trained, "var(--trained)");
    lab(dT, 2, x(by.trained.tokens) + 12, y(by.trained.rate) + 4, `Post-trained on Tinker, no reasoning: ${pct1(by.trained.rate)}, ~${Math.round(by.trained.tokens)} tokens`, { class: "t-strong" }, x(by.trained.tokens) + 16, y(by.trained.rate) - 2);
    const dR = dot(by.think, "var(--untrained)");
    lab(dR, 3, x(by.think.tokens) - 12, y(by.think.rate) - 10, `Untrained, reasoning: ${pct1(by.think.rate)}, ~${fmtInt(Math.round(by.think.tokens / 100) * 100)} tokens`, { class: "t-strong", "text-anchor": "end" }, x(by.think.tokens) - 16, y(by.think.rate) - 12);
    const dU = dot(by.untrained, "var(--untrained)");
    // Several off-the-shelf rings sit beside this point, so its label goes above it on a short leader.
    if (!narrow) S("line", { x1: x(by.untrained.tokens), x2: x(by.untrained.tokens), y1: y(by.untrained.rate) - 9, y2: y(0.105), stroke: "var(--ink-2)", "stroke-width": 1 }, dU);
    lab(dU, 4, x(by.untrained.tokens), y(0.12), `Untrained, told to answer at once: ${pct1(by.untrained.rate)}`, { "font-size": 11.5, class: "t-ink", "text-anchor": "middle" }, x(by.untrained.tokens) - 16, y(by.untrained.rate) - 12);
    if (keyEl) { keyEl.innerHTML = key.join(""); keyEl.hidden = !key.length; }
    // arrow from untrained-with-reasoning to trained
    const x0 = x(by.think.tokens), y0 = y(by.think.rate) + 10, x1 = x(by.trained.tokens) + 4, y1 = y(by.trained.rate) + 12;
    const cx = (x0 + x1) / 2, cy = Math.max(y0, y1) + (y(0.45) - y(0.8));
    const arrow = S("path", { d: `M${x0},${y0} Q${cx},${cy} ${x1},${y1}`, fill: "none", stroke: "var(--trained)", "stroke-width": 2, "stroke-dasharray": "1000", "stroke-dashoffset": "1000" }, svg);
    const len = arrow.getTotalLength ? arrow.getTotalLength() : 600;
    arrow.setAttribute("stroke-dasharray", `${len}`);
    const head = S("path", { d: "M0,0 L-9,-5 L-9,5 Z", fill: "var(--trained)", opacity: 0 }, svg);
    const ang = Math.atan2(y1 - cy, x1 - cx) * 180 / Math.PI;
    head.setAttribute("transform", `translate(${x1},${y1}) rotate(${ang})`);
    const al = T(svg, cx, narrow ? y(0.485) : (y0 + y1) / 2 + (cy - (y0 + y1) / 2) / 2 + 18, `after training: ${r1(DATA.headline.vs_reasoning.diff)} points higher (paired)`, Object.assign({ "text-anchor": "middle", "font-size": 11.5, opacity: 0, class: "t-ink" }, halo));
    return (t) => {
      c.reveal(1);
      c.fade(seg(t, 0.1, 0.6));
      pts.forEach((p, i) => p.setAttribute("opacity", seg(t, 0.3 + 0.12 * i, 0.7 + 0.12 * i)));
      const a = ease(seg(t, 1.8, 3.0));
      arrow.setAttribute("stroke-dashoffset", len * (1 - a));
      head.setAttribute("opacity", a >= 0.98 ? 1 : 0);
      al.setAttribute("opacity", seg(t, 2.8, 3.3));
    };
  }

  // ---------------------------------------------------------------- 6. verifier
  function setupVerify(fig) {
    const el = fig.querySelector('[data-chart="verify"]');
    const v = DATA.verify, ks = [1, 2, 4, 8];
    const xs = (k) => Math.log2(k) / 3;
    const colors = ["var(--trained)", "var(--aqua)", "var(--violet)"];
    const series = [];
    v.splits.forEach((s, i) => {
      series.push({ pts: ks.map((k) => ({ x: k, y: v.trained[s][k].fixed, lo: v.trained[s][k].lo, hi: v.trained[s][k].hi })), color: colors[i], label: `${v.labels[i]}, trained`, band: true, endLabel: pct1(v.trained[s][8].fixed) });
      series.push({ pts: ks.map((k) => ({ x: k, y: v.random[s][k].fixed })), color: colors[i], dash: "4 4", width: 1.4, r: 2.5, label: `${v.labels[i]}, random edits` });
    });
    const c = lineChart(el, { xs, yMax: 1, yTicks: [0, 0.25, 0.5, 0.75, 1], xTicks: ks.map((k) => ({ v: k, label: String(k) })), xLabel: "verified answers per network (k)", series,
      label: "Share of faulty networks fixed within k verified answers. " + v.splits.map((s, i) => `${v.labels[i]}: ${pct1(v.trained[s][1].fixed)} at k=1, ${pct1(v.trained[s][8].fixed)} at k=8; random edits ${pct1(v.random[s][8].fixed)} at k=8`).join("; ") });
    return (t) => { c.reveal(ease(seg(t, 0.2, 2.2))); c.fade(seg(t, 2.1, 2.5)); };
  }

  // ---------------------------------------------------------------- 7. propose and verify, replayed
  function setupLoop(fig) {
    const ie = DATA.improve_example;
    const ol = fig.querySelector(".proposals");
    ol.innerHTML = "";
    const lis = ie.proposals.map((p, i) => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="pn">#${i + 1}</span><span class="pj" title="${esc(p.answer)}">${esc(p.answer)}</span><span class="pv"></span>`;
      ol.appendChild(li);
      return li;
    });
    const val = fig.querySelector(".loop-acc-val"), lab = fig.querySelector(".loop-acc-label"), note = fig.querySelector(".loop-note");
    const fill = fig.querySelector(".loop .acc-fill"), basebar = fig.querySelector(".loop .acc-base");
    const errs0 = 1 - ie.acc_before;
    const t0 = 0.5, dt = 1.2, check = 0.55;
    return (t) => {
      let best = ie.acc_before, bestIdx = -1;
      lis.forEach((li, i) => {
        const ta = t0 + i * dt;
        const shown = t >= ta, checked = t >= ta + check;
        li.classList.toggle("in", shown);
        li.classList.remove("keep", "rej", "ok");
        const p = ie.proposals[i], pv = li.querySelector(".pv");
        if (!checked) { pv.textContent = shown ? "checking…" : ""; return; }
        if (!p.valid) { pv.textContent = "invalid answer"; li.classList.add("rej"); return; }
        if (p.acc <= ie.acc_before + 1e-9) { pv.textContent = `${pct1(p.acc)} · ${p.acc < ie.acc_before - 1e-9 ? "worse" : "no gain"}, rejected`; li.classList.add("rej"); return; }
        if (p.acc > best + 1e-9) {
          if (bestIdx >= 0) { lis[bestIdx].classList.remove("keep"); lis[bestIdx].classList.add("ok"); lis[bestIdx].querySelector(".pv").textContent = `${pct1(ie.proposals[bestIdx].acc)} · better, superseded`; }
          best = p.acc; bestIdx = i; li.classList.add("keep"); pv.textContent = `${pct1(p.acc)} · kept, best so far`;
        } else { li.classList.add("ok"); pv.textContent = `${pct1(p.acc)} · better, not the best`; }
      });
      const removed = (best - ie.acc_before) / errs0;
      lab.textContent = "Accuracy";
      note.textContent = `Bar: the network's starting errors (${pct1(errs0)} of inputs); blue = share removed by the kept edit, ${pct0(removed)}.`;
      val.textContent = best > ie.acc_before ? `${pct1(ie.acc_before)} → ${pct1(best)}` : pct1(ie.acc_before);
      basebar.style.width = "100%";
      fill.style.width = 100 * removed + "%";
      fill.setAttribute("title", "");
      const done = t >= t0 + (lis.length - 1) * dt + check + 0.4;
      if (done) note.textContent = `Kept proposal #${bestIdx + 1}. Bar: the network's starting errors (${pct1(errs0)} of inputs); blue = share removed, ${pct0(removed)}.`;
    };
  }

  // ---------------------------------------------------------------- 8. improve curve
  function setupImprove(fig) {
    const el = fig.querySelector('[data-chart="improve"]');
    const im = DATA.improve;
    const xs = (k) => Math.log2(k) / 7;
    const ser = (arr, color, label, endLabel) => ({ pts: arr.map((p) => ({ x: p.k, y: p.gap_closed, lo: p.gap_lo, hi: p.gap_hi, tip: `${gap1(p.gap_closed)} with ${p.k} proposal${p.k > 1 ? "s" : ""}` })), color, label, band: true, endLabel });
    const s2 = im.series.seed2;
    const c = lineChart(el, {
      xs, yMax: 0.25, yTicks: [0, 0.05, 0.1, 0.15, 0.2, 0.25], xTicks: [1, 8, 32, 128].map((k) => ({ v: k, label: String(k) })), xLabel: "proposals checked per network",
      hlines: [{ y: im.search.gap_closed, label: `Exhaustive search over simple edits ${gap1(im.search.gap_closed)}` }],
      series: [
        ser(im.series.random, "var(--gray-series)", "Random valid edits", gap1(im.series.random[im.series.random.length - 1].gap_closed)),
        ser(im.series.untrained, "var(--untrained)", "Untrained", gap1(im.series.untrained[1].gap_closed)),
        ser(im.series.trained, "var(--trained)", "RL-trained", gap1(im.series.trained[1].gap_closed)),
        { pts: [{ x: 8, y: s2[1].gap_closed, tip: `second seed ${gap1(s2[1].gap_closed)} with 8 proposals` }], color: "var(--trained)", ring: true, r: 5, label: "RL, second seed" },
      ],
      label: `Share of errors removed against proposals checked. With 8 proposals: RL-trained ${gap1(im.series.trained[1].gap_closed)}, second seed ${gap1(s2[1].gap_closed)}, random ${gap1(im.series.random[1].gap_closed)}, untrained ${gap1(im.series.untrained[1].gap_closed)}. Random with 32 and 128: ${gap1(im.series.random[2].gap_closed)} and ${gap1(im.series.random[3].gap_closed)}.`,
    });
    return (t) => { c.reveal(ease(seg(t, 0.2, 2.2))); c.fade(seg(t, 2.1, 2.5)); };
  }

  // ---------------------------------------------------------------- 9. pure RL collapse
  function setupCollapse(fig) {
    const el = fig.querySelector('[data-chart="collapse"]');
    const pr = DATA.pure_rl;
    const cols = { "2e-4": "var(--untrained)", "6e-5": "var(--violet)", "2e-5": "var(--gray-series)" };
    const maxStep = Math.max(...Object.values(pr).flat().map((p) => p.step));
    const c = lineChart(el, {
      xs: (s) => s / maxStep, yMax: 1, yTicks: [0, 0.5, 1], height: 220, right: 20,
      xTicks: Array.from({ length: maxStep + 1 }, (_, i) => ({ v: i, label: String(i) })), xLabel: "RL step",
      series: Object.keys(cols).map((lr) => ({ pts: pr[lr].map((p) => ({ x: p.step, y: p.noop })), color: cols[lr], label: `lr ${lr}` })),
      label: "Share of faulty training networks answered with no edit, per RL step. " + Object.keys(cols).map((lr) => `lr ${lr}: ${pr[lr].map((p) => pct0(p.noop)).join(", ")}`).join("; "),
    });
    return (t) => c.reveal(ease(seg(t, 0.2, 2.0)));
  }

  // ---------------------------------------------------------------- 10. image heads
  function setupHeads(fig) {
    const el = fig.querySelector('[data-chart="heads"]');
    const hs = DATA.imagehead;
    const counts = {};
    hs.forEach((h) => { const k = Math.round(h.change * 4) / 4; counts[k] = (counts[k] || 0) + 1; });
    const W = Math.max(300, cw(el)), H = 70 + 11 * Math.max(...Object.values(counts)), L = 20, R = 20;
    const lo = Math.min(-1, ...hs.map((h) => h.change)), hi = Math.max(5, ...hs.map((h) => h.change)) + 0.5;
    const svg = svgFor(el, W, H, "Change in test accuracy after the kept edit for each of the image classifiers: " + hs.map((h) => `${h.model} ${h.change >= 0 ? "+" : ""}${h.change.toFixed(2)} points`).join("; "));
    const x = (v) => L + ((v - lo) / (hi - lo)) * (W - L - R);
    const base = H - 40;
    for (let v = Math.ceil(lo); v <= hi; v++) {
      S("line", { x1: x(v), x2: x(v), y1: 4, y2: base + 6, class: v === 0 ? "axis" : "grid", "stroke-width": v === 0 ? 1.5 : 1 }, svg);
      T(svg, x(v), base + 22, `${v > 0 ? "+" : ""}${v}`, { "text-anchor": "middle", "font-size": 11 });
    }
    T(svg, W / 2, H - 2, "change in test accuracy (points)", { "text-anchor": "middle", "font-size": 11.5 });
    const stack = {};
    const dots = hs.map((h) => {
      const key = Math.round(h.change * 4) / 4;
      const k = (stack[key] = (stack[key] || 0) + 1) - 1;
      return S("circle", { cx: x(h.change), cy: base - 8 - k * 11, r: 4.8, fill: h.change > 0.5 ? "var(--trained)" : "var(--gray-series)", stroke: "var(--bg)", "stroke-width": 1.2, opacity: 0,
        "data-tip": `${h.model} (${h.dataset}): ${h.before.toFixed(2)}% → ${h.after.toFixed(2)}%, claimed ${h.claimed.toFixed(2)}%` }, svg);
    });
    return (t) => dots.forEach((d, i) => d.setAttribute("opacity", seg(t, 0.1 + 0.08 * i, 0.4 + 0.08 * i)));
  }

  // ---------------------------------------------------------------- Qwen slot (only from site/qwen.json)
  function wrapText(text, maxChars) {
    const words = String(text).split(/\s+/), lines = [];
    let cur = "";
    for (const w of words) { if (cur && (cur + " " + w).length > maxChars) { lines.push(cur); cur = w; } else cur = cur ? cur + " " + w : w; }
    if (cur) lines.push(cur);
    return lines;
  }
  function drawQwenBars(el, bars) {
    // labels sit above their bars (they can be long), wrapped to the chart width; the note follows in muted text
    const W = Math.max(300, cw(el)), L = 8, R = 64, top = 4, LH = 16;
    const maxChars = Math.max(24, Math.floor((W - L - 8) / 7.2));
    const rows = bars.map((b) => ({ b, lines: wrapText(b.label + (b.note ? ` (${b.note})` : ""), maxChars) }));
    const max = Math.min(1, Math.ceil(Math.max(...bars.map((b) => b.hi != null ? b.hi : b.value)) * 10 + 0.5) / 10);
    let y = top;
    rows.forEach((r) => { r.y0 = y + r.lines.length * LH + 4; y = r.y0 + 18 + 16; });
    const H = y + 24;
    const svg = svgFor(el, W, H, "MATH-500 accuracy by condition: " + bars.map((b) => `${b.label} ${pct1(b.value)}`).join("; "));
    const x = (v) => L + (v / max) * (W - L - R);
    const step = max > 0.5 ? 0.2 : max > 0.25 ? 0.1 : 0.05, ticks = []; for (let v = 0; v <= max + 1e-9; v += step) ticks.push(Math.round(v * 100) / 100);
    for (const v of ticks) { S("line", { x1: x(v), x2: x(v), y1: top, y2: H - 22, class: "grid" }, svg); T(svg, x(v), H - 6, pct0(v), { "text-anchor": "middle", "font-size": 11 }); }
    const items = rows.map(({ b, lines, y0 }) => {
      lines.forEach((ln, k) => T(svg, L, y0 - 6 - (lines.length - 1 - k) * LH, ln, Object.assign({ "font-size": 13, class: "t-ink" }, halo)));
      const r = S("rect", { x: L, y: y0, height: 18, rx: 4, fill: b.highlight ? "var(--trained)" : "var(--gray-series)", "data-tip": `${b.label}: ${pct1(b.value)}${b.lo != null ? ` [${r1(b.lo)}, ${r1(b.hi)}]` : ""}` }, svg);
      if (b.lo != null) {
        S("line", { x1: x(b.lo), x2: x(b.hi), y1: y0 + 9, y2: y0 + 9, stroke: "var(--ink)", "stroke-width": 1.5 }, svg);
        for (const v of [b.lo, b.hi]) S("line", { x1: x(v), x2: x(v), y1: y0 + 4, y2: y0 + 14, stroke: "var(--ink)", "stroke-width": 1.5 }, svg);
      }
      T(svg, x(b.hi != null ? b.hi : b.value) + 6, y0 + 14, pct1(b.value), Object.assign({ "font-size": 13, class: "t-strong" }, halo));
      return { r, v: b.value };
    });
    return (t) => items.forEach((it, i) => it.r.setAttribute("width", Math.max(2, x(it.v * ease(seg(t, 0.1 + 0.12 * i, 1.1 + 0.12 * i))) - L)));
  }
  function renderQwen(q) {
    if (!q || !Array.isArray(q.rows)) return;
    const sec = document.getElementById("qwen"), body = document.querySelector(".qwen-body");
    if (!sec || !body) return;  // the page states the real-LLM result in one line; the full table is in the README
    body.querySelector(".qwen-table tbody").innerHTML = q.rows.map((r) =>
      `<tr><th scope=row>${esc(r.label)}</th><td>${pct1(r.math500)}</td><td>[${r1(r.lo)}, ${r1(r.hi)}]</td><td>${esc(r.n)}</td><td class=muted>${esc(r.note || "")}</td></tr>`).join("");
    body.querySelector(".qwen-tablewrap").hidden = !q.rows.length;
    body.querySelector(".qwen-summary").textContent = q.summary || "";
    const ac = body.querySelector(".qwen-across");
    if (ac && q.across) { ac.textContent = q.across; ac.hidden = false; }
    if (Array.isArray(q.caveats) && q.caveats.length) {
      const cv = body.querySelector(".qwen-caveats");
      cv.innerHTML = q.caveats.map((c) => `<li>${esc(c)}</li>`).join("");
      cv.hidden = false;
    }
    sec.hidden = false; body.hidden = false;
    const tq = document.querySelector(".toc-qwen"); if (tq) tq.hidden = false;
    if (Array.isArray(q.bars) && q.bars.length) {
      const fig = document.getElementById("fig-qwen");
      fig.hidden = false;
      fig.setAttribute("data-anim", "qwen");
      register("qwen", 2.0, (f) => drawQwenBars(f.querySelector('[data-chart="qwen"]'), q.bars));
    }
  }

  // ---------------------------------------------------------------- section navigator
  function setupNav() {
    const links = [...document.querySelectorAll(".toc a[href^='#']")].filter((a) => a.getAttribute("href").length > 1 && !a.classList.contains("toc-home"));
    const targets = links.map((a) => document.querySelector(a.getAttribute("href"))).filter(Boolean);
    if (!("IntersectionObserver" in window)) return;
    const vis = new Map();
    const io = new IntersectionObserver((ents) => {
      ents.forEach((e) => vis.set(e.target.id, e.isIntersecting ? e.boundingClientRect.top : null));
      let best = null;
      for (const tgt of targets) { const v = vis.get(tgt.id); if (v != null) { best = tgt.id; break; } }
      if (best) links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + best));
    }, { rootMargin: "-60px 0px -55% 0px" });
    targets.forEach((tg) => io.observe(tg));
  }
  if (DATA.qwen) renderQwen(DATA.qwen);
  else if (location.protocol.startsWith("http") && !CLIP) fetch("qwen.json", { cache: "no-store" }).then((r) => (r.ok ? r.json() : null)).then(renderQwen).catch(() => {});
  setupNav();

  // ---------------------------------------------------------------- clip mode (video export)
  const CLIPS = {
    race: { get title() { return `Same number of tokens: one network, or ${DATA.race.n}`; }, sub: () => `Inkling 2 Small before and after post-training on Tinker. Left: ${fmtInt(DATA.race.budget)} tokens of reasoning on one network. Right: ${DATA.race.n} networks answered in ${fmtInt(DATA.race.used)} tokens, ${DATA.race.repaired} repaired. Same number of generated tokens, not wall-clock time.`,
      foot: () => `All ${Math.round(DATA.headline.trained.n_faulty)} faulty test networks: untrained + reasoning ${pct1(DATA.headline.reasoning.rate)} at ~${fmtInt(Math.round(DATA.headline.reasoning.tokens / 100) * 100)} tokens; trained, no reasoning ${pct1(DATA.headline.trained.rate)} at ~${Math.round(DATA.headline.trained.tokens)} tokens` },
    splits: { title: "Inkling 2 Small, post-trained on Tinker: repair rate on six test sets", sub: () => `Share of broken networks fixed from raw weights, no reasoning tokens. 95% intervals; black tick: statistics-only heuristic.`,
      foot: () => `Weak row: unseen fault types. GPT-6 Astra and GPT-6.1 Sol fix ${pct0(Math.min(...DATA.comparison.filter((c) => /^frontier_(sol|astra)_/.test(c.key)).map((c) => c.rate)))}-${pct0(Math.max(...DATA.comparison.filter((c) => /^frontier_(sol|astra)_/.test(c.key)).map((c) => c.rate)))} of the in-distribution networks.` },
    repair: { title: "What the trained model reads, and the edit it writes", sub: () => `A ${DATA.example.n_weights}-weight test network with two outgoing weights swapped.`, foot: () => "" },
    loop: { title: "RL alone, no labels: propose, verify, keep", sub: () => `The 8 proposals the RL-trained model sampled for one imperfect test network, each checked with forward passes.`,
      foot: () => `All ${Math.round(DATA.improve.series.trained[1].n)} test networks, 8 proposals: ${gap1(DATA.improve.series.trained[1].gap_closed)} of errors removed (random edits ${gap1(DATA.improve.series.random[1].gap_closed)}).` },
    disguise: { title: "Nothing to memorise: one function, five sets of weights", sub: () => `A compiled circuit from the gym under its own disguise functions. Outputs identical on all ${fmtInt(DATA.disguise.n_inputs)} inputs.`, foot: () => "" },
    card: { title: "", sub: () => "", foot: () => "" },
    tokens: { title: "Repair rate against tokens per network", sub: () => "Same in-distribution test networks. Hollow points: other models off the shelf (grey closed, orange open-weights).", foot: () => "" },
    train: { title: "Training on Tinker: validation repair rate", sub: () => "Supervised stage, then RL with the tinker-cookbook loop; checkpoint selected on validation.", foot: () => "" },
  };
  if (CLIP && CLIPS[CLIP]) {
    const fig = document.querySelector(`[data-anim="${CLIP}"]`);
    const frame = document.createElement("div");
    frame.className = "clip-frame clip-" + CLIP;
    const fw = +(params.get("w") || 1280), fh = +(params.get("h") || 720), zoom = +(params.get("zoom") || 1);
    frame.style.width = fw + "px"; frame.style.height = fh + "px";
    document.body.style.width = fw * zoom + "px"; document.body.style.height = fh * zoom + "px";
    if (zoom !== 1) frame.style.zoom = zoom;
    const c = CLIPS[CLIP];
    if (c.title) frame.innerHTML = `<p class="clip-title">${esc(c.title)}</p><p class="clip-sub">${esc(c.sub())}</p>`;
    frame.appendChild(fig);
    if (CLIP !== "card") {
      const foot = document.createElement("div");
      foot.className = "clip-foot";
      foot.innerHTML = `<span>${esc(c.foot())}</span><span>${esc(DATA.byline)}</span>`;
      frame.appendChild(foot);
    }
    document.body.appendChild(frame);
    document.body.classList.add("clip");
  }

  register("card", 2.6, setupCard);
  register("race", 17.4, setupRace);
  register("disguise", 13.5, setupDisguise);
  register("repair", 9, setupRepair);
  register("splits", 3, setupSplits);
  register("train", 3.8, setupTrain);
  register("tokens", 3.6, setupTokens);
  register("verify", 2.6, setupVerify);
  register("loop", 11.6, setupLoop);
  register("improve", 2.6, setupImprove);
  register("collapse", 2.2, setupCollapse);
  register("heads", 1.6, setupHeads);

  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => redrawAll());

  window.__clip = {
    names: Object.keys(ANIMS),
    dur: (n) => ANIMS[n].dur,
    seek: (n, t) => { const a = ANIMS[n]; a.t = t; a.draw(t); },
  };
})();
