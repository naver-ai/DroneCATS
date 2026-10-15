// Same episode, same model, two structures: commander + sub-agents vs. one model driving all four drones.
(() => {
  const root = document.getElementById("compare");
  if (!root) return;
  const DR = ["Drone1", "Drone2", "Drone3", "Drone4"];
  const COL = DC.drone;
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };
  const X0 = -75, X1 = 36, Y0 = 259, Y1 = 341;
  const P = (x, y) => [x - X0, Y1 - y];

  Promise.all([fetch("assets/replay/data.json").then(r => r.json()), fetch("assets/replay/compare.json").then(r => r.json())])
    .then(([M, C]) => init(M, C)).catch(() => { root.textContent = "Comparison data could not be loaded."; });

  function star(parent, x, y, r, fill, stroke) {
    const p = [];
    for (let j = 0; j < 10; j++) {
      const a = Math.PI / 2 + j * Math.PI / 5, rr = j % 2 ? r * 0.45 : r;
      p.push(`${x + rr * Math.cos(a)},${y - rr * Math.sin(a)}`);
    }
    return el("polygon", { points: p.join(" "), fill, stroke: stroke || DC.bg, "stroke-width": 0.3 }, parent);
  }

  function map(M, host) {
    const svg = el("svg", { viewBox: `0 0 ${X1 - X0} ${Y1 - Y0}`, class: "rp-svg" });
    host.appendChild(svg);
    el("rect", { x: 0, y: 0, width: X1 - X0, height: Y1 - Y0, class: "rp-bg" }, svg);
    for (const [x0, y0, x1, y1] of M.walls) {
      const [a, b] = P(x0, y0), [c, d] = P(x1, y1);
      el("line", { x1: a, y1: b, x2: c, y2: d, stroke: DC.wall, "stroke-width": 0.9, "stroke-linecap": "round" }, svg);
    }
    const [gx, gy] = P(M.goal.x, M.goal.y);
    el("circle", { cx: gx, cy: gy, r: 5, class: "rp-goal", "stroke-width": 0.35, "stroke-dasharray": "1.2 0.9" }, svg);
    const [mx, my] = P(-17, 300);
    for (const s of M.signs) {
      const [x, y] = P(s.x, s.y);
      if (s.kind === "red") {
        el("rect", { x: x - 1.6, y: y - 1.1, width: 3.2, height: 2.2, rx: 0.3, fill: DC.red }, svg);
        const vx = x - mx, vy = y - my, L = Math.hypot(vx, vy) || 1;
        const t = el("text", { x: x + 4.2 * vx / L, y: y + 4.2 * vy / L + 1.0, class: "rp-lbl red" }, svg); t.textContent = s.text;
      } else if (s.kind === "entrance") {
        el("rect", { x: x - 0.8, y: y - 1.6, width: 1.6, height: 3.2, rx: 0.3, fill: DC.blue }, svg);
        const t = el("text", { x: x - 2.1, y: y + 0.9, class: "rp-lbl blue" + (s.text === M.goal.text ? " goal" : ""), "text-anchor": "end" }, svg);
        t.textContent = s.text;
      } else if (s.kind === "end") {
        el("rect", { x: x - 0.6, y: y - 0.6, width: 1.2, height: 1.2, fill: DC.blue }, svg);
      }
    }
    star(svg, gx, gy, 1.8, DC.ink, DC.bg).setAttribute("class", "rp-goalstar");
    return svg;
  }

  const at = (tr, t) => {
    let lo = 0, hi = tr.length - 1;
    if (t <= tr[0][0]) return tr[0];
    if (t >= tr[hi][0]) return tr[hi];
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (tr[m][0] <= t) lo = m; else hi = m; }
    const a = tr[lo], b = tr[hi], f = (t - a[0]) / (b[0] - a[0]);
    let dy = b[3] - a[3]; if (dy > 180) dy -= 360; if (dy < -180) dy += 360;
    return [t, a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), a[3] + f * dy];
  };

  function init(M, C) {
    const sides = [
      { key: "commander", title: "Commander + sub-agents", sub: "Gemini 3.7 Flash commands; four Gemini Robotics-ER 2 sub-agents fly" },
      { key: "centralized", title: "One model drives all four", sub: "Gemini 3.7 Flash returns four actions per call" },
    ];
    root.innerHTML = `<div class="cp-grid">${sides.map(s => `<div class="cp-side" data-k="${s.key}">
        <div class="cp-head"><b>${s.title}</b><span class="cp-badge"></span></div>
        <div class="cp-sub">${s.sub}</div>
        <div class="rp-map cp-map"></div>
        </div>`).join("")}</div>
      <div class="rp-ctrl cp-ctrl"></div>`;
    const S = {};
    for (const s of sides) {
      const box = root.querySelector(`[data-k="${s.key}"]`);
      const svg = map(M, box.querySelector(".cp-map"));
      const gT = el("g", {}, svg), gD = el("g", {}, svg), gB = el("g", {}, svg);
      const trail = {}, body = {}, cone = {};
      for (const d of DR) {
        const [sx, sy] = P(...M.start[d]);
        el("circle", { cx: sx, cy: sy, r: 0.9, fill: DC.bg, stroke: COL[d], "stroke-width": 0.35 }, gT);
        trail[d] = el("polyline", { fill: "none", stroke: COL[d], "stroke-width": 0.7, "stroke-linejoin": "round", opacity: 0.95 }, gT);
        cone[d] = el("path", { fill: COL[d], opacity: 0.18 }, gB);
        const g = el("g", {}, gB);
        el("circle", { r: 1.9, fill: COL[d], stroke: DC.bg, "stroke-width": 0.45 }, g);
        const t = el("text", { y: 0.75, class: "rp-num" }, g); t.textContent = d.slice(-1);
        body[d] = g;
      }
      S[s.key] = { box, data: C[s.key], trail, body, cone, gD, badge: box.querySelector(".cp-badge"),
 };
    }
    const T_END = C.cap;
    const ctrl = root.querySelector(".cp-ctrl");
    ctrl.innerHTML = `<button class="rp-btn" data-k="play" type="button">Pause</button>
      <span class="rp-speeds">${[8, 16, 32].map(s => `<button class="rp-btn${s === 16 ? " on" : ""}" data-s="${s}" type="button">${s}×</button>`).join("")}</span>
      <input class="rp-range" type="range" min="0" max="${T_END}" step="0.1" value="0" aria-label="Comparison time">
      <span class="rp-clock"></span>`;
    let t = 0, playing = true, speed = 16, last = performance.now(), visible = true;
    const range = ctrl.querySelector(".rp-range"), clock = ctrl.querySelector(".rp-clock"), playBtn = ctrl.querySelector('[data-k="play"]');
    playBtn.onclick = () => { playing = !playing; playBtn.textContent = playing ? "Pause" : "Play"; if (playing && t >= T_END) t = 0; };
    ctrl.querySelectorAll("[data-s]").forEach(b => b.onclick = () => { speed = +b.dataset.s; ctrl.querySelectorAll("[data-s]").forEach(x => x.classList.toggle("on", x === b)); });
    range.oninput = () => { t = +range.value; draw(); };
    new IntersectionObserver(es => { visible = es[0].isIntersecting; }, { threshold: 0.05 }).observe(root);

    function draw() {
      for (const k in S) {
        const s = S[k], D = s.data;
        const done = D.success && t >= D.success.t;
        const tt = done ? D.success.t : t;        // a solved episode stops where it was solved
        for (const d of DR) {
          const tr = D.tracks[d];
          const pts = [];
          for (const r of tr) { if (r[0] > tt) break; const [x, y] = P(r[1], r[2]); pts.push(`${x.toFixed(2)},${y.toFixed(2)}`); }
          const [, x, y, yaw] = at(tr, tt);
          const [px, py] = P(x, y);
          pts.push(`${px.toFixed(2)},${py.toFixed(2)}`);
          s.trail[d].setAttribute("points", pts.join(" "));
          s.body[d].setAttribute("transform", `translate(${px} ${py})`);
          const h = yaw * Math.PI / 180, R = 7;
          let path = `M${px.toFixed(2)},${py.toFixed(2)}`;
          for (let j = 0; j <= 12; j++) { const a = h - Math.PI / 4 + j * Math.PI / 24; path += ` L${(px + R * Math.cos(a)).toFixed(2)},${(py - R * Math.sin(a)).toFixed(2)}`; }
          s.cone[d].setAttribute("d", path + " Z");
        }
        s.gD.innerHTML = "";
        for (const e of D.decl) {
          if (e.t > tt) break;
          const [x, y] = P(e.x, e.y);
          if (e.ok) star(s.gD, x, y, 2.6, DC.ok, DC.ink);
          else el("path", { d: `M${x - 1.1},${y - 1.1} L${x + 1.1},${y + 1.1} M${x - 1.1},${y + 1.1} L${x + 1.1},${y - 1.1}`, stroke: COL[e.drone], "stroke-width": 0.6 }, s.gD);
        }
        if (done) { s.badge.textContent = `arrived at ${D.success.t.toFixed(0)} s`; s.badge.className = "cp-badge ok"; }
        else if (t >= T_END) { s.badge.textContent = `not arrived by ${T_END.toFixed(0)} s`; s.badge.className = "cp-badge bad"; }
        else { s.badge.textContent = "flying"; s.badge.className = "cp-badge"; }
        s.box.classList.toggle("won", done);
      }
      range.value = t;
      clock.textContent = `${t.toFixed(0)} s / ${T_END.toFixed(0)} s`;
    }
    function tick(now) {
      const dt = (now - last) / 1000; last = now;
      if (playing && visible) {
        t += dt * speed;
        if (t > T_END) {
          t = T_END; draw(); playing = false; playBtn.textContent = "Replay";
          setTimeout(() => { if (!playing) { t = 0; playing = true; playBtn.textContent = "Pause"; } }, 3000);
        }
        draw();
      }
      requestAnimationFrame(tick);
    }
    const q = new URLSearchParams(location.search).get("ct");   // ?ct=200 opens paused at 200 s
    if (q !== null) { t = +q; playing = false; playBtn.textContent = "Play"; }
    draw();
    requestAnimationFrame(tick);
  }
})();
