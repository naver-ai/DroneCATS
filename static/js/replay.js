// Commander + sub-agents replay: one Three-sign sum episode, driven by the run's logs.
(() => {
  const root = document.getElementById("replay");
  if (!root) return;
  const DR = ["Drone1", "Drone2", "Drone3", "Drone4"];
  const COL = DC.drone;
  const ACT = DC.act;
  const NS = "http://www.w3.org/2000/svg";
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const num = d => d.slice(-1);
  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };

  fetch("assets/replay/data.json").then(r => r.json()).then(D => init(D)).catch(() => {
    root.querySelector(".rp-map").textContent = "Replay data could not be loaded.";
  });

  function init(D) {
    const T_END = Math.ceil(Math.max(...DR.map(d => D.tracks[d].at(-1)[0]))) + 3;
    // ---- map (x = north to the right, y = east upwards, as in the paper) ----
    const X0 = -75, X1 = 36, Y0 = 259, Y1 = 341;
    const P = (x, y) => [x - X0, Y1 - y];
    const svg = el("svg", { viewBox: `0 0 ${X1 - X0} ${Y1 - Y0}`, class: "rp-svg" });
    root.querySelector(".rp-map").appendChild(svg);
    el("rect", { x: 0, y: 0, width: X1 - X0, height: Y1 - Y0, class: "rp-bg" }, svg);
    for (const [x0, y0, x1, y1] of D.walls) {
      const [a, b] = P(x0, y0), [c, d] = P(x1, y1);
      el("line", { x1: a, y1: b, x2: c, y2: d, stroke: DC.wall, "stroke-width": 0.9, "stroke-linecap": "round" }, svg);
    }
    const [gx, gy] = P(D.goal.x, D.goal.y);
    el("circle", { cx: gx, cy: gy, r: 5, class: "rp-goal", "stroke-width": 0.35, "stroke-dasharray": "1.2 0.9" }, svg);
    for (const s of D.signs) {
      const [x, y] = P(s.x, s.y);
      if (s.kind === "red") {
        el("rect", { x: x - 1.6, y: y - 1.1, width: 3.2, height: 2.2, rx: 0.3, fill: DC.red }, svg);
        // the drone that reads a sign parks in front of it: put the number on the far side
        const [mx, my] = P(-17, 300), vx = x - mx, vy = y - my, L = Math.hypot(vx, vy) || 1;
        const t = el("text", { x: x + 4.2 * vx / L, y: y + 4.2 * vy / L + 1.0, class: "rp-lbl red" }, svg); t.textContent = s.text;
      } else if (s.kind === "entrance") {
        el("rect", { x: x - 0.8, y: y - 1.6, width: 1.6, height: 3.2, rx: 0.3, fill: DC.blue }, svg);
        const t = el("text", { x: x - 2.1, y: y + 0.9, class: "rp-lbl blue" + (s.text === D.goal.text ? " goal" : ""), "text-anchor": "end" }, svg);
        t.textContent = s.text;
      } else if (s.kind === "end") {
        el("rect", { x: x - 0.6, y: y - 0.6, width: 1.2, height: 1.2, fill: DC.blue }, svg);
      } else {
        el("rect", { x: x - 0.9, y: y - 0.9, width: 1.8, height: 1.8, fill: DC.ink, stroke: DC.idle, "stroke-width": 0.3, transform: `rotate(45 ${x} ${y})` }, svg);
      }
    }
    star(svg, gx, gy, 1.8, DC.ink, DC.bg).setAttribute("class", "rp-goalstar");
    const gTrail = el("g", {}, svg), gDecl = el("g", {}, svg), gDrone = el("g", {}, svg);
    const trail = {}, body = {}, cone = {};
    for (const d of DR) {
      const [sx, sy] = P(...D.start[d]);
      el("circle", { cx: sx, cy: sy, r: 0.9, fill: DC.bg, stroke: COL[d], "stroke-width": 0.35 }, gTrail);
      trail[d] = el("polyline", { fill: "none", stroke: COL[d], "stroke-width": 0.7, "stroke-linejoin": "round", "stroke-linecap": "round", opacity: 0.95 }, gTrail);
      cone[d] = el("path", { fill: COL[d], opacity: 0.18 }, gDrone);
      const g = el("g", {}, gDrone);
      el("circle", { r: 1.9, fill: COL[d], stroke: DC.bg, "stroke-width": 0.45 }, g);
      const t = el("text", { y: 0.75, class: "rp-num" }, g); t.textContent = num(d);
      body[d] = g;
    }
    const scale = el("g", {}, svg);
    el("line", { x1: 4, y1: Y1 - Y0 - 4, x2: 24, y2: Y1 - Y0 - 4, stroke: DC.idle, "stroke-width": 0.5 }, scale);
    const st = el("text", { x: 14, y: Y1 - Y0 - 5.4, class: "rp-lbl" }, scale); st.textContent = "20 m";

    const at = (tr, t) => {      // linear interpolation along a track [[t,x,y,yaw],...]
      let lo = 0, hi = tr.length - 1;
      if (t <= tr[0][0]) return tr[0];
      if (t >= tr[hi][0]) return tr[hi];
      while (hi - lo > 1) { const m = (lo + hi) >> 1; if (tr[m][0] <= t) lo = m; else hi = m; }
      const a = tr[lo], b = tr[hi], f = (t - a[0]) / (b[0] - a[0]);
      let dy = b[3] - a[3]; if (dy > 180) dy -= 360; if (dy < -180) dy += 360;
      return [t, a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), a[3] + f * dy];
    };

    // ---- cards ----
    const cmd = root.querySelector(".rp-cmd");
    const cards = {};
    const agents = root.querySelector(".rp-agents");
    for (const d of DR) {
      const c = document.createElement("div");
      c.className = "rp-agent";
      c.style.setProperty("--dc", COL[d]);
      c.innerHTML = `<div class="rp-ah"><span class="rp-dot"></span><b>Drone ${num(d)}</b><span class="rp-sub">sub-agent</span><span class="rp-state"></span></div>
        <div class="rp-cam r2-fade"><img alt=""><img alt=""><span class="rp-act"></span></div>
        <div class="rp-ol">order from the commander</div><div class="rp-order"></div>`;
      agents.appendChild(c);
      const [ia, ib] = c.querySelectorAll("img");
      cards[d] = { root: c, imgs: [ia, ib], front: 0, src: null, act: c.querySelector(".rp-act"), order: c.querySelector(".rp-order"), state: c.querySelector(".rp-state") };
    }

    // crossfade between decision frames instead of cutting; warm the cache first
    D.decisions.forEach((x, i) => setTimeout(() => { new Image().src = "assets/replay/" + x.img; }, i * 15));
    function show(c, src) {
      if (c.src === src) return;
      c.src = src;
      const back = c.imgs[1 - c.front];
      back.onload = () => { if (c.src !== src) return; back.classList.add("on"); c.imgs[c.front].classList.remove("on"); c.front = 1 - c.front; };
      back.src = src;
    }

    // ---- what passes between the commander and the sub-agents ----
    const events = [];
    const prevOrder = {};
    for (const c of D.calls) for (const d of DR) {
      const o = c.orders[d];
      if (o && o !== prevOrder[d]) events.push({ t: c.t, kind: "order", drone: d, text: c.short ? c.short[d] : o, full: o, call: c.call });
      prevOrder[d] = o;
    }
    for (const x of D.decisions) if (x.action === "finished") events.push({ t: x.t, kind: "report", drone: x.drone });
    events.push({ t: D.success.t, kind: "judge", drone: D.success.drone });
    events.sort((a, b) => a.t - b.t);

    const wbox = root.querySelector(".rp-wirebox"), wsvg = root.querySelector(".rp-wires");
    const wire = {}, gTok = el("g", {}, null);
    for (const d of DR) { wire[d] = el("path", { fill: "none", "stroke-width": 2 }, wsvg); wire[d].style.stroke = "var(--line2)"; }
    wsvg.appendChild(gTok);
    function layoutWires() {
      const b = wbox.getBoundingClientRect(), h = b.height;
      wsvg.setAttribute("viewBox", `0 0 ${b.width} ${h}`);
      DR.forEach((d, i) => {
        const r = cards[d].root.getBoundingClientRect();
        const ex = r.left + r.width / 2 - b.left, sx = b.width * (0.32 + 0.12 * i);
        wire[d].setAttribute("d", `M${sx},0 C${sx},${h * 0.55} ${ex},${h * 0.45} ${ex},${h}`);
      });
    }
    window.addEventListener("resize", () => { layoutWires(); draw(); });

    const logEl = root.querySelector(".rp-log ol");
    let logN = -1;
    const short = (x, n) => (x.length > n ? x.slice(0, n - 1) + "\u2026" : x);
    // the log lists the key moments, not every message: the orders already show on the cards
    const ICON = { read: "&#128065;", find: "&#128269;", sum: "&#931;", report: "&#9650;", ok: "&#9733;" };
    const moments = D.moments || [];
    const logLine = m => `<li class="m ${m.kind}"><span class="rp-t">${m.t.toFixed(0)} s</span><span class="r2-ic" style="--dc:${COL[m.drone] || DC.idle}">${ICON[m.kind] || ""}</span><span class="rp-msg">${esc(m.text)}</span></li>`;

    const strip = root.querySelector(".rp-strip");
    const SW = 1000, SH = 54, sx = tt => 40 + (SW - 50) * tt / T_END;
    const ssvg = el("svg", { viewBox: `0 0 ${SW} ${SH}`, class: "rp-ssvg", preserveAspectRatio: "none" });
    strip.appendChild(ssvg);
    el("line", { x1: 40, y1: SH / 2, x2: SW - 10, y2: SH / 2, class: "rp-sline", "stroke-width": 1 }, ssvg);
    const lab = (y, txt) => { const tt = el("text", { x: 2, y, class: "rp-slab" }, ssvg); tt.textContent = txt; };
    lab(15, "orders"); lab(47, "reports");
    for (const c of D.calls) el("path", { d: `M${sx(c.t) - 4},6 L${sx(c.t) + 4},6 L${sx(c.t)},16 Z`, fill: DC.order }, ssvg);
    for (const e of events) if (e.kind === "report") el("path", { d: `M${sx(e.t) - 4},48 L${sx(e.t) + 4},48 L${sx(e.t)},38 Z`, fill: COL[e.drone] }, ssvg);
    star(ssvg, sx(D.success.t), SH / 2, 7, DC.ok, DC.bg);
    const head = el("line", { x1: 40, y1: 2, x2: 40, y2: SH - 2, class: "rp-shead", "stroke-width": 2 }, ssvg);
    strip.addEventListener("click", ev => {
      const r = strip.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width * SW;
      t = Math.min(T_END, Math.max(0, (x - 40) / (SW - 50) * T_END)); draw();
    });

    // ---- controls ----
    const ctrl = root.querySelector(".rp-ctrl");
    ctrl.innerHTML = `<button class="rp-btn" data-k="play" type="button">Pause</button>
      <span class="rp-speeds">${[4, 8, 16].map(s => `<button class="rp-btn${s === 8 ? " on" : ""}" data-s="${s}" type="button">${s}×</button>`).join("")}</span>
      <input class="rp-range" type="range" min="0" max="${T_END}" step="0.1" value="0" aria-label="Replay time">
      <span class="rp-clock"></span>`;
    const chapters = [
      { t: D.calls[0].t, label: "Commander splits the drones" },
      { t: (D.calls.find(c => Object.values(c.notes).some(n => /confirmed|reads?\b.*\d/i.test(n))) || D.calls[2]).t, label: "First number read" },
      { t: (D.calls.find(c => Object.values(c.notes).some(n => new RegExp("= ?" + D.goal.text).test(n))) || D.calls[10]).t, label: `Sum ${D.goal.text}: Drone 4 sent in` },
      { t: D.success.t, label: "Drone 4 arrives" },
    ];
    const chap = root.querySelector(".rp-chapters");
    chap.innerHTML = chapters.map((c, i) => `<button class="rp-chip" data-i="${i}" type="button"><b>${c.t.toFixed(0)} s</b> ${esc(c.label)}</button>`).join("");

    let t = 0, playing = true, speed = 8, last = performance.now(), visible = true;
    const range = ctrl.querySelector(".rp-range"), clock = ctrl.querySelector(".rp-clock"), playBtn = ctrl.querySelector('[data-k="play"]');
    playBtn.onclick = () => { playing = !playing; playBtn.textContent = playing ? "Pause" : "Play"; if (playing && t >= T_END) t = 0; };
    ctrl.querySelectorAll("[data-s]").forEach(b => b.onclick = () => {
      speed = +b.dataset.s; ctrl.querySelectorAll("[data-s]").forEach(x => x.classList.toggle("on", x === b));
    });
    range.oninput = () => { t = +range.value; draw(); };
    chap.querySelectorAll(".rp-chip").forEach(b => b.onclick = () => { t = Math.max(0, chapters[+b.dataset.i].t - 1); draw(); });
    new IntersectionObserver(es => { visible = es[0].isIntersecting; }, { threshold: 0.05 }).observe(root);

    let lastCall = -1;
    function draw() {
      // map
      for (const d of DR) {
        const tr = D.tracks[d];
        const pts = [];
        for (const r of tr) { if (r[0] > t) break; const [x, y] = P(r[1], r[2]); pts.push(`${x.toFixed(2)},${y.toFixed(2)}`); }
        const [, x, y, yaw] = at(tr, t);
        const [px, py] = P(x, y);
        pts.push(`${px.toFixed(2)},${py.toFixed(2)}`);
        trail[d].setAttribute("points", pts.join(" "));
        body[d].setAttribute("transform", `translate(${px} ${py})`);
        const h = yaw * Math.PI / 180, R = 7;
        let path = `M${px.toFixed(2)},${py.toFixed(2)}`;
        for (let j = 0; j <= 12; j++) {
          const a = h - Math.PI / 4 + j * Math.PI / 24;
          path += ` L${(px + R * Math.cos(a)).toFixed(2)},${(py - R * Math.sin(a)).toFixed(2)}`;
        }
        cone[d].setAttribute("d", path + " Z");
      }
      gDecl.innerHTML = "";
      for (const e of D.decl) {
        if (e.t > t) break;
        const [x, y] = P(e.x, e.y);
        if (e.ok) star(gDecl, x, y, 2.6, DC.ok, DC.ink);
        else {
          el("path", { d: `M${x - 1.1},${y - 1.1} L${x + 1.1},${y + 1.1} M${x - 1.1},${y + 1.1} L${x + 1.1},${y - 1.1}`, stroke: COL[e.drone], "stroke-width": 0.6 }, gDecl);
        }
      }
      // commander
      let ci = -1;
      for (let i = 0; i < D.calls.length; i++) { if (D.calls[i].t <= t) ci = i; else break; }
      if (ci !== lastCall) {
        lastCall = ci;
        if (ci < 0) {
          cmd.innerHTML = `<div class="rp-ch"><b>Commander</b><span class="rp-sub">${esc(D.commander)}</span></div><p class="rp-idle">Sees the four drone views and a map, and writes one order per drone. First call at ${D.calls[0].t.toFixed(0)} s.</p>`;
        } else {
          const c = D.calls[ci];
          const trig = c.trigger === "declaration" ? "woken by a report" : c.trigger === "initial" ? "start" : "regular check";
          const sent = new Set(events.filter(e => e.kind === "order" && e.t === c.t).map(e => e.drone));
          cmd.innerHTML = `<div class="rp-ch"><b>Commander</b><span class="rp-sub">${esc(D.commander)}</span>
              <span class="rp-call">call ${c.call} &middot; ${c.t.toFixed(0)} s &middot; ${trig}</span></div>
            <div class="rp-know">What it concluded from the four views and the map</div>
            <div class="rp-rows">${DR.map(d => `<div class="rp-row" style="--dc:${COL[d]}">
              <span class="rp-tag">D${num(d)}</span><div class="rp-n">${esc((c.short_notes || c.notes)[d] || "")}</div>
              ${sent.has(d) ? '<span class="rp-sent">new order &darr;</span>' : '<span class="rp-sent off">no change</span>'}</div>`).join("")}</div>`;
          cmd.classList.remove("flash"); void cmd.offsetWidth; cmd.classList.add("flash");
        }
      }
      // messages in flight
      const dur = Math.max(4, speed * 1.4);
      gTok.innerHTML = "";
      for (const d of DR) { wire[d].style.stroke = "var(--line2)"; wire[d].setAttribute("stroke-width", 2); }
      for (const e of events) {
        if (e.t > t) break;
        const p = (t - e.t) / dur;
        if (p > 1 || e.kind === "judge") continue;
        const w = wire[e.drone], L = w.getTotalLength();
        if (!L) continue;
        const pt = w.getPointAtLength((e.kind === "order" ? p : 1 - p) * L);
        w.style.stroke = COL[e.drone]; w.setAttribute("stroke-width", 3);
        const txt = e.kind === "order" ? (e.text === "HOLD" ? "hold" : "order") : "finished";
        const fill = e.kind === "order" ? DC.order : DC.act.finished;
        const g = el("g", { transform: `translate(${pt.x} ${pt.y})` }, gTok);
        const wdt = txt.length * 7.2 + 16;
        el("rect", { x: -wdt / 2, y: -10, width: wdt, height: 20, rx: 10, fill, stroke: DC.bg, "stroke-width": 1.5 }, g);
        const tx = el("text", { x: 0, y: 4.5, class: "rp-tok" }, g); tx.textContent = txt;
      }
      // message log
      const shown = moments.filter(m => m.t <= t);
      if (shown.length !== logN) {
        logN = shown.length;
        logEl.innerHTML = shown.length ? shown.slice().reverse().map(logLine).join("") : '<li class="idle">The drones set off; no sign read yet.</li>';
      }
      head.setAttribute("x1", sx(t)); head.setAttribute("x2", sx(t));
      // sub-agents
      for (const d of DR) {
        let dec = null;
        for (const x of D.decisions) { if (x.t > t) break; if (x.drone === d) dec = x; }
        const c = cards[d];
        if (!dec) {
          c.imgs.forEach(i => i.classList.remove("on")); c.src = null; c.act.textContent = "waiting"; c.act.style.background = DC.idle;
          const first = events.find(e => e.kind === "order" && e.drone === d && e.t <= t);
          c.order.textContent = first ? first.text : "";
          c.state.textContent = ""; continue;
        }
        const src = "assets/replay/" + dec.img;
        show(c, src);
        let a = dec.action;
        if (a === "go" && dec.depth != null) a += ` ${(+dec.depth).toFixed(0)} m`;
        if (a === "rotate" && dec.angle != null) a += ` ${dec.angle > 0 ? "+" : ""}${(+dec.angle).toFixed(0)}°`;
        c.act.textContent = a; c.act.style.background = ACT[dec.action] || DC.idle;
        // the newest order the commander has sent this drone, not the one its last step ran on
        let ord = null;
        let full = null;
        for (const e of events) { if (e.t > t) break; if (e.kind === "order" && e.drone === d) { ord = e.text; full = e.full; } }
        ord = ord || dec.order;
        c.order.textContent = ord === "HOLD" ? "HOLD position" : ord;
        c.order.title = full || dec.order || "";
        const got = events.some(e => e.kind === "order" && e.drone === d && t >= e.t && t - e.t < Math.max(4, speed * 1.4));
        c.root.classList.toggle("got", got);
        const won = D.success.drone === d && t >= D.success.t;
        c.state.textContent = won ? "arrived" : dec.action === "finished" ? "declared" : "";
        c.root.classList.toggle("won", won);
      }
      range.value = t;
      clock.textContent = `${t.toFixed(0)} s / ${T_END} s`;
    }
    function tick(now) {
      const dt = (now - last) / 1000; last = now;
      if (playing && visible) {
        t += dt * speed;
        if (t > T_END) { t = T_END; draw(); playing = false; playBtn.textContent = "Replay"; setTimeout(() => { if (!playing) { t = 0; playing = true; playBtn.textContent = "Pause"; } }, 2500); }
        draw();
      }
      requestAnimationFrame(tick);
    }
    layoutWires();
    const q = new URLSearchParams(location.search).get("rt");   // ?rt=86 opens the replay paused at 86 s
    if (q !== null) { t = +q; playing = false; playBtn.textContent = "Play"; }
    draw();
    requestAnimationFrame(tick);
  }

  function star(parent, x, y, r, fill, stroke) {
    const p = [];
    for (let j = 0; j < 10; j++) {
      const a = Math.PI / 2 + j * Math.PI / 5, rr = j % 2 ? r * 0.45 : r;
      p.push(`${x + rr * Math.cos(a)},${y - rr * Math.sin(a)}`);
    }
    return el("polygon", { points: p.join(" "), fill, stroke: stroke || DC.bg, "stroke-width": 0.3 }, parent);
  }
})();
