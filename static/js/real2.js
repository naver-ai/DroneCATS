// Two real Tellos under a commander, replayed from the round's logs on one clock:
// every frame the judge server kept, the commander's calls, and the declarations.
(() => {
  const root = document.getElementById("real2");
  if (!root) return;
  const DR = ["Drone1", "Drone2"];
  const COL = { Drone1: DC.drone.Drone1, Drone2: DC.drone.Drone2 };
  const ACT = DC.act;
  const NS = "http://www.w3.org/2000/svg";
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const num = d => d.slice(-1);
  const short = (x, n) => (x.length > n ? x.slice(0, n - 1) + "…" : x);
  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };
  fetch("assets/real2/data.json").then(r => r.json()).then(init).catch(() => { root.textContent = "Replay data could not be loaded."; });

  function init(D) {
    // stop a few seconds after the accepted declaration: later calls only keep both drones holding
    const T_END = Math.ceil(D.success.t) + 8;
    D.calls = D.calls.filter(c => c.t <= T_END);
    root.innerHTML = `
      <div class="rp-mission"><b>Instruction</b> ${esc(D.mission)}</div>
      <div class="rp-main r2-main">
        <div class="rp-left"><div class="rp-map r2-map"></div>
          <div class="rp-log"><div class="rp-lh">Key moments <span>newest first</span></div><ol></ol></div></div>
        <div class="rp-right"><div class="rp-cmd"></div>
          <div class="rp-wirebox"><svg class="rp-wires"></svg><span class="rp-wl down">&#9660; orders</span><span class="rp-wl up">reports &#9650;</span></div>
          <div class="rp-agents"></div>
        </div>
      </div>
      <div class="rp-strip"></div><div class="rp-ctrl"></div><div class="rp-chapters"></div>`;

    // ---- events between the commander and the drones ----
    const events = [], prev = {};
    for (const c of D.calls) for (const d of DR) {
      const o = c.orders[d];
      if (o && o !== prev[d]) events.push({ t: c.t, kind: "order", drone: d, text: c.short[d], full: o });
      prev[d] = o;
    }
    for (const x of D.decl) events.push({ t: x.t, kind: "report", drone: x.drone, ok: x.ok });
    events.push({ t: D.success.t + 0.01, kind: "judge", drone: D.success.drone });
    events.sort((a, b) => a.t - b.t);

    // ---- cards: two stacked images per drone, crossfaded ----
    const cmd = root.querySelector(".rp-cmd"), agents = root.querySelector(".rp-agents"), cards = {};
    for (const d of DR) {
      const c = document.createElement("div");
      c.className = "rp-agent";
      c.style.setProperty("--dc", COL[d]);
      c.innerHTML = `<div class="rp-ah"><span class="rp-dot"></span><b>Drone ${num(d)}</b><span class="rp-sub">Tello</span><span class="rp-state"></span></div>
        <div class="rp-cam r2-fade"><img alt=""><img alt=""><span class="rp-act"></span><span class="r2-mark"></span></div>
        <div class="rp-ol">order from the commander</div><div class="rp-order"></div>`;
      agents.appendChild(c);
      const [a, b] = c.querySelectorAll("img");
      cards[d] = { root: c, imgs: [a, b], front: 0, src: null, act: c.querySelector(".rp-act"), mark: c.querySelector(".r2-mark"),
                   order: c.querySelector(".rp-order"), state: c.querySelector(".rp-state") };
    }
    // warm the cache so the crossfade never waits on the network, but only once
    // the replay is close to the screen: on a phone the page should not start
    // by pulling every frame
    const warm = () => { for (const d of DR) D.frames[d].forEach((f, i) => setTimeout(() => { new Image().src = "assets/real2/" + f.img; }, i * 15)); };
    const wio = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { wio.disconnect(); warm(); } }, { rootMargin: "800px 0px" });
    wio.observe(root);
    function show(c, src) {
      if (c.src === src) return;
      c.src = src;
      const back = c.imgs[1 - c.front];
      back.onload = () => { if (c.src !== src) return; back.classList.add("on"); c.imgs[c.front].classList.remove("on"); c.front = 1 - c.front; };
      back.src = src;
    }

    // ---- wires ----
    const wbox = root.querySelector(".rp-wirebox"), wsvg = root.querySelector(".rp-wires"), wire = {}, gTok = el("g", {});
    for (const d of DR) { wire[d] = el("path", { fill: "none", "stroke-width": 2 }, wsvg); wire[d].style.stroke = "var(--line2)"; }
    wsvg.appendChild(gTok);
    function layoutWires() {
      const b = wbox.getBoundingClientRect(), h = b.height;
      wsvg.setAttribute("viewBox", `0 0 ${b.width} ${h}`);
      DR.forEach((d, i) => {
        const r = cards[d].root.getBoundingClientRect(), ex = r.left + r.width / 2 - b.left, sx = b.width * (0.38 + 0.24 * i);
        wire[d].setAttribute("d", `M${sx},0 C${sx},${h * 0.55} ${ex},${h * 0.45} ${ex},${h}`);
      });
    }
    window.addEventListener("resize", () => { layoutWires(); draw(); });

    // ---- message log ----
    const logEl = root.querySelector(".rp-log ol");
    let logN = -1;
    const ICON = { read: "&#128065;", find: "&#128269;", sum: "&#931;", report: "&#9650;", ok: "&#9733;" };
    const logLine = m => `<li class="m ${m.kind}"><span class="rp-t">${m.t.toFixed(0)} s</span><span class="r2-ic" style="--dc:${COL[m.drone] || DC.idle}">${ICON[m.kind] || ""}</span><span class="rp-msg">${esc(m.text)}</span></li>`;

    // ---- timeline strip: calls on top, each drone's decisions below, declarations ----
    const strip = root.querySelector(".rp-strip");
    const SW = 1000, SH = 62, sx = tt => 70 + (SW - 80) * tt / T_END;
    const ssvg = el("svg", { viewBox: `0 0 ${SW} ${SH}`, class: "rp-ssvg r2-ssvg", preserveAspectRatio: "none" }, strip);
    const lab = (y, txt) => { el("text", { x: 2, y, class: "rp-slab" }, ssvg).textContent = txt; };
    lab(14, "commander"); lab(36, "Drone 1"); lab(56, "Drone 2");
    for (const c of D.calls) el("path", { d: `M${sx(c.t) - 4},4 L${sx(c.t) + 4},4 L${sx(c.t)},14 Z`, fill: DC.order }, ssvg);
    DR.forEach((d, i) => {
      const y = 32 + 20 * i;
      el("line", { x1: 70, y1: y, x2: SW - 10, y2: y, class: "rp-sline" }, ssvg);
      for (const f of D.frames[d]) if (f.kind && f.t >= 0) el("rect", { x: sx(f.t) - 2, y: y - 5, width: 4, height: 10, rx: 1, fill: ACT[f.kind] || DC.idle }, ssvg);
    });
    for (const x of D.decl) {
      const y = 32 + 20 * DR.indexOf(x.drone), X = sx(x.t);
      if (x.ok) star(ssvg, X, y, 8, DC.ok, DC.bg);
      else el("path", { d: `M${X - 4},${y + 4} L${X + 4},${y + 4} L${X},${y - 4} Z`, fill: COL[x.drone], stroke: DC.bg, "stroke-width": 0.8 }, ssvg);
    }
    const head = el("line", { y1: 2, y2: SH - 2, class: "rp-shead", "stroke-width": 2 }, ssvg);
    strip.addEventListener("click", ev => {
      const r = strip.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width * SW;
      t = Math.min(T_END, Math.max(0, (x - 70) / (SW - 80) * T_END)); draw();
    });

    // ---- map: the setup sheet's layout, and each drone's path from the commands it executed ----
    const L = D.layout, X0 = -1.4, X1 = 11.0, Y0 = -9.6, Y1 = 0.8, S = 10;
    const P = (x, y) => [(x - X0) * S, (Y1 - y) * S];
    const msvg = el("svg", { viewBox: `0 0 ${(X1 - X0) * S} ${(Y1 - Y0) * S}`, class: "rp-svg" }, root.querySelector(".r2-map"));
    el("rect", { width: (X1 - X0) * S, height: (Y1 - Y0) * S, class: "rp-bg" }, msvg);
    { const [a, b] = P(...L.doors[0]), [c2] = P(...L.doors[1]);
      el("line", { x1: a, y1: b, x2: c2, y2: b, class: "r2-door", "stroke-width": 3, "stroke-linecap": "round" }, msvg);
      el("text", { x: a - 4, y: b + 2, class: "r2-ml end" }, msvg).textContent = "doors"; }
    const goal = L.chairs.find(c => c.n === D.answer);
    { const [gx, gy] = P(goal.x, goal.y);
      el("circle", { cx: gx, cy: gy, r: 1.5 * S, fill: "rgba(3,199,90,0.07)", stroke: DC.ok, "stroke-width": 0.8, "stroke-dasharray": "3 2" }, msvg); }
    for (const c of L.chairs) {
      const [x, y] = P(c.x, c.y);
      el("rect", { x: x - 3.5, y: y - 3.5, width: 7, height: 7, rx: 1.5, class: "r2-chair", "stroke-width": 0.8 }, msvg);
      el("text", { x, y: y + 12, class: "r2-ml red" + (c.n === D.answer ? " goal" : "") }, msvg).textContent = c.n;
    }
    for (const st of L.stands) {
      const [x, y] = P(st.x, st.y);
      el("rect", { x: x - 2.5, y: y - 2.5, width: 5, height: 5, class: "r2-stand" }, msvg);
      el("text", { x, y: y - 5, class: "r2-ml blue" }, msvg).textContent = st.n;
    }
    const gTr = el("g", {}, msvg), gMk = el("g", {}, msvg), gDr = el("g", {}, msvg), trl = {}, bod = {}, con = {};
    for (const d of DR) {
      const [sx0, sy0] = P(...L.start[d]);
      el("circle", { cx: sx0, cy: sy0, r: 2, class: "r2-chair", stroke: COL[d], "stroke-width": 0.8 }, gTr);
      trl[d] = el("polyline", { fill: "none", stroke: COL[d], "stroke-width": 1.6, "stroke-linejoin": "round", "stroke-linecap": "round" }, gTr);
      con[d] = el("path", { fill: COL[d], opacity: 0.18 }, gDr);
      const g = el("g", {}, gDr);
      el("circle", { r: 4.2, fill: COL[d], stroke: DC.bg, "stroke-width": 1 }, g);
      el("text", { y: 1.7, class: "r2-mn" }, g).textContent = num(d);
      bod[d] = g;
    }
    { const [a] = P(8.4, 0), [b] = P(10.4, 0), y = 12;
      el("line", { x1: a, y1: y, x2: b, y2: y, class: "r2-scale", "stroke-width": 1 }, msvg);
      el("text", { x: (a + b) / 2, y: y - 3, class: "r2-ml" }, msvg).textContent = "2 m";
      el("text", { x: b, y: y + 9, class: "r2-ml r2-note end" }, msvg).textContent = "approximate positions"; }
    const pose = (tr, tt) => {
      if (tt <= tr[0][0]) return tr[0];
      for (let i = 1; i < tr.length; i++) if (tr[i][0] >= tt) {
        const a = tr[i - 1], b = tr[i], f = (tt - a[0]) / ((b[0] - a[0]) || 1);
        return [tt, a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), a[3] + f * (b[3] - a[3])];
      }
      return tr[tr.length - 1];
    };
    function drawMap() {
      for (const d of DR) {
        const tr = D.tracks[d], pts = [P(...L.start[d])];
        for (const r of tr) { if (r[0] > t) break; pts.push(P(r[1], r[2])); }
        const [, x, y, h] = pose(tr, t), [px, py] = P(x, y);
        pts.push([px, py]);
        trl[d].setAttribute("points", pts.map(p => p.map(v => v.toFixed(1)).join(",")).join(" "));
        bod[d].setAttribute("transform", `translate(${px.toFixed(1)} ${py.toFixed(1)})`);
        let path = `M${px.toFixed(1)},${py.toFixed(1)}`;
        for (let j = 0; j <= 12; j++) {
          const a = (h - 35 + j * 70 / 12) * Math.PI / 180;
          path += ` L${(px + 16 * Math.sin(a)).toFixed(1)},${(py - 16 * Math.cos(a)).toFixed(1)}`;
        }
        con[d].setAttribute("d", path + " Z");
      }
      gMk.innerHTML = "";
      for (const x of D.decl) {
        if (x.t > t) break;
        const [, dx, dy] = pose(D.tracks[x.drone], x.t), [X, Y] = P(dx, dy);
        if (x.ok) star(gMk, X, Y, 6, DC.ok, DC.ink);
      }
    }

    // ---- controls and chapters ----
    const ctrl = root.querySelector(".rp-ctrl");
    ctrl.innerHTML = `<button class="rp-btn" data-k="play" type="button">Pause</button>
      <span class="rp-speeds">${[2, 4, 8].map(s => `<button class="rp-btn${s === 4 ? " on" : ""}" data-s="${s}" type="button">${s}×</button>`).join("")}</span>
      <input class="rp-range" type="range" min="0" max="${T_END}" step="0.1" value="0" aria-label="Real two-drone replay time">
      <span class="rp-clock"></span>`;
    const noteAt = re => (D.calls.find(c => DR.some(d => re.test(c.notes[d] || ""))) || {}).t;
    const firstRej = D.decl.find(x => !x.ok);
    const chapters = [
      { t: D.calls[0].t, label: "Commander splits the drones" },
      { t: noteAt(/\b20\b/), label: "Drone 2 reads 20" },
      { t: noteAt(/\b5\b.*confirmed|confirmed.*\b5\b/i), label: "Drone 1 reads 5" },
      { t: noteAt(/54.*56.*58.*60/), label: "Drone 1 finds the chairs" },
      { t: noteAt(/= ?60/), label: "Drone 2 reads 35: sum 60" },
      firstRej && { t: firstRej.t, label: `Drone ${num(firstRej.drone)} waits at chair 58` },
      { t: D.success.t, label: "Drone 1 arrives at chair 60" },
    ].filter(c => c && c.t != null).sort((a, b) => a.t - b.t);
    const chap = root.querySelector(".rp-chapters");
    chap.innerHTML = chapters.map((c, i) => `<button class="rp-chip" data-i="${i}" type="button"><b>${c.t.toFixed(0)} s</b> ${esc(c.label)}</button>`).join("");

    let t = 0, playing = true, speed = 4, last = performance.now(), visible = true, lastCall = -2;
    const range = ctrl.querySelector(".rp-range"), clock = ctrl.querySelector(".rp-clock"), playBtn = ctrl.querySelector('[data-k="play"]');
    playBtn.onclick = () => { playing = !playing; playBtn.textContent = playing ? "Pause" : "Play"; if (playing && t >= T_END) t = 0; };
    ctrl.querySelectorAll("[data-s]").forEach(b => b.onclick = () => { speed = +b.dataset.s; ctrl.querySelectorAll("[data-s]").forEach(x => x.classList.toggle("on", x === b)); });
    range.oninput = () => { t = +range.value; draw(); };
    chap.querySelectorAll(".rp-chip").forEach(b => b.onclick = () => { t = Math.max(0, chapters[+b.dataset.i].t - 1); draw(); });
    new IntersectionObserver(es => { visible = es[0].isIntersecting; }, { threshold: 0.05 }).observe(root);

    function draw() {
      // commander
      let ci = -1;
      for (let i = 0; i < D.calls.length; i++) { if (D.calls[i].t <= t) ci = i; else break; }
      if (ci !== lastCall) {
        lastCall = ci;
        if (ci < 0) cmd.innerHTML = `<div class="rp-ch"><b>Commander</b><span class="rp-sub">${esc(D.commander)}</span></div><p class="rp-idle">Sees both drones' views and writes one order per drone; each drone is flown by a ${esc(D.subagent)} sub-agent. Three stands hold ${D.numbers.join(", ")}; the chairs show ${D.choices.join(", ")}.</p>`;
        else {
          const c = D.calls[ci];
          const trig = c.trigger === "declaration" ? "woken by a report" : c.trigger === "initial" ? "start" : "regular check";
          const sent = new Set(events.filter(e => e.kind === "order" && e.t === c.t).map(e => e.drone));
          cmd.innerHTML = `<div class="rp-ch"><b>Commander</b><span class="rp-sub">${esc(D.commander)}</span>
              <span class="rp-call">call ${c.call} &middot; ${c.t.toFixed(0)} s &middot; ${trig}</span></div>
            <div class="rp-know">What it concluded from the two views</div>
            <div class="rp-rows">${DR.map(d => `<div class="rp-row" style="--dc:${COL[d]}">
              <span class="rp-tag">D${num(d)}</span><div class="rp-n" title="${esc(c.notes[d] || "")}">${esc(c.short_notes[d] || "")}</div>
              ${sent.has(d) ? '<span class="rp-sent">new order &darr;</span>' : '<span class="rp-sent off">no change</span>'}</div>`).join("")}</div>`;
          cmd.classList.remove("flash"); void cmd.offsetWidth; cmd.classList.add("flash");
        }
      }
      // messages in flight
      const dur = Math.max(3, speed * 1.6);
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
        const g = el("g", { transform: `translate(${pt.x} ${pt.y})` }, gTok), wdt = txt.length * 7.2 + 16;
        el("rect", { x: -wdt / 2, y: -10, width: wdt, height: 20, rx: 10, fill: e.kind === "order" ? DC.order : DC.act.finished, stroke: DC.bg, "stroke-width": 1.5 }, g);
        el("text", { x: 0, y: 4.5, class: "rp-tok" }, g).textContent = txt;
      }
      const shown = D.moments.filter(m => m.t <= t);
      if (shown.length !== logN) {
        logN = shown.length;
        logEl.innerHTML = shown.length ? shown.slice().reverse().map(logLine).join("") : '<li class="idle">The drones set off; nothing read yet.</li>';
      }
      drawMap();
      // drones
      for (const d of DR) {
        const c = cards[d];
        let f = null, dec = null;
        for (const x of D.frames[d]) { if (x.t > t) break; f = x; if (x.kind) dec = x; }
        if (f) show(c, "assets/real2/" + f.img);
        c.act.textContent = dec ? dec.chip : "waiting"; c.act.style.background = dec ? ACT[dec.kind] || DC.idle : DC.idle;
        let ord = null;
        for (const e of events) { if (e.t > t) break; if (e.kind === "order" && e.drone === d) ord = e; }
        c.order.textContent = ord == null ? "" : ord.text === "HOLD" ? "HOLD position" : ord.text;
        c.order.title = ord ? ord.full : "";
        c.root.classList.toggle("got", events.some(e => e.kind === "order" && e.drone === d && t >= e.t && t - e.t < dur));
        let dl = null;
        for (const x of D.decl) { if (x.t > t) break; if (x.drone === d) dl = x; }
        const recent = dl && t - dl.t < 6;
        c.mark.textContent = recent ? (dl.ok ? "arrived ✓" : "reported") : "";
        c.mark.className = "r2-mark " + (recent ? (dl.ok ? "accepted" : "reported") : "");
        const won = D.success.drone === d && t >= D.success.t;
        c.state.textContent = won ? "arrived" : "";
        c.root.classList.toggle("won", won);
      }
      head.setAttribute("x1", sx(t)); head.setAttribute("x2", sx(t));
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
    const q = new URLSearchParams(location.search).get("r2t");   // ?r2t=82 opens it paused at 82 s
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
    return el("polygon", { points: p.join(" "), fill, stroke: stroke || DC.bg, "stroke-width": 0.6 }, parent);
  }
})();
