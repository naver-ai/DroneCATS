// Agent loop figure: one step of the DroneCATS Agent, replayed on a fixed clock.
(() => {
  const root = document.getElementById("agentfig");
  if (!root) return;
  const $ = id => root.querySelector("#" + id);
  const pulse = $("af-pulse"), jt = $("af-jt"), red = $("af-red"), img = $("af-img"), frameBox = root.querySelector(".af-frame");
  const JSON_TXT = '{"action":"go","point":[457,477],"depth_m":6}';
  const T = 8.4;                                    // seconds per step
  const seg = (a, b, t) => Math.min(1, Math.max(0, (t - a) / (b - a)));
  const ease = x => (x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2);
  const on = (id, v) => $(id).classList.toggle("on", v);
  const moves = [["af-e1", 0.45, 1.25], ["af-e2", 3.75, 4.35], ["af-e3", 4.75, 5.15], ["af-loop", 5.55, 7.15]];
  const len = Object.fromEntries(moves.map(([id]) => [id, $(id).getTotalLength()]));

  if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
    jt.textContent = JSON_TXT; red.classList.add("on"); $("af-pt").classList.add("on"); return;
  }

  let clock = 0, last = performance.now(), visible = true, cycle = 0;
  new IntersectionObserver(es => { visible = es[0].isIntersecting; }, { threshold: 0.1 }).observe(root);

  function draw(t) {
    frameBox.classList.toggle("on", t < 0.6 || t > 7.15);
    let shown = false;
    for (const [id, a, b] of moves) {
      if (t < a || t > b) continue;
      const p = $(id).getPointAtLength(ease(seg(a, b, t)) * len[id]);
      pulse.setAttribute("cx", p.x); pulse.setAttribute("cy", p.y); shown = true;
    }
    pulse.style.opacity = shown ? 1 : 0;
    on("af-e1", t > 0.45 && t < 1.4); on("af-ep", t > 0.45 && t < 1.4); on("af-prompt", t > 0.45 && t < 1.6);
    $("af-mllm").classList.toggle("think", t > 1.25 && t < 2.5); on("af-mllm", t > 1.2 && t < 2.6);
    const k = Math.round(seg(2.5, 3.4, t) * JSON_TXT.length);
    jt.textContent = t < 2.5 ? "" : JSON_TXT.slice(0, k) + (k < JSON_TXT.length ? "▍" : "");
    on("af-json", t > 2.5 && t < 3.9); on("af-ej", t > 2.45 && t < 3.6);
    red.classList.toggle("on", t > 3.25 && t < 7.2);
    red.style.clipPath = `inset(0 0 0 ${(100 - 100 * seg(3.25, 3.75, t)).toFixed(1)}%)`;
    $("af-pt").classList.toggle("on", t > 3.6 && t < 7.2);
    on("af-e2", t > 3.75 && t < 4.5); on("af-ctrl", t > 4.2 && t < 4.9);
    on("af-e3", t > 4.75 && t < 5.3); on("af-fly", t > 5.0 && t < 5.7);
    on("af-loop", t > 5.55 && t < 7.2); on("af-next", t > 5.55 && t < 7.2);
  }
  function tick(now) {
    const dt = (now - last) / 1000; last = now;
    if (visible) {
      clock += dt;
      const c = Math.floor((clock + T - 7.15) / T);   // the drone has moved: the next frame is a new view
      if (c !== cycle) { cycle = c; img.classList.toggle("shift"); }
      draw(clock % T);
    }
    requestAnimationFrame(tick);
  }
  const q = new URLSearchParams(location.search).get("at");   // ?at=3.5 freezes the figure at 3.5 s
  if (q !== null) { draw(+q); return; }
  requestAnimationFrame(tick);
})();
