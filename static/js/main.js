// Palette for the replays, read once from the CSS custom properties (static/css/palette.css).
const DC = (() => {
  const cs = getComputedStyle(document.documentElement);
  const v = n => cs.getPropertyValue(n).trim();
  return {
    drone: { Drone1: v("--d1"), Drone2: v("--d2"), Drone3: v("--d3"), Drone4: v("--d4") },
    act: { go: v("--go"), rotate: v("--rotate"), think: v("--think"), finished: v("--finished") },
    order: v("--ice"), ok: v("--green-500"), red: v("--crimson"), blue: v("--ice"), wall: v("--line2"),
    ink: v("--ink"), bg: v("--bg-0"), idle: v("--text-4"),
  };
})();

// multi-drone vs commander + sub-agents, Table 3 totals
const ROLES = [
  ["Gemini 3.7 Flash", 50.0, 93.3], ["Claude Opus 5", 46.7, 86.7], ["Gemini Robotics-ER 2", 53.3, 76.7],
  ["GPT-5", 13.3, 60.0], ["Qwen3.5-27B", 26.7, 53.3], ["Qwen3.5-9B", 36.7, 43.3], ["Qwen3.5-4B", 16.7, 43.3],
  ["Qwen3.5-2B", 3.3, 50.0], ["Cosmos3-Edge-2B", 0.0, 33.3],
];

function buildDumbbell() {
  const root = document.getElementById("dumbbell");
  if (!root) return;
  root.innerHTML = `<div class="db-legend"><span><i class="db-dot multi"></i>one model drives all four drones</span><span><i class="db-dot cmd"></i>the model commands four sub-agents</span></div>` +
    ROLES.map(([m, a, b]) => `<div class="db-row"><span class="db-name">${m}</span><div class="db-track">
      <span class="db-line" style="left:${a}%;width:${b - a}%"></span>
      <span class="db-dot multi" style="left:${a}%" title="${a}%"></span>
      <span class="db-dot cmd" style="left:${b}%" title="${b}%"></span>
      <span class="db-val" style="left:${b}%">${b.toFixed(0)}</span>
      <span class="db-val l${a < 8 ? " in" : ""}" style="left:${a}%">${a.toFixed(0)}</span></div></div>`).join("") +
    `<div class="db-row axis"><span class="db-name"></span><div class="db-track">${[0, 25, 50, 75, 100].map(v => `<span class="db-tick" style="left:${v}%">${v}</span>`).join("")}</div></div>`;
}

function navHighlight() {
  const links = [...document.querySelectorAll(".toc .links a")];
  const secs = links.map(a => document.querySelector(a.getAttribute("href"))).filter(Boolean);
  const io = new IntersectionObserver(es => {
    es.forEach(e => {
      if (e.isIntersecting) {
        links.forEach(a => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id));
      }
    });
  }, { rootMargin: "-40% 0px -55% 0px" });
  secs.forEach(s => io.observe(s));
}

function navState() {
  const nav = document.querySelector(".toc");
  const on = () => nav.classList.toggle("scrolled", window.scrollY > 24);
  window.addEventListener("scroll", on, { passive: true });
  on();
}

function copyBib() {
  const btn = document.getElementById("copy-bib");
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(document.getElementById("bib-text").textContent);
      btn.innerHTML = '<i class="fas fa-check"></i> Copied';
    } catch (e) {
      btn.textContent = "Select and copy";
    }
    setTimeout(() => { btn.innerHTML = '<i class="far fa-copy"></i> Copy'; }, 1600);
  });
}

// Videos are fetched whole and played from memory. Some static hosts answer
// without Range or Content-Length; a looping <video> then cannot seek back to
// the start and freezes, and iOS will not play it at all. Fetches go through a
// small queue, header first, so a phone is not asked for every clip at once.
const SMALL = window.matchMedia("(max-width: 700px)").matches;
const queue = [];
let active = 0;
function pump() {
  while (active < 2 && queue.length) {
    const v = queue.shift();
    const url = (SMALL && v.dataset.srcSmall) || v.dataset.src;
    active++;
    fetch(url).then(r => { if (!r.ok) throw new Error(r.status); return r.blob(); })
      .then(b => { v.src = URL.createObjectURL(new Blob([b], { type: "video/mp4" })); v.play().catch(() => {}); })
      .catch(() => { v.src = url; v.play().catch(() => {}); })
      .finally(() => { active--; pump(); });
  }
}
function loadVideo(v, first) {
  if (v.dataset.loaded) return;
  v.dataset.loaded = "1";
  first ? queue.unshift(v) : queue.push(v);
  pump();
}

function lazyVideos() {
  const vids = document.querySelectorAll("video[data-src]");
  const near = new IntersectionObserver(es => {
    es.forEach(e => { if (e.isIntersecting) { loadVideo(e.target); near.unobserve(e.target); } });
  }, { rootMargin: "600px 0px" });
  const seen = new IntersectionObserver(es => {
    es.forEach(e => {
      const v = e.target;
      if (!v.src) return;
      if (e.isIntersecting) { v.play().catch(() => {}); } else { v.pause(); }
    });
  }, { threshold: 0.1 });
  vids.forEach(v => { near.observe(v); seen.observe(v); });
  const hero = document.querySelector(".vhero-bg[data-src]");
  if (hero) loadVideo(hero, true);   // the header first, before anything else in view
}

// blocks rise in as they scroll into view
function reveal() {
  const els = document.querySelectorAll(".section:not(.tight) .wrap > *");
  if (!("IntersectionObserver" in window)) { els.forEach(e => e.classList.add("rv", "in")); return; }
  const io = new IntersectionObserver(es => {
    es.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.06 });
  els.forEach(e => { e.classList.add("rv"); io.observe(e); });
}

document.addEventListener("DOMContentLoaded", () => {
  reveal();
  buildDumbbell();
  navHighlight();
  navState();
  copyBib();
  lazyVideos();
});
