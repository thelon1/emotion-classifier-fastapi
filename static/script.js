const $ = s => document.querySelector(s);
const EMO = {
  sadness:  { e: '🥹', c: [91, 141, 239] },
  joy:      { e: '😃', c: [255, 200, 61] },
  love:     { e: '😍', c: [255, 93, 158] },
  anger:    { e: '😠', c: [255, 77, 61] },
  fear:     { e: '😨', c: [155, 107, 255] },
  surprise: { e: '😮', c: [47, 212, 192] }
};
// How the particle field behaves for each emotion
const MOOD = {
  neutral:  { c: [154, 148, 176], vy: -.08, j: .1, size: 1,   a: .5 },
  sadness:  { c: EMO.sadness.c,   vy: .55,  j: .1, size: .9,  a: .55 },
  joy:      { c: EMO.joy.c,       vy: -.75, j: .5, size: 1.3, a: .8 },
  love:     { c: EMO.love.c,      vy: -.25, j: .2, size: 1.2, a: .75, pulse: true },
  anger:    { c: EMO.anger.c,     vy: -.1,  j: 2.2, size: 1.1, a: .85 },
  fear:     { c: EMO.fear.c,      vy: .05,  j: 1.4, size: .9, a: .7, flicker: true },
  surprise: { c: EMO.surprise.c,  vy: -.2,  j: .5, size: 1.2, a: .8 }
};
const EXAMPLES = [
  'i feel like nobody noticed i was gone',
  'i just got the job and i cannot stop smiling',
  'you are the best thing that ever happened to me',
  'how dare they cancel my flight again',
  'i heard footsteps behind me in the dark',
  'wow i did not see that plot twist coming'
];
const cap = s => s[0].toUpperCase() + s.slice(1);
const rgb = c => `rgb(${c[0]},${c[1]},${c[2]})`;
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ---------- Mood field (canvas) ---------- */
const cv = $('#field'), cx = cv.getContext('2d');
let W, H, parts = [], mood = MOOD.neutral, col = [...MOOD.neutral.c], alpha = .5, size = 1;
const mouse = { x: -999, y: -999 };

function resize() {
  const d = Math.min(devicePixelRatio || 1, 2);
  W = innerWidth; H = innerHeight;
  cv.width = W * d; cv.height = H * d;
  cx.setTransform(d, 0, 0, d, 0, 0);
  const n = W < 700 ? 55 : 110;
  parts = Array.from({ length: n }, () => ({
    x: Math.random() * W, y: Math.random() * H, vx: 0, vy: 0,
    r: 1 + Math.random() * 2.2, ph: Math.random() * 6.28
  }));
}
addEventListener('resize', resize);
addEventListener('pointermove', e => { mouse.x = e.clientX; mouse.y = e.clientY; });

function burst() {
  parts.forEach(p => {
    const dx = p.x - W / 2, dy = p.y - H / 3, d = Math.hypot(dx, dy) || 1;
    const f = Math.min(14, 2200 / d);
    p.vx += dx / d * f * (.5 + Math.random()); p.vy += dy / d * f * (.5 + Math.random());
  });
}

function frame(t) {
  t /= 1000;
  for (let i = 0; i < 3; i++) col[i] += (mood.c[i] - col[i]) * .04;
  alpha += (mood.a - alpha) * .04; size += (mood.size - size) * .04;
  cx.clearRect(0, 0, W, H);
  for (const p of parts) {
    p.vx += (0 - p.vx) * .03 + (Math.random() - .5) * mood.j * .15;
    p.vy += (mood.vy - p.vy) * .03 + (Math.random() - .5) * mood.j * .15;
    const dx = p.x - mouse.x, dy = p.y - mouse.y, d2 = dx * dx + dy * dy;
    if (d2 < 16000) { const d = Math.sqrt(d2) || 1, f = (126 - d) / 126 * .9; p.vx += dx / d * f; p.vy += dy / d * f; }
    p.x += p.vx; p.y += p.vy;
    if (p.x < -10) p.x = W + 10; if (p.x > W + 10) p.x = -10;
    if (p.y < -10) p.y = H + 10; if (p.y > H + 10) p.y = -10;
    let r = p.r * size, a = alpha;
    if (mood.pulse) r *= 1 + .35 * Math.sin(t * 3 + p.ph);
    if (mood.flicker) a *= .55 + .45 * Math.sin(t * 18 + p.ph * 5);
    cx.beginPath(); cx.arc(p.x, p.y, Math.max(r, .3), 0, 6.283);
    cx.fillStyle = `rgba(${col[0] | 0},${col[1] | 0},${col[2] | 0},${a})`;
    cx.fill();
  }
  if (!reduce) requestAnimationFrame(frame);
}
resize();
reduce ? frame(0) : requestAnimationFrame(frame);

/* ---------- UI ---------- */
const ta = $('#text'), go = $('#go'), count = $('#count'), res = $('#result'), err = $('#err');
const hist = []; let busy = false;

EXAMPLES.forEach(t => {
  const b = document.createElement('button');
  b.className = 'chip'; b.textContent = t;
  b.onclick = () => { ta.value = t; sync(); analyze(); };
  $('#chips').append(b);
});

function sync() {
  count.textContent = `${ta.value.length} / 120`;
  go.disabled = busy || !ta.value.trim();
}
ta.addEventListener('input', sync);
ta.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); analyze(); } });
go.onclick = analyze;

async function analyze() {
  const text = ta.value.trim();
  if (!text || busy) return;
  busy = true; err.textContent = ''; go.classList.add('busy'); sync();
  try {
    const r = await fetch('/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text })
    });
    if (!r.ok) {
      throw new Error(r.status === 422 ? 'Add a sentence with letters or numbers.'
        : r.status === 503 ? 'The model is still loading. Try again in a few seconds.'
        : 'The server returned an error. Check its logs and try again.');
    }
    show(await r.json());
  } catch (e) {
    err.textContent = e instanceof TypeError ? 'Cannot reach the server. Check your connection and try again.' : e.message;
  } finally {
    busy = false; go.classList.remove('busy'); sync();
  }
}

function countUp(el, to, ms = 1100) {
  const s = performance.now();
  (function step(n) {
    const k = Math.min((n - s) / ms, 1), e = 1 - Math.pow(1 - k, 3);
    el.textContent = (to * e).toFixed(0);
    if (k < 1) requestAnimationFrame(step);
  })(s);
}

function show(d) {
  const key = d.predicated_emotion, E = EMO[key], pct = d.confidence * 100;
  document.body.style.setProperty('--accent', rgb(E.c));
  mood = MOOD[key]; if (!reduce) burst();

  const sorted = Object.entries(d.all_probabilities).sort((a, b) => b[1] - a[1]);
  const second = sorted[1][0];
  const letters = [...cap(key)].map((ch, i) => `<span style="--k:${i}">${ch}</span>`).join('');
  const rows = sorted.map(([k, v], i) =>
    `<li class="${i === 0 ? 'win' : ''}" style="--c:${rgb(EMO[k].c)};--i:${i}">
       <span>${cap(k)}</span><span class="tr"><span class="fl" data-w="${(v * 100).toFixed(1)}"></span></span>
       <span class="v">${(v * 100).toFixed(1)}%</span></li>`).join('');

  res.innerHTML = `
    <div class="top">
      <div class="ring"><svg viewBox="0 0 120 120"><circle class="trk" cx="60" cy="60" r="52"/><circle class="arc" cx="60" cy="60" r="52"/></svg>
        <span class="emo">${E.e}</span></div>
      <div class="verdict"><h2 aria-label="${cap(key)}">${letters}</h2>
        <p><b class="pct">0</b>% confident${pct < 60 ? `. It could also read as ${second}.` : ''}</p></div>
    </div>
    <ul class="probs">${rows}</ul>`;

  requestAnimationFrame(() => requestAnimationFrame(() => {
    res.querySelector('.arc').style.strokeDashoffset = 326.73 * (1 - d.confidence);
    res.querySelectorAll('.fl').forEach(f => f.style.width = f.dataset.w + '%');
  }));
  countUp(res.querySelector('.pct'), pct);
  res.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'nearest' });

  hist.unshift({ text: d.text, key }); hist.length = Math.min(hist.length, 5);
  $('#history').hidden = false;
  $('#hist').replaceChildren(...hist.map(h => {
    const b = document.createElement('button');
    b.textContent = `${EMO[h.key].e}  ${h.text}`;
    b.onclick = () => { ta.value = h.text; sync(); analyze(); };
    return b;
  }));
}

/* ---------- Server status (handles Render cold starts) ---------- */
async function health() {
  const s = $('#status');
  try {
    const j = await (await fetch('/health')).json();
    if (j.model_loaded) { s.classList.add('ok'); s.querySelector('b').textContent = 'Model ready'; return; }
    s.querySelector('b').textContent = 'Model is loading';
  } catch { s.querySelector('b').textContent = 'Server is waking up'; }
  setTimeout(health, 3000);
}
health(); sync();
