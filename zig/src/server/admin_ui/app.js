// The engine's admin page. It knows nothing the API does not say: settings come from /schema and /config, the tiles and header
// from /status, the charts from /series, memory from /ledger, requests, keys and the log from their routes. All text reaches the
// page through textContent or setAttribute, never as markup. The key lives in the variable `key` and in this tab's session storage (so a refresh keeps the sign-in); a sign-out or any refusal clears both.
const SVGNS = 'http://www.w3.org/2000/svg';
const IDLE_SIGN_OUT_MS = 30 * 60 * 1000;
const STALE_MS = 5000;
const base = location.pathname.replace(/\/ui(\/.*)?$/, '') + '/api/v1/';
const app = document.getElementById('app');

let key = null;
// The tab's session storage keeps the key across a refresh and nowhere else: not shared with other tabs, gone when the tab closes,
// never in a cookie or in local storage. Every access is guarded: a browser that refuses storage just asks again after a refresh.
const KEY_SLOT = 'tensorfold-admin-key';
// What is stored is {k, exp}: the key and the moment the idle sign-out would have signed it out (every touch moves it on). A key past
// its time is deleted and never used, so a refresh within the idle time signs straight back in and one after it asks again.
function keptKey() {
  try {
    const raw = window.sessionStorage.getItem(KEY_SLOT);
    if (!raw) return '';
    let v = null;
    try { v = JSON.parse(raw); } catch (e) { v = null; }
    if (v && typeof v.k === 'string' && v.k && typeof v.exp === 'number' && v.exp > Date.now()) return v.k;
    window.sessionStorage.removeItem(KEY_SLOT);
  } catch (e) { /* storage refused */ }
  return '';
}
function keepKey(v) { try { if (v) window.sessionStorage.setItem(KEY_SLOT, JSON.stringify({ k: v, exp: Date.now() + IDLE_SIGN_OUT_MS })); else window.sessionStorage.removeItem(KEY_SLOT); } catch (e) { /* storage refused: the sign-in lasts until a refresh */ } }
const S = { tick: 0, routes: [], scope: 'read', schema: null, config: null, status: null, statusAt: 0, fails: 0, tab: 'overview', group: null, filter: 'all', query: '', open: new Set(), cleanup: null, idleTimer: 0 };
const ROUTE_NAMES = () => new Set(S.routes.map((r) => r.path.slice(r.path.lastIndexOf('/v1/') + 4)));
const TABS = [['overview', 'Overview'], ['settings', 'Settings'], ['history', 'History'], ['activity', 'Requests & keys']];

// ---- element helpers: attributes and text only, never markup
function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  return el;
}
function svg(tag, attrs) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
  return el;
}
function clear(el) { el.replaceChildren(); return el; }
function icon(paths, w, hgt, attrs) {
  const el = svg('svg', Object.assign({ 'aria-hidden': 'true', width: w, height: hgt, viewBox: `0 0 ${w} ${hgt}`, fill: 'none' }, attrs));
  for (const p of paths) el.append(svg(p[0], p[1]));
  return el;
}

// ---- formatting: whole numbers with thousands separators, never "6.5k"
const whole = new Intl.NumberFormat('en-AU', { maximumFractionDigits: 0 });
const one = new Intl.NumberFormat('en-AU', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const loose = new Intl.NumberFormat('en-AU', { maximumFractionDigits: 3 });
const dayTime = new Intl.DateTimeFormat('en-AU', { timeZone: 'Australia/Sydney', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
const hm = new Intl.DateTimeFormat('en-AU', { timeZone: 'Australia/Sydney', hour: '2-digit', minute: '2-digit', hour12: false });
const dayName = new Intl.DateTimeFormat('en-AU', { timeZone: 'Australia/Sydney', day: 'numeric', month: 'short' });
const dayKey = new Intl.DateTimeFormat('en-CA', { timeZone: 'Australia/Sydney' });
const none = (x) => x == null || (typeof x === 'number' && !Number.isFinite(x));
const fmt = (x) => (none(x) ? '0' : whole.format(Number(x)));
const gib = (b) => (none(b) ? one.format(0) : one.format(Number(b) / 2 ** 30));
function dur(s) {
  if (none(s)) return '0 s';
  s = Math.floor(s);
  const d = Math.floor(s / 86400), hh = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  return d ? `${d} d ${hh} h` : hh ? `${hh} h ${m} min` : m ? `${m} min ${s % 60} s` : `${s} s`;
}
const utc = (ms) => new Date(ms).toISOString();
function whenText(ms) { return dayTime.format(new Date(ms)); }
function dayLabel(ms) {
  const k = dayKey.format(new Date(ms));
  if (k === dayKey.format(new Date())) return 'Today';
  if (k === dayKey.format(new Date(Date.now() - 86400000))) return 'Yesterday';
  return dayName.format(new Date(ms));
}
function when(ms) { return h('span', { title: utc(ms) + ' UTC (shown in Sydney time)', text: whenText(ms) }); }

// ---- the API
async function api(name, query, init) {
  const headers = { Authorization: 'Bearer ' + key };
  if (init && init.body) headers['Content-Type'] = 'application/json';
  const res = await fetch(base + name + (query ? '?' + query : ''), { cache: 'no-store', ...init, headers });
  let body = null;
  try { body = await res.json(); } catch (e) { body = null; }
  if (res.status === 401 && S.status) { refused('refused: the key was revoked, or admin was turned off'); throw new Error('refused'); }
  if (res.status === 404 && S.status) { refused('refused: the key was revoked, or admin was turned off'); throw new Error('refused'); }
  return { ok: res.ok, status: res.status, body };
}
function errText(r) { return r.body && r.body.error && r.body.error.message ? `${r.status} ${r.body.error.code || ''}: ${r.body.error.message}` : `HTTP ${r.status}`; }

// ---- login
// The sign-in screen never shows with a key stored: every way here wipes it. The one automatic retry is signIn's, at start.
function loginScreen(message) {
  stopAll();
  clearTimeout(S.idleTimer);
  key = null;
  keepKey('');
  const input = h('input', { type: 'password', id: 'key', autocomplete: 'off', spellcheck: 'false', 'aria-describedby': 'keynote' });
  const note = h('p', { id: 'keynote', class: 'err', role: 'alert', text: message || '' });
  const form = h('form', { class: 'login', onsubmit: (e) => { e.preventDefault(); signIn(input.value); input.value = ''; } },
    h('span', { class: 'word', text: 'TENSORFOLD' }),
    h('h1', { text: 'Sign in' }),
    h('label', { for: 'key' }, 'Admin key', h('span', { class: 'row' }, input, h('button', { type: 'button', class: 'btn sm', onclick: () => { input.type = input.type === 'password' ? 'text' : 'password'; } }, 'Show'))),
    h('p', { class: 'dim', text: 'The key is kept in this browser tab only, so a refresh keeps you signed in; closing the tab or signing out forgets it. A wrong key and a disabled admin look the same: the engine answers "not found" to both.' }),
    h('button', { class: 'btn go', type: 'submit' }, 'Sign in'),
    note);
  clear(app).append(form);
  input.focus();
  document.title = 'Sign in - TensorFold';
}
// `retries`: how many times an engine that did not answer is asked again (1 at start after a refresh, 0 for a typed key). A refusal is final.
async function signIn(value, retries) {
  const typed = value.trim();
  if (!typed) return;
  key = typed;
  const again = (why) => {
    if (!retries) return loginScreen(why);
    key = null;
    const note = document.getElementById('keynote');
    if (note) note.textContent = why + '; trying once more';
    setTimeout(() => signIn(typed, retries - 1), 1500);
  };
  try {
    const res = await fetch(base + 'status', { cache: 'no-store', headers: { Authorization: 'Bearer ' + key } });
    if (res.status === 404 || res.status === 401) return loginScreen('refused: wrong key, or admin is not enabled on this engine');
    if (!res.ok) return again(`the engine answered ${res.status}`);
    S.status = await res.json();
    S.statusAt = Date.now();
  } catch (e) { return again('engine unreachable: ' + e.message); }
  keepKey(typed);
  await boot();
}
function refused(message) { S.status = null; loginScreen(message); } // (a refusal forgets the stored key)
function signOut() { S.status = null; S.schema = null; S.config = null; loginScreen(''); }
function touch() {
  clearTimeout(S.idleTimer);
  if (key) { S.idleTimer = setTimeout(() => signOut(), IDLE_SIGN_OUT_MS); keepKey(key); }
}
for (const ev of ['pointerdown', 'keydown']) addEventListener(ev, touch, true);

// ---- boot: routes, schema and config once; status every second
async function boot() {
  touch();
  const [routes, schema, config] = await Promise.all([api('routes'), api('schema'), api('config')]);
  if (!routes.ok || !schema.ok || !config.ok) return loginScreen('the engine did not answer the first reads: ' + errText(!routes.ok ? routes : !schema.ok ? schema : config));
  S.routes = routes.body.routes;
  S.schema = schema.body;
  S.config = config.body;
  indexSchema();
  S.scope = S.status && S.status.capabilities ? S.status.capabilities.scope : 'read';
  const wanted = decodeURIComponent(location.hash.slice(1));
  S.tab = TABS.some((t) => t[0] === wanted) ? wanted : 'overview';
  shell();
  poll();
}
function indexSchema() {
  S.byId = new Map(S.schema.options.map((o) => [o.id, o]));
  S.byName = new Map(S.schema.options.map((o) => [o.name, o]));
  S.byEnv = new Map(S.schema.options.map((o) => [o.env, o]));
  S.cfgById = new Map(S.config.values.map((v) => [v.id, v]));
}

// ---- one poll loop feeds every box: the status each second, and with it the requests table; the series and the memory record every
// fifth second. A page registers `refs.pull` (what it fetches beyond the status) and `refs.update` (what it writes into its own boxes
// in place). Nothing here ever clears and rebuilds a view: a view is built once, when its tab is chosen.
let pollTimer = 0;
async function poll() {
  clearTimeout(pollTimer);
  if (!key) return;
  if (document.hidden) { pollTimer = setTimeout(poll, 1000); return; }
  try {
    const r = await api('status');
    if (!r.ok) throw new Error(errText(r));
    S.status = r.body;
    S.statusAt = Date.now();
    S.fails = 0;
    S.tick++;
    if (refs.pull) { try { await refs.pull(S.tick); } catch (e) { /* the status poll shows an outage; a failed side read leaves its box as it was */ } }
  } catch (e) {
    if (!key) return;
    S.fails++;
  }
  onStatus();
  pollTimer = setTimeout(poll, S.fails ? Math.min(15000, 1000 * 2 ** (S.fails - 1)) : 1000);
}
addEventListener('visibilitychange', () => { if (!document.hidden && key) poll(); });
function stopAll() { clearTimeout(pollTimer); if (S.cleanup) { S.cleanup(); S.cleanup = null; } }

// text and tip writers that touch the DOM only when the value moved, so a hovered tip, a selection and a screen reader are left alone
function setText(el, text) { const t = String(text); if (el.textContent !== t) el.textContent = t; }
function setTip(el, text) { const t = text || ''; if (el.getAttribute('title') !== t) { if (t) el.setAttribute('title', t); else el.removeAttribute('title'); } }
function setColor(el, color) { if (el.style.color !== color) el.style.color = color; }
function setWidth(el, pct) { const w = Math.max(0, Math.min(100, pct)).toFixed(1) + '%'; if (el.style.width !== w) el.style.width = w; }

// ---- shell: the header strip (engine, Turbo, the prompt-vs-generation balance, the key, the clock), section nav, the view, the staged bar
let refs = {};
function strip(label, tip) {
  const v = h('span', { class: 'v' }), q = h('span', { class: 'q' });
  return { el: h('div', { class: 'htile', 'data-tile': label.toLowerCase().split(' ')[0] }, h('span', { class: 'k', text: label, title: tip || null }), v, q), v, q };
}
function shell() {
  stopAll();
  refs = {};
  refs.blind = h('div', { class: 'bar blind', role: 'alert', hidden: true });
  refs.ro = h('div', { class: 'bar ro', text: 'Read-only: this key may not change settings.', hidden: true });
  refs.tray = h('section', { class: 'tray', 'aria-label': 'Changes not yet applied', hidden: true });
  refs.sub = h('span', { class: 'sub' });
  refs.nav = h('nav', { class: 'sections', 'aria-label': 'Sections' }, TABS.map(([id, label]) => h('button', { type: 'button', 'data-tab': id, onclick: () => selectTab(id) }, label)));
  refs.main = h('main', { id: 'view', tabindex: '-1' });
  const head = headerTiles();
  refs.head = head.update;
  clear(app).append(
    h('header', { class: 'top' }, h('div', {}, h('div', { class: 'brand' }, h('span', { class: 'word', text: 'TENSORFOLD' }), refs.sub), refs.nav, head.el)),
    refs.blind, refs.ro, refs.main, refs.tray);
  selectTab(S.tab, true);
  onStatus();
  drawTray();
}
function selectTab(id, first) {
  if (S.cleanup) { S.cleanup(); S.cleanup = null; }
  S.tab = id;
  if (!first) history.replaceState(null, '', '#' + encodeURIComponent(id));
  for (const b of refs.nav.children) {
    if (b.dataset.tab === id) b.setAttribute('aria-current', 'page'); else b.removeAttribute('aria-current');
  }
  document.title = TABS.find((t) => t[0] === id)[1] + ' - TensorFold';
  refs.update = null; refs.pull = null;
  clear(refs.main);
  try {
    S.cleanup = { overview, settings: settingsPage, history: historyPage, activity: activityPage }[id]() || null;
  } catch (e) { refs.main.append(h('p', { class: 'err', role: 'alert', text: 'This section failed to draw: ' + e.message })); }
  if (!first) refs.main.focus({ preventScroll: true });
}

// ---- every poll: the header, the blind bar, and whatever the open page keeps live
function onStatus() {
  if (!refs.head) return;
  const st = S.status;
  const stale = !st || Date.now() - S.statusAt > STALE_MS || S.fails > 0;
  refs.blind.hidden = !(S.fails > 0);
  if (S.fails > 0) setText(refs.blind, `The engine has not answered for ${Math.round((Date.now() - S.statusAt) / 1000)} s: what is shown is the last good reading, dimmed.`);
  if (st) {
    refs.ro.hidden = st.read_only === false;
    const hh = st.health || {};
    const ranks = st.memory && st.memory.ranks ? st.memory.ranks.length : null;
    setText(refs.sub, [st.model_name || hh.model, ranks ? `${ranks} × DGX Spark` : null, st.release].filter(Boolean).join(' · '));
  }
  refs.head(st, stale);
  if (refs.update) refs.update(stale);
}

// The header strip: one tile each, a value and a short line, the detail on hover. Built once; onStatus writes into it.
function headerTiles() {
  const eng = strip('ENGINE', 'Whether the engine answers and for how long it has been up (its own health record).');
  const tb = strip('TURBO'), bal = strip('BALANCE');
  const who = strip('KEY', 'The label of the key this tab is signed in with. The key itself is kept in this tab only.');
  const clock = strip('LOCAL · AEST', 'Sydney time.');
  const signOutBtn = h('button', { type: 'button', class: 'linkbtn', onclick: signOut }, 'sign out');
  who.q.append(signOutBtn);
  const turbo = turboTile(tb), balance = balanceTile(bal);
  const el = h('div', { class: 'tiles' }, eng.el, tb.el, bal.el, who.el, clock.el);
  function update(st, stale) {
    if (!st) return;
    const hh = st.health || {};
    const down = S.fails > 0;
    eng.el.classList.toggle('bad', down);
    eng.el.classList.toggle('warm', !down && !!hh.warming);
    setText(eng.v, down ? 'Not answering' : hh.warming ? 'Warming up' : 'Serving');
    setText(eng.q, 'up ' + dur(hh.uptime_s));
    const label = st.capabilities && st.capabilities.label;
    setText(who.v, label || 'signed in');
    setText(clock.v, dayTime.format(new Date()));
    turbo.paint();
    balance.sync();
    el.classList.toggle('stale', stale);
  }
  return { el, update };
}

// Turbo: a prompt pass of 8,192 rows with split matrix multiplies (TF_DS_PASS_ROWS=8192, TF_DS_GEMM_HALVES=1), a speed bought with KV
// pool. The tile shows the state in force and, on the other state's button, a confirm before it stages the two settings like any change
// (Check, Apply, restart). Every figure is the engine's measured record (status.turbo), never the page's own; with no record it stays off.
let balanceRepaint = () => {};
let turboRepaint = () => {};
function turboTile(t) {
  const pr = S.byName.get('tf_ds_pass_rows'), gh = S.byName.get('tf_ds_gemm_halves');
  t.el.hidden = !pr || !gh;
  const TURBO = { pr: '8192', gh: '1' }, PLAIN = { pr: '4096', gh: '0' };
  const btn = h('button', { type: 'button', class: 'vbtn' });
  const yes = h('button', { type: 'button', class: 'mini go' }, 'stage'), no = h('button', { type: 'button', class: 'mini' }, 'cancel');
  const qline = h('span', { class: 'qt' });
  t.v.append(btn);
  t.q.append(qline, yes, no);
  let confirming = false, last = '';
  const gate = () => {
    const rec = S.status && S.status.turbo;
    const measured = !!(rec && rec.measured === true);
    return { rec, measured, runsOn: same(currentOf(pr), TURBO.pr) && same(currentOf(gh), TURBO.gh), shownOn: same(shownValue(pr), TURBO.pr) && same(shownValue(gh), TURBO.gh), can: canWrite(pr) && canWrite(gh) };
  };
  const goTo = (on) => {
    const want = on ? TURBO : PLAIN;
    stage(pr, same(want.pr, pr.default) ? 'unset' : want.pr);
    stage(gh, same(want.gh, gh.default) ? 'unset' : want.gh);
  };
  btn.addEventListener('click', () => { const g = gate(); if (g.measured && g.can) { confirming = true; last = ''; paint(); } });
  no.addEventListener('click', () => { confirming = false; last = ''; paint(); });
  yes.addEventListener('click', () => { const g = gate(); confirming = false; last = ''; goTo(!g.shownOn); });
  const pct = (x) => (Math.round(Number(x) * 10) / 10).toString();
  function paint() {
    if (!pr || !gh) return;
    const g = gate(), rec = g.rec || {};
    const sig = JSON.stringify([rec, g.runsOn, g.shownOn, g.can, confirming, T.staged.size, g.measured]);
    if (sig === last) return;
    last = sig;
    const staged = g.shownOn !== g.runsOn;
    btn.textContent = `Turbo ${g.shownOn ? 'on' : 'off'}`;
    btn.disabled = !g.measured || !g.can || confirming;
    btn.classList.toggle('on', g.shownOn);
    qline.hidden = confirming;
    yes.hidden = no.hidden = !confirming;
    yes.textContent = g.shownOn ? 'turn off at restart' : 'turn on at restart';
    qline.textContent = !g.measured ? 'not measured here' : staged ? 'staged: Check, Apply' : `x${Number(rec.long_speedup).toFixed(2)} · -${pct(rec.pool_cost_pct)} % KV`;
    qline.classList.toggle('warn', staged);
    const lines = !g.measured ? ['Turbo is not measured on this build: the switch stays off until a measured record is filled in.'] : [
      `Turbo: a prompt pass of 8,192 rows with split matrix multiplies (TF_DS_PASS_ROWS=8192, TF_DS_GEMM_HALVES=1) against 4,096 rows and none. ${g.runsOn ? 'On' : 'Off'} now${g.runsOn === (same(pr.default, TURBO.pr) && same(gh.default, TURBO.gh)) ? ' (the default)' : ' (set by the operator)'}. Applies at the next restart.`,
      `Long prompts x${Number(rec.long_speedup).toFixed(3)} (${rec.build || 'measured'}), short prompts x${Number(rec.short_speedup).toFixed(3)}${rec.short_build ? ' (' + rec.short_build + ')' : ''}.`,
      rec.pool_cost_positions != null ? `KV pool cost: ${fmt(rec.pool_cost_positions)} positions = ${pct(rec.pool_cost_pct)} % of the pool${rec.pool_normal_positions != null ? ` (${fmt(rec.pool_normal_positions)} against ${fmt(rec.pool_turbo_positions)})` : ''}; free memory is unchanged.` : null,
      rec.beside_prompt_ratio != null ? `Beside generating streams: the prompt x${Number(rec.beside_prompt_ratio).toFixed(3)}, the streams x${Number(rec.beside_streams_ratio).toFixed(3)}${rec.beside_build ? ' (' + rec.beside_build + ')' : ''}.` : null,
      rec.replies_equal === true ? 'Replies byte for byte equal.' : null,
      rec.what || rec.source_run ? `Measured on ${rec.build || '?'}${rec.source_run ? ': ' + rec.source_run : ''}.` : null].filter(Boolean);
    setTip(t.el, lines.join('\n'));
  }
  turboRepaint = () => { last = ''; paint(); };
  return { paint };
}
// The prompt-vs-generation balance: TF_DS_ROUNDS_BESIDE, the generation rounds run after each prompt step of a long prompt, through the
// same staged bar as the Settings page. Position 0 is off (the setting unset). The engine refuses a round count while the old prompt
// share is below 1, so the slider is disabled then and the tip says why.
function balanceTile(t) {
  const o = S.byName.get('tf_ds_rounds_beside'), share = S.byName.get('tensorfold_prefill_share');
  t.el.hidden = !o;
  if (!o) return { sync() {} };
  const slider = h('input', { type: 'range', min: '0', max: '16', step: '1', value: '0', 'aria-label': o.label || o.name });
  t.v.append(slider);
  const val = h('span', { class: 'qt' });
  t.q.append(val);
  const shown = () => { const v = shownValue(o); const n = v == null || v === '' ? 0 : Number(v); return Number.isFinite(n) ? n : 0; };
  const text = (n) => (n === 0 ? 'off' : `${n} round${n === 1 ? '' : 's'}`);
  slider.addEventListener('input', () => { setText(val, text(Number(slider.value))); });
  slider.addEventListener('change', () => stage(o, Number(slider.value) === 0 ? 'unset' : String(Number(slider.value))));
  function sync() {
    const n = shown();
    const sv = share ? shownValue(share) : null;
    const blocked = sv != null && sv !== '' && Number(sv) < 1;
    slider.disabled = !canWrite(o) || blocked;
    if (document.activeElement !== slider) { slider.max = String(Math.max(16, n)); if (slider.value !== String(n)) slider.value = String(n); setText(val, text(n)); }
    val.classList.toggle('warn', T.staged.has(o.name));
    setTip(t.el, `Prompt vs generation: how many generation rounds run after each prompt step of a long prompt (${o.env}). More rounds keep replies streaming while a prompt is read; fewer finish the prompt sooner; 0 is one prompt step, then one round. Default ${defaultText(o)}.` + (blocked ? ` The old prompt share ${sv} is set; clear it in Settings first.` : '') + (T.staged.has(o.name) ? ' A change is staged: Check, then Apply.' : ''));
  }
  balanceRepaint = sync;
  return { sync };
}

// ---- the board's plots: a label and the value now, a gutter of axis numbers, a plot of fixed-bucket points. The engine gives fixed
// buckets (a bucket keeps its value as the window moves) with the mean and the peak of each. Zero sits inside the plot; the scale
// fits the data (the window's top + 10 %, rounded up to 1, 1.5, 2, 3, 4, 5, 6 or 8 times a power of ten); no text over the plot.
const PLOT_PAD = 4; // of 100: the room above the top and below zero
function niceTop(x) {
  if (!(x > 0)) return 1;
  const mag = 10 ** Math.floor(Math.log10(x));
  for (const s of [1, 1.5, 2, 3, 4, 5, 6, 8, 10]) if (s * mag >= x) return s * mag;
  return 10 * mag;
}
// A thin smooth path through points [x, y] (monotone cubic: it never overshoots a peak or dips below a flat run). `pts` run left to right.
function smoothPath(pts) {
  const n = pts.length;
  if (!n) return '';
  const q = (v) => v.toFixed(1);
  if (n === 1) return `M${q(pts[0][0])},${q(pts[0][1])}`;
  const dx = [], sl = [];
  for (let i = 0; i < n - 1; i++) { dx.push(pts[i + 1][0] - pts[i][0]); sl.push((pts[i + 1][1] - pts[i][1]) / dx[i]); }
  const m = new Array(n);
  m[0] = sl[0]; m[n - 1] = sl[n - 2];
  for (let i = 1; i < n - 1; i++) m[i] = sl[i - 1] * sl[i] <= 0 ? 0 : (sl[i - 1] + sl[i]) / 2;
  for (let i = 0; i < n - 1; i++) {
    if (sl[i] === 0) { m[i] = 0; m[i + 1] = 0; continue; }
    const a = m[i] / sl[i], b = m[i + 1] / sl[i], r = a * a + b * b;
    if (r > 9) { const t = 3 / Math.sqrt(r); m[i] = t * a * sl[i]; m[i + 1] = t * b * sl[i]; }
  }
  let d = `M${q(pts[0][0])},${q(pts[0][1])}`;
  for (let i = 0; i < n - 1; i++) {
    const w = dx[i] / 3;
    d += ` C${q(pts[i][0] + w)},${q(pts[i][1] + m[i] * w)} ${q(pts[i + 1][0] - w)},${q(pts[i + 1][1] - m[i + 1] * w)} ${q(pts[i + 1][0])},${q(pts[i + 1][1])}`;
  }
  return d;
}
const plotY = (frac) => 100 - PLOT_PAD - Math.min(1, Math.max(0, frac)) * (100 - 2 * PLOT_PAD);
function plotShell(label, tip, aria) {
  const now = h('span', { class: 'pn num', text: '0' });
  const g = [h('span', { class: 'g top' }), h('span', { class: 'g mid' }), h('span', { class: 'g zero', text: '0' })];
  g[0].style.top = (PLOT_PAD) + '%'; g[1].style.top = '50%'; g[2].style.top = (100 - PLOT_PAD) + '%';
  const gutter = h('div', { class: 'gutter', title: tip || null }, ...g);
  const plotSvg = svg('svg', { 'aria-hidden': 'true', viewBox: '0 0 600 100', preserveAspectRatio: 'none' });
  for (const f of [0, 0.5, 1]) plotSvg.append(svg('line', { x1: '0', x2: '600', y1: String(plotY(f)), y2: String(plotY(f)), stroke: f === 0 ? '#2a3033' : '#1a1e1f', 'stroke-width': '1', 'stroke-dasharray': f === 0 ? '' : '3 3', 'vector-effect': 'non-scaling-stroke' }));
  const hair = h('div', { class: 'hair', 'aria-hidden': 'true', hidden: true });
  const tipBox = h('div', { class: 'tip', 'aria-hidden': 'true', hidden: true });
  const plot = h('div', { class: 'plot', role: 'img', 'aria-label': aria }, plotSvg, hair, tipBox);
  const xlab = h('div', { class: 'xlab', 'aria-hidden': 'true' });
  const el = h('div', { class: 'pbox' }, h('div', { class: 'phead' }, h('span', { class: 'pl', text: label, title: tip || null }), now), h('div', { class: 'pbody' }, gutter, plot), h('div', { class: 'pfoot' }, h('span'), xlab));
  return { el, now, g, plotSvg, hair, tipBox, plot, xlab };
}
function hoverIndex(plot, n, e) {
  const r = plot.getBoundingClientRect();
  return Math.round(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)) * (n - 1));
}
function xlabels(xlab, t0, stepMs, n) {
  const want = [0, 1, 2, 3, 4].map((k) => hm.format(new Date(t0 + (n - 1) * stepMs * k / 4)));
  while (xlab.children.length < 5) xlab.append(h('span'));
  want.forEach((w, i) => setText(xlab.children[i], w));
}
function chart(label, tip, color, unit, name) {
  const s = plotShell(label, tip, label);
  const area = svg('path', { fill: color, 'fill-opacity': '0.10', stroke: 'none' });
  const line = svg('path', { fill: 'none', stroke: color, 'stroke-width': '1.25', 'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'vector-effect': 'non-scaling-stroke' });
  const goalLine = svg('line', { x1: '0', x2: '600', y1: '-10', y2: '-10', stroke: '#e0a33e', 'stroke-width': '1.2', 'stroke-dasharray': '6 4', 'vector-effect': 'non-scaling-stroke' });
  s.plotSvg.append(area, line, goalLine);
  const dot = h('div', { class: 'dot', 'aria-hidden': 'true', hidden: true });
  dot.style.background = color;
  s.plot.append(dot);
  const state = { pts: [], peaks: [], t0: 0, step: 1, top: 1 };
  s.plot.addEventListener('pointermove', (e) => {
    const n = state.pts.length;
    if (n < 2) return;
    const i = hoverIndex(s.plot, n, e), v = state.pts[i], pk = state.peaks[i];
    const end = state.t0 + i * state.step * 1000;
    s.hair.hidden = s.tipBox.hidden = false;
    s.hair.style.left = (i / (n - 1) * 100) + '%';
    s.tipBox.style.left = Math.min(88, Math.max(12, i / (n - 1) * 100)) + '%';
    dot.hidden = v == null;
    if (v != null) { dot.style.left = (i / (n - 1) * 100) + '%'; dot.style.top = plotY(v / state.top) + '%'; }
    clear(s.tipBox).append(h('div', { class: 'dim', text: `${dayTime.format(new Date(end - state.step * 1000))} to ${dayTime.format(new Date(end))}` }), h('div', { class: 'num', text: v == null ? 'no reading' : `${fmt(v)} ${unit}${state.step > 1 ? ' average' : ''}` }), pk != null && state.step > 1 ? h('div', { class: 'num dim', text: `peak ${fmt(pk)} ${unit}` }) : null);
  });
  s.plot.addEventListener('pointerleave', () => { s.hair.hidden = dot.hidden = s.tipBox.hidden = true; });
  return {
    el: s.el, name,
    setNow(v) { setText(s.now, `${fmt(v)} ${unit}`); },
      const pts = body.points.map((p) => (p == null ? null : Number(p)));
      const peaks = (body.peaks || []).map((p) => (p == null ? null : Number(p)));
      const n = pts.length;
      const real = pts.filter((p) => p != null);
      const top = niceTop(Math.max(0, goal || 0, ...peaks.filter((p) => p != null), ...real) * 1.1);
      const lastMs = body.last_ms != null ? Number(body.last_ms) : atMs;
      Object.assign(state, { pts, peaks, top, step: body.step_s, t0: lastMs - (n - 1) * body.step_s * 1000 });
      let d = '', a = '', run = [];
      const base = plotY(0).toFixed(1);
      const flush = () => {
        if (run.length) {
          d += (d ? ' ' : '') + smoothPath(run);
          a += `${a ? ' ' : ''}${smoothPath(run)} L${run[run.length - 1][0].toFixed(1)},${base} L${run[0][0].toFixed(1)},${base} Z`;
        }
        run = [];
      };
      pts.forEach((p, i) => { if (p == null) { flush(); return; } run.push([i / Math.max(1, n - 1) * 600, plotY(p / top)]); });
      flush();
      line.setAttribute('d', d);
      area.setAttribute('d', a);
      const gy = goal ? plotY(goal / top).toFixed(1) : '-10';
      setText(s.g[0], whole.format(top)); setText(s.g[1], whole.format(top / 2));
      xlabels(s.xlab, state.t0, body.step_s * 1000, n);
      s.plot.setAttribute('aria-label', `${label}, last ${Math.round(n * body.step_s / 60)} minutes, ${real.length ? 'highest ' + whole.format(Math.max(...real)) + ' ' + unit : 'no readings yet'}`);
    },
  };
}
// The KV pool held by each seat: one stacked layer a seat (the first slots of the engine's requests table), fixed buckets, the top of the
// plot the pool's whole size. Hover reads one bucket.
const SEAT_COLORS = ['#3f7fc4', '#9b7fd6', '#c8782c', '#5cc28a', '#e6c27a', '#d9738f', '#4cc3c9', '#a7aeb2'];
function poolChart() {
  const s = plotShell('HELD BY EACH SEAT', 'The KV positions each seat holds (its prompt and the reply so far), stacked; the top of the plot is the whole pool.', 'KV pool held by each seat');
  const layers = [];
  for (let k = 15; k >= 0; k--) {
    const color = SEAT_COLORS[k % SEAT_COLORS.length];
    const area = svg('path', { fill: color, 'fill-opacity': '0.30', stroke: 'none' });
    const edge = svg('path', { fill: 'none', stroke: color, 'stroke-width': '1', 'stroke-linejoin': 'round', 'vector-effect': 'non-scaling-stroke' });
    s.plotSvg.append(area, edge);
    layers[k] = { area, edge };
  }
  const st = { seats: [], n: 0, t0: 0, step: 1, pool: 1 };
  s.plot.addEventListener('pointermove', (e) => {
    if (st.n < 2) return;
    const i = hoverIndex(s.plot, st.n, e);
    s.hair.hidden = s.tipBox.hidden = false;
    s.hair.style.left = (i / (st.n - 1) * 100) + '%';
    s.tipBox.style.left = Math.min(88, Math.max(12, i / (st.n - 1) * 100)) + '%';
    const end = st.t0 + i * st.step * 1000;
    const held = st.seats.map((sv, k) => [k, sv[i] || 0]).filter((x) => x[1] > 0);
    const total = held.reduce((a, x) => a + x[1], 0);
    clear(s.tipBox).append(h('div', { class: 'dim', text: `${dayTime.format(new Date(end - st.step * 1000))} to ${dayTime.format(new Date(end))}` }),
      h('div', { class: 'num', text: `${fmt(Math.round(total))} positions · ${one.format(100 * total / st.pool)} % of the pool` }),
      ...held.map(([k, v]) => h('div', { class: 'num dim', text: `seat ${k + 1}: ${fmt(Math.round(v))}` })));
  });
  s.plot.addEventListener('pointerleave', () => { s.hair.hidden = s.tipBox.hidden = true; });
  return {
    el: s.el,
    draw(body, poolTokens) {
      const rows = (body.seats || []).map((sv) => (sv.points || []).map((p) => (p == null ? 0 : Number(p))));
      const n = rows.length ? rows[0].length : 0;
      const pool = Math.max(1, Number(poolTokens) || 1);
      Object.assign(st, { seats: rows, n, pool, step: body.step_s, t0: (body.last_ms != null ? Number(body.last_ms) : Date.now()) - (n - 1) * body.step_s * 1000 });
      const cum = [];
      rows.forEach((sv, k) => { cum[k] = sv.map((v, i) => v + (k ? cum[k - 1][i] : 0)); });
      const base = plotY(0).toFixed(1);
      rows.forEach((sv, k) => {
        const run = cum[k].map((v, i) => [i / Math.max(1, n - 1) * 600, plotY(v / pool)]);
        const d = smoothPath(run);
        const any = sv.some((v) => v > 0);
        layers[k].edge.setAttribute('d', any ? d : '');
        layers[k].area.setAttribute('d', any && run.length ? `${d} L${run[run.length - 1][0].toFixed(1)},${base} L${run[0][0].toFixed(1)},${base} Z` : '');
      });
      const lastTotal = rows.reduce((a, sv) => a + (sv[n - 1] || 0), 0);
      setText(s.now, `${one.format(n ? 100 * lastTotal / pool : 0)} %`);
      setText(s.g[0], `${whole.format(pool / 1e6 * 10) / 10} M`); setText(s.g[1], `${whole.format(pool / 2e6 * 10) / 10} M`);
      xlabels(s.xlab, st.t0, body.step_s * 1000, n);
    },
  };
}

// ---- temperatures: one fixed colour scale, green at 45 °C or under through yellow (65) and orange (80) to bright red at 90 and over.
// The zones' own critical trip is 104 °C. A reading the host did not give has no colour.
const TEMP_STOPS = [[45, [78, 201, 160]], [65, [230, 194, 122]], [80, [232, 135, 58]], [90, [255, 59, 48]]];
const TEMP_TRIP_C = 104;
function tempColor(c) {
  if (none(c)) return '';
  const x = Number(c);
  if (x <= TEMP_STOPS[0][0]) return `rgb(${TEMP_STOPS[0][1].join(',')})`;
  for (let i = 1; i < TEMP_STOPS.length; i++) {
    if (x <= TEMP_STOPS[i][0]) {
      const [t0, c0] = TEMP_STOPS[i - 1], [t1, c1] = TEMP_STOPS[i], f = (x - t0) / (t1 - t0);
      return `rgb(${c0.map((v, k) => Math.round(v + (c1[k] - v) * f)).join(',')})`;
    }
  }
  return `rgb(${TEMP_STOPS[TEMP_STOPS.length - 1][1].join(',')})`;
}
const TEMP_TIP = `Colour: green at 45 °C or under, yellow near 65, orange near 80, bright red at 90 and over; the zones trip at ${TEMP_TRIP_C} °C.`;

// ---- NODE HEALTH: one card a Spark, everything for that Spark grouped. The engine's own readings: rank 0 samples itself, the others
// report over the engine's link. A figure the host did not give reads 0 in its unit, never a dash (a dash flipping with 0 is noise). The GPU clock is shown with the limit NVML names
// and the reasons it gives for holding it down; nothing here changes either (this page does not change the clock).
function nodeCard(rank) {
  const f1 = (x) => one.format(none(x) ? 0 : Number(x));
  const f0 = (x) => whole.format(none(x) ? 0 : Number(x));
  const v = {};
  // a row: a label, one to three values, a tip for the row; `temp` marks a value the colour scale paints
  const row = (key, label, tip, n, temp) => {
    const cells = Array.from({ length: n }, () => h('span', { class: 'num', text: '0' }));
    if (temp) cells.forEach((c) => { c.dataset.temp = key; });
    v[key] = cells;
    return h('div', { class: 'nrow', 'data-row': key, title: tip }, h('span', { class: 'nl', text: label }), h('span', { class: 'nv' }, cells));
  };
  const group = (title, ...rows) => h('div', { class: 'ngroup' }, h('span', { class: 'gt', text: title }), ...rows);
  const wBar = h('div', { class: 'w' }), mBar = h('div', { class: 'm' }), kBar = h('div', { class: 'k' });
  const stack = h('div', { class: 'stack', role: 'img' }, wBar, mBar, kBar, h('div', { class: 'f' }));
  const ageEl = h('span', { class: 'age', 'data-age': '', hidden: true });
  const role = h('span', { class: 'role', text: rank === 0 ? 'rank 0 · API' : 'rank ' + rank });
  const el = h('div', { class: 'ncard', 'data-node': String(rank) },
    h('div', { class: 'nhead' }, h('span', { class: 'nn', text: 'SPARK ' + (rank + 1) }), h('span', { class: 'nr' }, ageEl, role)),
    group('MEMORY',
      row('mem', 'Free / total', `Memory this Spark has free now (MemAvailable) of its total, in GiB. From /proc/meminfo on this Spark, as it last reported it.`, 2),
      stack,
      row('split', 'Weights · work · KV', `How the memory was divided when the engine started: the model weights, the engine's working memory, the KV pool (from the memory record at the start; the KV pool can grow or shrink since).`, 3)),
    group('TEMPERATURE',
      row('tz', 'Zones max', `The hottest of this Spark's ACPI thermal zones, in °C (/sys/class/thermal). ${TEMP_TIP}`, 1, true),
      row('tg', 'GPU', `The GPU die temperature in °C (NVML). ${TEMP_TIP}`, 1, true),
      row('tn', 'NVMe', `The NVMe drive's temperature in °C (hwmon). ${TEMP_TIP}`, 1, true)),
    group('GPU',
      row('gb', 'Busy', `GPU utilisation in per cent over NVML's last sample window.`, 1),
      row('gc', 'Clock', `The GPU's clock now and the highest clock NVML names, in MHz. Shown only: this page cannot change the clock. The reasons NVML gives for holding it down show here when any is set.`, 1),
      row('gp', 'Power', `GPU power draw in watts (NVML).`, 1)),
    group('CPU',
      row('cl', 'Load', `One-minute load average over the CPU count (/proc/loadavg).`, 1),
      row('cb', 'Busy', `CPU busy time in per cent over all CPUs since the last report (/proc/stat).`, 1)),
    group('NVME',
      row('nt', 'Temp', `The NVMe drive's temperature in °C.`, 1, true),
      row('nw', 'Wear', `The NVMe drive's wear, per cent of its rated life used (SMART, read once a minute).`, 1)),
    group('RAILS',
      row('ra', 'Rail A', `The first RoCE port function (mlx5_0): bits in and out per second, from its port counters between two reports, in bps, Kbps, Mbps or Gbps as the rate needs (decimal: 1 Gbps = 10^9 bits a second); and its errors and discards since boot.`, 3),
      row('rb', 'Rail B', `The second RoCE port function (mlx5_2): bits in and out per second, in bps, Kbps, Mbps or Gbps as the rate needs, and its errors and discards since boot.`, 3)));
  const groups = el.querySelectorAll('.ngroup');
  groups[0].classList.add('wide'); groups[groups.length - 1].classList.add('wide'); // the memory and the rails take the card's whole width
  let prev = null;
  // each rail's last rate in and out: a status poll that brings no new report shows it again (a poll between two reports
  // computed no rate and the cell flipped between a rate and nothing); before the first two reports it is 0
  const held = [[0, 0], [0, 0]];
  const bps = (now, before, dt) => (before == null || now == null || dt <= 0 || Number(now) < Number(before) ? null : (Number(now) - Number(before)) * 8 / dt);
  const put = (cells, texts) => cells.forEach((c, i) => setText(c, texts[i]));
  const paintTemp = (cell, c) => { setColor(cell, tempColor(c)); setText(cell, f1(c) + ' °C'); };
  return {
    el,
    update(r, nh, ledger) {
      const have = r && r.age_ms != null;
      const old = have && (S.fails > 0 || r.age_ms > 6000);
      el.classList.toggle('old', !!old || !have);
      ageEl.hidden = !(old || (r && !have));
      if (!ageEl.hidden) setText(ageEl, have ? `no report ${Math.round(r.age_ms / 1000)} s` : 'no report');
      const led = ledger && ledger.get(rank);
      if (led) {
        setWidth(wBar, 100 * led.w / led.total); setWidth(mBar, 100 * led.m / led.total); setWidth(kBar, 100 * led.k / led.total);
        stack.setAttribute('aria-label', `Spark ${rank + 1}: weights ${gib(led.w)} GiB, working memory ${gib(led.m)} GiB, KV pool ${gib(led.k)} GiB`);
        put(v.split, [gib(led.w), gib(led.m), gib(led.k)]);
      }
      if (!have) return;
      const gpu = r.gpu || {}, nv = r.nvme || {};
      const zs = (r.zones_c || []).filter((z) => z != null);
      const zoneTip = (r.zones_c || []).map((z, i) => (z == null ? null : `${(nh.zone_names && nh.zone_names[i]) || 'zone ' + i}: ${one.format(z)} °C`)).filter(Boolean).join('\n');
      put(v.mem, [gib(r.mem_available_bytes), gib(r.mem_total_bytes) + ' GiB']);
      paintTemp(v.tz[0], zs.length ? Math.max(...zs) : null);
      setTip(el.querySelector('[data-row="tz"]'), `The hottest of this Spark's ACPI thermal zones, in °C (/sys/class/thermal). ${TEMP_TIP}${zoneTip ? '\n' + zoneTip : ''}`);
      paintTemp(v.tg[0], gpu.temp_c); paintTemp(v.tn[0], nv.temp_c); paintTemp(v.nt[0], nv.temp_c);
      put(v.gb, [f0(gpu.util_pct) + ' %']);
      put(v.gc, [f0(gpu.clock_mhz) + (gpu.clock_max_mhz == null ? '' : ' / ' + f0(gpu.clock_max_mhz))]);
      setColor(v.gc[0], gpu.reasons && gpu.reasons.length ? '#e0a33e' : '');
      setTip(el.querySelector('[data-row="gc"]'), `The GPU's clock now and the highest clock NVML names, in MHz. Shown only: this page cannot change the clock.` + (gpu.reasons && gpu.reasons.length ? `\nHeld down by: ${gpu.reasons.join(', ')}` : '\nNo clock limit event is set.'));
      put(v.gp, [f1(gpu.power_w) + ' W']);
      put(v.cl, [f1(r.load1) + (r.cpus > 0 ? ' / ' + r.cpus : '')]);
      put(v.cb, [f1(r.cpu_busy_pct) + ' %']);
      put(v.nw, [f0(nv.wear_pct) + ' %']);
      const at = S.statusAt - r.age_ms, fresh = !prev || at > prev.at, dt = prev && at > prev.at ? (at - prev.at) / 1000 : 0;
      [['ra', 0], ['rb', 1]].forEach(([k, i]) => {
        const rl = (r.rails || [])[i] || {}, bf = prev && prev.rails[i];
        if (fresh) {
          const rx = bps(rl.rx_bytes, bf && bf.rx_bytes, dt), tx = bps(rl.tx_bytes, bf && bf.tx_bytes, dt);
          if (rx != null) held[i][0] = rx;
          if (tx != null) held[i][1] = tx;
        }
        put(v[k], [fmtRate(held[i][0], 'in'), fmtRate(held[i][1], 'out'), `${f0(Number(rl.errors || 0) + Number(rl.discards || 0))} err`]);
        setColor(v[k][2], rl.errors != null && Number(rl.errors) + Number(rl.discards || 0) > 0 ? '#ff6b5e' : '');
      });
      if (fresh) prev = { at, rails: r.rails || [] };
    },
  };
  // a rate in bits a second with its unit scaled to it (decimal, as link speeds are named: the rails are 200 Gbps ports)
  function fmtRate(x, w) {
    if (none(x)) x = 0;
    const [d, u] = x >= 1e9 ? [1e9, 'Gbps'] : x >= 1e6 ? [1e6, 'Mbps'] : x >= 1e3 ? [1e3, 'Kbps'] : [1, 'bps'];
    return `${u === 'bps' ? f0(x) : one.format(x / d)} ${u} ${w}`;
  }
}

// ---- the Overview: a status board's front page, fed only by the engine's admin API. Built once; `update` writes into the boxes.
const posFmt = (v) => (none(v) ? '0' : v >= 1e6 ? (v / 1e6).toFixed(2) + ' M' : v >= 1e3 ? (v / 1e3).toFixed(0) + ' K' : whole.format(v));
// a request's age as m:ss, or h:mm:ss past an hour (the lines' column is narrow and the digits do not jump)
function clockText(s) { s = Math.max(0, Math.floor(s)); const hh = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = String(s % 60).padStart(2, '0'); return hh ? `${hh}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`; }
const secFmt = (s) => (none(s) ? '0' : s >= 100 ? whole.format(s) : s >= 10 ? one.format(s) : (Math.round(s * 100) / 100).toFixed(2));
function box(name, title, tip, ...kids) {
  const right = h('span', { class: 'bright' });
  const head = h('div', { class: 'bhead' }, h('span', { class: 'bt', text: title, title: tip || null }), right);
  return { el: h('section', { class: 'box', 'data-box': name, 'aria-label': title }, head, ...kids), right };
}
// a grid of tiles: a label, one value with its unit, one short line; the detail goes on hover (the board's rule: no prose, nothing wraps)
function tileGrid(specs, cls) {
  const t = {};
  const el = h('div', { class: 'tgrid' + (cls ? ' ' + cls : '') }, specs.map(([key, label]) => {
    const v = h('b', { class: 'num', text: '0' }), u = h('span', { class: 'u' }), q = h('span', { class: 'q' });
    const cell = h('div', { class: 'tile', 'data-tile': key }, h('span', { class: 'k', text: label }), h('span', { class: 'val' }, v, u), q);
    t[key] = { cell, v, u, q };
    return cell;
  }));
  return {
    el,
    set(key, value, unit, sub, tip) { const x = t[key]; setText(x.v, value); setText(x.u, unit ? ' ' + unit : ''); setText(x.q, sub || ''); setTip(x.cell, tip); },
const quantile = (arr, p) => { if (!arr.length) return null; const a = arr.slice().sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.floor(p * a.length))]; };

function linesBox() {
  const COLS = [['#', 'The seat: requests in flight, oldest first. The engine does not name a lane, so a seat is the position in that order.'], ['PHASE', 'DECODE: producing its reply. PREFILL: its prompt is being read now. QUEUED: submitted, waiting for the engine. The engine tells prefill and queue apart only through the one prompt pass it reports.'],
    ['STAGE', 'The request\'s context: the purple part is its prompt, the white part the reply so far. A prefilling request shows how much of its prompt the pass has read.'], ['PROMPT tok', 'Prompt tokens of the request.'], ['OUT tok', 'Reply tokens so far.'],
    ['WHO', 'The label of the key that sent the request.'], ['CACHED', 'Prompt tokens restored from a cache. The engine counts them when the request ends, so a line in flight shows a dash; the finished requests are in Requests & keys.'],
    ['TOK/S', 'Reply tokens a second for this request, from the change in its token count between two readings (smoothed). For a prefilling request, the prompt pass\'s speed.'], ['WAIT', 'About how long it waited before its first token (read once per second, so ±1 s; a dash for a request this page first saw already generating). A queued or prefilling request shows how long it has waited so far.'], ['TOTAL', 'How long the request has been in the engine.']];
  const sub = h('span', { class: 'bsub' });
  const grid = h('div', { class: 'lgrid' });
  const headRow = h('div', { class: 'lrow lhead' }, COLS.map(([c, tip]) => h('span', { text: c, title: tip })));
  const b = box('lines', 'LINES', 'Every seat the engine has, always shown: what each one is doing, its stage, its speed and its tokens.', headRow, grid);
  b.right.append(sub);
  const rows = [];
  const mkRow = (i) => {
    const bar = h('span', { class: 'stg' }, h('i', { class: 'pre' }), h('i', { class: 'dec' }));
    const c = COLS.map(() => h('span'));
    c[2] = bar;
    const el = h('div', { class: 'lrow', 'data-line': String(i) }, c);
    setText(c[0], i + 1);
    return { el, c, pre: bar.children[0], dec: bar.children[1] };
  };
  return {
    el: b.el,
    update(st, inflight, seen) {
      const hh = st.health || {}, live = hh.live || {}, max = hh.max_batch_size;
      const used = inflight.length;
      const n = Math.min(32, Math.max(none(max) ? 0 : max, used));
      while (rows.length < n) { const r = mkRow(rows.length); rows.push(r); grid.append(r.el); }
      while (rows.length > n) rows.pop().el.remove();
      setText(sub, none(max) ? `${used} in use` : `${used} of ${max} in use`);
      const p = live.prompt_pass;
      const firstWaiting = inflight.findIndex((x) => x.state !== 'decode');
      rows.forEach((r, i) => {
        const x = inflight[i];
        const c = r.c;
        r.el.classList.toggle('idle', !x);
        if (!x) {
          setText(c[1], 'idle'); setColor(c[1], '#5a6165'); setWidth(r.pre, 0); setWidth(r.dec, 0);
          for (const k of [3, 4, 5, 6, 7, 8, 9]) setText(c[k], '');
          return;
        }
        const m = seen.get(x.id) || {};
        const dec = x.state === 'decode';
        const filling = !dec && i === firstWaiting && !!p;
        setText(c[1], dec ? 'DECODE' : filling ? 'PREFILL' : 'QUEUED');
        setColor(c[1], dec ? '#e8eaec' : filling ? '#9d8bd0' : '#e0a33e');
        const ctx = Math.max(1, Number(x.prompt_tokens) + Number(x.generated_tokens));
        if (filling) { setWidth(r.pre, 100 * p.computed / Math.max(1, p.total)); setWidth(r.dec, 0); } else if (dec) { setWidth(r.pre, 100 * x.prompt_tokens / ctx); setWidth(r.dec, 100 * x.generated_tokens / ctx); } else { setWidth(r.pre, 0); setWidth(r.dec, 0); }
        setText(c[3], whole.format(x.prompt_tokens));
        setText(c[4], whole.format(x.generated_tokens));
        setText(c[5], x.key_label);
        setText(c[6], '0');
        setText(c[7], dec ? one.format(m.tps == null ? 0 : m.tps) : whole.format(filling && !none(live.prefill_tokens_per_second) ? live.prefill_tokens_per_second : 0));
        setText(c[8], dec ? `~${one.format(m.waitS == null ? 0 : m.waitS)} s` : `${one.format(x.age_s)} s`);
        setText(c[9], clockText(x.age_s));
        setTip(r.el, `Seat ${i + 1}: ${x.endpoint}, ${x.priority}, key ${x.key_label}; holds ${whole.format(x.kv_tokens)} KV positions.`);
      });
    },
  };
}

function healthBox() {
  const cards = new Map();
  const grid = h('div', { class: 'ngrid' });
  const none_ = h('p', { class: 'dim', text: 'not reported by this engine', hidden: true });
  const b = box('health', 'NODE HEALTH', 'Each Spark\'s health as it last reported it: a block is sent when a value moves, and at least every 2 s. Rank 0 samples itself.', grid, none_);
  const sub = h('span', { class: 'bsub' });
  b.right.append(sub);
  return {
    el: b.el,
    update(st, ledger) {
      const nh = st.node_health, rows = nh && nh.ranks ? nh.ranks : [];
      b.el.hidden = false;
      none_.hidden = rows.length > 0;
      for (const r of rows) {
        if (!cards.has(r.rank)) { const c = nodeCard(r.rank); cards.set(r.rank, c); grid.append(c.el); }
        cards.get(r.rank).update(r, nh, ledger);
      }
      const ok = rows.filter((r) => r.age_ms != null && r.age_ms <= 6000).length;
      setText(sub, rows.length ? `${ok} of ${rows.length} reporting` : '');
    },
  };
}

// The speed features, each by its schema label, on or off, grouped by the schema's group; the env name and the reason on hover. The
// section is built once and its rows are updated in place, so opening it is never undone by an update.
const GROUP_NAMES = { serving: 'Serving', transport: 'Transport', cache: 'Cache & SSD', prompt: 'Prompt passes', decode: 'Generation', memory: 'Memory', goals: 'Goals', glm: 'GLM', debug: 'Debug', other: 'Other' };
function speedBox() {
  const det = h('details', { class: 'sfbox' });
  const sum = h('summary', {});
  const body = h('div', { class: 'sfbody' });
  det.append(sum, body);
  const b = box('speed', 'SPEED FEATURES', 'The engine\'s essential speed features, each on or off. Hover a line for its setting, the reason and what it costs.', det);
  let sig = '', rowEls = [];
  return {
    el: b.el,
    update(st) {
      b.el.hidden = !Array.isArray(fs);
      const label = (f) => { const o = S.byEnv && S.byEnv.get(f.env); return (o && o.label) || f.name; };
      const grp = (f) => { const o = S.byEnv && S.byEnv.get(f.env); return GROUP_NAMES[o && o.group] || 'Other'; };
      const next = fs.map((f) => f.env + '|' + grp(f)).join(',');
      if (next !== sig) {
        sig = next;
        const by = new Map();
        fs.forEach((f, i) => { const g = grp(f); if (!by.has(g)) by.set(g, []); by.get(g).push(i); });
        rowEls = [];
        const kids = [];
        for (const [g, idx] of [...by.entries()].sort((x, y) => x[0].localeCompare(y[0]))) {
          kids.push(h('span', { class: 'gt', text: g.toUpperCase() }));
          for (const i of idx) {
            const nm = h('span', { class: 'sl' }), chip = h('span', { class: 'chipx' });
            const el = h('div', { class: 'srow2', 'data-env': fs[i].env }, nm, chip);
            rowEls[i] = { el, nm, chip };
            kids.push(el);
          }
        }
        body.replaceChildren(...kids);
      }
      const on = fs.filter((f) => f.on).length, bad = fs.filter((f) => !f.on && !f.asked).length;
      setText(sum, `${on} on · ${fs.length - on} off` + (bad ? ` · ${bad} not asked for` : ''));
      b.right.classList.toggle('warn', bad > 0);
      fs.forEach((f, i) => {
        const r = rowEls[i];
        setText(r.nm, label(f));
        setText(r.chip, f.on ? 'ON' : f.asked ? 'OFF · chosen' : 'OFF');
        r.el.classList.toggle('off', !f.on && !f.asked);
        r.el.classList.toggle('chosen', !f.on && !!f.asked);
        setTip(r.el, `${f.env}\n${f.reason}${f.on ? '' : '\nWithout it: ' + f.cost}`);
      });
    },
}

function kvBox() {
  const tiles = tileGrid([['pool', 'POOL positions'], ['use', 'IN USE'], ['window', 'WINDOW'], ['target', 'FREE TARGET']]);
  const fill = h('div', { class: 'fill' });
  const meter = h('div', { class: 'meter', role: 'img' }, fill);
  const pool = poolChart();
  const b = box('kv', 'KV POOL', 'The KV cache: positions (tokens) the engine can hold at once, and how much of it the requests in flight hold. With the growing pool the size moves.', tiles.el, meter, pool.el);
  const state = h('span', { class: 'bsub' });
  b.right.append(state);
  return {
    el: b.el, pool,
    update(st, inflight) {
      const kv = st.kv_pool;
      b.el.hidden = !kv;
      if (!kv) return;
      const tot = Number(kv.tokens_active), bpt = Number(kv.bytes_per_token) || 0;
      const used = inflight.reduce((a, x) => a + Number(x.kv_tokens), 0);
      const perRank = tot * bpt / 2 ** 30;
      const free = kv.mode === 'free_target';
      setText(state, free ? 'grows and shrinks · keeps ' + (kv.free_target_gib || '?') + ' GiB free' : 'fixed size');
      tiles.set('pool', posFmt(tot), '', bpt ? `${one.format(perRank)} GiB a rank` : '', `${whole.format(tot)} positions are in the pool now (asked for ${whole.format(kv.tokens_requested)}); at ${bpt} bytes a position that is ${one.format(perRank)} GiB on each Spark.`);
      tiles.set('use', posFmt(used), '', `${one.format(100 * used / Math.max(1, tot))} % of pool`, `The requests in flight hold ${whole.format(used)} positions (their prompt and the reply so far).`);
      tiles.set('window', posFmt(kv.usable_window), '', 'one request', `The longest context one request may use: ${whole.format(kv.usable_window)} tokens. The pool holds ${one.format(tot / Math.max(1, kv.usable_window))} such requests.`);
      const lr = kv.last_resize;
      tiles.set('target', free ? kv.free_target_gib || 0 : 'OFF', free ? 'GiB' : '', free ? (lr ? `last: ${lr.verdict.replace(/_/g, ' ')}` : 'no resize yet') : 'fixed pool', free ? `The engine resizes the pool to keep this much memory free on each Spark. Growth it leaves for a prompt pass: ${kv.growth_gib} GiB.` + (lr ? `\nLast resize (version ${lr.version}): ${lr.verdict.replace(/_/g, ' ')}, from ${whole.format(lr.from_tokens)} to a fit of ${whole.format(lr.fit_tokens)} positions.` : '') : 'The pool has a size set in Settings; it does not move.');
      setWidth(fill, 100 * used / Math.max(1, tot));
      meter.setAttribute('aria-label', `${one.format(100 * used / Math.max(1, tot))} % of the KV pool in use`);
    },
  };
}

function tierBox() {
  const tiles = tileGrid([['stored', 'STORED'], ['writes', 'WRITES'], ['restores', 'RESTORES'], ['restored', 'RESTORED'], ['wait', 'RESTORE WAIT'], ['wear', 'DISK WEAR'], ['records', 'RECORDS']], 'c4');
  const b = box('tier', 'SSD TIER', 'The prompt states kept on the SSD of node 1, restored instead of read again.', tiles.el);
  const state = h('span', { class: 'bsub' });
  b.right.append(state);
  return {
    el: b.el,
    update(st) {
      const hh = st.health || {}, pc = hh.prompt_cache, sp = hh.spill;
      const t = pc && pc.spill && pc.spill[0];
      b.el.hidden = !sp;
      if (!sp) return;
      const on = sp.state === 'on';
      setText(state, sp.state);
      b.right.classList.toggle('warn', !on);
      if (!t || !pc) { for (const k of ['stored', 'writes', 'restores', 'restored', 'wait', 'wear', 'records']) tiles.set(k, '0', '', ''); return; }
      const nh = st.node_health && st.node_health.ranks && st.node_health.ranks.find((r) => r.rank === 0);
      const wear = nh && nh.nvme && nh.nvme.wear_pct;
      const avgMs = pc.restore_waits > 0 ? pc.restore_wait_ns / pc.restore_waits / 1e6 : null;
      tiles.set('stored', gib(t.stored_bytes), 'GiB', `of ${gib(t.cap_bytes)} GiB`, `The unique bytes the tier holds on disk now, of the ${gib(t.cap_bytes)} GiB it may use.`);
      tiles.set('writes', whole.format(t.writes_committed), '', `${whole.format(t.writes_pending)} waiting`, `Prompt states written and published since the engine started: ${gib(t.written_bytes)} GiB written; ${whole.format(t.writes_pending)} queued, not yet published.`);
      tiles.set('restores', whole.format(pc.restores_begun), '', `${whole.format(pc.disk_hits)} disk hits`, `Disk restores begun since the engine started; ${whole.format(pc.disk_hits)} lookups found a state on disk longer than memory had.`);
      tiles.set('restored', posFmt(pc.restored_tokens), '', 'not recomputed', `Prompt tokens restored from disk instead of computed, since the engine started.`);
      tiles.set('wait', whole.format(avgMs == null ? 0 : avgMs), 'ms', `${whole.format(pc.restore_waits)} waits`, `The average time a request waited for a restore (${whole.format(pc.restore_waits)} waits, the longest ${whole.format((pc.restore_wait_max_ns || 0) / 1e6)} ms).`);
      tiles.set('wear', whole.format(none(wear) ? 0 : wear), '%', 'node 1 NVMe', 'The share of the NVMe drive\'s rated life used (SMART), as node 1 reports it.');
      tiles.set('records', whole.format(t.points), '', 'points on disk', 'The published prompt states (points) in the tier\'s index.');
    },
  };
}

function rateBox() {
  const tiles = tileGrid([['decode', 'DECODE tok/s'], ['stream', 'USER tok/s'], ['prefill', 'PREFILL tok/s'], ['itl', 'TOKEN GAP ms'], ['ttft', 'TTFT p99 s']], 'c5');
  const b = box('rate', 'RATE', 'How fast the engine runs now.', tiles.el);
  return {
    el: b.el,
    update(st, recent) {
      const live = (st.health || {}).live || {};
      const running = none(live.connections) ? 0 : live.connections - (live.waiting || 0);
      const dec = live.decode_tokens_per_second;
      const per = running > 0 && !none(dec) ? dec / running : null;
      const ttfts = (recent || []).map((x) => x.ttft_s).filter((x) => !none(x));
      tiles.set('decode', fmt(dec), '', `${running} stream${running === 1 ? '' : 's'}`, 'Tokens a second the engine generates for all streams together (the engine\'s own live counter).');
      tiles.set('stream', one.format(none(per) ? 0 : per), '', 'mean a user', 'The decode rate divided by the streams generating: what one user gets on average.');
      tiles.set('itl', whole.format(none(per) || per <= 0 ? 0 : 1000 / per), '', 'per token', 'The mean time between two tokens of one stream: 1000 over the per-stream rate.');
      tiles.set('ttft', ttfts.length ? secFmt(quantile(ttfts, 0.99)) : '0', '', ttfts.length ? `p50 ${secFmt(quantile(ttfts, 0.5))} s` : 'none finished', `Time to first token over the last ${ttfts.length} finished requests the engine remembers (a prompt of any length).`);
    },
  };
}

function transportBox() {
  const list = h('div', { class: 'trows' });
  const b = box('transport', 'LINKS', 'How the Sparks exchange data, as the engine decided at its start.', list);
  const rows = [];
  let sig = '';
  return {
    el: b.el,
    update(st) {
      const tp = st.transport;
      b.el.hidden = !tp;
      if (!tp) return;
      const p = tp.prompt_exchanges, g = tp.generation_exchanges, w = tp.link_watch;
      const peers = (st.memory && st.memory.ranks ? st.memory.ranks.length : 0) - 1;
      const how = (waits) => (waits ? 'the network card waits on data' : 'proxy threads wait on data');
      const items = [
        ['Prompt exchanges', p.known ? 'ON' : 'IDLE', p.known ? '' : 'idle', p.known ? `Rocinante; ${how(p.waits_in_card)}${p.arrival_signal ? '; arrival signal: ' + p.arrival_signal : ''}` : 'The engine recorded nothing about them at its start.'],
        ['Generation exchanges', !g.known ? 'IDLE' : g.on ? 'ON' : 'OFF', !g.known ? 'idle' : g.on ? '' : 'warn', !g.known ? 'The engine recorded nothing about them at its start.' : g.on ? `RDMA rings; ${how(g.waits_in_card)}; ${g.arrival_signal ? 'arrival signal: ' + g.arrival_signal : 'no generation word (completion events)'}` : 'No RDMA ring: generation exchanges use the other transport.'],
        ['Link watch', w.links > 0 ? (peers > 0 && w.links >= peers ? `${w.links} of ${peers}` : String(w.links)) : 'OFF', w.links > 0 ? '' : 'idle', 'If a Spark stops answering, the others stop within about 6 seconds and say which one.'],
      ];
      const next = items.length + '';
      if (sig !== next) { sig = next; items.forEach(() => { const k = h('span', { class: 'nl' }), s = h('span', { class: 'chipx' }); rows.push({ el: h('div', { class: 'srow2' }, k, s), k, s }); }); list.replaceChildren(...rows.map((r) => r.el)); }
      items.forEach(([name, state, cls, tip], i) => { setText(rows[i].k, name); setText(rows[i].s, state); rows[i].el.classList.toggle('off', cls === 'warn'); rows[i].el.classList.toggle('chosen', cls === 'idle'); setTip(rows[i].el, tip); });
    },
  };
}

function overview() {
  const lines = linesBox(), health = healthBox(), speed = speedBox(), kv = kvBox(), tier = tierBox(), rate = rateBox(), links = transportBox();
  const cd = chart('DECODE', 'Tokens a second the engine generates, all streams, the last 15 minutes. Fixed 10-second buckets: a bucket keeps its value as the window moves; hover for the bucket\'s mean and peak.', '#e8eaec', 'tok/s', 'decode_tps');
  const wrap = (name, c) => h('section', { class: 'box plotbox', 'data-box': name }, c.el);
  refs.main.append(h('div', { class: 'board' },
    h('div', { class: 'col' }, lines.el, health.el, speed.el),
    h('div', { class: 'col' }, kv.el, tier.el, rate.el, wrap('decode', cd), wrap('prefill', cp), links.el)));
  const seen = new Map(); // request id -> { gen, at, tps, waitS }
  let inflight = [], recent = [], ledger = null, alive = true;
  const track = (body, at) => {
    inflight = body.in_flight.slice().sort((a, b) => b.age_s - a.age_s);
    recent = body.recent.slice(0, 50);
    const ids = new Set(inflight.map((x) => x.id));
    for (const id of [...seen.keys()]) if (!ids.has(id)) seen.delete(id);
    for (const x of inflight) {
      const m = seen.get(x.id) || { gen: Number(x.generated_tokens), at, tps: null, waitS: null, early: Number(x.generated_tokens) === 0 };
      const dt = (at - m.at) / 1000;
      if (dt >= 0.5 && Number(x.generated_tokens) >= m.gen) {
        const inst = (Number(x.generated_tokens) - m.gen) / dt;
        m.tps = m.tps == null ? inst : 0.6 * m.tps + 0.4 * inst;
        m.gen = Number(x.generated_tokens); m.at = at;
      }
      if (m.early && m.waitS == null && Number(x.generated_tokens) > 0) m.waitS = Number(x.age_s);
      seen.set(x.id, m);
    }
  };
  refs.pull = async (tick) => {
    const names = ROUTE_NAMES();
    if (names.has('requests')) { const r = await api('requests'); if (alive && r.ok) track(r.body, Date.now()); }
    if (!names.has('series') || (tick % 5 !== 1 && S.chartsDrawn)) return;
    const [d, p, q] = await Promise.all(['decode_tps', 'prefill_tps', 'seat_kv'].map((n) => api('series', `name=${n}&seconds=900&step=10`)));
    if (!alive) return;
    const at = S.status && S.status.at_ms ? Number(S.status.at_ms) : Date.now();
    if (d.ok) cd.draw(d.body, at);
    if (p.ok) cp.draw(p.body, at);
    if (q.ok && q.body.seats) kv.pool.draw(q.body, S.status && S.status.kv_pool ? S.status.kv_pool.tokens_active : 0);
    S.chartsDrawn = true;
    if (names.has('ledger') && (!ledger || tick % 30 === 1)) {
      const l = await api('ledger');
      if (alive && l.ok && l.body.ranks && l.body.ranks.length) {
        const by = new Map();
        for (const e of l.body.entries) {
          if (!by.has(e.rank)) by.set(e.rank, { w: 0, k: 0, m: 0, total: 1 });
          const b = by.get(e.rank), v = Number(e.device_bytes);
          if (e.feature_id === 'weights') b.w += v; else if (e.feature_id === 'kv_pool') b.k += v; else b.m += v;
        }
        for (const rk of l.body.ranks) if (by.has(rk.rank)) by.get(rk.rank).total = Number(rk.device_bytes) || 1;
        ledger = by;
      }
    }
  };
  refs.update = (stale) => {
    refs.main.classList.toggle('stale', stale);
    const st = S.status; if (!st) return;
    lines.update(st, inflight, seen);
    health.update(st, ledger);
    speed.update(st);
    kv.update(st, inflight);
    tier.update(st);
    rate.update(st, recent);
    links.update(st);
    const live = (st.health || {}).live || {};
    cd.setNow(live.decode_tokens_per_second);
    cp.setNow(live.prefill_tokens_per_second);
  };
  S.chartsDrawn = false;
  refs.update(false);
  return () => { alive = false; refs.update = null; refs.pull = null; };
}

// ---- settings
const WRITE = { staged: new Map(), reverts: null, sheet: null, outcome: null, busy: false, reason: '', persist: null };
const T = WRITE;
const same = (a, b) => a === b || (a != null && b != null && a !== '' && b !== '' && !Number.isNaN(Number(a)) && Number(a) === Number(b));
function cfgOf(o) { return S.cfgById.get(o.id) || {}; }
// what the engine runs (or has been asked to run) for an option, as text; null when the code decides
function currentOf(o) {
  const c = cfgOf(o);
  if (o.secret) return null;
  if (c.requested != null) return c.requested;
  return c.value != null ? c.value : o.default;
}
function defaultText(o) {
  if (o.default_text) return o.default_text;
  if (o.default == null) return 'Chosen by the engine';
  if (o.kind === 'flag') return o.default === '1' ? 'On' : 'Off';
  return o.default === '' ? '(empty)' : (o.unit ? `${o.default} ${o.unit}` : o.default);
}
function applyTag(o) {
  const live = (o.apply === 'H' || o.apply === 'A') && o.live !== false;
  if (live) return ['Applies live', 'live'];
  if (o.apply === 'P') return ['Set per request', ''];
  if (o.apply === 'I' || o.apply === 'N') return ['Applies when idle', 'live'];
  return ['Needs a restart', 'restart'];
}
function writable() { const c = S.status && S.status.capabilities; return !!(c && c.writable_classes && c.writable_classes.length) && ROUTE_NAMES().has('set'); }
function canWrite(o) { return writable() && !o.secret && o.who.length > 0 && S.status.capabilities.writable_classes.includes(o.apply); }
function shownValue(o) { return T.staged.has(o.name) ? (T.staged.get(o.name).value === 'unset' ? o.default : T.staged.get(o.name).value) : currentOf(o); }
function isChanged(o) { const v = shownValue(o); const d = o.default; if (o.secret) return cfgOf(o).source && cfgOf(o).source !== 'default'; return v != null && !same(v, d) && !(d == null && v === ''); }
function range(o) {
  if (o.kind === 'enum' || o.kind === 'flag') return '';
  const f = (x) => whole.format(x);
  if (o.min != null && o.max != null) return `From ${f(o.min)} to ${f(o.max)}`;
  if (o.min != null) return `${f(o.min)} or more`;
  if (o.max != null) return `Up to ${f(o.max)}`;
  return '';
}
function stage(o, value) {
  const cur = currentOf(o);
  const nothingSet = cfgOf(o).source === 'default' || cfgOf(o).value == null;
  if (value === 'unset' ? nothingSet : same(value, cur)) T.staged.delete(o.name); else T.staged.set(o.name, { o, value });
  T.sheet = null; T.outcome = null;
  if (!T.staged.size) T.reverts = null;
  drawTray();
}
function unstage(name) { T.staged.delete(name); T.sheet = null; if (!T.staged.size) T.reverts = null; drawTray(); }

// one row of the list; sync() repaints what a staged change touches without rebuilding the inputs (so focus stays put)
function settingRow(o) {
  const [tag, tagCls] = applyTag(o);
  const fid = 'set-' + o.id;
  const allowed = canWrite(o);
  const changedChip = h('span', { class: 'chip chg', text: 'Changed from default', hidden: true });
  let control, setControl = () => {};
  const val = () => { const v = shownValue(o); return v == null ? '' : String(v); };
  if (o.secret) {
    control = h('span', { class: 'mono dim', text: cfgOf(o).source && cfgOf(o).source !== 'default' ? 'Set (never shown)' : 'Not set' });
  } else if (!allowed) {
    control = h('span', { class: 'num', id: fid, text: o.kind === 'flag' ? (val() === '1' ? 'On' : 'Off') : val() === '' ? defaultText(o) : o.unit ? `${val()} ${o.unit}` : val() });
  } else if (o.kind === 'flag') {
    const sw = h('button', { type: 'button', role: 'switch', id: fid, class: 'btn sm', 'aria-checked': 'false' });
    const paint = () => { const on = val() === '1'; sw.setAttribute('aria-checked', on ? 'true' : 'false'); sw.textContent = on ? 'On' : 'Off'; };
    sw.addEventListener('click', () => stage(o, val() === '1' ? '0' : '1'));
    setControl = paint; paint(); control = sw;
  } else if (o.kind === 'enum') {
    const sel = h('select', { id: fid, class: 'pick' }, o.values.map((v) => h('option', { value: v, text: v === '' ? '(empty)' : v })));
    if (!o.values.includes(val())) sel.prepend(h('option', { value: val(), text: val() === '' ? defaultText(o) : val() }));
    sel.addEventListener('change', () => stage(o, sel.value));
    setControl = () => { sel.value = val(); sel.classList.toggle('staged', T.staged.has(o.name)); }; setControl(); control = sel;
  } else {
    const input = h('input', { id: fid, type: o.kind === 'text' ? 'text' : 'number', step: o.kind === 'float' ? 'any' : '1', autocomplete: 'off', spellcheck: 'false' });
    if (o.min != null) input.min = o.min;
    if (o.max != null) input.max = o.max;
    input.addEventListener('change', () => { if (input.value !== '') stage(o, input.value); });
    const box = h('div', { class: 'field' }, input, o.unit ? h('span', { class: 'u', text: o.unit }) : null);
    setControl = () => { if (document.activeElement !== input) input.value = val(); box.classList.toggle('staged', T.staged.has(o.name)); }; setControl(); control = box;
  }
  const defBtn = h('button', { type: 'button', class: 'linkbtn', hidden: true, onclick: () => stage(o, 'unset') }, 'Use default');
  const dots = h('button', { type: 'button', class: 'dots', 'aria-expanded': S.open.has(o.id) ? 'true' : 'false', 'aria-label': 'Technical details for ' + o.label }, icon([['circle', { cx: 2, cy: 2, r: 1.6 }], ['circle', { cx: 8, cy: 2, r: 1.6 }], ['circle', { cx: 14, cy: 2, r: 1.6 }]], 16, 4, { fill: '#c3c8cb' }));
  const c = cfgOf(o);
  const tech = h('dl', { class: 'tech', hidden: !S.open.has(o.id) },
    [['Environment variable', o.env], ['Setting id', String(o.id)],
      ['Current value from', (c.source || 'default') + (c.other ? ` (${c.other.source === 'env' ? 'replaced the environment\'s ' : 'the environment kept its place over engine.cfg\'s '}${c.other.value})` : '')],
      ['Allowed', range(o) || (o.values.length ? o.values.map((v) => v || '(empty)').join(', ') : o.kind === 'flag' ? 'On or off' : 'Any text')],
      ['Takes effect', (S.schema.apply_classes[o.apply] || {}).text && tagCls === 'live' ? 'Live: ' + S.schema.apply_classes[o.apply].text : tagCls === 'restart' ? 'At the next engine start' : tag]].map(([k, v]) => h('div', {}, h('dt', { text: k }), h('dd', { text: v }))));
  dots.addEventListener('click', () => { const on = !S.open.has(o.id); if (on) S.open.add(o.id); else S.open.delete(o.id); dots.setAttribute('aria-expanded', on ? 'true' : 'false'); tech.hidden = !on; });
  const rg = range(o);
  const row = h('div', { class: 'srow', 'data-id': o.id },
    h('div', { class: 'scols' },
      h('div', { class: 'sname', title: (o.doc || o.label || o.name) + '\n' + o.env }, h('div', { class: 't' }, allowed || o.secret ? h('label', { for: fid, text: o.label || o.name }) : h('span', { class: 'lbl', text: o.label || o.name }), h('span', { class: 'chip ' + tagCls, text: tag }), changedChip),
        o.secret ? h('p', { class: 'note', text: 'A key: it is never shown or changed here.' }) : !allowed ? h('p', { class: 'note', text: writable() ? 'This setting cannot be changed from this page.' : 'This key may not change settings.' }) : null),
      h('div', { class: 'sval' }, control, rg ? h('span', { class: 'range', text: rg }) : null),
      h('div', { class: 'sdef' }, h('span', { text: defaultText(o) }), defBtn),
      dots),
    tech);
  row.sync = () => {
    const staged = T.staged.has(o.name);
    row.classList.toggle('is-staged', staged);
    setControl();
    const ch = isChanged(o);
    changedChip.hidden = !ch;
    defBtn.hidden = !(allowed && (ch || staged)) || o.secret;
  };
  row.sync();
  return row;
}
function settingsPage() {
  const listed = S.schema.tabs && S.schema.tabs.length ? S.schema.tabs.slice() : [];
  // every option the API serves is reachable: one whose tab the schema's list does not name gets a group of its own
  for (const n of new Set(S.schema.options.map((o) => o.tab))) if (!listed.some((t) => t.name === n)) listed.push({ name: n, title: n, about: '' });
  const tabs = listed;
  if (!S.group || !tabs.some((t) => t.name === S.group)) S.group = tabs[0].name;
  const counts = new Map();
  for (const o of S.schema.options) counts.set(o.tab, (counts.get(o.tab) || 0) + 1);
  const groupNav = h('nav', { class: 'groups', 'aria-label': 'Setting groups' });
  const title = h('h1'), about = h('p', { class: 'dim' });
  const search = h('input', { id: 'q', type: 'search', placeholder: 'Search every setting by name or description', autocomplete: 'off', value: S.query });
  const filters = [['all', 'All'], ['changed', 'Changed from default'], ['live', 'Applies live']];
  const seg = h('div', { class: 'seg', role: 'group', 'aria-label': 'Show' });
  const list = h('div', { class: 'srows' });
  const goals = ROUTE_NAMES().has('goals') ? goalsRow() : null;
  const status = h('p', { class: 'note', role: 'status' });
  let rows = [];
  const paintNav = () => {
    clear(groupNav).append(h('span', { class: 'label', text: 'Groups' }), ...tabs.map((t) => h('button', { type: 'button', 'aria-current': t.name === S.group && !S.query ? 'true' : 'false', onclick: () => { S.group = t.name; S.query = ''; search.value = ''; draw(); } }, h('span', { text: t.name }), h('span', { class: 'n', text: String(counts.get(t.name) || 0) }))));
    clear(seg).append(...filters.map(([id, name]) => h('button', { type: 'button', 'aria-pressed': S.filter === id ? 'true' : 'false', onclick: () => { S.filter = id; draw(); } }, name)));
  };
  const draw = () => {
    paintNav();
    const q = S.query.trim().toLowerCase();
    const cur = tabs.find((t) => t.name === S.group);
    title.textContent = q ? 'Search results' : cur.title || cur.name;
    about.textContent = q ? 'Across every group.' : cur.about || '';
    let opts = q ? S.schema.options : S.schema.options.filter((o) => o.tab === S.group);
    if (q) opts = opts.filter((o) => ((o.label || '') + ' ' + (o.doc || '') + ' ' + o.name + ' ' + o.env).toLowerCase().includes(q));
    if (S.filter === 'changed') opts = opts.filter(isChanged);
    if (S.filter === 'live') opts = opts.filter((o) => applyTag(o)[1] === 'live');
    rows = opts.map(settingRow);
    if (goals) goals.el.hidden = q !== '' || S.group !== 'Goals';
    clear(list).append(h('div', { class: 'scols shead label', 'aria-hidden': 'true' }, h('span', { text: 'Setting' }), h('span', { text: 'Value' }), h('span', { text: 'Default' }), h('span')), ...rows);
    status.textContent = rows.length ? `${rows.length} setting${rows.length === 1 ? '' : 's'} shown` : 'No setting matches.';
  };
  search.addEventListener('input', () => { S.query = search.value; draw(); });
  refs.main.append(h('div', { class: 'settings' }, groupNav,
    h('div', { class: 'sbody' }, h('div', { class: 'sname' }, title, about),
      h('div', { class: 'toolbar' }, h('label', { class: 'sr', for: 'q', text: 'Search settings' }), h('div', { class: 'search' }, icon([['circle', { cx: 7, cy: 7, r: 5 }], ['path', { d: 'M11 11l3.5 3.5' }]], 16, 16, { stroke: '#8b9296', 'stroke-width': '1.6', 'stroke-linecap': 'round' }), search), seg),
      ...(goals ? [goals.el] : []), list, status)));
  draw();
  settingsRepaint = () => rows.forEach((r) => r.sync());
  let alive = true;
  if (goals) { goals.load(() => alive); S.goalsTimer = setInterval(() => { if (!goals.el.hidden) goals.load(() => alive); }, 5000); }
  return () => { alive = false; clearInterval(S.goalsTimer); settingsRepaint = () => {}; };
}
// The goals controller's verdict (alert only: the engine measures and says what it would take, it changes nothing), as one compact row in
// the Goals group. Its chart (the metric against the goal's dashed line) exists only while a goal is set.
function goalsRow() {
  const pill = h('span', { class: 'state idle' }, h('i', { class: 'dotc' }), 'loading');
  const facts = h('span', { class: 'gfacts num' });
  const holder = h('div', { class: 'gchart' });
  const el = h('div', { class: 'goalsrow', 'data-goals': '', title: 'The goal controller: an operator sets a target such as tokens a second for each user; the engine only reports whether it meets it and changes nothing. Figures marked ESTIMATED come from the model priors in the record, not from this build.' }, pill, facts, holder);
  const word = { off: 'Off', idle: 'No goal set', meeting: 'Meeting the goal', short: 'Short of the goal', unreachable: 'Cannot be reached' };
  const cls = { meeting: 'state', short: 'state warn', unreachable: 'state bad' };
  let gc = null;
  const g1 = (g, n) => (g.goals || []).find((x) => x.name === n);
  return {
    el,
    async load(alive) {
      const g = await api('goals');
      if (!alive() || !g.ok) return;
      const b = g.body, m = b.measured || {}, pre = b.mode === 'prefill_first';
      const dg = g1(b, 'decode'), pg = g1(b, 'prefill');
      pill.className = cls[b.state] || 'state idle';
      const t = word[b.state] || b.state;
      if (pill.lastChild.textContent !== t) pill.lastChild.textContent = t;
      setText(facts, `${dg ? 'decode ' + fmt(dg.target) + ' tok/s a user' : 'no decode goal'} · ${pg ? 'prompt ' + fmt(pg.target) + ' tok/s' : 'no prompt goal'} · measured ${none(m.decode_user_tps) ? '-' : one.format(m.decode_user_tps)} tok/s a user, prompt ${fmt(m.prefill_tps)}` + (b.met_fraction == null ? '' : ` · met in ${Math.round(100 * b.met_fraction)} %`));
      setTip(facts, b.reachability || '');
      const goal = pre ? (pg && pg.target) : (dg && dg.target);
      if (!goal || b.state === 'off' || !ROUTE_NAMES().has('series')) { holder.replaceChildren(); gc = null; return; }
      if (!gc) { gc = chart('AGAINST THE GOAL', 'The metric against the goal\'s dashed line, the last 15 minutes.', '#4ec9a0', 'tok/s', 'decode_tps'); holder.append(gc.el); }
      const [d, r, p] = await Promise.all(['decode_tps', 'running', 'prefill_tps'].map((n) => api('series', `name=${n}&seconds=900&step=10`)));
      if (!alive() || !d.ok || !r.ok || !p.ok) return;
      const body = pre ? p.body : { ...d.body, points: d.body.points.map((v, i) => (v == null || r.body.points[i] == null ? null : Number(v) / Math.max(1, Number(r.body.points[i])))) };
      gc.draw(body, S.status && S.status.at_ms ? Number(S.status.at_ms) : Date.now(), goal);
      gc.setNow(pre ? m.prefill_tps : m.decode_user_tps);
    },
  };
}
let settingsRepaint = () => {};

// ---- the staged-changes bar: Discard, Check (the engine's dry run), Apply
function changeLine(name, to) { const o = S.byName.get(name); return `${o ? o.label || o.name : name}: ${to}`; }
function drawTray() {
  settingsRepaint();
  balanceRepaint();
  turboRepaint();
  const tray = refs.tray;
  if (!tray) return;
  tray.hidden = !T.staged.size && !T.outcome;
  clear(tray);
  if (tray.hidden) return;
  const n = T.staged.size;
  const items = [...T.staged.values()].map(({ o, value }) => `${o.label || o.name}: ${valueWord(o, currentOf(o))} → ${value === 'unset' ? 'default (' + defaultText(o) + ')' : valueWord(o, value)}`);
  const cfg = !!(S.config && S.config.engine_cfg);
  persist.addEventListener('change', () => { T.persist = persist.checked; });
  const reason = h('input', { type: 'text', maxlength: '200', placeholder: 'Why (kept in the history)', 'aria-label': 'Reason', value: T.reason });
  reason.addEventListener('input', () => { T.reason = reason.value; });
  const anyRestart = T.sheet && T.sheet.resolved.some((r) => r.apply_mode !== 'live');
  const checkLine = !T.sheet ? (n ? 'Press Check to have the engine validate before applying.' : '') : T.sheet.ok ? 'Checked by the engine: valid' + (anyRestart ? '. Some changes take effect at the next start.' : ', applies live.') : 'The engine refuses part of this:';
  const msg = h('div', { class: 'msg' },
    n ? h('b', { text: (T.reverts != null ? `Undo of version ${T.reverts}: ` : '') + `${n} change${n === 1 ? '' : 's'} not yet applied` }) : null,
    n ? h('span', { class: 'dim', text: items.join(' · ') }) : null,
    h('span', { class: T.sheet && !T.sheet.ok ? 'err' : T.sheet ? 'okc' : 'faint', text: checkLine }),
    T.sheet && T.sheet.refused.length ? h('ul', { class: 'err' }, T.sheet.refused.map((r) => h('li', { text: `${changeLine(r.name, r.value ?? '')}`.replace(/: $/, '') + ' - ' + r.message }))) : null,
    T.sheet ? h('ul', { class: 'dim' }, T.sheet.warnings.filter((w) => w.code !== 'restart_needed').map((w) => h('li', { text: `${(S.byName.get(w.name) || {}).label || w.name}: ${w.code.replace(/_/g, ' ')}` + (w.requires ? ' (' + w.requires + ')' : '') }))) : null,
    T.sheet && T.sheet.derived && T.sheet.derived.kv_pool ? h('span', { class: 'dim', text: `KV pool: ${fmt(T.sheet.derived.kv_pool.tokens_before)} to ${fmt(T.sheet.derived.kv_pool.tokens_after)} tokens (estimated)` }) : null,
    T.outcome);
  tray.append(h('div', {}, msg, n ? reason : null, n ? h('label', { class: 'keep' }, persist, 'Keep after a restart') : null,
    h('div', { class: 'toolbar' },
      h('button', { type: 'button', class: 'btn', onclick: discard }, 'Discard'),
      n ? h('button', { type: 'button', class: 'btn check', disabled: T.busy, onclick: checkStaged }, 'Check') : null,
      n ? h('button', { type: 'button', class: 'btn go', disabled: T.busy || !T.sheet || !T.sheet.ok, onclick: applyStaged }, 'Apply') : null)));
}
function discard() { T.staged.clear(); T.sheet = null; T.outcome = null; T.reverts = null; drawTray(); }
function body() {
  const set = {};
  for (const { o, value } of T.staged.values()) set[o.name] = value === 'unset' ? null : value;
  return set;
}
async function checkStaged() {
  T.busy = true; T.outcome = null; drawTray();
  try {
    const r = await api('validate', '', { method: 'POST', body: JSON.stringify({ set: body() }) });
    T.sheet = r.ok ? r.body : null;
    if (!r.ok) T.outcome = h('p', { class: 'err', role: 'alert', text: errText(r) });
  } catch (e) { T.outcome = h('p', { class: 'err', role: 'alert', text: String(e.message) }); }
  T.busy = false; drawTray();
}
async function applyStaged() {
  T.busy = true; drawTray();
  const req = { set: body(), if_version: Number(S.config.config_version), reason: T.reason || undefined, persist: !!S.config.engine_cfg && T.persist !== false };
  if (T.reverts != null) req.reverts_version = T.reverts;
  try {
    const r = await api('set', '', { method: 'POST', body: JSON.stringify(req) });
    if (r.ok) {
      const pend = r.body.outcome === 'pending';
      T.outcome = h('p', { class: pend ? 'dim' : 'okc', role: 'status', text: (r.body.noop ? 'No change: the engine already stands there.' : `Version ${r.body.config_version}: ` + r.body.changes.map((c) => `${(S.byName.get(c.name) || {}).label || c.name} ${c.state.replace(/_/g, ' ')}`).join(', ')) + (pend ? ' (waiting for the engine to take it)' : '') });
      await reloadConfig();
    } else {
      const e = r.body && r.body.error || {};
      const lines = (e.details && e.details.refused || []).map((x) => h('li', { text: `${(S.byName.get(x.name) || {}).label || x.name}: ${x.message}` }));
      T.outcome = h('div', { class: 'err', role: 'alert' }, h('p', { text: errText(r) }), lines.length ? h('ul', {}, lines) : null,
        e.code === 'version_conflict' ? h('p', { text: 'The settings changed since this was drawn up: the page has read them again; check and apply once more.' }) : null);
      if (e.code === 'version_conflict') await reloadConfig();
      T.sheet = null;
    }
  } catch (e) { T.outcome = h('p', { class: 'err', role: 'alert', text: String(e.message) }); }
  T.busy = false; drawTray();
}
async function reloadConfig() {
  const c = await api('config');
  if (!c.ok) return;
  S.config = c.body;
  S.cfgById = new Map(S.config.values.map((v) => [v.id, v]));
  if (S.tab === 'settings') selectTab('settings', true);
  balanceRepaint();
}

// ---- history: one row per change, with the setting's label, from -> to, the key, the result, and an undo
// "Most CUDA graphs kept: 128 → 8", and how many more the same change touched
function recentLine(x) {
  const c = x.changes[0], o = S.byId.get(Number(c.id)) || S.byName.get(c.name);
  const more = x.changes.length - 1;
  return `${changeWhat(c)}: ${valueWord(o, c.from)} → ${valueWord(o, c.to)}` + (more > 0 ? ` and ${more} more` : '');
}
function changeWhat(c) { const o = S.byId.get(Number(c.id)) || S.byName.get(c.name); return (o && (o.label || o.name)) || c.name; }
function valueWord(o, v) {
  if (v == null || (v === '' && o && (o.kind === 'int' || o.kind === 'float'))) return o && o.default_text ? o.default_text : o && o.default ? valueWord(o, o.default) : 'Not set';
  if (o && o.kind === 'flag') return v === '1' ? 'On' : v === '0' ? 'Off' : v;
  return v === '' ? '(empty)' : o && o.unit ? `${v} ${o.unit}` : v;
}
function resultText(x) {
  if (x.outcome === 'refused') return { kind: 'refused', short: 'Refused', long: 'Refused' };
  if (x.endpoint === 'start') return { kind: 'restart', short: 'Edited in engine.cfg', long: 'Edited in engine.cfg while stopped' };
  const kept = (x.changes || []).some((c) => c.persisted !== false) ? ' · saved to engine.cfg' : '';
  if (x.apply === 'live') return { kind: 'live', short: 'Applied live', long: 'Applied live' + kept };
  if (x.apply === 'next_start') return { kind: 'restart', short: 'At the next start', long: 'At the next start' + kept };
  if (x.apply === 'mixed') return { kind: 'live', short: 'Live and at next start', long: 'Part live, part at the next start' + kept };
  return { kind: 'live', short: x.outcome || 'Recorded', long: (x.outcome || 'Recorded') + kept };
}
function historyPage() {
  const box = h('div', { class: 'card flush' }, h('p', { class: 'pad dim', text: 'Loading' }));
  let filter = 'all', data = null, undone = new Set();
  const seg = h('div', { class: 'seg', role: 'group', 'aria-label': 'Show' });
  const canUndo = writable();
  const undo = async (v) => {
    const p = await api('revert', '', { method: 'POST', body: JSON.stringify({ history_version: v }) });
    if (!p.ok) { T.outcome = h('p', { class: 'err', role: 'alert', text: errText(p) }); drawTray(); return; }
    T.staged.clear(); T.reverts = v; T.sheet = null; T.outcome = null;
    for (const [name, val] of Object.entries(p.body.plan.set)) { const o = S.byName.get(name); if (o) T.staged.set(name, { o, value: val == null ? 'unset' : String(val) }); }
    const skipped = (p.body.skipped || []).map((sk) => `${(S.byName.get(sk.name) || {}).label || sk.name}: ${sk.reason.replace(/_/g, ' ')}`);
    if (skipped.length) T.outcome = h('p', { class: 'dim', text: 'Skipped: ' + skipped.join('; ') });
    undone.add(v);
    drawTray();
    checkStaged();
    paint();
  };
  const paint = () => {
    clear(seg).append(...[['all', 'All'], ['live', 'Live'], ['restart', 'At restart'], ['refused', 'Refused']].map(([id, name]) => h('button', { type: 'button', 'aria-pressed': filter === id ? 'true' : 'false', onclick: () => { filter = id; paint(); } }, name)));
    if (!data) return;
    const flat = [];
    for (const x of data.rows || []) {
      const res = resultText(x);
      const cs = (x.changes || []).length ? x.changes : [];
      cs.forEach((c, i) => flat.push({ x, c, res, first: i === 0 }));
    }
    const shown = flat.filter((r) => filter === 'all' || r.res.kind === filter || (filter === 'live' && r.x.apply === 'mixed'));
    const head = ['When', 'Setting', 'Change', 'Key', 'Result', ''].map((c, i) => h('th', { scope: 'col', class: i === 5 ? 'r' : '' }, i === 5 ? h('span', { class: 'sr', text: 'Actions' }) : c));
    const trs = shown.map(({ x, c, res, first }) => {
      const o = S.byId.get(Number(c.id)) || S.byName.get(c.name);
      const ms = Number(x.at_ms);
      const word = undone.has(Number(x.version)) ? 'Undone' : res.short;
      return h('tr', {},
        h('td', { class: 'mono' }, first && x.at_ms ? [h('div', { class: 'num' }, when(ms)), h('div', { class: 'note', text: dayLabel(ms) })] : ''),
        h('td', {}, h('div', { text: changeWhat(c) }), h('div', { class: 'note', text: o ? o.tab : '' })),
        h('td', { class: 'num' }, h('span', { class: 'dim', text: valueWord(o, c.from) }), h('span', { class: 'faint', 'aria-hidden': 'true', text: '  →  ' }), h('span', { text: valueWord(o, c.to) })),
        h('td', { class: 'mono', text: first ? x.label || '' : '' }),
        h('td', {}, first ? [h('span', { class: 'state ' + (undone.has(Number(x.version)) ? 'idle' : res.kind === 'refused' ? 'bad' : res.kind === 'restart' ? 'warn' : '') }, h('i', { class: 'dotc' }), word), h('div', { class: 'note', text: [res.long.replace(res.short, '').replace(/^ · /, ''), x.reason].filter(Boolean).join(' · ') })] : ''),
        h('td', { class: 'r' }, first && canUndo && x.endpoint !== 'start' && x.outcome !== 'refused' && !undone.has(Number(x.version)) ? h('button', { type: 'button', class: 'btn sm', onclick: () => undo(Number(x.version)), 'aria-label': `Undo version ${x.version}` }, 'Undo') : null));
    });
    clear(box).append(h('div', { class: 'scroll' }, h('table', { class: 't' }, h('thead', {}, h('tr', {}, head)), h('tbody', {}, trs.length ? trs : h('tr', {}, h('td', { colspan: 6, class: 'dim', text: data.note || 'No change is recorded.' }))))),
      ...(Number(data.damaged) > 0 ? [h('p', { class: 'pad note', text: `${data.damaged} line(s) of the file are no row and were skipped.` })] : []));
  };
  refs.main.append(h('div', { class: 'head' }, h('div', { class: 'sname' }, h('h1', { text: 'History' }), h('p', { class: 'dim', text: 'Every settings change, who made it and how it took effect. Undo puts back the value from before that change.' })), seg), box,
    h('p', { class: 'note', text: 'Times are Sydney time; hover a time for UTC. The record is engine.history.jsonl beside engine.cfg.' }));
  paint();
  let alive = true;
  (async () => {
    const r = await api('history', 'limit=200');
    if (!alive) return;
    if (!r.ok) { clear(box).append(h('p', { class: 'pad err', text: errText(r) })); return; }
    data = r.body;
    if (data.memory_only) refs.main.append(h('p', { class: 'note', text: data.note }));
    paint();
  })();
  return () => { alive = false; };
}

// ---- requests and keys, and the engine log (admin keys only: the panel exists only where the routes do)
function activityPage() {
  const names = ROUTE_NAMES();
  const reqBody = h('tbody'), reqNote = h('span', { class: 'dim' });
  const th = (c, r) => h('th', { scope: 'col', class: r ? 'r' : '' }, c);
  const reqCard = h('section', { class: 'card flush', 'aria-labelledby': 'req-h' },
    h('div', { class: 'head pad' }, h('h2', { id: 'req-h', text: 'Recent requests' }), reqNote),
    h('div', { class: 'scroll' }, h('table', { class: 't' }, h('thead', {}, h('tr', {}, [th('Ended'), th('Key'), th('Endpoint'), th('Prompt', 1), th('From cache', 1), th('Reply', 1), th('First token', 1), th('Reply speed', 1), th('Ended by')])), reqBody)));
  const keysList = h('div'), keyNote = h('p', { class: 'note' }), reloadBtn = h('button', { type: 'button', class: 'btn sm' }, 'Reload keys');
  const keysCard = h('section', { class: 'card', 'aria-labelledby': 'keys-h' }, h('div', { class: 'head' }, h('h2', { id: 'keys-h', text: 'Keys' }), names.has('keys/reload') && S.scope === 'admin' ? reloadBtn : null), keysList, keyNote);
  let level = 'info';
  const logOut = h('div', { class: 'log', role: 'log', tabindex: '0', 'aria-label': 'Engine log' }), logNote = h('p', { class: 'note' });
  const seg = h('div', { class: 'seg sm', role: 'group', 'aria-label': 'Log level' });
  const logCard = h('section', { class: 'card', 'aria-labelledby': 'log-h' }, h('div', { class: 'head' }, h('h2', { id: 'log-h', text: 'Engine log' }), seg), logOut, logNote);
  const hasLogs = names.has('logs');
  refs.main.append(h('div', { class: 'sname' }, h('h1', { text: 'Requests & keys' }), h('p', { class: 'dim', text: hasLogs ? 'The last requests the engine finished, the keys that may call it, and the engine’s own log.' : 'The last requests the engine finished and the keys that may call it. The engine log is shown to admin keys only.' })),
    reqCard, h('div', { class: 'grid two' }, names.has('keys') ? keysCard : null, hasLogs ? logCard : null));
  let alive = true, since = 0;
  const paintSeg = () => clear(seg).append(...[['info', 'All'], ['warn', 'Warnings']].map(([id, name]) => h('button', { type: 'button', 'aria-pressed': level === id ? 'true' : 'false', onclick: () => { level = id; since = 0; clear(logOut); paintSeg(); pullLog(); } }, name)));
  paintSeg();
  const endWord = { stop: 'Finished', eos: 'Finished', length: 'Length limit', cancelled: 'Cancelled', canceled: 'Cancelled', error: 'Error', tool_calls: 'Tool call' };
  async function pullReq() {
    if (!alive || document.hidden) return;
    try {
      const r = await api('requests');
      if (!alive || !r.ok) return;
      const rec = r.body.recent.slice(0, 50);
      reqNote.textContent = `Last ${rec.length}, newest first` + (r.body.untracked ? ` · ${fmt(r.body.untracked)} were not tracked (the table was full)` : '');
      const cell = (x, cls) => h('td', { class: cls || 'num r', text: x });
      clear(reqBody).append(...(rec.length ? rec.map((x) => h('tr', { class: 'num' },
        h('td', { class: 'num' }, when(x.end_ms)), cell(x.key_label, 'mono'), cell(x.endpoint, 'mono dim'), cell(fmt(x.prompt_tokens)), cell(fmt(x.cached_tokens || 0), 'num r dim'), cell(fmt(x.generated_tokens)),
        cell(one.format(none(x.ttft_s) ? 0 : x.ttft_s) + ' s'), cell(fmt(x.decode_tps) + ' tok/s'), cell(endWord[x.finish] || String(x.finish).replace(/_/g, ' '), 'dim'))) : [h('tr', {}, h('td', { colspan: 9, class: 'dim', text: 'No request has finished yet.' }))]));
  const access = { admin: ['Full control', 'Reads everything and may change settings.'], agent: ['Limited changes', 'May change the settings its profile allows.'], read: ['Read only', 'Reads status; changes nothing.'] };
  async function pullKeys() {
    if (!alive || document.hidden || !names.has('keys') || S.scope !== 'admin') { if (S.scope !== 'admin') clear(keysList).append(h('p', { class: 'dim', text: 'Keys are listed to admin keys only.' })); return; }
    try {
      const k = await api('keys');
      if (!alive || !k.ok) return;
      const rows = [...k.body.admin_keys.map((x) => [x.label, ...(access[x.scope] || [x.scope, '']), 'This page and the admin API']), ...k.body.api_keys.map((x) => [x.label, 'Model calls', `${fmt(x.requests)} repl${Number(x.requests) === 1 ? 'y' : 'ies'} since the engine started.`, ''])];
      clear(keysList).append(...rows.map((r) => h('div', { class: 'keyrow' }, h('span', { class: 'mono', text: r[0] }), h('span', { class: 'chip', text: r[1] }), h('span', { class: 'full', text: r[2] }))));
      keyNote.textContent = 'Keys live in files on node1; the engine never shows a key, only its label.';
    } catch (e) { /* as above */ }
  }
  reloadBtn.addEventListener('click', async () => {
    reloadBtn.disabled = true;
    try { const r = await api('keys/reload', '', { method: 'POST', body: '{}' }); reloadBtn.textContent = r.ok ? 'Reloaded' : 'Reload failed'; if (!r.ok) keyNote.textContent = errText(r); await pullKeys(); } catch (e) { reloadBtn.textContent = 'Reload failed'; }
    reloadBtn.disabled = false;
  });
  async function pullLog() {
    if (!alive || !hasLogs) return;
    if (!document.hidden) {
      try {
        const r = await api('logs', `since=${since}&limit=300&level=${level}`);
        if (!alive) return;
        if (!r.ok) logNote.textContent = errText(r);
        else {
          const b = r.body;
          if (b.dropped) logOut.append(h('div', { class: 'dim' }, h('span'), h('span'), `${fmt(b.dropped)} lines were overwritten before this page read them`));
          for (const l of b.lines) logOut.append(h('div', { title: 'seq ' + l.seq }, h('span', { class: 't', title: utc(Number(l.at_ms)) + ' UTC', text: whenText(Number(l.at_ms)) }), h('span', { class: l.level === 'info' ? 'faint' : 'lv-' + l.level, text: l.level === 'warn' ? 'warning' : l.level }), h('span', { text: l.text })));
          while (logOut.childElementCount > 600) logOut.firstChild.remove();
          since = b.next_seq - 1;
          logOut.scrollTop = logOut.scrollHeight;
          logNote.textContent = 'The newest lines the engine kept; each is cut at 512 bytes.';
        }
      } catch (e) { logNote.textContent = 'The engine is not answering.'; }
    }
  }
  pullReq(); pullKeys(); pullLog();
  const t1 = setInterval(pullReq, 2000), t2 = setInterval(pullKeys, 10000), t3 = setInterval(pullLog, 2000);
  return () => { alive = false; clearInterval(t1); clearInterval(t2); clearInterval(t3); };
}

// ---- start: the key this tab kept (a refresh), else the page asks for it
{
  const kept = keptKey();
  loginScreen('');
  if (kept) signIn(kept, 1);
}
