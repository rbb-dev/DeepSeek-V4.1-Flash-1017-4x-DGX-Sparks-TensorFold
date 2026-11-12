// Renders the admin page in headless Chrome against fixture API answers (fixture.mjs), under the engine's own Content-Security-Policy,
// drives it, and checks what a source scan cannot: that a poll updates boxes in place, that an open section, a focused control and
// every box survive a run of updates, that every option of the schema is reachable in the Settings tab, and that nothing in the
// Overview is clipped, wrapped, or shows a literal entity. Not a build: it needs node and google-chrome, no engine.
// usage: node check.mjs [--shots <dir>] [--widths 1600,1280] [--polls 6]
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { answer, KEY, VISIBLE } from './fixture.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const UI = join(here, '../../src/server/admin_ui');
const arg = (n, d) => { const i = process.argv.indexOf(n); return i > 0 ? process.argv[i + 1] : d; };
const SHOTS = arg('--shots', null), WIDTHS = arg('--widths', '1600,1280').split(',').map(Number), POLLS = Number(arg('--polls', 6));
const CSP = "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const state = { silent: null, apiCalls: 0 };
const server = createServer((req, res) => {
  const u = new URL(req.url, 'http://x');
  const file = { '/admin/ui/': ['index.html', 'text/html;charset=utf-8'], '/admin/ui/app.js': ['app.js', 'text/javascript;charset=utf-8'], '/admin/ui/app.css': ['app.css', 'text/css;charset=utf-8'] }[u.pathname];
  if (file) { res.writeHead(200, { 'Content-Type': file[1], 'Content-Security-Policy': CSP, 'Cache-Control': 'no-store' }); res.end(readFileSync(join(UI, file[0]))); return; }
  const m = /^\/admin\/api\/v1\/(.+)$/.exec(u.pathname);
  if (m) {
    if (req.headers.authorization !== 'Bearer ' + KEY) { res.writeHead(404, { 'Content-Type': 'application/json' }); res.end('{"error":{"code":"not_found","message":"unknown path"}}'); return; }
    state.apiCalls++;
    const body = answer(m[1], u.search, Date.now(), { silent: state.silent });
    res.writeHead(body ? 200 : 404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(body || { error: { code: 'not_found', message: 'unknown path' } }));
    return;
  }
  res.writeHead(404); res.end();
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const PAGE = `http://127.0.0.1:${server.address().port}/admin/ui/`;

const PROF = join(tmpdir(), 'admin-ui-check-prof-' + process.pid);
mkdirSync(PROF, { recursive: true });
const PORT = 9400 + (process.pid % 500);
const chrome = spawn('taskset', ['-c', '0-1', 'nice', '-n', '19', 'google-chrome', '--headless=new', '--disable-gpu', '--no-first-run', `--user-data-dir=${PROF}`, `--remote-debugging-port=${PORT}`, '--hide-scrollbars', 'about:blank'], { stdio: 'ignore', detached: true });
let code = 0;
const finish = () => { try { process.kill(-chrome.pid, 'SIGTERM'); } catch { /* gone */ } server.close(); setTimeout(() => process.exit(code), 300); };
setTimeout(() => { console.log('TIMEOUT'); code = 2; finish(); }, 120000);

let ver;
for (let i = 0; i < 60 && !ver; i++) { await sleep(200); try { ver = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json(); } catch { /* not up yet */ } }
const target = ver && ver.find((t) => t.type === 'page');
if (!target) { console.log('NO PAGE TARGET'); code = 2; finish(); await sleep(5000); }
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r));
let id = 0; const pending = new Map(); const errors = [];
ws.addEventListener('message', (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); return; }
  if (m.method === 'Runtime.exceptionThrown') errors.push('EXCEPTION ' + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text).slice(0, 300));
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') errors.push('CONSOLE.ERROR ' + JSON.stringify(m.params.args.map((a) => a.value ?? a.description)).slice(0, 300));
  if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') errors.push('LOG ' + m.params.entry.text.slice(0, 300));
});
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (expression) => { const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true }); if (r.result?.exceptionDetails) throw new Error('eval: ' + JSON.stringify(r.result.exceptionDetails).slice(0, 300)); return r.result?.result?.value; };
await send('Runtime.enable'); await send('Page.enable'); await send('Log.enable');
const keep = JSON.stringify({ k: KEY, exp: Date.now() + 3600e3 });
await send('Page.addScriptToEvaluateOnNewDocument', { source: `try { sessionStorage.setItem('tensorfold-admin-key', ${JSON.stringify(keep)}); } catch (e) {}` });

let failed = 0;
const check = (name, ok, detail) => { console.log((ok ? 'PASS ' : 'FAIL ') + name + (detail ? '  [' + detail + ']' : '')); if (!ok) failed++; };
async function shoot(name) {
  if (!SHOTS) return;
  mkdirSync(SHOTS, { recursive: true });
  const h = await ev('Math.ceil(document.documentElement.scrollHeight)');
  const w = await ev('window.innerWidth');
  await send('Emulation.setDeviceMetricsOverride', { width: w, height: Math.min(4000, h), deviceScaleFactor: 1, mobile: false });
  const png = await send('Page.captureScreenshot', { format: 'png' });
  writeFileSync(join(SHOTS, name + '.png'), Buffer.from(png.result.data, 'base64'));
  console.log('SHOT ' + join(SHOTS, name + '.png'));
}
const viewport = (w, h = 1000) => send('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: 1, mobile: false });

// text clipped, wrapped, past its box, or showing a literal entity, in the Overview's boxes (the board's render_check rules)
const LINT = `(() => {
  const out = [], root = document.getElementById('view'); let n = 0;
  for (const el of root.querySelectorAll('span,div,button,b,i,summary,label')) {
    const own = [...el.childNodes].some((x) => x.nodeType === 3 && x.nodeValue.trim()); if (!own) continue;
    const cs = getComputedStyle(el); if (cs.display === 'none' || cs.visibility === 'hidden') continue;
    const r = el.getBoundingClientRect(); if (!r.width) continue; n++;
    const txt = el.textContent.trim().replace(/\\s+/g, ' ').slice(0, 70);
    if (cs.display !== 'inline' && el.scrollWidth > el.clientWidth + 1) out.push('CLIP ' + el.scrollWidth + '>' + el.clientWidth + ' | ' + txt);
    const lh = parseFloat(cs.lineHeight) || parseFloat(cs.fontSize) * 1.3;
    if (el.getClientRects().length > 1 || (cs.display !== 'inline' && el.children.length === 0 && r.height > lh * 1.6)) out.push('WRAP h=' + r.height.toFixed(0) + ' | ' + txt);
    const card = el.closest('.tile,.ncard,.htile,.box'); if (card) { const q = card.getBoundingClientRect(); if (r.right > q.right + 1) out.push('ESCAPES ' + r.right.toFixed(0) + '>' + q.right.toFixed(0) + ' | ' + txt); }
    if (/&#39;|&amp;|&quot;|&lt;|\\{\\{|undefined|NaN|\\bnull\\b|\\[object/.test(el.textContent)) out.push('LITERAL | ' + txt);
  }
  for (const el of root.querySelectorAll('[title]')) { const t = el.getAttribute('title'); if (!t) out.push('EMPTY-TITLE'); else if (/&#39;|&amp;|undefined|NaN|\\[object/.test(t)) out.push('LITERAL-TITLE | ' + t.slice(0, 70)); }
  if (document.documentElement.scrollWidth > window.innerWidth + 1) out.push('PAGE SCROLLS SIDEWAYS ' + document.documentElement.scrollWidth + '>' + window.innerWidth);
  return { checked: n, issues: [...new Set(out)] };
})()`;

await viewport(WIDTHS[0]);
await send('Page.navigate', { url: PAGE });
await sleep(3500);

// ---- 1. the Overview draws, with the board's boxes, and no exception
const boxes = await ev(`[...document.querySelectorAll('[data-box]')].map((e) => e.dataset.box)`);
check('overview draws without an exception or a console error', errors.length === 0, errors.slice(0, 3).join(' | '));
for (const b of ['lines', 'kv', 'tier', 'rate', 'decode', 'prefill', 'health', 'speed']) check('box ' + b + ' exists', boxes.includes(b));
for (const b of ['requests-table', 'memory-table']) check('the old ' + b + ' is gone', !boxes.includes(b));
const nodeCards = await ev(`document.querySelectorAll('[data-box="health"] [data-node]').length`);
check('node health holds one card per Spark', nodeCards === 4, String(nodeCards));
const lineRows = await ev(`document.querySelectorAll('[data-box="lines"] [data-line]').length`);
check('every seat is a line, always visible', lineRows === 8, String(lineRows));
const noProse = await ev(`(() => { const t = document.getElementById('view').innerText; return ['Turn Turbo on', 'Prompt vs generation balance', 'Updated just now', 'Memory on each Spark', 'Speed features'].filter((s) => t.includes(s)); })()`);
check('none of the rejected prose sentences is on the page', noProse.length === 0, noProse.join(', '));
const tempColors = await ev(`[...document.querySelectorAll('[data-box="health"] [data-temp]')].map((e) => e.dataset.temp + ':' + getComputedStyle(e).color)`);
check('temperatures are colour coded', tempColors.length >= 12 && new Set(tempColors.map((s) => s.split(':')[1])).size > 3, tempColors.slice(0, 4).join(' '));
const tips = await ev(`(() => { const rows = [...document.querySelectorAll('[data-box="health"] [data-node]:first-child [data-row]')]; return [rows.length, rows.filter((r) => !(r.getAttribute('title') || '').trim()).length]; })()`);
check('every node health row has a tip', tips[0] >= 12 && tips[1] === 0, tips.join('/'));
const ages = await ev(`document.getElementById('view').innerText.match(/no report|Updated/g) || []`);
check('no card shows an age while its Spark reports', ages.length === 0, ages.join(','));
const turbo = await ev(`(() => { const t = document.querySelector('[data-tile="turbo"]'); return t ? t.innerText.replace(/\\s+/g, ' ') : null; })()`);
check('turbo is a compact tile with a short qualifier', !!turbo && /Turbo/i.test(turbo) && turbo.length < 60, turbo);
for (const w of WIDTHS) {
  await viewport(w); await sleep(700);
  const lint = await ev(LINT);
  check(`overview layout at ${w} px: nothing clipped, wrapped or literal (${lint.checked} texts)`, lint.issues.length === 0, lint.issues.slice(0, 6).join(' || '));
  await shoot(`overview-${w}`);
}
await viewport(WIDTHS[0]);

// ---- 2. updates are in place: boxes, an open section, a focused control and a dragged slider survive a run of polls
await ev(`(() => { const d = document.querySelector('[data-box="speed"] details'); if (d) d.open = true; })()`);
await ev(`(() => { const s = document.querySelector('[data-tile="balance"] input'); if (s) s.focus(); window.__ids = [...document.querySelectorAll('[data-box],[data-node],[data-line]')]; window.__removed = 0;
  new MutationObserver((ms) => { for (const m of ms) for (const n of m.removedNodes) if (n.nodeType === 1 && (n.dataset.box || n.dataset.node || n.dataset.line)) window.__removed++; }).observe(document.getElementById('view'), { childList: true, subtree: true });
  window.__before = document.getElementById('view').innerText; })()`);
// a number the engine did not report reads 0 in its unit, never a dash, at ANY moment of the polls (the rails flipped between a rate and a dash when a poll brought no new report)
await ev(`(() => { window.__dashes = []; const re = /^\\s*[-\\u2013\\u2014]\\s*$|(^|\\s)[-\\u2013\\u2014] (in|out)\\b/;
  const look = (el) => { if (el && el.nodeType === 1 && el.classList.contains('num') && re.test(el.textContent)) window.__dashes.push(el.textContent.trim()); };
  document.querySelectorAll('#view .num').forEach(look);
  new MutationObserver((ms) => { for (const m of ms) look(m.target.nodeType === 3 ? m.target.parentElement : m.target); }).observe(document.getElementById('view'), { childList: true, subtree: true, characterData: true }); })()`);
const callsBefore = state.apiCalls;
await sleep(POLLS * 1000 + 600);
const dashes = await ev(`window.__dashes`);
check('no number shows a dash at any poll: a missing value reads 0 in its unit', dashes.length === 0, `${dashes.length} seen: ${[...new Set(dashes)].slice(0, 6).join(' | ')}`);
const after = await ev(`({ open: !!(document.querySelector('[data-box="speed"] details') || {}).open, same: [...document.querySelectorAll('[data-box],[data-node],[data-line]')].length === window.__ids.length && window.__ids.every((e) => e.isConnected),
  removed: window.__removed, focus: !!document.activeElement && document.activeElement.type === 'range', changed: document.getElementById('view').innerText !== window.__before })`);
check(`the poll ran ${POLLS} times`, state.apiCalls - callsBefore >= POLLS * 2, String(state.apiCalls - callsBefore));
check('the values changed (the page is live)', after.changed);
check('an open Speed features section stays open across updates', after.open);
check('no box, card or line is replaced by an update', after.same && after.removed === 0, `removed ${after.removed}`);
check('a focused control keeps its focus across updates', after.focus);
// a Spark that stops reporting shows its age in red, and only then
state.silent = 2; await sleep(2600);
const silent = await ev(`(() => { const c = document.querySelector('[data-node="2"] [data-age]'); return c ? [c.textContent, getComputedStyle(c).color, c.hidden] : null; })()`);
check('a Spark silent for over 6 s shows its age in red', !!silent && !silent[2] && /\d/.test(silent[0]), JSON.stringify(silent));
state.silent = null;

// ---- 3. the Settings tab reaches every option the API serves
await ev(`document.querySelector('nav button[data-tab="settings"]').click()`); await sleep(500);
const tabs = await ev(`[...document.querySelectorAll('nav.groups button')].map((b) => b.firstChild.textContent)`);
const seen = new Map();
for (const t of tabs) {
  await ev(`[...document.querySelectorAll('nav.groups button')].find((b) => b.firstChild.textContent === ${JSON.stringify(t)}).click()`); await sleep(150);
  for (const r of await ev(`[...document.querySelectorAll('.srow')].map((r) => r.dataset.id)`)) seen.set(r, t);
}
check(`every option the API serves is in the Settings tab (${VISIBLE.length})`, seen.size === VISIBLE.length, String(seen.size));
for (const n of ['tf_ds_split64n', 'tf_ds_kv_vmm', 'tf_ds_kv_free_target_gib']) { const o = VISIBLE.find((x) => x.name === n); check(`${n} shows in Settings`, !!o && seen.has(String(o.id))); }
const rowText = await ev(`(() => { const r = document.querySelector('.srow'); return r ? [r.innerText.replace(/\\s+/g, ' ').slice(0, 160), !!r.querySelector('[title]')] : null; })()`);
check('a settings row shows label, value, default and apply mode, with the doc on hover', !!rowText && rowText[1], JSON.stringify(rowText));
await ev(`[...document.querySelectorAll('nav.groups button')].find((b) => b.firstChild.textContent === 'Goals').click()`); await sleep(900);
const goals = await ev(`(() => { const g = document.querySelector('[data-goals]'); return g ? g.innerText.replace(/\\s+/g, ' ') : null; })()`);
check('Goals is one compact row in the Settings tab, with no chart while no goal is set', !!goals && !(await ev(`!!document.querySelector('[data-goals] svg')`)), goals);
await shoot('settings-' + WIDTHS[0]);

// ---- 4. the other tabs still draw
for (const t of ['history', 'activity']) {
  await ev(`document.querySelector('nav button[data-tab="${t}"]').click()`); await sleep(700);
  check(`tab ${t} draws`, (await ev(`document.getElementById('view').innerText.length`)) > 40);
}
check('no exception or console error anywhere (including the CSP)', errors.length === 0, errors.slice(0, 3).join(' | '));
console.log(failed ? `\n${failed} FAILED` : '\nALL PASSED');
code = failed ? 1 : 0;
finish();
