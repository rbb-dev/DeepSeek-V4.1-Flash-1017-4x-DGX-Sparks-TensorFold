function writable() { const c = S.status && S.status.capabilities; return !!(c && c.writable_classes && c.writable_classes.length) && ROUTE_NAMES().has('set'); }
function unstage(name) { T.staged.delete(name); T.sheet = null; if (!T.staged.size) T.reverts = null; drawTray(); }
  } else if (o.kind === 'enum') {
  } else {
    if (o.min != null) input.min = o.min;
    if (o.max != null) input.max = o.max;
}
function drawTray() {
  const tray = refs.tray;
  if (!tray) return;
  tray.hidden = !T.staged.size && !T.outcome;
  clear(tray);
  persist.addEventListener('change', () => { T.persist = persist.checked; });
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
  if (T.reverts != null) req.reverts_version = T.reverts;
  try {
    const r = await api('set', '', { method: 'POST', body: JSON.stringify(req) });
    if (r.ok) {
      const pend = r.body.outcome === 'pending';
      await reloadConfig();
    } else {
      const e = r.body && r.body.error || {};
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
}
  const canUndo = writable();
  const undo = async (v) => {
    const p = await api('revert', '', { method: 'POST', body: JSON.stringify({ history_version: v }) });
    T.staged.clear(); T.reverts = v; T.sheet = null; T.outcome = null;
    drawTray();
    checkStaged();
  };
  (async () => {
  })();
}
