// The admin API's shapes as fixtures, served from memory so the admin page can be rendered and driven in a browser without an
// engine. Values move with the clock (tokens grow, temperatures drift), so a page that redraws itself on every poll shows it.
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
export const SCHEMA = JSON.parse(readFileSync(join(here, '../../config-schema.json'), 'utf8'));
export const VISIBLE = SCHEMA.options.filter((o) => !o.hidden);
export const KEY = 'fixture-admin-key';
const T0 = Date.now();
const GIB = 2 ** 30;

export function schemaBody() {
  const apply_classes = {};
  for (const [k, text] of [['P', 'per request'], ['H', 'live: rank 0'], ['A', 'live on all four ranks'], ['I', 'when idle'], ['N', 'now'], ['R', 'at a restart']]) apply_classes[k] = { text, now_ok: k !== 'R', apply_time: { value_s: null, label: 'unmeasured' } };
  const options = VISIBLE.map((o) => ({ ...o, who: o.secret ? [] : ['admin'], values: o.values || [] }));
  return { schema_version: SCHEMA.schema_version, count: options.length, apply_classes, tabs: SCHEMA.tabs, options, retired: [] };
}
export function configBody() {
  return {
    schema_version: SCHEMA.schema_version, config_version: 7, active_version: 7, bootstrapped: true, engine_cfg: { path: '/etc/engine.cfg' },
    values: VISIBLE.map((o) => ({ id: o.id, name: o.name, env: o.env, source: 'default', value: null, default: o.default, default_text: o.default_text, other: null, requested: null, state: 'in_effect', set_by: null, since_version: null })),
  };
}
const route = (n) => ({ path: '/admin/api/v1/' + n, methods: ['GET'] });
export const routesBody = () => ({ routes: ['status', 'schema', 'config', 'series', 'requests', 'ledger', 'history', 'goals', 'keys', 'logs', 'routes', 'validate', 'set', 'revert'].map(route) });

const FEATURES = [
  ['Split matrix multiplies', 'TF_DS_GEMM_HALVES', false, true, 'set to 0', 'a long prompt is a little slower'],
  ['Prompt pass rows', 'TF_DS_PASS_ROWS', true, true, '4096 rows', ''],
  ['CUDA graphs', 'TF_DS_GRAPHS', true, true, 'captured', ''],
  ['Proxy nap', 'TF_DS_RDMA_WFE', true, true, 'on', ''],
  ['KV pool growth', 'TF_DS_KV_VMM', true, true, 'on', ''],
  ['Engram read ahead', 'TF_DS_ENGRAM_AHEAD_PASS', false, false, 'no cache room', 'the next layer waits for its tables'],
];
export function statusBody(now = Date.now(), opts = {}) {
  const s = (now - T0) / 1000;
  const wob = (k, a) => Math.sin(s / 7 + k) * a;
  const gen = (id) => Math.floor(40 + s * (18 + id));
  // a worker reports every 2 s (the engine's peer report period), so a poll between two reports serves the same report again,
  // older, with the same counters; rank 0 samples itself on every poll
  const repAt = (r) => (r === 0 ? now : T0 + Math.floor((now - T0) / 2000) * 2000);
  const rank = (r) => ({
    rank: r, age_ms: opts.silent === r ? 12000 : now - repAt(r) + 300 + 100 * r, mem_available_bytes: Math.floor((14 + r + wob(r, 0.5)) * GIB), mem_total_bytes: Math.floor(121.7 * GIB),
    load1: 4.2 + wob(r, 1), cpu_busy_pct: 31 + wob(r, 5), cpus: 20,
    zones_c: [48 + wob(r, 2) + r * 6, 52 + r * 9, 44 + wob(r, 3) + r * 12, 51, 49, 47, 50, 46].map((v) => Math.round(v * 10) / 10),
    gpu: { temp_c: 52 + r * 11 + Math.round(wob(r, 3)), util_pct: 71 + Math.round(wob(r, 10)), clock_mhz: 2200, clock_max_mhz: 2411, power_w: 61.4 + wob(r, 4), reasons: r === 3 ? ['sw_power_cap'] : [], idle: false },
    nvme: { temp_c: 38 + r * 13, wear_pct: r },
    rails: ['A', 'B'].map((n, i) => { const sr = (repAt(r) - T0) / 1000; return { rail: n, rx_bytes: Math.floor(sr * (1.1e9 + i * 3e8) * (1 + r / 10)), tx_bytes: Math.floor(sr * (1.0e9 + i * 2e8)), errors: 0, discards: 0 }; }),
  });
  return {
    admin_api: 'v1', api: { major: 1, minor: 4 }, at_ms: now, at: new Date(now).toISOString(), read_only: false,
    capabilities: { scope: 'admin', label: 'admin', writable_classes: ['H', 'A', 'R'], modes: ['live', 'next_start'] },
    idle: { idle: false, idle_for_s: null, basis: 'server' },
    health: {
      status: 'ok', model: 'model', max_batch_size: 8, warming: false, uptime_s: 7200 + s,
      live: { connections: 6, waiting: 1, decode_tokens_per_second: Math.round((212 + wob(1, 20)) * 10) / 10, prefill_tokens_per_second: Math.round(3100 + wob(2, 200)), prompt_pass: { computed: Math.floor(40000 + (s * 3000) % 60000), total: 131072, resumed: 0, seconds: 12.5 }, prompt_tokens_computed_total: 9000000 },
      prompt_cache: { restores_begun: 14, restored_tokens: 1450000, restore_waits: 14, restore_wait_ns: 14 * 780e6, disk_hits: 15, spill: [{ rank: 0, points: 203, stored_bytes: Math.floor(41 * GIB), cap_bytes: 300 * GIB, writes_committed: 590, writes_pending: 1, written_bytes: Math.floor(96 * GIB), reads: 31 }] },
      spill: { state: 'on' },
    },
    config: { config_version: 7, active_version: 7, schema_version: SCHEMA.schema_version, ranks: [{ rank: 0, version: 7 }], pending: [] },
    kv_pool: { mode: 'free_target', tokens_active: 18022400, tokens_requested: 18022400, usable_window: 1048526, streams_x_context: 8388208, extra_tokens: 9634192, free_target_gib: '12.0', growth_gib: '2.8', bytes_per_token: 890, last_resize: { version: 3, verdict: 'grow', from_tokens: 16777216, fit_tokens: 18022400, rank: 0 } },
    memory: { basis: 'at the start', ranks: [0, 1, 2, 3].map((r) => ({ rank: r, device_bytes: Math.floor(121.7 * GIB), free_at_open_gib: '100.0', free_target_gib: '12.0', ledger_gib: '90.0' })) },
    speed_features: FEATURES.map(([name, env, on, asked, reason, cost]) => ({ name, env, on, asked, reason, cost })),
    release: 'T15', host_memory: { basis: 'now', ranks: [] },
    node_health: { basis: 'as each Spark last reported it', version: 1, zone_names: ['acpitz', 'ch0', 'ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6'], ranks: [0, 1, 2, 3].map(rank) },
    turbo: { source: 'built in', measured: true, build: 'the four Sparks', what: 'a prompt pass of 8,192 rows with split matrix multiplies', long_speedup: 1.11, short_speedup: 1.056, beside_prompt_ratio: 0.965, beside_streams_ratio: 1.18, pool_normal_positions: 18350080, pool_turbo_positions: 10420224, pool_cost_positions: 7929856, pool_cost_pct: 43.2, replies_equal: true, source_run: 'a cold ~152K-token prompt of real text, 7,468 against 6,729 tokens a second' },
    model_name: 'DeepSeek V4.1 Flash 1017',
    transport: { prompt_exchanges: { known: true, waits_in_card: true, arrival_signal: 'generation word' }, generation_exchanges: { known: true, on: true, waits_in_card: false, arrival_signal: 'generation word' }, link_watch: { links: 3 } },
    goals: { state: 'off' }, settings: { schema_version: SCHEMA.schema_version, options: VISIBLE.length, from_environment: 0, from_engine_cfg: 0, engine_cfg: null }, admin: {},
  };
}
export function requestsBody(now = Date.now()) {
  const s = (now - T0) / 1000;
  const live = [0, 1, 2, 3, 4].map((i) => ({
    id: 100 + i, slot: i, kv_tokens: 5000 + 90000 * i + Math.floor(s * 20), key_label: ['gateway', 'tools', 'gateway', 'board', 'tools'][i], endpoint: 'chat', priority: 'foreground', stream: true,
    age_s: 30 + 41 * (5 - i) + (s % 1000), prompt_tokens: 4000 + 90000 * i, generated_tokens: i === 4 ? 0 : Math.floor(40 + s * (18 + i)), state: i === 4 ? 'submitted' : 'decode',
  }));
  live.push({ id: 110, slot: 5, kv_tokens: 131072, key_label: 'tools', endpoint: 'messages', priority: 'foreground', stream: true, age_s: 12, prompt_tokens: 131072, generated_tokens: 0, state: 'submitted' });
  const recent = Array.from({ length: 30 }, (_, i) => ({ id: 90 - i, key_label: 'gateway', endpoint: 'chat', prompt_tokens: 3000 + i * 700, cached_tokens: i % 3 ? 2000 : 0, generated_tokens: 300 + i, ttft_s: 0.4 + i * 0.35, decode_tps: 41 + (i % 5), total_s: 12 + i, end_ms: now - i * 20000, finish: 'stop' }));
  return { in_flight: live, recent, untracked: 0 };
}
function series(name, now, n = 90, step = 10) {
  const base = name === 'prefill_tps' ? 3000 : 210;
  const points = Array.from({ length: n }, (_, i) => (i < 6 ? null : Math.round(base * (0.6 + 0.4 * Math.abs(Math.sin(i / 9)) + (i % 7) / 40))));
  return { name, step_s: step, points, peaks: points.map((p) => (p == null ? null : Math.round(p * 1.15))), last_ms: now };
}
export function seriesBody(q, now = Date.now()) {
  if (q.get('name') === 'seat_kv') return { step_s: 10, last_ms: now, seats: Array.from({ length: 16 }, (_, k) => ({ points: Array.from({ length: 90 }, (_, i) => (k < 5 ? 20000 * (k + 1) * (0.5 + 0.5 * Math.sin(i / 11 + k)) : 0)) })) };
  return series(q.get('name'), now);
}
export const ledgerBody = () => ({
  ranks: [0, 1, 2, 3].map((r) => ({ rank: r, device_bytes: Math.floor(121.7 * GIB), free_at_ready: Math.floor(14 * GIB) })),
  entries: [0, 1, 2, 3].flatMap((r) => [{ rank: r, feature_id: 'weights', device_bytes: Math.floor(68 * GIB) }, { rank: r, feature_id: 'kv_pool', device_bytes: Math.floor(12 * GIB) }, { rank: r, feature_id: 'work', device_bytes: Math.floor(11 * GIB) }]),
});
export const goalsBody = () => ({ state: 'idle', mode: 'decode_first', reachability: 'No goal is set.', measured: { decode_user_tps: 41.2, prefill_tps: 3100, judged_s: 60 }, achievable: { decode_user_max: 60, prefill_max: 3400 }, goals: [], met_fraction: null, n: null, n_wanted: null, n_cap: 16 });

export function answer(path, search, now = Date.now(), opts = {}) {
  const q = new URLSearchParams(search);
  switch (path) {
    case 'routes': return routesBody();
    case 'schema': return schemaBody();
    case 'config': return configBody();
    case 'status': return statusBody(now, opts);
    case 'requests': return requestsBody(now);
    case 'series': return seriesBody(q, now);
    case 'ledger': return ledgerBody();
    case 'goals': return goalsBody();
    case 'history': return { rows: [], note: 'No setting has been changed yet.' };
    case 'keys': return { admin_keys: [{ label: 'admin', scope: 'admin' }], api_keys: [{ label: 'gateway', requests: 120 }] };
    case 'logs': return { lines: [], next_seq: 1, dropped: 0 };
    default: return null;
  }
}
