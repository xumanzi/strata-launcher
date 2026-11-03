const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function api() {
  const source = fs.readFileSync('strata_monitor_extension.js', 'utf8');
  const context = {globalThis: null, document: null};
  context.globalThis = context;
  vm.runInNewContext(source, context);
  return context.StrataMonitorUsage;
}

test('context thresholds use normal warning and danger states', () => {
  const usage = api();
  assert.equal(usage.usageLevel(69.9), 'normal');
  assert.equal(usage.usageLevel(70), 'warning');
  assert.equal(usage.usageLevel(90), 'danger');
});

test('key resolution prefers the monitor session key and falls back to the chat store', () => {
  const source = fs.readFileSync('strata_monitor_extension.js', 'utf8');
  const load = (sessionStorage, localStorage) => {
    const context = {globalThis: null, document: null, sessionStorage, localStorage};
    context.globalThis = context;
    vm.runInNewContext(source, context);
    return context.StrataMonitorUsage;
  };
  const sessionFirst = load({getItem: () => 'sess-key'}, {getItem: () => JSON.stringify('chat-key')});
  assert.equal(sessionFirst.resolveKey(), 'sess-key');
  const chatFallback = load({getItem: () => null}, {getItem: () => JSON.stringify('chat-key')});
  assert.equal(chatFallback.resolveKey(), 'chat-key');
  const rawFallback = load({getItem: () => null}, {getItem: () => 'raw-key'});
  assert.equal(rawFallback.resolveKey(), 'raw-key');
  const none = load({getItem: () => null}, {getItem: () => null});
  assert.equal(none.resolveKey(), '');
});

test('chat page targets the Monitor tab while the standalone page targets its panel', () => {
  const source = fs.readFileSync('strata_monitor_extension.js', 'utf8');
  const chatView = {querySelector: () => null, firstChild: null};
  const chatDocument = {getElementById: (id) => (id === 'view-monitor' ? chatView : null), querySelector: () => null};
  const chatContext = {globalThis: null, document: Object.assign(chatDocument, {readyState: "loading", addEventListener: () => {}}), sessionStorage: {getItem: () => null}, localStorage: {getItem: () => null}};
  chatContext.globalThis = chatContext;
  vm.runInNewContext(source, chatContext);
  const chatTarget = chatContext.StrataMonitorUsage.findTarget();
  assert.equal(chatTarget.tabbed, true);
  assert.equal(chatTarget.view, chatView);
  assert.equal(chatTarget.host, chatView);
  const inner = {firstChild: null};
  const withInner = {querySelector: (sel) => (sel === '.monitor' ? inner : null), firstChild: null};
  const nestedDocument = {getElementById: (id) => (id === 'view-monitor' ? withInner : null), querySelector: () => null};
  const nestedContext = {globalThis: null, document: Object.assign(nestedDocument, {readyState: "loading", addEventListener: () => {}}), sessionStorage: {getItem: () => null}, localStorage: {getItem: () => null}};
  nestedContext.globalThis = nestedContext;
  vm.runInNewContext(source, nestedContext);
  assert.equal(nestedContext.StrataMonitorUsage.findTarget().host, inner);
  const panelHost = {parentNode: {insertBefore: () => {}}};
  const standaloneDocument = {getElementById: () => null, querySelector: (sel) => (sel === '.panel' ? panelHost : null)};
  const standaloneContext = {globalThis: null, document: Object.assign(standaloneDocument, {readyState: "loading", addEventListener: () => {}}), sessionStorage: {getItem: () => null}, localStorage: {getItem: () => null}};
  standaloneContext.globalThis = standaloneContext;
  vm.runInNewContext(source, standaloneContext);
  const standalone = standaloneContext.StrataMonitorUsage.findTarget();
  assert.equal(standalone.tabbed, false);
  assert.equal(standalone.host, panelHost);
});

test('snapshot view contains context request totals and compression savings', () => {
  const view = api().viewModel({
    context: {used_tokens: 15000, limit_tokens: 32768, percent: 45.8},
    last_request: {input_tokens: 2100, output_tokens: 834, total_tokens: 2934, precise: true},
    totals: {input_tokens: 86420, output_tokens: 21308, total_tokens: 107728},
    requests: {total: 37, successful: 35, failed: 1, active: 1},
    compression: {count: 3, cache_hits: 1, saved_tokens: 41672,
      last: {before_tokens: 20000, after_tokens: 9000, saved_tokens: 11000}},
    updated_at: 1,
  });
  assert.equal(view.context, '15K / 32.77K');
  assert.equal(view.last, '2.1K / 834 / 2.93K');
  assert.equal(view.total, '107.73K');
  assert.equal(view.saved, '41.67K');
  assert.equal(view.requests, '37 次 · 成功 35 · 失败 1 · 处理中 1');
  assert.equal(view.level, 'normal');
});

test('token units handle thresholds, rounding and invalid values', () => {
  const format = api().tokenNumber;
  for (const [value, expected] of [[0, '0'], [999, '999'], [1000, '1K'],
    [12345, '12.35K'], [999999, '1M'], [1250000, '1.25M'],
    [1000000000, '1B'], [1000000000000, '1T'], [-1, '0'], [NaN, '0'], [Infinity, '0']]) {
    assert.equal(format(value), expected);
  }
});


test('monitor automatically uses current launcher key instead of stale browser keys', async () => {
  const context = {document: null, sessionStorage: {getItem: () => 'old-key'},
    localStorage: {getItem: () => JSON.stringify('old-chat-key')},
    fetch: async (url) => {
      assert.equal(url, '/__strata_launcher/local-key');
      return {ok: true, json: async () => ({api_key: 'current-launcher-key'})};
    }};
  vm.runInNewContext(fs.readFileSync('strata_monitor_extension.js', 'utf8'), context);
  const headers = await context.StrataMonitorUsage.authHeaders();
  assert.equal(headers.Authorization, 'Bearer current-launcher-key');
});

test('empty launcher key overrides stale browser key and remote clients fall back', async () => {
  for (const local of [true, false]) {
    const context = {document: null, sessionStorage: {getItem: () => 'remote-key'},
      fetch: async () => ({ok: local, json: async () => ({api_key: ''})})};
    vm.runInNewContext(fs.readFileSync('strata_monitor_extension.js', 'utf8'), context);
    const headers = await context.StrataMonitorUsage.authHeaders();
    assert.equal(headers.Authorization, local ? undefined : 'Bearer remote-key');
  }
});

test('live display includes uncommitted counts and retains exact hover values', () => {
  const view = api().viewModel({current_request: {input_tokens: 2000, output_tokens: 123,
      total_tokens: 2123, source: 'engine', in_progress: true},
    last_request: {output_tokens: 1}, totals: {output_tokens: 10},
    display_totals: {input_tokens: 4000, output_tokens: 133, total_tokens: 4133}});
  assert.equal(view.last, '2K / 123 / 2.12K');
  assert.equal(view.lastExact, '2,000 / 123 / 2,123');
  assert.equal(view.totalExact, '4,133');
  assert.match(view.precision, /生成中/);
});

test('API key rotation is recovered automatically on unauthorized usage response', async () => {
  let current = 'first', calls = 0;
  const context = {document: null, fetch: async (url, options) => {
    if (url.endsWith('/local-key')) return {ok: true, json: async () => ({api_key: current})};
    calls++;
    return {status: options.headers.Authorization === `Bearer ${current}` ? 200 : 401};
  }};
  vm.runInNewContext(fs.readFileSync('strata_monitor_extension.js', 'utf8'), context);
  const usage = context.StrataMonitorUsage;
  assert.equal((await usage.authorizedFetch('/__strata_launcher/usage')).status, 200);
  current = 'second';
  assert.equal((await usage.authorizedFetch('/__strata_launcher/usage')).status, 200);
  assert.equal(calls, 3);
});

test('monitor opens with fresh data, polls each second, and does not overlap requests', async () => {
  const elements = {};
  let tick, changed, polls = 0, release;
  const view = {hidden: true, querySelector: () => null, insertBefore: () => {}};
  const document = {hidden: false, readyState: 'loading', head: {append() {}},
    addEventListener() {}, getElementById: id => id === 'view-monitor' ? view : elements[id],
    createElement: () => ({set innerHTML(text) {
      for (const match of text.matchAll(/id="([^"]+)"/g)) elements[match[1]] = {style: {}};
    }})};
  const context = {document, confirm: () => true, setInterval: (callback, ms) => {
    assert.equal(ms, 1000); tick = callback;
  }, MutationObserver: class {constructor(callback) {changed = callback;} observe() {}},
    fetch: async url => {
      if (url.endsWith('/local-key')) return {ok: true, json: async () => ({api_key: 'test-key'})};
      polls++;
      return new Promise(resolve => {release = data => resolve({ok: true, status: 200, json: async () => data});});
    }};
  vm.runInNewContext(fs.readFileSync('strata_monitor_extension.js', 'utf8'), context);
  context.StrataMonitorUsage.install();
  assert.equal(polls, 0);
  view.hidden = false;
  changed();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(polls, 1);
  tick();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(polls, 1);
  release({current_request: {input_tokens: 20, output_tokens: 5, total_tokens: 25, in_progress: true, source: 'engine'}});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(elements['su-last'].textContent, '20 / 5 / 25');
  assert.equal(elements['su-last'].title, '精确值：20 / 5 / 25');
  tick();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(polls, 2);
  release({current_request: {input_tokens: 20, output_tokens: 9, total_tokens: 29, in_progress: true, source: 'engine'}});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(elements['su-last'].textContent, '20 / 9 / 29');
});
