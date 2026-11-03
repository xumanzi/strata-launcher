const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function extension() {
  const source = fs.readFileSync('strata_web_extension.js', 'utf8');
  const controls = {'attach-btn': {}, file: {}};
  const context = {
    location: {hostname: '192.168.1.50'},
    store: {set() {}},
    $: (id) => controls[id] ||= {},
    health: {images: true}, loadHealth: async () => {},
    addFiles() {}, send() {}, attachments: [], renderAttachments() {}, toast() {},
    globalThis: null,
  };
  context.globalThis = context;
  vm.runInNewContext(source, context);
  return context.StrataLauncherWeb;
}

test('video sampling stays within an eight-frame request budget', () => {
  const video = extension();
  const times = video.frameTimes(60);
  assert.equal(times.length, 8);
  assert.ok(times[0] > 0);
  assert.ok(times.at(-1) < 60);
  assert.ok(times.every((time, i) => i === 0 || time > times[i - 1]));
});

test('video eligibility rejects text-only models and oversized clips', () => {
  const video = extension();
  const clip = {name: 'scene.mp4', type: 'video/mp4', size: 5_000_000};
  assert.equal(video.videoError(clip, false), '此模型未启用读图，不能分析视频画面。');
  assert.match(video.videoError({...clip, size: 101_000_000}, true), /100 MB/);
  assert.equal(video.videoError(clip, true), '');
});

test('localhost fills the running key while a LAN page does not request it', async () => {
  const source = fs.readFileSync('strata_web_extension.js', 'utf8');
  let requests = 0, saved = '', refreshed = 0;
  const controls = {'api-key': {value: ''}};
  const context = {
    location: {hostname: '127.0.0.1'},
    fetch: async (path) => { requests++; assert.equal(path, '/__strata_launcher/local-key');
      return {ok: true, json: async () => ({api_key: 'current-model-key'})}; },
    store: {set: (_name, value) => { saved = value; }},
    $: (id) => controls[id] ||= {},
    loadHealth: async () => { refreshed++; }, loadMcp: async () => {},
    health: {images: false}, addFiles() {}, send() {}, attachments: [], renderAttachments() {}, toast() {},
  };
  context.globalThis = context;
  vm.runInNewContext(source, context);
  await new Promise(setImmediate);
  assert.equal(requests, 1);
  assert.equal(saved, 'current-model-key');
  assert.equal(controls['api-key'].value, saved);
  assert.ok(refreshed >= 1);
  context.location.hostname = '192.168.1.50';
  vm.runInNewContext(source, context);
  await new Promise(setImmediate);
  assert.equal(requests, 1);
});
