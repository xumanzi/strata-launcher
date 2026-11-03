const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function api() {
  const source = fs.readFileSync('strata_code_preview.js', 'utf8');
  const context = {globalThis: null, document: null};
  context.globalThis = context;
  vm.runInNewContext(source, context);
  return context.StrataCodePreview;
}

test('recognizes previewable fenced languages only', () => {
  const preview = api();
  assert.equal(preview.languageOf({className: 'language-html'}), 'html');
  assert.equal(preview.languageOf({className: 'hljs language-javascript'}), 'javascript');
  assert.equal(preview.languageOf({className: 'language-python'}), '');
});

test('builds a complete document with css javascript and restrictive CSP', () => {
  const html = api().buildDocument({html: '<h1>Hi</h1>', css: 'h1{color:red}',
    js: 'document.body.dataset.ok=1'});
  assert.match(html, /<!doctype html>/i);
  assert.match(html, /Content-Security-Policy/);
  assert.match(html, /default-src 'none'/);
  assert.match(html, /h1\{color:red\}/);
  assert.match(html, /document\.body\.dataset\.ok=1/);
});

test('augments a complete html document instead of nesting it inside another body', () => {
  const html = api().buildDocument({html: '<!doctype html><html><head><title>X</title></head><body>ok</body></html>',
    css: 'body{color:red}', js: 'document.body.dataset.ready=1'});
  assert.equal((html.match(/<!doctype html>/gi) || []).length, 1);
  assert.equal((html.match(/<html/gi) || []).length, 1);
  assert.match(html, /Content-Security-Policy/);
  assert.match(html, /document\.body\.dataset\.ready=1/);
});

test('sandbox allows scripts without same-origin or navigation', () => {
  const value = api().sandboxAttributes();
  assert.equal(value, 'allow-scripts');
  assert.doesNotMatch(value, /allow-same-origin|allow-top-navigation|allow-popups/);
});

test('source provides idempotent enhancement and destroys iframe on stop', () => {
  const source = fs.readFileSync('strata_code_preview.js', 'utf8');
  assert.match(source, /dataset\.strataPreview/);
  assert.match(source, /frame\.replaceWith/);
  assert.match(source, /event\.source\s*!==\s*frame\.contentWindow/);
});

