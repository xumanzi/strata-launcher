// Token and context statistics shown inside the WebUI Monitor tab (#view-monitor). The
// standalone /api-monitor page was retired; the panel-host fallback stays for older Strata installs.
(function () {
  "use strict";
  const number = value => new Intl.NumberFormat().format(Number(value) || 0);
  function tokenNumber(value) {
    const amount = Math.max(0, Number(value) || 0);
    if (!Number.isFinite(amount)) return '0';
    const units = ['', 'K', 'M', 'B', 'T'];
    let scaled = amount, unit = 0;
    while (scaled >= 1000 && unit < units.length - 1) {
      scaled /= 1000; unit++;
    }
    if (unit && Math.round(scaled * 100) / 100 >= 1000 && unit < units.length - 1) {
      scaled /= 1000; unit++;
    }
    return new Intl.NumberFormat('en-US', {maximumFractionDigits: unit ? 2 : 0}).format(scaled) + units[unit];
  }
  let launcherKey = null;
  let keyRequest = null;
  async function loadLauncherKey() {
    if (launcherKey !== null) return launcherKey;
    if (!keyRequest) keyRequest = (async () => {
      try {
        const response = await fetch('/__strata_launcher/local-key', {
          cache: 'no-store', credentials: 'same-origin'});
        if (response.ok) {
          const data = await response.json();
          if (typeof data.api_key === 'string') launcherKey = data.api_key;
        }
      } catch (_) { /* Remote clients retain the configured chat key. */ }
      return resolveKey();
    })();
    try { return await keyRequest; } finally { keyRequest = null; }
  }
  async function authHeaders() {
    const key = await loadLauncherKey();
    return key ? {Authorization: `Bearer ${key}`} : {};
  }
  async function authorizedFetch(url, options = {}) {
    const send = async () => fetch(url, {...options, cache: 'no-store', credentials: 'same-origin',
      headers: {...(options.headers || {}), ...await authHeaders()}});
    let response = await send();
    if (response.status === 401) {
      launcherKey = null; // the running model or its API key may have changed
      response = await send();
      if (response.status === 401 && launcherKey === null) {
        // Remote browsers cannot read /local-key. Try the current WebUI field/store
        // before the retired standalone monitor's session key.
        const candidates = [];
        try { candidates.push(document.getElementById('api-key')?.value); } catch (_) {}
        try {
          const value = localStorage.getItem('strata.apikey');
          try { candidates.push(JSON.parse(value)); } catch (_) { candidates.push(value); }
        } catch (_) {}
        for (const key of candidates.filter(key => typeof key === 'string' && key)) {
          response = await fetch(url, {...options, cache: 'no-store', credentials: 'same-origin',
            headers: {...(options.headers || {}), Authorization: `Bearer ${key}`}});
          if (response.status !== 401) break;
        }
      }
    }
    return response;
  }
  function usageLevel(percent) {
    return percent >= 90 ? "danger" : percent >= 70 ? "warning" : "normal";
  }
  // The standalone monitor page keeps its key in sessionStorage; the chat page stores it via
  // Strata's own store (JSON-encoded in localStorage). Accept either so both pages authenticate.
  function resolveKey() {
    if (launcherKey !== null) return launcherKey;
    let key = "";
    try { key = sessionStorage.getItem("strata.monitor.key") || ""; } catch (e) { /* storage unavailable */ }
    if (key) return key;
    try {
      const stored = localStorage.getItem("strata.apikey");
      if (typeof stored === "string" && stored) {
        try { const parsed = JSON.parse(stored); key = typeof parsed === "string" ? parsed : stored; }
        catch (e) { key = stored; }
      }
    } catch (e) { /* storage unavailable */ }
    return key;
  }
  // Chat page: render inside the Monitor tab (#view-monitor) so the panel only shows when that
  // tab is open. The legacy standalone monitor host fallback keeps older pages working.
  function findTarget() {
    const view = document.getElementById("view-monitor");
    if (view) return {view, host: view.querySelector(".monitor") || view, tabbed: true};
    const panel = document.querySelector(".panel") || document.querySelector("main");
    return panel && panel.parentNode ? {host: panel, tabbed: false} : null;
  }
  function viewModel(data) {
    const context = data.context || {}, last = data.current_request || data.last_request || {};
    const totals = data.display_totals || data.totals || {}, requests = data.requests || {}, compression = data.compression || {};
    return {
      context: `${tokenNumber(context.used_tokens)} / ${tokenNumber(context.limit_tokens)}`,
      contextExact: `${number(context.used_tokens)} / ${number(context.limit_tokens)}`,
      percent: Number(context.percent) || 0,
      level: usageLevel(Number(context.percent) || 0),
      last: `${tokenNumber(last.input_tokens)} / ${tokenNumber(last.output_tokens)} / ${tokenNumber(last.total_tokens)}`,
      lastExact: `${number(last.input_tokens)} / ${number(last.output_tokens)} / ${number(last.total_tokens)}`,
      precision: last.in_progress ? (last.source === 'waiting' ? '等待引擎统计…' :
        '生成中 · 本次未结算' + (last.parallel_requests > 1 ? ' · 多请求以各自 usage 更新' : '')) :
        last.precise === false ? "未收到最终准确统计" : "准确",
      input: tokenNumber(totals.input_tokens), output: tokenNumber(totals.output_tokens), total: tokenNumber(totals.total_tokens),
      inputExact: number(totals.input_tokens), outputExact: number(totals.output_tokens), totalExact: number(totals.total_tokens),
      requests: `${number(requests.total)} 次 · 成功 ${number(requests.successful)} · 失败 ${number(requests.failed)} · 处理中 ${number(requests.active)}`,
      compression: `${number(compression.count)} 次 · 缓存命中 ${number(compression.cache_hits)}`,
      saved: tokenNumber(compression.saved_tokens),
      savedExact: number(compression.saved_tokens),
      lastCompression: compression.last ? `${tokenNumber(compression.last.before_tokens)} → ${tokenNumber(compression.last.after_tokens)}（节省 ${tokenNumber(compression.last.saved_tokens)}）` : "尚未触发",
      updated: data.updated_at ? new Date(data.updated_at * 1000).toLocaleTimeString() : "—",
    };
  }
  function install() {
    if (!document || document.getElementById("strata-usage-panel")) return;
    const target = findTarget();
    if (!target) return;
    const style = document.createElement("style");
    style.textContent = `#strata-usage-panel{margin:0 0 24px;padding:18px 20px;border:1px solid var(--st-line);border-radius:14px;background:var(--st-surface)}#strata-usage-panel .su-head{display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap}.su-grid{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:12px;margin-top:15px}.su-item{padding:12px;background:var(--st-bg);border-radius:9px}.su-item span{display:block;color:var(--st-ink-muted);font-size:12px}.su-item b{display:block;margin-top:4px;font-size:18px;font-weight:500}.su-bar{height:9px;margin-top:9px;border-radius:9px;background:var(--st-line);overflow:hidden}.su-bar i{display:block;height:100%;background:var(--st-accent)}#strata-usage-panel.warning .su-bar i{background:var(--st-warn)}#strata-usage-panel.danger .su-bar i{background:var(--st-danger)}.su-status{color:var(--st-ink-muted);font-size:12px}@media(max-width:950px){.su-grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:600px){.su-grid{grid-template-columns:1fr}}`;
    document.head.append(style);
    const panel = document.createElement("section");
    panel.id = "strata-usage-panel";
    panel.innerHTML = `<div class="su-head"><div><h2>Token 使用量</h2><div class="su-status" id="su-status">正在读取…</div></div><div class="controls"><button id="su-refresh">刷新</button><button id="su-reset">清零本次统计</button></div></div><div class="su-grid"><div class="su-item"><span>当前上下文</span><b id="su-context">—</b><div class="su-bar"><i id="su-progress"></i></div></div><div class="su-item"><span>最近输入 / 输出 / 合计</span><b id="su-last">—</b><small id="su-precision"></small></div><div class="su-item"><span>累计输入 / 输出</span><b id="su-io">—</b></div><div class="su-item"><span>累计 Token</span><b id="su-total">—</b></div><div class="su-item"><span>API 请求</span><b id="su-requests">—</b></div><div class="su-item"><span>上下文压缩</span><b id="su-compression">—</b></div><div class="su-item"><span>累计节省</span><b id="su-saved">—</b></div><div class="su-item"><span>最近一次压缩</span><b id="su-last-compression">—</b></div></div>`;
    if (target.tabbed) target.host.insertBefore(panel, target.host.firstChild || null);
    else target.host.parentNode.insertBefore(panel, target.host);
    const set = (id, value, exact) => {
      const element = document.getElementById(id);
      element.textContent = value;
      if (exact !== undefined) element.title = `精确值：${exact}`;
    };
    let refreshing = false;
    async function refresh() {
      if (target.tabbed && target.view.hidden) return;   // chat page: only poll while the Monitor tab is open
      if (document.hidden || refreshing) return;
      refreshing = true;
      try {
        const response = await authorizedFetch("/__strata_launcher/usage");
        const data = await response.json();
        if (!response.ok) throw new Error(data.error?.message || `HTTP ${response.status}`);
        const view = viewModel(data);
        panel.className = view.level;
        set("su-context", `${view.context} · ${view.percent.toFixed(1)}%`, view.contextExact);
        document.getElementById("su-progress").style.width = `${Math.min(100, view.percent)}%`;
        set("su-last", view.last, view.lastExact); set("su-precision", view.precision);
        set("su-io", `${view.input} / ${view.output}`, `${view.inputExact} / ${view.outputExact}`);
        set("su-total", view.total, view.totalExact);
        set("su-requests", view.requests); set("su-compression", view.compression);
        set("su-saved", view.saved, view.savedExact); set("su-last-compression", view.lastCompression);
        set("su-status", `${data.current_request ? '生成中 · 累计含本次未结算' : '统计已结算'} · 每秒刷新 · 更新于 ${view.updated}`);
        const experts = document.getElementById('slots-text');
        if (experts) {
          try {
            const response = await authorizedFetch('/metrics');
            const metrics = response.ok ? await response.json() : {};
            const engine = metrics.engine || {};
            if (engine.arch === 'inkling2_loop') {
              experts.textContent = engine.expert_slots != null
                ? `${number(engine.expert_slots)} / ${number(engine.expert_total)} 个专家`
                : '当前引擎未提供专家驻留统计';
            }
          } catch (_) { /* Usage accounting remains available during a metrics retry. */ }
        }
      } catch (error) { set("su-status", `统计暂不可用：${error.message || error}`); }
      finally { refreshing = false; }
    }
    document.getElementById("su-refresh").onclick = refresh;
    document.getElementById("su-reset").onclick = async () => {
      if (!confirm("只清零本次 Token 统计，不会删除聊天记录或模型。继续吗？")) return;
      try {
        const response = await authorizedFetch("/__strata_launcher/usage/reset", {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"});
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        await refresh();
      } catch (error) { set('su-status', `清零失败：${error.message || error}`); }
    };
    refresh();
    setInterval(refresh, 1000);
    document.addEventListener('visibilitychange', refresh);
    if (target.tabbed && typeof MutationObserver !== 'undefined') {
      new MutationObserver(refresh).observe(target.view, {attributes: true, attributeFilter: ['hidden']});
    }
  }
  globalThis.StrataMonitorUsage = {tokenNumber, usageLevel, viewModel, resolveKey, loadLauncherKey, authHeaders, authorizedFetch, findTarget, install};
  if (typeof document !== "undefined" && document) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", install, {once: true});
    else install();
  }
})();
