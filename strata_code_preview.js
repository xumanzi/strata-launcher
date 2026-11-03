// Sandboxed HTML/CSS/JavaScript/SVG preview for generated code blocks.
(function () {
  "use strict";
  const PREVIEWABLE = new Set(["html", "css", "javascript", "js", "svg"]);
  function languageOf(code) {
    const match = String(code?.className || "").match(/(?:^|\s)language-([\w-]+)/i);
    const language = (match?.[1] || "").toLowerCase();
    return PREVIEWABLE.has(language) ? (language === "js" ? "javascript" : language) : "";
  }
  function sandboxAttributes() { return "allow-scripts"; }
  function safeScript(value) { return String(value || "").replace(/<\/script/gi, "<\\/script"); }
  function safeStyle(value) { return String(value || "").replace(/<\/style/gi, "<\\/style"); }
  function buildDocument(parts) {
    const body = String(parts?.html || "");
    const css = safeStyle(parts?.css);
    const js = safeScript(parts?.js);
    const bridge = safeScript(`addEventListener("error",e=>parent.postMessage({source:"strata-preview",error:e.message},"*"));addEventListener("unhandledrejection",e=>parent.postMessage({source:"strata-preview",error:String(e.reason)},"*"));`);
    const policy = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data: blob:; media-src data: blob:; style-src 'unsafe-inline'; script-src 'unsafe-inline' blob:; font-src data:">`;
    const styles = `<style>html,body{min-height:100%;margin:0;font-family:system-ui,sans-serif}${css}</style>`;
    const scripts = `<script>${bridge}<\/script><script>${js}<\/script>`;
    if (/<html[\s>]/i.test(body)) {
      let documentText = body;
      documentText = /<\/head>/i.test(documentText) ? documentText.replace(/<\/head>/i, `${policy}${styles}</head>`) : documentText.replace(/<html[^>]*>/i, match => `${match}<head>${policy}${styles}</head>`);
      documentText = /<\/body>/i.test(documentText) ? documentText.replace(/<\/body>/i, `${scripts}</body>`) : documentText.replace(/<\/html>/i, `<body>${scripts}</body></html>`);
      return documentText;
    }
    return `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">${policy}${styles}</head><body>${body}${scripts}</body></html>`;
  }
  function download(name, content) {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([content], {type: "text/html;charset=utf-8"}));
    link.download = name;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }
  function openPreview(initial) {
    document.getElementById("strata-preview-dialog")?.remove();
    const overlay = document.createElement("div");
    overlay.id = "strata-preview-dialog";
    overlay.innerHTML = `<style>#strata-preview-dialog{position:fixed;inset:0;z-index:99999;background:#0009;padding:3vh 3vw;box-sizing:border-box}.sp-shell{height:94vh;background:var(--st-surface,#fff);color:var(--st-ink,#182033);border-radius:14px;display:flex;flex-direction:column;overflow:hidden}.sp-bar{display:flex;align-items:center;gap:8px;padding:10px 14px;border-bottom:1px solid var(--st-line,#ccd2dd);flex-wrap:wrap}.sp-bar strong{margin-right:auto}.sp-bar button{padding:7px 12px}.sp-work{display:grid;grid-template-columns:minmax(300px,42%) 1fr;min-height:0;flex:1}.sp-edit{display:grid;grid-template-rows:repeat(3,1fr);gap:1px;background:var(--st-line,#ccd2dd)}.sp-field{display:flex;flex-direction:column;min-height:0;background:var(--st-surface,#fff)}.sp-field label{padding:5px 9px;font-size:12px}.sp-field textarea{flex:1;resize:none;border:0;padding:9px;font:12px/1.5 ui-monospace,monospace;background:var(--st-bg,#f5f7fa);color:inherit}.sp-view{display:flex;flex-direction:column;min-width:0}.sp-view iframe{flex:1;width:100%;border:0;background:white}.sp-errors{max-height:90px;overflow:auto;color:#b42318;font:12px/1.4 ui-monospace,monospace;padding:6px 9px}.sp-shell:fullscreen{height:100vh;border-radius:0}@media(max-width:800px){#strata-preview-dialog{padding:0}.sp-shell{height:100vh;border-radius:0}.sp-work{grid-template-columns:1fr;grid-template-rows:48% 52%}}</style><div class="sp-shell"><div class="sp-bar"><strong>代码效果预览</strong><button data-action="run">运行</button><button data-action="stop">停止</button><button data-action="full">全屏</button><button data-action="download">下载 HTML</button><button data-action="close">关闭</button></div><div class="sp-work"><div class="sp-edit"><div class="sp-field"><label>HTML / SVG</label><textarea data-code="html"></textarea></div><div class="sp-field"><label>CSS</label><textarea data-code="css"></textarea></div><div class="sp-field"><label>JavaScript</label><textarea data-code="js"></textarea></div></div><div class="sp-view"><iframe title="代码效果"></iframe><div class="sp-errors" aria-live="polite"></div></div></div></div>`;
    document.body.append(overlay);
    for (const key of ["html", "css", "js"]) overlay.querySelector(`[data-code="${key}"]`).value = initial[key] || "";
    let frame = overlay.querySelector("iframe");
    const errors = overlay.querySelector(".sp-errors");
    function documentText() { return buildDocument(Object.fromEntries(["html", "css", "js"].map(key => [key, overlay.querySelector(`[data-code="${key}"]`).value]))); }
    function stop() {
      const fresh = document.createElement("iframe"); fresh.title = "代码效果";
      frame.replaceWith(fresh); frame = fresh; errors.textContent = "";
    }
    function run() { stop(); frame.setAttribute("sandbox", sandboxAttributes()); frame.srcdoc = documentText(); }
    const onMessage = event => {
      if (event.source !== frame.contentWindow || event.data?.source !== "strata-preview") return;
      const lines = errors.textContent ? errors.textContent.split("\n") : [];
      if (lines.length < 50) errors.textContent = [...lines, event.data.error].filter(Boolean).join("\n");
    };
    addEventListener("message", onMessage);
    overlay.querySelector('[data-action="run"]').onclick = run;
    overlay.querySelector('[data-action="stop"]').onclick = stop;
    overlay.querySelector('[data-action="full"]').onclick = () => overlay.querySelector(".sp-shell").requestFullscreen?.();
    overlay.querySelector('[data-action="download"]').onclick = () => download("strata-preview.html", documentText());
    overlay.querySelector('[data-action="close"]').onclick = () => { removeEventListener("message", onMessage); stop(); overlay.remove(); };
    run();
  }
  function enhance(root) {
    if (!root?.querySelectorAll) return;
    root.querySelectorAll("pre code").forEach(code => {
      const language = languageOf(code);
      if (!language || code.dataset.strataPreview) return;
      code.dataset.strataPreview = "1";
      const pre = code.parentElement;
      const button = document.createElement("button");
      button.type = "button"; button.textContent = "预览效果"; button.className = "strata-preview-button";
      button.onclick = () => openPreview({
        html: language === "html" || language === "svg" ? code.textContent : "<main id=\"app\"></main>",
        css: language === "css" ? code.textContent : "",
        js: language === "javascript" ? code.textContent : "",
      });
      pre.insertAdjacentElement("beforebegin", button);
    });
  }
  globalThis.StrataCodePreview = {languageOf, sandboxAttributes, buildDocument, enhance, openPreview};
  if (typeof document !== "undefined" && document) {
    const start = () => { enhance(document); new MutationObserver(records => records.forEach(record => record.addedNodes.forEach(node => enhance(node.nodeType === 1 ? node : null)))).observe(document.body, {subtree: true, childList: true}); };
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start, {once: true}); else start();
  }
})();
