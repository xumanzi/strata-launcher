// Launcher additions to Strata's own WebUI. This file is served by the local gateway.
(function () {
  "use strict";
  // Apply the launcher's default only when this browser has no saved chat preference.
  if (typeof store.get === "function" && !store.get("sampling", null)) {
    fetch("/__strata_launcher/preferences", {cache: "no-store", credentials: "same-origin"})
      .then((response) => response.ok ? response.json() : null)
      .then((pref) => {
        if (pref && ["high", "medium", "low"].includes(pref.reasoning_effort)) {
          settings.thinking = pref.reasoning_effort;
          if (typeof loadDrawer === "function") loadDrawer(settings);
        }
      })
      .catch(() => {});
  }
  const localHost = ["127.0.0.1", "localhost", "::1"].includes(location.hostname);
  if (localHost) {
    fetch("/__strata_launcher/local-key", {cache: "no-store", credentials: "same-origin"})
      .then((response) => { if (!response.ok) throw new Error("local key unavailable"); return response.json(); })
      .then(({api_key}) => {
        if (typeof api_key !== "string") return;
        store.set("apikey", api_key);
        $("api-key").value = api_key;
        loadHealth().then(loadMcp);
      })
      .catch(() => {});
  }

  const MAX_VIDEO_BYTES = 100_000_000;
  const MAX_VIDEO_SECONDS = 120;
  const MAX_VIDEO_FRAMES = 8;
  const isVideo = (file) => (file.type || "").startsWith("video/") || /\.(mp4|webm|mov)$/i.test(file.name || "");
  const previousLoadHealth = loadHealth;
  loadHealth = async function (...args) {
    const result = await previousLoadHealth(...args);
    if (health.images) $("attach-btn").title = "上传图片或视频；视频只分析抽出的画面，不含声音";
    return result;
  };
  loadHealth();

  function videoError(file, imagesEnabled) {
    if (!imagesEnabled) return "此模型未启用读图，不能分析视频画面。";
    if (file.size > MAX_VIDEO_BYTES) return "视频超过 100 MB；请选择较短的片段。";
    return "";
  }

  function frameTimes(duration) {
    const count = Math.min(MAX_VIDEO_FRAMES, Math.max(2, Math.ceil(duration / 4)));
    return Array.from({length: count}, (_, index) => duration * (index + 0.5) / count);
  }

  function waitForMetadata(video) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("读取视频信息超时。")), 15000);
      video.onloadedmetadata = () => { clearTimeout(timer); resolve(); };
      video.onerror = () => { clearTimeout(timer); reject(new Error("浏览器无法解码此视频；请尝试 MP4 或 WebM。")); };
      video.load();
    });
  }

  function seek(video, seconds) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error("读取视频画面超时。")), 15000);
      video.addEventListener("seeked", () => { clearTimeout(timer); resolve(); }, {once: true});
      video.currentTime = seconds;
    });
  }

  async function sampledFrames(file) {
    const video = document.createElement("video");
    video.preload = "auto";
    video.muted = true;
    const url = URL.createObjectURL(file);
    try {
      video.src = url;
      await waitForMetadata(video);
      const duration = video.duration;
      if (!Number.isFinite(duration) || duration <= 0 || duration > MAX_VIDEO_SECONDS)
        throw new Error("视频需在 2 分钟以内，并且浏览器能够读取时长。");
      const width = video.videoWidth, height = video.videoHeight;
      if (!width || !height) throw new Error("浏览器无法读取视频画面。");
      const scale = Math.min(1, 1024 / Math.max(width, height));
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(width * scale));
      canvas.height = Math.max(1, Math.round(height * scale));
      const context = canvas.getContext("2d");
      const times = frameTimes(duration);
      const frames = [];
      for (const seconds of times) {
        await seek(video, seconds);
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        frames.push({kind: "image", name: `${file.name} · ${seconds.toFixed(1)} 秒`,
                     url: canvas.toDataURL("image/jpeg", 0.78)});
      }
      return {frames, times};
    } finally {
      video.pause();
      video.removeAttribute("src");
      video.load();
      URL.revokeObjectURL(url);
    }
  }

  let pendingVideos = 0;
  const previousAddFiles = addFiles;
  addFiles = function (files) {
    const items = [...files];
    previousAddFiles(items.filter((file) => !isVideo(file)));
    for (const file of items.filter(isVideo)) {
      const error = videoError(file, health.images);
      if (error) { toast("warn", "无法读取视频", error); continue; }
      pendingVideos++;
      toast("info", "正在提取视频画面", `${file.name}：最多 8 帧，不读取声音。`, 6000);
      sampledFrames(file).then(({frames, times}) => {
        attachments.push({kind: "file", name: `${file.name} 的画面时间点`,
          text: `视频 ${file.name} 已按时间顺序抽取 ${frames.length} 张画面；时间点（秒）：${times.map((t) => t.toFixed(1)).join("、")}。这些是静态画面，不含声音。`});
        attachments.push(...frames);
        renderAttachments();
        toast("success", "视频画面已就绪", "现在可输入问题并发送。", 5000);
      }).catch((cause) => toast("warn", "视频处理失败", cause.message || String(cause), 7000))
        .finally(() => { pendingVideos--; });
    }
  };
  const previousSend = send;
  send = function (...args) {
    if (pendingVideos) { toast("info", "请稍等", "视频画面还在提取中。"); return; }
    return previousSend(...args);
  };
  globalThis.StrataLauncherWeb = {frameTimes, videoError};
})();
