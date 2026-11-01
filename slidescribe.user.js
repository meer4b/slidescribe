// ==UserScript==
// @name         SlideScribe for Panopto
// @namespace    https://github.com/meer4b/slidescribe
// @version      1.0.0
// @description  Export Panopto lectures as study notes: every slide image and slide text, plus a timestamped transcript (Panopto captions if available, otherwise Whisper running locally in your browser).
// @author       meer4b
// @license      MIT
// @homepageURL  https://github.com/meer4b/slidescribe
// @supportURL   https://github.com/meer4b/slidescribe/issues
// @downloadURL  https://raw.githubusercontent.com/meer4b/slidescribe/main/slidescribe.user.js
// @updateURL    https://raw.githubusercontent.com/meer4b/slidescribe/main/slidescribe.user.js
// @match        https://*.panopto.eu/Panopto/Pages/Viewer.aspx*
// @match        https://*.panopto.com/Panopto/Pages/Viewer.aspx*
// @match        https://*.panopto.eu/Panopto/Pages/Embed.aspx*
// @match        https://*.panopto.com/Panopto/Pages/Embed.aspx*
// @grant        none
// @sandbox      raw
// @inject-into  page
// @run-at       document-idle
// ==/UserScript==

(function () {
  'use strict';

  const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.1';
  const MODELS = {
    'base.en (fast)': 'onnx-community/whisper-base.en',
    'small.en (more accurate, ~3x slower)': 'onnx-community/whisper-small.en',
  };

  // ---------- helpers ----------
  const deliveryId = () => new URLSearchParams(location.search).get('id');
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const fmt = t => { t = Math.max(0, Math.floor(t)); const h = Math.floor(t / 3600), m = Math.floor(t % 3600 / 60), s = t % 60; return (h ? h + ':' + String(m).padStart(2, '0') : m) + ':' + String(s).padStart(2, '0'); };
  const safeName = s => String(s || 'lecture').replace(/[\\/:*?"<>|]+/g, '_').trim().slice(0, 120);
  const save = (blob, name) => { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 5000); };
  const blobToDataURL = b => new Promise(r => { const fr = new FileReader(); fr.onload = () => r(fr.result); fr.readAsDataURL(b); });

  async function deliveryInfo(extra = {}) {
    const body = new URLSearchParams({ deliveryId: deliveryId(), isEmbed: 'false', responseType: 'json', ...extra });
    const r = await fetch('/Panopto/Pages/Viewer/DeliveryInfo.aspx', { method: 'POST', body, credentials: 'include' });
    if (!r.ok) throw new Error('DeliveryInfo failed: HTTP ' + r.status);
    return r.json();
  }

  // ---------- slides ----------
  // Panopto ObjectSequenceNumber: low 16 bits = slide number, high bits = build/animation step.
  // Group consecutive entries of the same slide; keep the LAST build (fully revealed slide).
  function groupSlides(timestamps) {
    const ts = (timestamps || []).filter(t => t.ObjectIdentifier && t.ObjectSequenceNumber).sort((a, b) => a.Time - b.Time);
    const segs = [];
    for (const t of ts) {
      const key = t.ObjectIdentifier + ':' + ((+t.ObjectSequenceNumber) & 0xffff);
      const last = segs[segs.length - 1];
      if (last && last.key === key) { last.img = t; if ((t.Data || '').length > last.text.length) last.text = t.Data || ''; }
      else segs.push({ key, start: t.Time, title: t.Caption || '', text: t.Data || '', img: t, transcript: [] });
    }
    return segs;
  }

  async function fetchSlideImages(segs, status) {
    for (let i = 0; i < segs.length; i++) {
      status(`Downloading slide ${i + 1}/${segs.length}…`);
      const t = segs[i].img;
      try {
        const r = await fetch(`/Panopto/Pages/Viewer/Image.aspx?id=${encodeURIComponent(t.ObjectIdentifier)}&number=${encodeURIComponent(t.ObjectSequenceNumber)}`, { credentials: 'include' });
        segs[i].dataUrl = r.ok ? await blobToDataURL(await r.blob()) : null;
      } catch { segs[i].dataUrl = null; }
    }
  }

  // ---------- transcript ----------
  async function existingCaptions() {
    try {
      const caps = await deliveryInfo({ getCaptions: 'true', language: '0' });
      if (Array.isArray(caps) && caps.length) return caps.map(c => ({ start: c.Time ?? 0, text: (c.Caption || '').trim() })).filter(c => c.text);
    } catch { }
    return null;
  }

  function audioUrl(d) {
    const pod = (d.PodcastStreams || []).find(s => s.StreamUrl);
    if (pod) return pod.StreamUrl;
    const http = [...(d.PodcastStreams || []), ...(d.Streams || [])].find(s => s.StreamHttpUrl);
    return http ? http.StreamHttpUrl : null;
  }

  async function whisperTranscribe(url, modelId, status) {
    status('Downloading audio…');
    const buf = await (await fetch(url)).arrayBuffer();
    status('Decoding audio…');
    const audio = (await new OfflineAudioContext(1, 16000, 16000).decodeAudioData(buf)).getChannelData(0);

    status('Loading Whisper model (first time downloads it, then cached)…');
    const tf = await import(TRANSFORMERS_URL);
    const gpu = !!navigator.gpu;
    const asr = await tf.pipeline('automatic-speech-recognition', modelId,
      gpu ? { device: 'webgpu', dtype: { encoder_model: 'fp32', decoder_model_merged: 'q4' } } : { device: 'wasm', dtype: 'q8' });

    const SR = 16000, BLOCK = 300 * SR; // 5-minute blocks so we can show progress
    const total = audio.length, out = [];
    for (let off = 0; off < total; off += BLOCK) {
      status(`Transcribing ${fmt(off / SR)} / ${fmt(total / SR)}${gpu ? '' : ' (no WebGPU: slow CPU mode)'}…`);
      const res = await asr(audio.subarray(off, Math.min(total, off + BLOCK)), { chunk_length_s: 30, stride_length_s: 5, return_timestamps: true });
      for (const c of res.chunks || [{ timestamp: [0], text: res.text }]) {
        const text = (c.text || '').trim();
        if (text) out.push({ start: off / SR + (c.timestamp?.[0] ?? 0), text });
      }
    }
    return out;
  }

  // ---------- output ----------
  function attach(segs, lines) {
    if (!segs.length) return [{ start: 0, title: 'Transcript', text: '', transcript: lines }];
    for (const l of lines) {
      let i = segs.length - 1;
      while (i > 0 && segs[i].start > l.start) i--;
      segs[i].transcript.push(l);
    }
    return segs;
  }

  function buildMarkdown(name, src, segs) {
    let md = `# ${name}\n\nTranscript source: ${src}\n\n`;
    segs.forEach((s, i) => {
      md += `## Slide ${i + 1} [${fmt(s.start)}] ${s.title}\n\n`;
      if (s.text && s.text !== s.title) md += `**Slide text:**\n${s.text.split(/\r\n?|\n/).map(x => x.trim()).filter(Boolean).map(x => '- ' + x).join('\n')}\n\n`;
      if (s.transcript.length) md += `**Lecturer:** ${s.transcript.map(l => l.text).join(' ')}\n\n`;
    });
    return md;
  }

  function buildHTML(name, src, segs) {
    const body = segs.map((s, i) => `
<section>
  <h2><span class="t">${fmt(s.start)}</span> Slide ${i + 1}: ${esc(s.title)}</h2>
  ${s.dataUrl ? `<img src="${s.dataUrl}" alt="Slide ${i + 1}">` : ''}
  ${s.text && s.text !== s.title ? `<p class="st"><b>Slide text:</b><br>${s.text.split(/\r\n?|\n/).map(x => esc(x.trim())).filter(Boolean).join('<br>')}</p>` : ''}
  ${s.transcript.length ? `<div class="tr">${s.transcript.map(l => `<p><span class="t">${fmt(l.start)}</span> ${esc(l.text)}</p>`).join('')}</div>` : ''}
</section>`).join('');
    return `<!doctype html><html><head><meta charset="utf-8"><title>${esc(name)}</title><style>
body{font:16px/1.55 system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#222}
section{border-bottom:1px solid #ddd;padding:1.5rem 0;break-inside:avoid-page}
img{max-width:100%;border:1px solid #ccc;border-radius:6px}
.t{font:12px monospace;color:#888;margin-right:.5em}.st{color:#555}.tr p{margin:.3em 0}
</style></head><body><h1>${esc(name)}</h1><p><i>Transcript source: ${esc(src)}</i></p>${body}</body></html>`;
  }

  // ---------- main ----------
  async function run(modelId, status, btns) {
    btns.forEach(b => b.disabled = true);
    try {
      status('Reading lecture info…');
      const info = await deliveryInfo();
      const d = info.Delivery;
      const name = safeName(d.SessionName);
      const segs = groupSlides(d.Timestamps);

      let lines = await existingCaptions(), src = 'Panopto captions';
      if (!lines) {
        const url = audioUrl(d);
        if (!url) throw new Error('No downloadable audio stream found for this recording.');
        lines = await whisperTranscribe(url, modelId, status);
        src = 'Whisper (' + modelId.split('/')[1] + ', in-browser)';
      }

      await fetchSlideImages(segs, status);
      const all = attach(segs, lines);
      status('Saving files…');
      save(new Blob([buildHTML(d.SessionName, src, all)], { type: 'text/html' }), name + ' - notes pack.html');
      save(new Blob([buildMarkdown(d.SessionName, src, all)], { type: 'text/markdown' }), name + ' - transcript.md');
      status(`Done: ${segs.length} slides, ${lines.length} transcript lines.`);
    } catch (e) {
      console.error('[LectureExport]', e);
      status('Error: ' + e.message);
    } finally { btns.forEach(b => b.disabled = false); }
  }

  async function downloadAudio(status) {
    try {
      status('Fetching audio…');
      const d = (await deliveryInfo()).Delivery;
      const url = audioUrl(d);
      if (!url) throw new Error('No downloadable audio stream found.');
      save(await (await fetch(url)).blob(), safeName(d.SessionName) + '.mp4');
      status('Audio saved.');
    } catch (e) { status('Error: ' + e.message); }
  }

  // ---------- UI ----------
  function ui() {
    if (document.getElementById('lx-panel')) return;
    const p = document.createElement('div');
    p.id = 'lx-panel';
    p.style.cssText = 'position:fixed;right:16px;bottom:90px;z-index:999999;background:#fff;color:#222;border:1px solid #bbb;border-radius:10px;padding:10px 12px;font:13px system-ui,sans-serif;box-shadow:0 4px 16px rgba(0,0,0,.2);width:250px';
    p.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px"><b>SlideScribe</b><span id="lx-x" style="cursor:pointer;color:#888">✕</span></div>
      <label style="font-size:12px">Whisper model (used only if no captions)</label>
      <select id="lx-model" style="width:100%;margin:4px 0 8px">${Object.keys(MODELS).map(k => `<option>${k}</option>`).join('')}</select>
      <button id="lx-go" style="width:100%;padding:6px;margin-bottom:6px;cursor:pointer">Export slides + transcript</button>
      <button id="lx-aud" style="width:100%;padding:6px;cursor:pointer">Download audio only (.mp4)</button>
      <div id="lx-s" style="margin-top:8px;font-size:12px;color:#555;min-height:1em"></div>`;
    document.body.appendChild(p);
    const status = m => { p.querySelector('#lx-s').textContent = m; };
    const go = p.querySelector('#lx-go'), aud = p.querySelector('#lx-aud');
    go.onclick = () => run(MODELS[p.querySelector('#lx-model').value], status, [go, aud]);
    aud.onclick = () => downloadAudio(status);
    p.querySelector('#lx-x').onclick = () => p.remove();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', ui); else ui();
  // expose for debugging/testing
  window.__lectureExport = { run, groupSlides, deliveryInfo, buildMarkdown };
})();
