/**
 * app.js — Real-Time AI Avatar Streaming Client
 *
 * Protocol:
 *   Client→Server: [0x01][int16_LE PCM at 24kHz, FRAME_SIZE=1920 samples]
 *   Server→Client: [0x01][seq:4B][int16_LE PCM] — audio
 *                  [0x02][seq:4B][JPEG bytes]    — video frame
 *                  [0x03][token:4B]              — text token
 *                  [0xFE]                        — keepalive
 *
 * Audio playback uses Web Audio API with a PCM scheduling queue for
 * glitch-free continuous playback.
 *
 * Lip-sync: audio seq N and video seq N are rendered at aligned timestamps.
 * Audio plays immediately; video frames are held until the corresponding
 * audio timestamp is reached on the AudioContext clock.
 */

'use strict';

// ── Constants ────────────────────────────────────────────────────────────────
const SAMPLE_RATE  = 24000;
const FRAME_SIZE   = 1920;          // 1 Moshi step = 80ms at 24kHz
const MSG_AUDIO    = 0x01;
const MSG_VIDEO    = 0x02;
const MSG_TEXT     = 0x03;
const MSG_KEEP     = 0xFE;
const NUM_VIS_BARS = 16;

// ── State ────────────────────────────────────────────────────────────────────
let sessionId     = null;
let ws            = null;
let audioCtx      = null;
let mediaStream   = null;
let scriptNode    = null;
let workletNode   = null;

// Audio playback scheduling
let audioQueue    = [];   // [{seq, buffer: AudioBuffer}]
let nextPlayTime  = 0;    // AudioContext time for next packet

// Video frame buffer for lip-sync
let frameQueue    = [];   // [{seq, blob}] 
let lastVideoSeq  = -1;

// Stats
let frameCount    = 0;
let audioPktCount = 0;
let fpsTimer      = null;
let fpsCount      = 0;
let lastFpsTime   = performance.now();
let audioLatencies= [];

const canvas  = document.getElementById('avatar-canvas');
const ctx2d   = canvas.getContext('2d');

// ── UI helpers ────────────────────────────────────────────────────────────────
function log(msg, type = 'info') {
  const box  = document.getElementById('log-box');
  const line = document.createElement('div');
  line.className = `log-${type}`;
  const ts = new Date().toLocaleTimeString('en', {hour12:false});
  line.textContent = `${ts}  ${msg}`;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
  // Keep at most 80 lines
  while (box.children.length > 80) box.removeChild(box.firstChild);
}

function setStatus(state, label) {
  const pill = document.getElementById('status-pill');
  const text = document.getElementById('status-text');
  pill.className = `status-pill ${state}`;
  text.textContent = label;
}

function setIdle(text, sub, spinning = true) {
  document.getElementById('idle-text').textContent = text;
  document.getElementById('idle-sub').textContent  = sub;
  const ring = document.getElementById('idle-ring');
  ring.className = spinning ? 'idle-ring' : 'idle-ring ready';
}

function showOverlay(show) {
  const ov = document.getElementById('video-overlay');
  ov.classList.toggle('hidden', !show);
}

function setStat(id, val, cls = '') {
  const el = document.getElementById(id);
  if (el) { el.textContent = val; el.className = `metric-value ${cls}`; }
}

// ── Queue bar rendering ───────────────────────────────────────────────────────
const QUEUES = [
  {id:'audio', label:'Audio Q', max: 8},
  {id:'token', label:'Token Q', max:16},
  {id:'emb',   label:'Emb Q',   max:32},
  {id:'frame', label:'Frame Q', max:30},
  {id:'send',  label:'Send Q',  max:60},
];

function initQueueBars() {
  const wrap = document.getElementById('queue-bars');
  QUEUES.forEach(q => {
    const div = document.createElement('div');
    div.className = 'queue-bar-wrap';
    div.innerHTML = `
      <div class="queue-label">
        <span>${q.label}</span>
        <span id="qv-${q.id}">0/${q.max}</span>
      </div>
      <div class="queue-bar">
        <div class="queue-bar-fill" id="qb-${q.id}" style="width:0%"></div>
      </div>`;
    wrap.appendChild(div);
  });
}

function updateQueueBar(id, val, max) {
  const vEl = document.getElementById(`qv-${id}`);
  const bEl = document.getElementById(`qb-${id}`);
  if (vEl) vEl.textContent = `${val}/${max}`;
  if (bEl) bEl.style.width = `${Math.min(100, (val/max)*100).toFixed(1)}%`;
}

// ── Audio visualiser ──────────────────────────────────────────────────────────
function initVisBars() {
  const vis = document.getElementById('audio-vis');
  for (let i = 0; i < NUM_VIS_BARS; i++) {
    const bar = document.createElement('div');
    bar.className = 'vis-bar';
    bar.style.height = '4px';
    bar.style.flex = '1';
    vis.appendChild(bar);
  }
}

function updateVis(pcmInt16) {
  const bars = document.querySelectorAll('.vis-bar');
  const step = Math.floor(pcmInt16.length / NUM_VIS_BARS);
  bars.forEach((bar, i) => {
    let rms = 0;
    for (let j = i * step; j < (i+1)*step && j < pcmInt16.length; j++) {
      const s = pcmInt16[j] / 32768;
      rms += s * s;
    }
    rms = Math.sqrt(rms / step);
    const h = Math.max(4, Math.min(38, rms * 200));
    bar.style.height = `${h}px`;
  });
}

// ── Session start ─────────────────────────────────────────────────────────────
async function startSession() {
  try {
    const resp = await fetch('/session/start', {method:'POST'});
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    sessionId = data.session_id;
    document.getElementById('session-id-display').textContent = sessionId;
    log(`Session created: ${sessionId.slice(0,12)}…`, 'ok');
    setStatus('connected', 'Session ready');
    document.getElementById('btn-mic').disabled = false;
    document.getElementById('btn-stop').disabled = false;
    setIdle('Upload your face image', 'Then click Start Talking', false);
  } catch (err) {
    log(`Session start failed: ${err}`, 'error');
  }
}

// ── Image upload ──────────────────────────────────────────────────────────────
async function uploadImage(file) {
  if (!sessionId) {
    log('Start a session first.', 'warn');
    return;
  }
  const fd = new FormData();
  fd.append('file', file);
  try {
    setIdle('Uploading image…', 'Encoding reference frame', true);
    const resp = await fetch(`/session/${sessionId}/image`, {method:'POST', body:fd});
    if (!resp.ok) throw new Error(await resp.text());
    log('Face image uploaded. AvatarForcing ready.', 'ok');
    setIdle('Ready! Click "Start Talking"', 'Mic → Moshi → Bridge → AvatarForcing', false);
  } catch (err) {
    log(`Image upload failed: ${err}`, 'error');
    setIdle('Upload failed', String(err), false);
  }
}

// ── WebSocket connect ─────────────────────────────────────────────────────────
function connectWS() {
  if (!sessionId) return;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url   = `${proto}://${location.host}/ws/${sessionId}`;
  ws = new WebSocket(url);
  ws.binaryType = 'arraybuffer';
  log(`Connecting WS → ${url}`, 'info');

  ws.onopen = () => {
    log('WebSocket connected.', 'ok');
    setStatus('streaming', 'Streaming');
  };

  ws.onmessage = (ev) => {
    const buf  = new Uint8Array(ev.data);
    const type = buf[0];
    switch (type) {
      case MSG_AUDIO: handleAudio(buf); break;
      case MSG_VIDEO: handleVideo(buf); break;
      case MSG_TEXT:  break; // optional
      case MSG_KEEP:  break; // keepalive — ignore
    }
  };

  ws.onclose = (ev) => {
    log(`WS closed (${ev.code})`, 'warn');
    setStatus('', 'Disconnected');
    ws = null;
  };

  ws.onerror = (ev) => {
    log('WS error.', 'error');
  };
}

// ── Audio receive & playback ──────────────────────────────────────────────────
function handleAudio(buf) {
  // [0x01][seq:4B big-endian][int16 PCM]
  const seq = (buf[1]<<24 | buf[2]<<16 | buf[3]<<8 | buf[4]) >>> 0;
  const raw = buf.slice(5);
  const pcm = new Int16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2);

  audioPktCount++;
  setStat('stat-audio-pkts', audioPktCount);
  setStat('stat-seq', seq);
  updateVis(pcm);

  if (!audioCtx) return;

  // Convert int16 → float32
  const f32 = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) f32[i] = pcm[i] / 32768.0;

  const ab = audioCtx.createBuffer(1, f32.length, SAMPLE_RATE);
  ab.copyToChannel(f32, 0);

  // Schedule playback
  const now = audioCtx.currentTime;
  if (nextPlayTime < now) nextPlayTime = now + 0.04; // 40ms ahead

  const src = audioCtx.createBufferSource();
  src.buffer = ab;
  src.connect(audioCtx.destination);
  src.start(nextPlayTime);

  // Record scheduled time for lip-sync
  audioQueue.push({seq, scheduledAt: nextPlayTime});
  if (audioQueue.length > 200) audioQueue.shift();

  nextPlayTime += ab.duration;

  // Latency stat
  const latMs = (nextPlayTime - now) * 1000;
  audioLatencies.push(latMs);
  if (audioLatencies.length > 20) audioLatencies.shift();
  const avgLat = audioLatencies.reduce((a,b)=>a+b,0) / audioLatencies.length;
  const latCls = avgLat < 150 ? 'good' : avgLat < 300 ? 'warn' : 'bad';
  setStat('stat-audio-lat', `${avgLat.toFixed(0)}ms`, latCls);
}

// ── Video receive & rendering ──────────────────────────────────────────────────
function handleVideo(buf) {
  // [0x02][seq:4B big-endian][JPEG bytes]
  const seq  = (buf[1]<<24 | buf[2]<<16 | buf[3]<<8 | buf[4]) >>> 0;
  const jpeg = buf.slice(5);

  frameCount++;
  fpsCount++;
  setStat('stat-frames', frameCount);

  // Decode JPEG → ImageBitmap → render
  const blob = new Blob([jpeg], {type: 'image/jpeg'});
  createImageBitmap(blob).then(bmp => {
    // Find closest audio scheduled time for this seq
    let renderTime = null;
    for (const a of audioQueue) {
      if (a.seq >= seq) { renderTime = a.scheduledAt; break; }
    }

    if (renderTime && audioCtx) {
      const delay = Math.max(0, renderTime - audioCtx.currentTime) * 1000;
      setTimeout(() => renderFrame(bmp, seq), delay);
    } else {
      renderFrame(bmp, seq);
    }
  }).catch(() => {});
}

function renderFrame(bmp, seq) {
  showOverlay(false);
  ctx2d.drawImage(bmp, 0, 0, canvas.width, canvas.height);
  bmp.close();
  lastVideoSeq = seq;
}

// FPS counter
setInterval(() => {
  const now   = performance.now();
  const dt    = (now - lastFpsTime) / 1000;
  const fps   = fpsCount / dt;
  fpsCount    = 0;
  lastFpsTime = now;
  const fpsCls = fps > 20 ? 'good' : fps > 10 ? 'warn' : 'bad';
  setStat('stat-fps', fps > 0 ? `${fps.toFixed(1)} fps` : '—', fpsCls);
}, 2000);

// ── Microphone capture → WebSocket ───────────────────────────────────────────
async function startMic() {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    connectWS();
    await new Promise(r => setTimeout(r, 500));
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate:   SAMPLE_RATE,
        echoCancellation: true,
        noiseSuppression: true,
      }
    });

    audioCtx = new AudioContext({sampleRate: SAMPLE_RATE});
    nextPlayTime = audioCtx.currentTime + 0.1;

    // Use AudioWorklet for low-latency mic capture at FRAME_SIZE boundaries
    await audioCtx.audioWorklet.addModule('/static/mic-worklet.js').catch(() => {
      // Fallback to ScriptProcessor if AudioWorklet fails
      useFallbackProcessor();
      return;
    });

    const src = audioCtx.createMediaStreamSource(mediaStream);
    workletNode = new AudioWorkletNode(audioCtx, 'mic-processor', {
      processorOptions: { frameSize: FRAME_SIZE }
    });

    workletNode.port.onmessage = (ev) => {
      sendAudioChunk(ev.data);
    };

    src.connect(workletNode);
    workletNode.connect(audioCtx.destination);

    document.getElementById('mic-ring').classList.add('active');
    log('Microphone active. Streaming…', 'ok');
    setStatus('streaming', 'Live');

  } catch (err) {
    log(`Mic error: ${err}`, 'error');
  }
}

function useFallbackProcessor() {
  // ScriptProcessor fallback (deprecated but widely supported)
  const bufSize = 4096;
  const src = audioCtx.createMediaStreamSource(mediaStream);
  scriptNode = audioCtx.createScriptProcessor(bufSize, 1, 1);
  let accumBuf = new Float32Array(0);

  scriptNode.onaudioprocess = (ev) => {
    const input  = ev.inputBuffer.getChannelData(0);
    const merged = new Float32Array(accumBuf.length + input.length);
    merged.set(accumBuf);
    merged.set(input, accumBuf.length);
    accumBuf = merged;

    while (accumBuf.length >= FRAME_SIZE) {
      sendAudioChunk(accumBuf.slice(0, FRAME_SIZE));
      accumBuf = accumBuf.slice(FRAME_SIZE);
    }
  };

  src.connect(scriptNode);
  scriptNode.connect(audioCtx.destination);
  log('Using ScriptProcessor fallback.', 'warn');
}

function sendAudioChunk(float32chunk) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  // Convert float32 → int16
  const int16 = new Int16Array(float32chunk.length);
  for (let i = 0; i < float32chunk.length; i++) {
    int16[i] = Math.max(-32768, Math.min(32767, float32chunk[i] * 32768));
  }
  // Pack: [0x01][int16 bytes]
  const buf = new Uint8Array(1 + int16.byteLength);
  buf[0] = MSG_AUDIO;
  buf.set(new Uint8Array(int16.buffer), 1);
  ws.send(buf);
}

// ── Stop ──────────────────────────────────────────────────────────────────────
function stopAll() {
  if (workletNode) { workletNode.disconnect(); workletNode = null; }
  if (scriptNode)  { scriptNode.disconnect();  scriptNode  = null; }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  if (audioCtx)    { audioCtx.close(); audioCtx = null; }
  if (ws)          { ws.close(); ws = null; }
  document.getElementById('mic-ring').classList.remove('active');
  setStatus('connected', 'Stopped');
  log('Session stopped.', 'warn');
  showOverlay(true);
  setIdle('Stopped', 'Click Start Session to restart', false);
}

// ── Status polling ────────────────────────────────────────────────────────────
setInterval(async () => {
  if (!sessionId) return;
  try {
    const r = await fetch(`/session/${sessionId}/status`);
    if (!r.ok) return;
    const d = await r.json();
    updateQueueBar('audio', d.audio_queue,  8);
    updateQueueBar('token', d.token_queue, 16);
    updateQueueBar('emb',   d.emb_queue,   32);
    updateQueueBar('frame', d.frame_queue, 30);
    updateQueueBar('send',  d.send_queue,  60);
  } catch (_) {}
}, 2000);

// ── Event wiring ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initQueueBars();
  initVisBars();

  const uploadZone = document.getElementById('upload-zone');
  const imgInput   = document.getElementById('img-input');

  uploadZone.addEventListener('click', () => imgInput.click());
  imgInput.addEventListener('change', (ev) => {
    const file = ev.target.files[0];
    if (!file) return;
    // Preview
    const url = URL.createObjectURL(file);
    uploadZone.innerHTML = `<img src="${url}" />`;
    uploadZone.classList.add('has-image');
    uploadImage(file);
  });

  // Drag-and-drop support
  uploadZone.addEventListener('dragover', e => { e.preventDefault(); });
  uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      imgInput.dispatchEvent(Object.assign(new Event('change'), { target: { files: [file] } }));
    }
  });

  document.getElementById('btn-start').disabled = false;

  document.getElementById('btn-start').addEventListener('click', async () => {
    await startSession();
    connectWS();
  });

  document.getElementById('btn-mic').addEventListener('click', startMic);
  document.getElementById('btn-stop').addEventListener('click', stopAll);
});
