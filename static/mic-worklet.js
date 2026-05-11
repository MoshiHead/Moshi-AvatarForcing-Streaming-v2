/**
 * mic-worklet.js — AudioWorklet processor for low-latency mic capture.
 *
 * Accumulates microphone samples at exactly FRAME_SIZE=1920 boundaries
 * and posts Float32Array chunks to the main thread for WebSocket transmission.
 *
 * This file must be served at /static/mic-worklet.js.
 */

class MicProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this._frameSize = (options.processorOptions && options.processorOptions.frameSize) || 1920;
    this._buf = new Float32Array(0);
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const ch     = input[0];                         // Float32Array, 128 samples
    const merged = new Float32Array(this._buf.length + ch.length);
    merged.set(this._buf);
    merged.set(ch, this._buf.length);
    this._buf = merged;

    while (this._buf.length >= this._frameSize) {
      const chunk = this._buf.slice(0, this._frameSize);
      this._buf   = this._buf.slice(this._frameSize);
      this.port.postMessage(chunk, [chunk.buffer]);
    }

    return true;  // keep processor alive
  }
}

registerProcessor('mic-processor', MicProcessor);
