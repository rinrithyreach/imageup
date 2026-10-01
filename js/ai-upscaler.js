/* ==========================================================================
   ImageUp AI — ai-upscaler.js
   Real AI super-resolution, running on the user's own device.

   The model is Real-ESRGAN "general x4v3" (SRVGGNetCompact), a trained
   super-resolution network, executed with ONNX Runtime Web — WebGPU where the
   browser offers it (not on iPhone or iPad, see gpuAllowed), multi-threaded
   WebAssembly otherwise. The image never leaves the page.

   All of the heavy work happens in js/upscale-worker.js, on its own thread,
   so the interface never freezes while the network runs. This file is the
   controller: it owns the worker, converts between canvases and raw pixel
   buffers, and keeps a running measure of how fast this device actually is.

   It exposes one object, window.ImageUpAI:

     await ImageUpAI.load(onProgress)             download + prepare the model
     await ImageUpAI.upscale(src, w, h, scale, o) returns the upscaled pixels
     ImageUpAI.getInfo()                          model / backend details
     ImageUpAI.estimateSeconds(w, h, scale)       rough "this will take…"

   Sections
   01. Configuration
   02. Module state
   03. Utilities
   04. The worker
   05. Loading
   06. Upscaling
   07. Public API
   ========================================================================== */

window.ImageUpAI = (function () {
  'use strict';

  /* ========================================================================
     01. CONFIGURATION
     ======================================================================== */

  const CONFIG = {
    /** The worker that owns the runtime and the model. */
    workerUrl: 'js/upscale-worker.js',

    /** Mirrors of the worker's own settings, needed for planning and estimates. */
    modelName: 'Real-ESRGAN general x4v3',
    modelScale: 4,
    tileSize: 128,

    /** Fallback throughput (source pixels per second) before anything is measured. */
    defaultThroughput: 25000,
    throughputKey: 'imageup-throughput',

    /**
     * Set while a job runs on the graphics card and cleared when it ends.
     * Still set when the page next loads, it means the browser killed the
     * page mid-job, and gpuOffKey then keeps this device on the CPU.
     */
    gpuJobKey: 'imageup-gpu-job',
    gpuOffKey: 'imageup-gpu-off'
  };


  /* ========================================================================
     02. MODULE STATE
     ======================================================================== */

  const state = {
    worker: null,
    loadPromise: null,
    ready: false,
    backend: null,
    threads: 1,
    throughput: null,      // measured source px/s, remembered between visits
    gpuCrashNotice: false, // the last page died mid-GPU-job; told to the user once

    // Callbacks for whatever the worker is doing right now.
    onLoadProgress: null,
    onRunProgress: null,
    resolveLoad: null,
    rejectLoad: null,
    resolveRun: null,
    rejectRun: null
  };


  /* ========================================================================
     03. UTILITIES
     ======================================================================== */

  function makeCanvas(width, height) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas;
  }

  function releaseCanvas(canvas) {
    if (canvas && canvas.tagName === 'CANVAS') {
      canvas.width = 0;
      canvas.height = 0;
    }
  }

  function readThroughput() {
    if (state.throughput) return state.throughput;
    try {
      const stored = Number(window.localStorage.getItem(CONFIG.throughputKey));
      if (stored > 0) state.throughput = stored;
    } catch (err) { /* private mode */ }
    return state.throughput || CONFIG.defaultThroughput;
  }

  function writeThroughput(pixelsPerSecond) {
    if (!(pixelsPerSecond > 0)) return;
    // Smooth it, so one cold run does not dominate the estimate.
    const previous = state.throughput || pixelsPerSecond;
    state.throughput = previous * 0.6 + pixelsPerSecond * 0.4;
    try {
      window.localStorage.setItem(CONFIG.throughputKey, String(Math.round(state.throughput)));
    } catch (err) { /* private mode */ }
  }

  function fail(message, userMessage) {
    const error = new Error(message);
    error.userMessage = userMessage;
    return error;
  }

  /** localStorage, doing nothing where it is unavailable (private mode). */
  function storageGet(key) {
    try { return window.localStorage.getItem(key); } catch (err) { return null; }
  }
  function storageSet(key, value) {
    try { window.localStorage.setItem(key, value); } catch (err) { /* private mode */ }
  }
  function storageRemove(key) {
    try { window.localStorage.removeItem(key); } catch (err) { /* private mode */ }
  }

  /**
   * iPhone, iPod or iPad, in any browser — all of them are Safari underneath.
   * An iPad asks for desktop sites and calls itself a Mac, but a Mac has no
   * touch screen.
   */
  function isAppleMobile() {
    const agent = navigator.userAgent;
    return /iPhone|iPad|iPod/.test(agent) ||
      (/Macintosh/.test(agent) && navigator.maxTouchPoints > 1);
  }

  /**
   * Whether the graphics card may be used.
   *
   * Not on iPhone or iPad. Safari there offers WebGPU and builds the session
   * on it, but running the model gets the page closed by iOS partway through
   * the tiles ("This webpage was reloaded because a problem occurred") — a
   * crash, not an error, so the worker's own CPU fallback never gets the
   * chance to step in.
   *
   * Nor anywhere that has happened before: a GPU job still marked as running
   * when the page loads never finished, whatever the browser.
   */
  function gpuAllowed() {
    return !isAppleMobile() && storageGet(CONFIG.gpuOffKey) !== '1';
  }

  function checkForGpuCrash() {
    if (storageGet(CONFIG.gpuJobKey) === null) return;
    storageRemove(CONFIG.gpuJobKey);
    storageSet(CONFIG.gpuOffKey, '1');
    state.gpuCrashNotice = true;
  }

  /** Pulls raw RGBA out of any drawable source. */
  function readPixels(source, width, height) {
    let canvas = source;
    let temporary = null;

    if (!source || source.tagName !== 'CANVAS') {
      temporary = makeCanvas(width, height);
      temporary.getContext('2d').drawImage(source, 0, 0, width, height);
      canvas = temporary;
    }

    const imageData = canvas.getContext('2d', { willReadFrequently: true })
      .getImageData(0, 0, width, height);

    if (temporary) releaseCanvas(temporary);
    return imageData;
  }


  /* ========================================================================
     04. THE WORKER
     ======================================================================== */

  function createWorker() {
    if (state.worker) return state.worker;

    if (typeof window.Worker !== 'function') {
      throw fail('Web Workers unavailable',
        'This browser cannot run the AI model. Try a recent version of Chrome, Edge, Firefox or Safari.');
    }

    let worker;
    try {
      worker = new Worker(CONFIG.workerUrl);
    } catch (err) {
      // file:// refuses to start workers at all, which is the usual cause.
      if (window.location.protocol === 'file:') {
        throw fail('Worker blocked on file://',
          'Open this page over http:// — browsers block local file access, so the model cannot load by double-clicking index.html. The README explains how.');
      }
      throw fail('Worker failed to start: ' + err.message,
        'Could not start the AI engine. Reload the page and try again.');
    }

    worker.onmessage = handleWorkerMessage;
    worker.onerror = (event) => {
      const error = window.location.protocol === 'file:'
        ? fail('Worker error on file://',
            'Open this page over http:// — browsers block local file access, so the model cannot load by double-clicking index.html. The README explains how.')
        : fail('Worker error: ' + (event.message || 'unknown'),
            'The AI engine stopped unexpectedly. Reload the page and try again.');
      settleLoad(null, error);
      settleRun(null, error);
    };

    state.worker = worker;
    return worker;
  }

  function settleLoad(value, error) {
    const resolve = state.resolveLoad;
    const reject = state.rejectLoad;
    state.resolveLoad = null;
    state.rejectLoad = null;
    if (error && reject) reject(error);
    else if (!error && resolve) resolve(value);
  }

  function settleRun(value, error) {
    const resolve = state.resolveRun;
    const reject = state.rejectRun;
    state.resolveRun = null;
    state.rejectRun = null;
    state.onRunProgress = null;
    if (error && reject) reject(error);
    else if (!error && resolve) resolve(value);
  }

  function handleWorkerMessage(event) {
    const message = event.data;

    switch (message.type) {
      case 'load-progress':
        if (state.onLoadProgress) {
          state.onLoadProgress({ phase: message.phase, ratio: message.ratio, download: message.download });
        }
        break;

      case 'ready':
        state.ready = true;
        state.backend = message.backend;
        state.threads = message.threads;
        settleLoad(true, null);
        break;

      case 'progress':
        if (state.onRunProgress) state.onRunProgress(message);
        break;

      case 'done':
        settleRun(message, null);
        break;

      case 'error': {
        const error = message.cancelled
          ? fail('Cancelled by user', null)
          : fail(message.message, message.userMessage);
        if (message.cancelled) error.cancelled = true;
        // A failure before 'ready' is a load failure; after it, a run failure.
        if (state.rejectLoad) {
          state.loadPromise = null;
          settleLoad(null, error);
        }
        settleRun(null, error);
        break;
      }

      default:
        break;
    }
  }


  /* ========================================================================
     05. LOADING
     ======================================================================== */

  /**
   * Starts the worker, downloads the runtime and the model, and builds the
   * inference session. Safe to call repeatedly — the work happens once.
   *
   * @param {(info: {phase: string, ratio: number}) => void} [onProgress]
   */
  function load(onProgress) {
    if (state.ready) return Promise.resolve(true);

    state.onLoadProgress = onProgress || state.onLoadProgress;

    if (state.loadPromise) return state.loadPromise;

    state.loadPromise = new Promise((resolve, reject) => {
      state.resolveLoad = resolve;
      state.rejectLoad = reject;
      try {
        createWorker().postMessage({ type: 'init', allowGpu: gpuAllowed() });
      } catch (err) {
        state.resolveLoad = null;
        state.rejectLoad = null;
        reject(err);
      }
    });

    state.loadPromise.catch(() => { state.loadPromise = null; });
    return state.loadPromise;
  }


  /* ========================================================================
     06. UPSCALING
     ======================================================================== */

  /** How many network passes a requested scale needs. */
  function passesFor(scale) {
    return scale > CONFIG.modelScale ? 2 : 1;
  }

  /** Total source pixels the network will look at for this job. */
  function workPixels(width, height, scale) {
    const first = width * height;
    if (passesFor(scale) === 1) return first;
    return first + first * CONFIG.modelScale * CONFIG.modelScale;
  }

  /** A rough "this will take about N seconds", refined by past runs. */
  function estimateSeconds(width, height, scale) {
    return workPixels(width, height, scale) / readThroughput();
  }

  /**
   * Largest amount of pixel memory the job will hold at once, in bytes.
   *
   * 2x and 4x hold one buffer: the result. 8x needs two passes, so the 4x
   * intermediate is alive while the 8x result is being filled. The source
   * pixels are alive for the first pass too.
   *
   * Worth checking before starting: running out halfway wastes whatever time
   * has already been spent.
   */
  function estimatePeakBytes(width, height, scale) {
    const source = width * height * 4;
    const result = (width * scale) * (height * scale) * 4;

    if (passesFor(scale) === 1) return source + result;

    const intermediate = (width * CONFIG.modelScale) * (height * CONFIG.modelScale) * 4;
    return intermediate + result;
  }

  /**
   * Upscales an image with the neural network.
   *
   * @param {CanvasImageSource} source  ImageBitmap, HTMLImageElement or canvas
   * @param {number} width              source width in pixels
   * @param {number} height             source height in pixels
   * @param {2|4|8} scale
   * @param {{onProgress?: Function, onLoadProgress?: Function, isCancelled?: () => boolean}} [options]
   * @returns {Promise<{rgba: Uint8ClampedArray, width: number, height: number}>}
   *          raw RGBA at exactly width*scale x height*scale
   */
  async function upscale(source, width, height, scale, options) {
    const opts = options || {};
    await load(opts.onLoadProgress);

    if (state.resolveRun) {
      throw fail('A job is already running', 'One image is already being upscaled.');
    }

    const imageData = readPixels(source, width, height);

    // Let the caller stop a job that is already in the worker.
    let cancelPoll = null;
    if (opts.isCancelled) {
      cancelPoll = setInterval(() => {
        if (opts.isCancelled() && state.worker) {
          state.worker.postMessage({ type: 'cancel' });
        }
      }, 120);
    }

    // Marked for as long as the graphics card has the job, so a page the
    // browser kills partway through is recognised when it comes back.
    const onGpu = state.backend === 'webgpu';
    if (onGpu) storageSet(CONFIG.gpuJobKey, '1');

    let result;
    try {
      result = await new Promise((resolve, reject) => {
        state.resolveRun = resolve;
        state.rejectRun = reject;
        state.onRunProgress = opts.onProgress || null;

        // The pixel buffer is transferred, not copied — the hand-off is free.
        state.worker.postMessage({
          type: 'run',
          rgba: imageData.data.buffer,
          width: width,
          height: height,
          scale: scale
        }, [imageData.data.buffer]);
      });
    } finally {
      if (cancelPoll) clearInterval(cancelPoll);
      if (onGpu) storageRemove(CONFIG.gpuJobKey);
    }

    if (result.spentMs > 0) {
      writeThroughput(result.pixelsDone / (result.spentMs / 1000));
    }

    // The worker already reduced to the requested scale, so these pixels are
    // final. They are handed back as raw bytes rather than in a canvas: a
    // browser canvas stops at roughly 268 megapixels in Chrome and 124 in
    // Firefox, which an 8x job passes easily, and putting the result in one
    // would cap what the app can produce for no gain. Nothing downstream
    // needs a canvas — the file is written from these bytes and the on-screen
    // preview is drawn from them too.
    //
    // The buffer came across by transfer, so wrapping it costs nothing.
    return {
      rgba: new Uint8ClampedArray(result.rgba),
      width: result.width,
      height: result.height
    };
  }


  /* ========================================================================
     07. PUBLIC API
     ======================================================================== */

  function getInfo() {
    return {
      modelName: CONFIG.modelName,
      modelScale: CONFIG.modelScale,
      backend: state.backend,
      threads: state.threads,
      loaded: state.ready,
      throughput: Math.round(readThroughput())
    };
  }

  function isLoaded() {
    return state.ready;
  }

  function dispose() {
    if (state.worker) {
      state.worker.terminate();
      state.worker = null;
    }
    state.ready = false;
    state.loadPromise = null;
    return Promise.resolve();
  }

  /** True once, on the first call after a page that died mid-GPU-job. */
  function takeCrashNotice() {
    const notice = state.gpuCrashNotice;
    state.gpuCrashNotice = false;
    return notice;
  }

  checkForGpuCrash();

  // Closing or leaving the page mid-job is not a crash.
  window.addEventListener('pagehide', () => storageRemove(CONFIG.gpuJobKey));

  return {
    CONFIG: CONFIG,
    load: load,
    upscale: upscale,
    isLoaded: isLoaded,
    getInfo: getInfo,
    estimateSeconds: estimateSeconds,
    estimatePeakBytes: estimatePeakBytes,
    passesFor: passesFor,
    isAppleMobile: isAppleMobile,
    takeCrashNotice: takeCrashNotice,
    dispose: dispose
  };
})();
