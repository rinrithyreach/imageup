/* ==========================================================================
   ImageUp AI — upscale-worker.js
   The whole super-resolution job, off the main thread.

   Neural network inference is a long stretch of blocking work. Run on the
   page's thread it freezes scrolling, animation and clicks for as long as
   each tile takes. So everything lives here instead: loading the runtime,
   fetching the model, tiling, inference, and compositing the result.

   The page and this worker exchange exactly two large buffers per job —
   the source pixels in, the finished pixels out — and both are *transferred*
   rather than copied, so the hand-off costs nothing. Everything between is
   small progress messages.

   Protocol
     page → worker   {type:'init', allowGpu}           load runtime + model
     page → worker   {type:'run', rgba, width, height, scale}
     page → worker   {type:'cancel'}
     worker → page   {type:'load-progress', phase, ratio, download}
     worker → page   {type:'ready', backend, threads}
     worker → page   {type:'progress', tilesDone, tilesTotal, pass, passes, secondsLeft}
     worker → page   {type:'done', rgba, width, height, spentMs, pixelsDone}
     worker → page   {type:'error', message, userMessage}
   ========================================================================== */

/* global ort */
'use strict';

const CONFIG = {
  ortBase: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',

  /**
   * The two builds of the runtime. The WebGPU one also runs on the CPU, and
   * is what a CPU fallback mid-job uses; the CPU-only one is loaded when the
   * graphics card is not going to be used at all — half the download, and
   * half as much for the browser to compile and hold.
   *
   * `wasm` is the runtime's WebAssembly binary — by far the largest download,
   * and the one the user waits on. ONNX Runtime fetches it itself during
   * session creation, which would leave a long silent gap, so it is fetched
   * here first with progress. That warms the HTTP cache and the runtime's
   * own request is then served from it. `wasmBytes` is its uncompressed size,
   * for the progress bar; `download` is what actually crosses the network.
   */
  gpuBuild: {
    script: 'ort.webgpu.min.js',
    wasm: 'ort-wasm-simd-threaded.asyncify.wasm',   // what ort.webgpu.min.js loads as of 1.30
    wasmBytes: 26781914,
    download: '5.5 MB'
  },
  cpuBuild: {
    script: 'ort.wasm.min.js',
    wasm: 'ort-wasm-simd-threaded.wasm',
    wasmBytes: 14239897,
    download: '3.1 MB'
  },

  /**
   * Real-ESRGAN general x4v3. This copy has nine bytes changed against the
   * upstream export — the output declaration reused the input's dimension
   * names, which made WebGPU refuse to run it. No weights were touched;
   * assets/models/LICENSE-Real-ESRGAN.md records exactly what changed.
   */
  modelUrl: '../assets/models/realesr-general-x4v3.onnx',
  modelScale: 4,
  modelBytes: 4871181,
  tileSize: 128,
  tileOverlap: 8,
  maxThreads: 4
};

let session = null;
let backend = null;
let threads = 1;
let initPromise = null;
let cancelled = false;

/**
 * The model file is kept after loading so the session can be rebuilt on the
 * CPU without downloading anything again. Some GPUs accept the model, build a
 * session, and only then fail to actually run it — see recoverOnCpu().
 */
let modelBytes = null;
let triedCpuFallback = false;

/** Set by the page: false keeps the graphics card out of it entirely. */
let gpuAllowed = true;

const post = (message, transfers) => self.postMessage(message, transfers || []);
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

function fail(message, userMessage) {
  const error = new Error(message);
  error.userMessage = userMessage;
  return error;
}


/* --------------------------------------------------------------------------
   Loading the runtime and the model
   -------------------------------------------------------------------------- */

/**
 * importScripts() is a no-cors request, which a cross-origin-isolated page
 * (COEP: require-corp) refuses. Fetching the source with CORS and running it
 * from a same-origin blob URL works in both cases.
 */
async function loadRuntime(build) {
  const url = CONFIG.ortBase + build.script;

  try {
    self.importScripts(url);
    if (self.ort) return;
  } catch (err) { /* blocked — fall through to the CORS path */ }

  try {
    const response = await fetch(url, { mode: 'cors' });
    if (!response.ok) throw new Error('HTTP ' + response.status);
    const source = await response.text();
    const blobUrl = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
    try {
      self.importScripts(blobUrl);
    } finally {
      URL.revokeObjectURL(blobUrl);
    }
  } catch (err) {
    throw fail('ONNX Runtime failed to load: ' + err.message,
      'Could not load the AI runtime. Check your internet connection and reload — it is only downloaded once.');
  }

  if (!self.ort) {
    throw fail('ONNX Runtime did not define ort',
      'The AI runtime loaded but did not start. Try reloading the page.');
  }
}

/**
 * Downloads the runtime's WebAssembly binary so the wait is visible.
 *
 * The bytes are streamed and discarded — the point is purely to fill the
 * browser cache, so that when ONNX Runtime fetches the same URL during
 * session creation it is served locally and returns immediately.
 *
 * Entirely best-effort: if anything goes wrong the runtime just downloads it
 * the way it always did, and the only loss is the progress bar.
 */
async function prefetchRuntimeBinary(build, onProgress) {
  try {
    const response = await fetch(CONFIG.ortBase + build.wasm, { mode: 'cors' });
    if (!response.ok || !response.body || !response.body.getReader) return;

    // Compressed in transit, so Content-Length understates it; the
    // uncompressed size is what the reader will actually hand us.
    const total = build.wasmBytes;
    const reader = response.body.getReader();
    let received = 0;

    for (;;) {
      const step = await reader.read();
      if (step.done) break;
      received += step.value.length;
      if (onProgress) onProgress(Math.min(received / total, 1));
    }
  } catch (err) {
    /* best-effort — session creation will fetch it normally */
  }
}

async function detectWebGPU() {
  if (!self.navigator || !self.navigator.gpu) return false;
  try {
    return !!(await self.navigator.gpu.requestAdapter());
  } catch (err) {
    return false;
  }
}

/** Streams the model so the first-run download can show real progress. */
async function fetchModel() {
  let response;
  try {
    response = await fetch(CONFIG.modelUrl);
  } catch (err) {
    throw fail('Model fetch failed: ' + err.message,
      'Could not load the AI model file. Check that assets/models/ was uploaded with the site.');
  }

  if (!response.ok) {
    throw fail('Model fetch returned ' + response.status,
      'The AI model file could not be found (HTTP ' + response.status + ').');
  }

  const total = Number(response.headers.get('content-length')) || CONFIG.modelBytes;

  if (!response.body || !response.body.getReader) {
    const buffer = await response.arrayBuffer();
    post({ type: 'load-progress', phase: 'model', ratio: 1 });
    return new Uint8Array(buffer);
  }

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;

  for (;;) {
    const step = await reader.read();
    if (step.done) break;
    chunks.push(step.value);
    received += step.value.length;
    post({ type: 'load-progress', phase: 'model', ratio: Math.min(received / total, 1) });
  }

  const bytes = new Uint8Array(received);
  let offset = 0;
  for (let i = 0; i < chunks.length; i += 1) {
    bytes.set(chunks[i], offset);
    offset += chunks[i].length;
  }
  return bytes;
}

function init() {
  if (initPromise) return initPromise;

  initPromise = (async function () {
    // Decided first, because it picks which build of the runtime to load.
    const useGpu = gpuAllowed && await detectWebGPU();
    const build = useGpu ? CONFIG.gpuBuild : CONFIG.cpuBuild;
    const progress = (ratio) =>
      post({ type: 'load-progress', phase: 'runtime', ratio: ratio, download: build.download });

    progress(0);
    await loadRuntime(build);

    // The big one. Fetched here rather than silently inside session creation.
    await prefetchRuntimeBinary(build, progress);
    progress(1);

    ort.env.wasm.wasmPaths = CONFIG.ortBase;
    ort.env.wasm.proxy = false;              // we already are the worker
    ort.env.logLevel = 'error';
    threads = (self.crossOriginIsolated && self.navigator.hardwareConcurrency)
      ? Math.min(CONFIG.maxThreads, self.navigator.hardwareConcurrency)
      : 1;
    ort.env.wasm.numThreads = threads;

    post({ type: 'load-progress', phase: 'model', ratio: 0 });
    const bytes = await fetchModel();
    modelBytes = bytes;
    post({ type: 'load-progress', phase: 'session', ratio: 0 });

    const providers = useGpu ? ['webgpu', 'wasm'] : ['wasm'];
    try {
      session = await ort.InferenceSession.create(bytes, {
        executionProviders: providers,
        graphOptimizationLevel: 'all'
      });
      backend = useGpu ? 'webgpu' : 'wasm';
    } catch (err) {
      // A GPU that advertises itself but cannot compile the graph is not rare.
      if (!useGpu) {
        throw fail('Session creation failed: ' + err.message,
          'This browser could not start the AI model. Try a recent version of Chrome, Edge, Firefox or Safari.');
      }
      try {
        session = await ort.InferenceSession.create(bytes, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all'
        });
        backend = 'wasm';
      } catch (innerErr) {
        throw fail('Session creation failed: ' + innerErr.message,
          'This browser could not start the AI model. Try a recent version of Chrome, Edge, Firefox or Safari.');
      }
    }

    post({ type: 'load-progress', phase: 'session', ratio: 1 });
    post({ type: 'ready', backend: backend, threads: threads });
  })();

  initPromise.catch(() => { initPromise = null; });
  return initPromise;
}


/* --------------------------------------------------------------------------
   Tiling
   -------------------------------------------------------------------------- */

/**
 * Core tiles, each with a margin of real surrounding pixels. The network sees
 * the margin for context; it is trimmed off before the tile is written, which
 * is what makes the boundaries invisible.
 */
function planTiles(width, height, tileSize, overlap) {
  const tiles = [];

  for (let y = 0; y < height; y += tileSize) {
    for (let x = 0; x < width; x += tileSize) {
      const coreW = Math.min(tileSize, width - x);
      const coreH = Math.min(tileSize, height - y);
      const padLeft = Math.min(overlap, x);
      const padTop = Math.min(overlap, y);
      const padRight = Math.min(overlap, width - (x + coreW));
      const padBottom = Math.min(overlap, height - (y + coreH));

      tiles.push({
        coreX: x, coreY: y, coreW: coreW, coreH: coreH,
        readX: x - padLeft, readY: y - padTop,
        readW: coreW + padLeft + padRight,
        readH: coreH + padTop + padBottom,
        padLeft: padLeft, padTop: padTop
      });
    }
  }
  return tiles;
}

function countTiles(width, height, tileSize) {
  return Math.ceil(width / tileSize) * Math.ceil(height / tileSize);
}

/** RGBA region → planar RGB float32 in 0..1, straight out of the big buffer. */
function readTile(rgba, sourceWidth, x, y, width, height) {
  const plane = width * height;
  const out = new Float32Array(plane * 3);

  for (let row = 0; row < height; row += 1) {
    let src = ((y + row) * sourceWidth + x) * 4;
    let dst = row * width;
    for (let col = 0; col < width; col += 1, src += 4, dst += 1) {
      out[dst] = rgba[src] / 255;
      out[plane + dst] = rgba[src + 1] / 255;
      out[plane * 2 + dst] = rgba[src + 2] / 255;
    }
  }
  return out;
}

/**
 * Planar RGB float32 → RGBA bytes, writing only the tile's core into the
 * destination. Uint8ClampedArray clamps out-of-range values for us.
 */
function writeTile(rgba, destWidth, data, outWidth, outHeight, cropX, cropY, cropW, cropH, destX, destY) {
  const plane = outWidth * outHeight;

  for (let row = 0; row < cropH; row += 1) {
    let src = (cropY + row) * outWidth + cropX;
    let dst = ((destY + row) * destWidth + destX) * 4;
    for (let col = 0; col < cropW; col += 1, src += 1, dst += 4) {
      rgba[dst] = data[src] * 255;
      rgba[dst + 1] = data[plane + src] * 255;
      rgba[dst + 2] = data[plane * 2 + src] * 255;
      rgba[dst + 3] = 255;
    }
  }
}

/**
 * The same, but halving as it writes — each 2x2 block of the network's output
 * becomes one destination pixel.
 *
 * This is what makes 2x and 8x affordable. The network only produces 4x, so a
 * 2x result used to mean building the whole 4x image and shrinking it at the
 * end: four times the pixels of the result, held in one buffer. Reducing here
 * means the oversized image never exists — only the tile does.
 *
 * For an exact 2:1 reduction, averaging the block *is* the correct resample,
 * so nothing is lost by doing it per tile. Tile cores are always an even
 * number of output pixels and start on even offsets, so no block ever
 * straddles two tiles.
 */
function writeTileHalved(rgba, destWidth, data, outWidth, outHeight, cropX, cropY, cropW, cropH, destX, destY) {
  const plane = outWidth * outHeight;
  const plane2 = plane * 2;
  const halfW = cropW >> 1;
  const halfH = cropH >> 1;

  for (let row = 0; row < halfH; row += 1) {
    const top = (cropY + row * 2) * outWidth + cropX;
    const bottom = top + outWidth;
    let dst = ((destY + row) * destWidth + destX) * 4;

    for (let col = 0; col < halfW; col += 1, dst += 4) {
      const a = top + col * 2;
      const b = a + 1;
      const c = bottom + col * 2;
      const d = c + 1;

      // 255 / 4 — average the block and scale to bytes in one step.
      rgba[dst] = (data[a] + data[b] + data[c] + data[d]) * 63.75;
      rgba[dst + 1] = (data[plane + a] + data[plane + b] + data[plane + c] + data[plane + d]) * 63.75;
      rgba[dst + 2] = (data[plane2 + a] + data[plane2 + b] + data[plane2 + c] + data[plane2 + d]) * 63.75;
      rgba[dst + 3] = 255;
    }
  }
}

/** Allocates a pixel buffer, or explains in plain words why it could not. */
function allocate(bytes, whatFor) {
  try {
    return new Uint8ClampedArray(bytes);
  } catch (err) {
    throw fail('Could not allocate ' + bytes + ' bytes for the ' + whatFor,
      'This image needs about ' + Math.round(bytes / 1048576) + ' MB of memory to hold the ' +
      whatFor + ', and this device could not provide it. Try a lower upscale level, or a smaller image.');
  }
}


/* --------------------------------------------------------------------------
   Inference
   -------------------------------------------------------------------------- */

const CANCELLED = 'IMAGEUP_CANCELLED';

/**
 * Turns a failure inside session.run() into an error that says what actually
 * went wrong.
 *
 * Running out of memory is only one reason inference can fail, and reporting
 * every failure as memory pressure sends people hunting the wrong problem —
 * retrying with ever smaller images against a fault that has nothing to do
 * with size. A graphics driver refusing the model is the other common cause,
 * and unlike memory it is recoverable: see recoverOnCpu().
 */
function inferenceFailure(err, tileNumber, tileCount) {
  const detail = String((err && err.message) || err);
  const outOfMemory = /out of memory|failed to allocate|allocation failed|OOM/i.test(detail);

  const error = fail(
    'Inference failed on tile ' + tileNumber + ' of ' + tileCount + ': ' + detail,
    outOfMemory
      ? 'The AI model ran out of memory partway through this image. ' +
        'Try a lower upscale level, or a smaller image.'
      : 'The AI model could not finish this image. Reload the page and try again — ' +
        'if it keeps happening, try a different browser.'
  );
  error.inference = true;
  error.outOfMemory = outOfMemory;
  return error;
}

/**
 * Rebuilds the session on the CPU after the graphics card failed mid-job.
 *
 * A driver that compiles the model and then cannot execute it is not rare,
 * and there is nothing the user can do about it. The model file is still in
 * memory, so this costs a second or two and the job simply starts again —
 * several times slower, but it finishes instead of failing.
 */
async function recoverOnCpu() {
  triedCpuFallback = true;

  if (session && session.release) {
    try { await session.release(); } catch (err) { /* already gone */ }
  }

  session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all'
  });
  backend = 'wasm';
  post({ type: 'ready', backend: backend, threads: threads, fellBack: true });
}

/**
 * Runs the network across one image.
 * @param {1|2} reduce  1 keeps the network's 4x output, 2 halves it to 2x as
 *                      each tile is written (see writeTileHalved).
 */
async function runPass(rgba, width, height, context, reduce) {
  const scale = CONFIG.modelScale;
  const outWidth = (width * scale) / reduce;
  const outHeight = (height * scale) / reduce;
  const out = allocate(outWidth * outHeight * 4, reduce === 1 ? 'result' : 'reduced result');

  const tiles = planTiles(width, height, CONFIG.tileSize, CONFIG.tileOverlap);
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  for (let i = 0; i < tiles.length; i += 1) {
    if (cancelled) throw new Error(CANCELLED);

    const tile = tiles[i];
    const input = readTile(rgba, width, tile.readX, tile.readY, tile.readW, tile.readH);

    const started = Date.now();
    let result;
    try {
      result = await session.run({
        [inputName]: new ort.Tensor('float32', input, [1, 3, tile.readH, tile.readW])
      });
    } catch (err) {
      throw inferenceFailure(err, i + 1, tiles.length);
    }
    context.spentMs += Date.now() - started;
    context.pixelsDone += tile.readW * tile.readH;

    const tensor = result[outputName];
    const write = reduce === 1 ? writeTile : writeTileHalved;
    write(
      out, outWidth,
      tensor.data, tensor.dims[3], tensor.dims[2],
      tile.padLeft * scale, tile.padTop * scale,
      tile.coreW * scale, tile.coreH * scale,
      (tile.coreX * scale) / reduce, (tile.coreY * scale) / reduce
    );

    context.tilesDone += 1;
    postProgress(context);

    // Yields the worker's event loop so a cancel message can arrive.
    await tick();
  }

  return { rgba: out, width: outWidth, height: outHeight };
}

function postProgress(context) {
  let secondsLeft = null;
  if (context.pixelsDone && context.spentMs) {
    const rate = context.pixelsDone / (context.spentMs / 1000);
    const left = context.pixelsTotal - context.pixelsDone;
    secondsLeft = left > 0 ? left / rate : 0;
  }

  post({
    type: 'progress',
    tilesDone: context.tilesDone,
    tilesTotal: context.tilesTotal,
    ratio: context.tilesTotal ? context.tilesDone / context.tilesTotal : 0,
    pass: context.pass,
    passes: context.passes,
    secondsLeft: secondsLeft
  });
}

/**
 * Everything the job needs to report progress, sized up front so the bar and
 * the time estimate are meaningful from the very first tile.
 */
function makeContext(message, passes) {
  const sourcePixels = message.width * message.height;

  return {
    tilesDone: 0,
    tilesTotal: countTiles(message.width, message.height, CONFIG.tileSize) +
      (passes === 2
        ? countTiles(message.width * CONFIG.modelScale, message.height * CONFIG.modelScale, CONFIG.tileSize)
        : 0),
    pixelsDone: 0,
    pixelsTotal: passes === 2
      ? sourcePixels + sourcePixels * CONFIG.modelScale * CONFIG.modelScale
      : sourcePixels,
    spentMs: 0,
    pass: 1,
    passes: passes
  };
}

/**
 * Runs the whole job.
 *
 * The network only does 4x, so each requested scale is a plan of passes. The
 * halving happens inside the last pass rather than afterwards, so the
 * oversized image is never held in memory:
 *   2x -> one pass, halved   (4x / 2)
 *   4x -> one pass
 *   8x -> two passes, the second halved   (4x then 16x / 2)
 *
 * The source buffer is only ever read, so this can safely be called twice —
 * which is what the CPU rescue in run() relies on.
 */
async function runPlan(message, context) {
  let current = new Uint8ClampedArray(message.rgba);
  let width = message.width;
  let height = message.height;

  const plan = context.passes === 2 ? [1, 2] : [message.scale === 2 ? 2 : 1];

  for (let pass = 0; pass < plan.length; pass += 1) {
    context.pass = pass + 1;
    const result = await runPass(current, width, height, context, plan[pass]);
    // Reassigning drops the last reference to the previous buffer, so the
    // intermediate can be collected as soon as this pass is done reading it.
    current = result.rgba;
    width = result.width;
    height = result.height;
  }

  return { rgba: current, width: width, height: height };
}

async function run(message) {
  await init();

  const passes = message.scale > CONFIG.modelScale ? 2 : 1;
  let context = makeContext(message, passes);
  postProgress(context);

  let result;
  try {
    result = await runPlan(message, context);
  } catch (err) {
    // A graphics card that accepted the model but cannot run it is worth one
    // retry on the CPU. Anything else — cancellation, genuine memory
    // pressure, a CPU run that already failed — is reported as it happened.
    const worthRetrying = err.inference && !err.outOfMemory &&
      backend === 'webgpu' && !triedCpuFallback && !cancelled;
    if (!worthRetrying) throw err;

    await recoverOnCpu();
    context = makeContext(message, passes);
    postProgress(context);
    result = await runPlan(message, context);
  }

  post({
    type: 'done',
    rgba: result.rgba.buffer,
    width: result.width,
    height: result.height,
    spentMs: context.spentMs,
    pixelsDone: context.pixelsDone
  }, [result.rgba.buffer]);
}


/* --------------------------------------------------------------------------
   Message handling
   -------------------------------------------------------------------------- */

self.onmessage = async function (event) {
  const message = event.data;

  if (message.type === 'cancel') {
    cancelled = true;
    return;
  }

  if (message.type === 'init') {
    if (message.allowGpu === false) gpuAllowed = false;
    try {
      await init();
    } catch (err) {
      post({ type: 'error', message: err.message, userMessage: err.userMessage });
    }
    return;
  }

  if (message.type === 'run') {
    cancelled = false;
    try {
      await run(message);
    } catch (err) {
      if (err && err.message === CANCELLED) {
        post({ type: 'error', message: CANCELLED, userMessage: null, cancelled: true });
      } else {
        post({ type: 'error', message: err.message, userMessage: err.userMessage });
      }
    } finally {
      cancelled = false;
    }
  }
};
