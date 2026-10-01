# ImageUp AI — Image Upscaler

An image upscaler that runs **real AI super-resolution in the browser**, built with
**plain HTML5, CSS3 and vanilla JavaScript**. No React, no Vue, no Tailwind, no Bootstrap,
no jQuery, no build step, no backend, no API key.

**User flow:** Upload image → choose 2× / 4× / 8× → upscale → compare before/after → download.

The upscaling is done by **Real-ESRGAN** (`realesr-general-x4v3`), a trained super-resolution
neural network, executed with **ONNX Runtime Web** on the visitor's own machine — WebGPU when
the browser offers it, multi-threaded WebAssembly otherwise. The image is never uploaded.

---

## Contents

1. [Quick start](#quick-start)
2. [Project structure](#project-structure)
3. [Features](#features)
4. [How the AI works](#how-the-ai-works)
5. [Performance](#performance)
6. [File size](#file-size)
7. [Making it faster: cross-origin isolation](#making-it-faster-cross-origin-isolation)
8. [Deploying](#deploying)
9. [Configuration reference](#configuration-reference)
10. [Using a server-side API instead](#using-a-server-side-api-instead)
11. [Privacy](#privacy)
12. [Limits and known constraints](#limits-and-known-constraints)
13. [Motion](#motion)
14. [Accessibility](#accessibility)
15. [Browser support](#browser-support)
16. [Code map](#code-map)
17. [Credits and licences](#credits-and-licences)

---

## Quick start

> **You must serve the folder over `http://` or `https://`.**
> Double-clicking `index.html` will *not* work any more: browsers block `fetch()` on
> `file://` URLs, so the page cannot read the model file. The app detects this and shows a
> clear message rather than failing silently.

```bash
# Node (no framework, no install)
npx serve .

# or Python 3
python -m http.server 8000
```

Then open <http://localhost:8000>.

**First run** downloads about 10 MB — roughly 5.5 MB of ONNX Runtime from the CDN and the
4.6 MB model that ships with this project. Both are cached by the browser afterwards, so
every later visit starts instantly.

The download starts on its own as soon as the page goes idle, so it is usually finished
before a visitor has chosen a file. Both parts report real progress while they run — see
[Why the wait is visible](#why-the-wait-is-visible).

There is nothing to install, compile, or configure. No API key, no account, no server.

---

## Project structure

```text
ai-image-upscaler/
│
├── index.html                          # All markup: header, hero, upscaler tool
├── css/
│   └── style.css                       # Design tokens, components, animations, media queries
├── js/
│   ├── upscale-worker.js               # The AI engine — runtime, model, tiling, inference
│   ├── ai-upscaler.js                  # Controller — owns the worker, converts pixels
│   └── app.js                          # The interface — upload, options, compare, download
├── assets/
│   ├── models/
│   │   ├── realesr-general-x4v3.onnx   # The super-resolution network (4.6 MB)
│   │   └── LICENSE-Real-ESRGAN.md      # BSD-3-Clause, upstream licence
│   ├── images/                         # Static images (screenshots, og-image, samples)
│   └── icons/                          # favicon.svg + standalone SVG icons
└── README.md
```

The three JavaScript files are deliberately separate. `upscale-worker.js` runs on its own
thread and touches no DOM at all. `ai-upscaler.js` owns that worker and knows nothing about
the interface beyond canvases. `app.js` never talks to the model directly.

---

## Features

**Upload**

- Click to browse, drag & drop, or paste from the clipboard (Ctrl/Cmd + V)
- Accepts JPG, JPEG, PNG and WEBP, up to 30 MB
- Validates MIME type, file size and that the file really decodes as an image
- Friendly toast errors: *Unsupported Image*, *File Too Large*, *Invalid Image*

**Upscale**

- 2× / 4× / 8× selectable cards (2× is the default) with full keyboard support
- Output resolution **and an estimated running time** update instantly on every change
- The model starts downloading as soon as you pick an image, so pressing Upscale does not wait
- **Real progress**: tile *n* of *m*, a percentage that means something, and a live ETA
- **Cancel** stops a long job cleanly and leaves your image loaded

**Compare & download**

- Draggable before/after slider — mouse, touch and keyboard (arrows / Home / End)
- Zoom controls (100 % → 400 %), with the divider staying aligned while zoomed
- Result card: original size, upscaled size, scale factor, format, file sizes, and which
  backend actually ran (`Real-ESRGAN ×4 · WebGPU` or `· WASM ×4`)
- **Large downloads land at a size set per level: about 17 MB at 2×, 51 MB at 4×, 100 MB at
  8× (± 10%).** A PNG written by the app itself: stored uncompressed when it is no bigger than
  that (its size is then known before the job runs), and otherwise reduced only as far as it
  takes to land near the target — losslessly, unless no PNG of the image can get that small,
  when it is a full-resolution JPEG at the highest quality that does
- **The file size is measured and shown before you download**, not after
- Filenames like `photo-4x-upscaled.png` (`.jpg` in the JPEG case)
- `Upscale Another Image` resets everything back to the empty upload state

**Everything else**

- Dark and light themes, chosen automatically from the OS setting — there is no switch
- Fully responsive: desktop, tablet and mobile layouts
- Smooth in-page scrolling and motion throughout: panels rise as they swap, cards and result
  figures arrive in sequence, the headline's gradient drifts on a slow loop, the comparison
  divider glides on keyboard steps, buttons give way under a press, and a newly measured file
  size fades in
- Respects `prefers-reduced-motion` — every one of those is neutralised, the busy indicators
  deliberately excepted

---

## How the AI works

### The model

`realesr-general-x4v3` is the compact general-purpose variant of **Real-ESRGAN**
(Wang et al., 2021) — an SRVGGNetCompact generator trained to reconstruct plausible detail
when enlarging an image 4×. It is 4.6 MB of weights, which is what makes it practical to ship
to a browser. It takes `[1, 3, H, W]` float32 in the 0–1 range and returns `[1, 3, H*4, W*4]`.

This is genuinely different from resizing. Bicubic or Lanczos interpolation averages the
pixels you already have, so edges stay soft; the network was trained on millions of
degraded/clean image pairs and *reconstructs* edges and texture. The difference is obvious on
type, thin lines and hard boundaries.

**The shipped file has nine bytes corrected.** The upstream export declares the graph's output
with the same symbolic dimension names as its input (`batch_size, 3, height, width` for both),
which tells ONNX Runtime the output is the same size as the input. The WebGPU backend believes
it, allocates an input-sized buffer, and then refuses to run:

    Shape mismatch attempting to re-use buffer. {1,90,120,3} != {1,360,480,3}

Renaming the output's two dimensions to distinct symbols lets the runtime size the output at
run time instead of guessing. No weights were touched, and both backends produce identical
numbers from the corrected file. `assets/models/LICENSE-Real-ESRGAN.md` records the change and
how to reproduce it.

### Scale mapping

The network only knows one factor, 4×. The three buttons map onto it like this:

| Button | What runs | Network passes |
| --- | --- | --- |
| 2× | one 4× pass, halved to 2× as each tile is written | 1 |
| 4× | one 4× pass | 1 |
| 8× | a 4× pass, then a second 4× pass whose tiles are halved to 8× | 2 |

Running the model and reducing gives a visibly better 2× than any plain resize, because the
detail is reconstructed first and then resampled. The 8× path is expensive — the second pass
processes 16× as many pixels as the first — which is why the interface says it is two passes
and estimates the time before you commit.

**The reduction happens per tile, not at the end.** Building the whole 4× image and shrinking
it afterwards would mean holding four times the pixels of the actual result in one buffer — a
2× of a 5000 × 3000 photo needed 916 MB that way, which starves the runtime and makes ordinary
tiles fail to allocate. Halving each tile as it comes out of the network means the oversized
image never exists. Peak memory dropped 4× for 2× jobs and 3.4× for 8× jobs, and the result is
identical: for an exact 2:1 reduction, averaging the 2 × 2 block *is* the correct resample, and
tile cores are always an even number of pixels on an even offset, so no block straddles a seam.

### Tiling

Whole images are not fed to the network. They are cut into 128 × 128 tiles, each read with an
8-pixel margin of surrounding context. The network runs on the padded tile, the margin is
trimmed off the output, and only the core is pasted into the result. That context is what
makes tile boundaries invisible — measured across a tile edge, the pixel difference is
indistinguishable from any neighbouring pair of columns.

Tiling also keeps peak memory low and gives the progress bar something honest to count.

### Why the wait is visible

The first run has to fetch about 10 MB before a single pixel can be processed, and on a slow
connection that is a long time to look at a page that says nothing. Measured cold, the setup
once took **58 seconds, 57.5 of them in a single phase with an indeterminate bar** — the point
where ONNX Runtime quietly downloads its 5.5 MB WebAssembly binary during session creation.

That download is now fetched by `upscale-worker.js` itself, streamed, with progress reported
per chunk. It is discarded immediately — the point is only to fill the browser cache, so the
runtime's own request for the same URL is served locally and returns at once. The same phase
now produces around 300 progress updates instead of none, and everything after it is short.

Three things keep the page honest while it works:

- every phase that *can* be counted is counted — the engine download, the model download, and
  then tile *n* of *m* with a live estimate;
- a running clock is appended to the detail line, so even the one genuinely uncountable step
  (compiling the model) visibly ticks;
- the whole download starts when the page goes idle, not when Upscale is pressed, so in normal
  use it is already finished by the time anyone clicks.

### Why it does not freeze the page

Inference is a long stretch of blocking work. Run on the page's own thread it locks up
scrolling, animation and clicks for as long as each tile takes — over a second per tile on the
WebAssembly path.

So the entire job lives in `js/upscale-worker.js`, on its own thread: loading the runtime,
fetching the model, tiling, inference and compositing. The page and the worker exchange exactly
two large buffers per job — the source pixels in, the finished pixels out — and both are
**transferred** rather than copied, so the hand-off costs nothing. Everything in between is
small progress messages.

Measured during a 27-second job: the worst main-thread frame gap was **47 ms**, with zero gaps
over 100 ms across 1613 frames. Throughput was unchanged from running it on the page's thread.

ONNX Runtime has a built-in `proxy` mode that also moves inference to a worker. It was tried
and measured **2.6× slower**, because it crosses the thread boundary once per `run()` call with
a copy each way. Owning the worker and keeping the whole tile loop inside it avoids that.

### Backends

On session creation the worker asks for a WebGPU adapter. If it gets one, inference runs on the
GPU; otherwise it falls back to WebAssembly, multi-threaded where the page allows it. The result
card always names the backend that actually ran.

There are two fallbacks, because there are two ways a GPU can let you down. One is refusing the
model outright, which shows up when the session is built. The other is accepting the model,
building a session, and only failing once inference starts — a driver-level problem the visitor
can do nothing about. So the worker keeps the model file in memory after loading: if a run
fails on the GPU for any reason other than memory pressure, it rebuilds the session on the CPU
and starts the job again. It takes several times longer, and the result card changes to
`WASM`, but an image comes out.

Genuine out-of-memory errors are *not* retried — the CPU would fail the same way, slower.

---

## Performance

Measured on this project against the same model and the same tiles, on an AMD RDNA2 GPU with
4 CPU cores. Throughput counts **source** pixels per second:

| Backend | Throughput | 800 × 600 at 4× | 1920 × 1080 at 4× |
| --- | --- | --- | --- |
| WebGPU | ~210 000 px/s | ~2.3 s | ~10 s |
| WebAssembly, 4 threads (cross-origin isolated) | ~46 000 px/s | ~10 s | ~45 s |
| WebAssembly, 1 thread (default hosting) | ~13 000 px/s | ~37 s | ~2.7 min |

WebGPU is roughly **4.6× faster than four WebAssembly threads**, and it is the path almost
every visitor will take. A 600 × 450 image at 2× finishes in about 1.7 seconds end to end.

Tile size barely moves throughput on either backend — between 96 and 512 pixels it varied by
about 13% on the GPU and under 10% on WebAssembly. 128 is kept because the gains above it are
small while the costs are not: a 256-pixel tile takes ~310 ms on the GPU against ~98 ms, which
means a third as many progress updates and a correspondingly slower response to **Cancel**.

The app measures its own throughput as it runs, remembers it in `localStorage`, and uses it to
estimate the next job. The first estimate on a new device is a guess; the second is accurate.

Two things dominate the cost: **source pixels** (not output pixels) and **number of passes**.
A 4000 × 3000 photo at 4× is 12 megapixels of input — several minutes even on the fast path.
The interface shows the estimate above the Upscale button, and Cancel is always available.

---

## File size

The app saves **PNG**, written by the app itself rather than through `canvas.toBlob`, and
brings large files to a size set per upscale level (`targetOutputBytes`, ± `outputBandFraction`):

| Level | Target | Band |
| --- | --- | --- |
| 2× | **17 MB** | 15–19 MB |
| 4× | **51 MB** | 46–56 MB |
| 8× | **100 MB** | 90–110 MB |

There is no format menu and no quality slider (see [The size targets](#the-size-targets)).

### Why the PNG is written by hand

`canvas.toBlob` gives no control over compression. A compressed PNG of a 4000 × 3000 result
came out around **15 MB** against **46 MB** of actual pixels — fine in general, but useless when
a specific file size is the requirement, and impossible to predict in advance because a PNG's
size depends on the content of the picture.

`encodePngStored()` in `js/app.js` writes the file directly, storing the pixels through
deflate's **stored** block type, which copies bytes verbatim instead of compressing them. That
is an ordinary part of the zlib format, so the output is a perfectly normal PNG — every decoder
reads it and the pixels are byte-identical to a compressed one. It is simply not squeezed.

Two things follow. The file is as large as the pixels genuinely are, and its size is
arithmetic rather than a guess:

    63 bytes of framing  +  ceil(raw / 65535) × 5  +  raw
    where raw = (1 + width × 4) × height

The 63 is the signature, IHDR, the IDAT length, tag and CRC, the zlib header and Adler-32,
and IEND. The five bytes a block costs are deflate's stored-block header, and 65535 is the
most one such block may carry.

The formula is exact, not approximate: `pngByteLength()` and the file `encodePngStored()`
actually writes agree to the byte at every size tested. That is what lets the line above the
Upscale button read *"about 30 seconds · 45.8 MB PNG"* before the job has run.

It is worth saying what the formula is *not* measured against. Node's `zlib.deflateSync(…,
{ level: 0 })` produces a slightly **larger** stream past the first block — it splits stored
blocks well below the 65535 ceiling, so it pays the five-byte header more often (90 bytes more
on a 1.2 MB stream). Both are valid; ours simply fills each block. The check that matters is
the other direction: `zlib.inflateSync()` reads what this encoder writes and returns the
scanlines byte for byte.

### Reaching a target size

Since an uncompressed pixel costs four bytes, a target maps straight onto a pixel count —
**40 MB is 10.5 MP, 50 MB is 13.1 MP**. These are the uncompressed sizes. Anything past the
top of its level's band (19 MB at 2×, 56 MB at 4×, 110 MB at 8×) is brought to about that
level's target instead (see below), so for the larger entries the file you get is about
17, 51 or 100 MB rather than what the table says:

| Source | 2× | 4× | 8× |
| --- | --- | --- | --- |
| 600 × 450 | 4.1 MB | 16.5 MB | 65.9 MB |
| **1000 × 750** | 11.4 MB | **45.8 MB** | 183.1 MB |
| 1200 × 900 | 16.5 MB | 65.9 MB | 263.7 MB |
| **2000 × 1500** | **45.8 MB** | 183.1 MB | 732.5 MB |

### How it is verified

The encoder is checked by decoding its own output back through the browser and comparing
against the source canvas. Both sizes that matter are covered: a small file of two deflate
blocks, and a 2400 × 1800 one of **264 blocks and 4.3 million pixels**, which is what actually
exercises the block splitting. Both report a maximum channel difference of **zero**, across all
four channels. Chunk structure, the PNG signature, the IHDR fields and the zlib header are
asserted too.

Encoding runs at roughly 52 MB/s, so a 45.8 MB file takes under a second. It reads the result's
bytes directly rather than pulling them back out of a canvas.

### The size targets

Large results are brought to about the level's `targetOutputBytes` — 17 MB at 2×, 51 MB at 4×,
100 MB at 8× — give or take `outputBandFraction` (10 %) of it, and reduced only as far as that
takes. `encodeOutput()` tries, in order (figures below are for 4×, band 46–56 MB):

1. **PNG, stored whole** — when it is no bigger than 56 MB as it is. Exactly as before, and the
   size is shown before the job runs. A result smaller than 46 MB also stays its real size:
   nothing is padded to look bigger.
2. **PNG, stored whole without the alpha channel** — when the image has no transparency and
   leaving that channel out (a quarter of the bytes) lands it in 46–56 MB. Still uncompressed.
3. **PNG, partly compressed** — `encodePngHybrid()` stores the top rows as they are and
   compresses the rest (Paeth filter, then the browser's own raw deflate). Deflate allows stored
   and compressed blocks in one stream, so moving that one boundary sets the size anywhere
   between fully compressed and fully stored, and the file stays lossless. The split is worked
   out from a sample of about 4 % of the rows (`sampleCompressionRatio()`,
   `storedRowsFor()`); if the file misses the band, the real ratio of the rows actually
   compressed sets a second, closer split.
4. **JPEG, full resolution, at the highest quality under 56 MB** — only when no PNG of the image
   can get that small. Whole percentages are searched, so it lands as close under the top as the
   format allows.

Measured end to end in the browser, downloading the real file. These were taken when every
level shared one 50 MB target (45–55 MB), before the per-level targets:

| 4× result | Uncompressed | Download |
| --- | --- | --- |
| 1080 × 1080 → 4320 × 4320 | 71.2 MB | **53.4 MB** PNG, uncompressed, alpha left out — pixel-identical |
| 1300 × 875 → 5200 × 3500 (wallpaper photo) | 69.4 MB | **52.1 MB** PNG, uncompressed, alpha left out — pixel-identical |
| 1000 × 750 → 4000 × 3000 | 45.8 MB | **45.8 MB** PNG, unchanged |
| 1500 × 1000 → 6000 × 4000, pure noise | 91.6 MB | **30.3 MB** JPEG at 99 % |

Partly compressed files, checked in Node against an independent decoder, landed at 49.7–50.0 MB
for results of 57 MB to 105 MB, every pixel identical. Pure random noise is the one input that
cannot reach the band: no PNG of it gets under 55 MB, and JPEG jumps from 30 MB at 99 % to over
55 MB at 100 %, so it gets the highest quality that fits.

The interface says which one you got: the result card shows the format and *Lossless ·
uncompressed*, *Lossless · partly compressed* or *JPEG · 99 % quality*, and the file-size hint
explains why. When the file will be brought to the target, the line above the Upscale button
reads *file about 51 MB* (or 17 / 100 MB, by level) instead of an exact figure.

The JPEG needs a canvas as large as the image, so it is only attempted up to
`maxStoredBytes` (512 MB of pixels) — the memory the job is already planned with. Past that,
or if the browser cannot make a canvas that large, the smallest lossless PNG is kept even though
it is over the band, and the hint says so; a lower upscale level then gives a smaller file.

**A larger file is not a better picture.** The pixels are identical to a compressed PNG; only
the storage differs, and the interface says so rather than letting the number imply otherwise.
The one honest way to add real detail is a higher upscale level, which adds real pixels.

---

## Making it faster: cross-origin isolation

Multi-threaded WebAssembly needs `SharedArrayBuffer`, which browsers only expose on a
**cross-origin isolated** page. That single change was worth **3.6×** in the table above.

Serve the site with these two headers:

```http
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

The page is already written for it — the ONNX Runtime `<script>` tag uses
`crossorigin="anonymous"` so it still loads under `require-corp`, and `ai-upscaler.js` checks
`self.crossOriginIsolated` and raises the thread count by itself. Nothing else to change.

Netlify (`_headers`), Vercel (`vercel.json`), Cloudflare Pages and nginx can all set these.
If you cannot set headers, the site still works — just single-threaded.

---

## Deploying

It is a static folder. Any static host works: GitHub Pages, Netlify, Vercel, Cloudflare Pages,
S3, nginx, Apache.

Two things to check on your host:

1. **`.onnx` is served as a binary file.** Most hosts do this correctly. If yours returns
   `text/html` for unknown extensions, add a MIME mapping for `.onnx`
   (`application/octet-stream` is fine).
2. **`assets/models/` is actually uploaded.** Some deploy pipelines skip large binaries.

If you would rather not depend on a CDN at all, download the ONNX Runtime `dist` files and
point `CONFIG.ortBase` in `js/ai-upscaler.js` at your own copy. You need `ort.webgpu.min.js`
plus the `ort-wasm-simd-threaded.jsep.*` files from the same version.

---

## Configuration reference

**`js/upscale-worker.js`** — the engine:

| Option | Default | What it does |
| --- | --- | --- |
| `ortBase` | jsDelivr CDN URL | Where ONNX Runtime Web is loaded from |
| `modelUrl` | `../assets/models/realesr-general-x4v3.onnx` | The network weights, relative to the worker |
| `modelScale` | `4` | The factor the network natively produces |
| `tileSize` | `128` | Tile edge in source pixels. Throughput is nearly flat from 96 to 512 on both backends, so this is chosen for responsive progress and Cancel rather than speed |
| `tileOverlap` | `8` | Context margin per tile; raise it if you ever see seams |
| `maxThreads` | `4` | WASM thread ceiling on a cross-origin-isolated page |
| `defaultThroughput` | `25000` | px/s assumed before anything has been measured (in `ai-upscaler.js`) |

**`js/app.js`** — the interface:

| Option | Default | What it does |
| --- | --- | --- |
| `engine` | `'local-ai'` | `'local-ai'` runs the model here; `'api'` posts to your backend |
| `apiEndpoint` | `'/api/upscale'` | Only used when `engine` is `'api'` |
| `maxFileSize` | `30 * 1024 * 1024` | Upload limit in bytes |
| `maxPeakBytes` | `2048 * 1024 * 1024` | Ceiling on the memory one job may hold, checked before starting. Halved on devices that report less through `navigator.deviceMemory` |
| `targetOutputBytes` | `{ 2: 17 MB, 4: 51 MB, 8: 100 MB }` | The size a large download is brought to at each upscale level; files no bigger than the band's top are left as they are |
| `outputBandFraction` | `0.1` | How far from the target, as a share of it, still counts as there (± 10 %: 15–19, 46–56, 90–110 MB) |
| `maxStoredBytes` | `512 * 1024 * 1024` | Memory set aside for the file while it is built; the JPEG fallback is not attempted past it |
| `slowJobSeconds` | `45` | Above this estimate, the time hint turns amber |
| `defaultScale` | `2` | Scale selected on load and after a reset |

Colors, radii, shadows and spacing are CSS custom properties at the top of `css/style.css`
(`:root` for dark, `@media (prefers-color-scheme: light)` for light). Change `--brand-1` /
`--brand-2` / `--brand-3` to rebrand the whole site.

---

## Using a server-side API instead

Running the model on the visitor's device is free, private and needs no backend — but it uses
their CPU/GPU, and a big image takes real time. If you would rather do the work server-side
(a bigger model, a GPU box, or a paid provider), the seam is already there.

### 1. Point the frontend at your own endpoint

```js
// js/app.js
const CONFIG = {
  engine: 'api',                     // was 'local-ai'
  apiEndpoint: '/api/upscale',       // your proxy, not the provider
  …
};
```

`upscaleWithApi()` already POSTs a `FormData` body with `image` (the original file) and
`scale` (2, 4 or 8), and expects an image back as the response body. It decodes that into the
same raw-pixel shape the local engine returns, so the UI needs no changes.

### 2. Write the proxy

**Never put a private API key in frontend JavaScript.** Anything shipped to the browser can be
read by anyone — view-source, DevTools, the network tab. A leaked key means someone else
spends your quota. The key must live on a server:

```js
// server/upscale-proxy.js  —  run with: node server/upscale-proxy.js
import http from 'node:http';

const API_KEY = process.env.UPSCALE_API_KEY;      // never in frontend code
const PROVIDER_URL = 'https://api.example-upscaler.com/v1/upscale';

http.createServer(async (req, res) => {
  if (req.method !== 'POST' || !req.url.startsWith('/api/upscale')) {
    res.writeHead(404).end();
    return;
  }

  const upstream = await fetch(PROVIDER_URL, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${API_KEY}`,
      'Content-Type': req.headers['content-type']
    },
    body: req,
    duplex: 'half'
  });

  if (!upstream.ok) { res.writeHead(upstream.status).end(); return; }
  res.writeHead(200, { 'Content-Type': upstream.headers.get('content-type') });
  res.end(Buffer.from(await upstream.arrayBuffer()));
}).listen(3000);
```

The same shape works as a serverless function or a few lines of PHP. Add a request size limit,
a per-IP rate limit, an allow-list of scales, and a check that the upload really is an image.

Any HTTP API that takes an image and returns an image fits: Replicate, Stability AI, Clipdrop,
Deep Image, or your own Real-ESRGAN instance behind FastAPI. If a provider returns a job id to
poll instead of an image, put the polling loop inside `upscaleWithApi()` — nothing outside
that function needs to know.

**If you switch to `'api'`, you have to add a disclosure yourself.** The page used to carry a
note under the tool reading *"Your image is never uploaded — there is no server"*, which flipped
to `LIVE API` and an upload warning whenever `engine` was `'api'`. That note has since been
removed from the interface, so nothing on the page now tells a visitor where their image goes.

Nothing currently makes a false claim, because nothing makes any claim at all. But sending
someone's photo to a third party without saying so is its own problem, and it is the kind of
thing privacy law in several places treats as mandatory. Put a visible line near the upload
control before you connect an external service.

---

## Privacy

With the default `engine: 'local-ai'`, the image never leaves the browser tab. It is read with
`URL.createObjectURL()`, decoded with `createImageBitmap()`, cut into tiles, run through the
network in WebAssembly or WebGPU, and encoded back to a `Blob` for download. There is no
server, no upload, no analytics and no storage. Closing the tab discards everything.

The only network requests the page makes are for the ONNX Runtime files (from the CDN) and the
model file (from your own host). Neither carries any part of your image.

This is no longer stated anywhere in the interface — the note that described it was removed —
so this README is the only place that records it. If you want visitors to know, add it back.

Object URLs are revoked as soon as they are no longer needed, and intermediate canvases are
released (`canvas.width = 0`) so large images do not pile up in memory.

---

## Limits and known constraints

- **Must be served over http/https.** `file://` cannot fetch the model. The app detects this
  and says so rather than failing silently.
- **Upload:** 30 MB, JPG / JPEG / PNG / WEBP.
- **Output: no canvas ceiling.** The result is held as raw bytes, never in a canvas, so the
  browser's canvas limits do not decide what can be produced. Those limits are real and
  awkward — Chrome stops at about 268 megapixels of area and 65,535 pixels a side, Firefox at
  about 124 MP and 32,767, Safari at 16,384 — and while the result lived in a canvas they set
  the ceiling. A 2560 × 1708 photo at 8× is 20480 × 13664, which is 280 MP: past Chrome's area
  limit by a few percent, and refused for that reason alone even though the machine had room
  for it. Nothing downstream ever needed the canvas. The file is written from the bytes, and
  the comparison view gets its own reduced copy.
- **What is left is memory**, which is the honest limit: an 8× job holds its 4× intermediate
  and its final result at once, plus the file it writes. That total is estimated before
  starting and checked against `maxPeakBytes`, trimmed to the device where
  `navigator.deviceMemory` reports one.
- **A refusal names the level that fits.** "Try a lower upscale level" left people guessing
  which one, and on a large photo two of the three can be out of reach. The message reads
  *"… would need about 1.87 GB of memory to build, more than this device can be relied on for.
  4× (10240 × 6832) will work."* — worked out by testing each smaller level against the same
  limits.
- **Time, not just size, is the real limit.** Large sources take minutes on the WebAssembly
  path. The estimate is shown before you start and Cancel is always available.
- **Memory is checked up front.** A job that would need more pixel memory than a browser tab can
  be relied on for is refused with the figure, rather than failing partway through and wasting
  the time already spent.
- **Mobile Safari** has far less memory. Large jobs may fail there even under the configured
  limits; the error toast covers this case.
- **The comparison view shows a reduced copy** of anything past 4096 pixels a side, because
  nothing bigger can be seen in a frame that size even zoomed in. It says so on the AFTER tag —
  *"preview at 1/5 size"* — rather than passing itself off as the full result. The download is
  always the full-size pixels.
- **First load needs internet** for the ONNX Runtime files. After that the browser cache
  covers it. Self-host the runtime if you need true offline use.
- **Export is PNG only.** Uploads still accept JPG, JPEG, PNG and WEBP; the download is always
  a PNG. If a browser cannot encode one at the requested size, the error says so rather than
  saving an empty file.
- **A GPU that fails mid-run is recovered from, not reported.** The job restarts on the CPU and
  the result card shows `WASM` instead of `WebGPU`. The only sign is that it took longer.

---

## Motion

All of it lives in section 17 of `css/style.css`, rather than scattered through the component
rules, so the page's movement can be read, retuned or deleted in one place. Three details are
easy to get wrong and worth recording.

**Fill mode is `backwards`, never `both`.** An entrance animation that holds its final frame
keeps `transform: none` applied at animation priority, which outranks the element's own rules —
silently killing the hover lift on every card it touches. `backwards` applies the opening frame
during the stagger delay and then hands the element back to its normal styles. Verified by
hovering a card through the DevTools protocol, so `:hover` genuinely applies: the card reports
`matrix(1, 0, 0, 1, 0, -2)`, the two-pixel lift, intact.

**The comparison divider eases, except while dragging.** Easing a pointer drag leaves the
divider trailing the cursor, which reads as broken rather than smooth. The frame already carries
`is-dragging` for that window, and the transition is switched off inside it — 0.2 s normally,
`0s` while dragging.

**The headline loop runs on `alternate`, not on repeat.** The gradient half of the hero title
is laid out at twice the headline's width and slides by exactly one width, nine seconds each
way. Playing it forward and then backwards is what removes the seam: the colours at the turn
are the ones already on screen, where a one-way loop has to jump from its last frame back to
its first, and that flicks however slowly it runs.

The easing has to be symmetric for the same reason, so this is the one place that does not use
`--ease`. That curve is an ease-out; played backwards it sets off fast from where the forward
run had just come to rest, and the mismatch shows as a tug at each turn. `--ease-inout` rests
at zero on both ends, so the speeds meet.

Checked by rendering the page at fixed points in the animation and reading the glyph colours
back out of the screenshots. "Quality" travels from violet to cyan across a sweep (a distance
of 115 in RGB), and frames one full round trip apart — 18.6 s and 36.6 s — come back to the
same colours within 2 of 255, which is the renderer's own run-to-run jitter at a fixed time
rather than drift. Under emulated `reduce` the sweep is frozen: seven seconds of animation
move it by nothing at all, and the headline falls back to its plain full-width gradient.

Any element that animates is checked to settle at full opacity. A value that updates many times
a second is never re-animated on every tick; `restartAnimation()` is called only when the text
genuinely changes, because restarting a fade continuously leaves it permanently mid-flight and
therefore invisible.

---

## Accessibility

- Semantic landmarks (`header`, `main`, `section`) and a skip link
- The uploader is a real `role="button"` with `tabindex="0"`, operable with Enter and Space;
  the native `<input type="file">` stays in the DOM for assistive tech
- The scale picker is a proper `role="radiogroup"` with arrow-key navigation and roving `tabindex`
- The progress bar is a real `role="progressbar"` with `aria-valuenow`, and it only goes
  indeterminate when progress genuinely cannot be measured
- The comparison divider is a `role="slider"` with `aria-valuenow` / `aria-valuetext`, movable
  with arrow keys, Home and End
- Errors are announced through `role="alert"`, status messages through `role="status"`
- Visible `:focus-visible` outlines everywhere; contrast meets WCAG AA in both themes
- `prefers-reduced-motion` disables decorative animation while keeping the loader legible

---

## Browser support

Chrome, Edge, Firefox and Safari — current versions and one back. WebGPU is used where
available (Chrome/Edge 113+, Safari 18+); everything else falls back to WebAssembly, which is
supported everywhere the rest of the app runs.

Internet Explorer is not supported.

---

## Code map

**`js/upscale-worker.js`** — the engine. Runs on its own thread, touches no DOM. Loads the
runtime, streams the model, plans and runs the tiles, and posts progress back. The message
protocol is documented at the top of the file.

**`js/ai-upscaler.js`** — the controller, no DOM knowledge beyond canvases. Owns the worker,
reads pixels out of the source canvas, hands the result back as raw bytes, and keeps the
throughput measurement.

Public surface: `load(onProgress)`, `upscale(source, w, h, scale, options)`, `getInfo()`,
`estimateSeconds(w, h, scale)`, `passesFor(scale)`, `isLoaded()`, `dispose()`.

**`js/app.js`** — the interface:

```text
01. Configuration            09. Drag & Drop             17. Canvas Encoding
02. DOM Elements             10. Clipboard Paste         18. Before/After Comparison
03. Application State        11. Image Validation        19. Zoom
04. Small Utilities          12. Metadata & Decoding     20. Download & file size
05. Toasts / Errors          13. Upscale Selection       21. Reset
06. Navigation & Scrolling   14. Output Resolution       22. Init
07. Panels (state machine)   15. Upscale Processing
08. File Upload              16. Upscaling Engine
```

The page scripts are wrapped so nothing leaks onto `window` except `ImageUpAI`. UI state lives in one
`state` object; runtime bookkeeping (drag state, cancellation flag, cached output blob) lives in
`runtime`.

`css/style.css` follows the same idea, with a table of contents at the top and all responsive
rules collected in section 18.

---

## Credits and licences

- **Real-ESRGAN** — Xintao Wang et al., [BSD-3-Clause](assets/models/LICENSE-Real-ESRGAN.md).
  Paper: *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data*
  ([arXiv:2107.10833](https://arxiv.org/abs/2107.10833)).
  The ONNX export of `realesr-general-x4v3` bundled here came from
  [Heliosoph/realesrgan-onnx](https://huggingface.co/Heliosoph/realesrgan-onnx) on Hugging Face,
  with a nine-byte correction to its output shape declaration so that WebGPU will run it —
  described in full in the licence file linked above.
- **ONNX Runtime Web** — Microsoft, MIT licence, loaded from jsDelivr.
- The site code itself: use it however you like.
