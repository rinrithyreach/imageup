/* ==========================================================================
   ImageUp AI — app.js
   Vanilla JavaScript. No libraries, no build step.

   Sections
   01. Configuration
   02. DOM Elements
   03. Application State
   04. Small Utilities
   05. Toasts / Error Handling
   06. Navigation & Smooth Scrolling
   07. Panels (UI state machine)
   08. File Upload
   09. Drag & Drop
   10. Clipboard Paste
   11. Image Validation
   12. Image Metadata & Decoding
   13. Upscale Selection
   14. Output Resolution
   15. Upscale Processing (UI flow)
   16. Upscaling Engine  <-- swap the engine here
   17. Canvas Encoding Helpers
   18. Before/After Comparison
   19. Zoom
   20. Download
   21. Reset
   22. Init
   ========================================================================== */

(function () {
  'use strict';

  /* ========================================================================
     01. CONFIGURATION
     ======================================================================== */

  const CONFIG = {
    /**
     * Which upscaling engine to use.
     *   'local-ai' → a real super-resolution network running on this device
     *                (Real-ESRGAN via ONNX Runtime Web — see js/ai-upscaler.js)
     *   'api'      → POST the image to your own backend/proxy (see section 17)
     */
    engine: 'local-ai',

    /** Only used when engine is 'api'. Never call a key-protected API from here. */
    apiEndpoint: '/api/upscale',

    /** Upload limits. */
    maxFileSize: 30 * 1024 * 1024,           // 30MB
    acceptedTypes: ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'],
    acceptedExtensions: ['jpg', 'jpeg', 'png', 'webp'],

    /**
     * Output limits.
     *
     * There is only one now, and it is memory. The result is held as raw
     * bytes rather than in a canvas, so the browser's canvas ceilings — about
     * 268 megapixels in Chrome, 124 in Firefox, and a per-side cap that
     * differs again — no longer decide what can be produced. What decides it
     * is whether the device can hold the pixels while the job runs.
     *
     * Checked before starting, because failing halfway through a job that
     * takes minutes is the worst possible outcome.
     */
    maxPeakBytes: 2048 * 1024 * 1024,        // 2GB, trimmed on smaller devices

    /**
     * The same ceiling on iPhone and iPad, where it has to be far lower.
     *
     * iOS gives a Safari tab much less than 2GB, and past its limit it does
     * not fail an allocation the app could catch — it closes the page
     * ("This webpage was reloaded because a problem occurred"). No browser
     * on iOS reports the device's memory, so this one figure covers them all.
     */
    iosMaxPeakBytes: 1024 * 1024 * 1024,     // 1GB

    /**
     * Above this, the file is written with real compression instead of being
     * stored whole.
     *
     * Storing is what makes the size predictable and large, but it costs as
     * many bytes as the pixels themselves — and holding a second copy of a
     * gigabyte of pixels is exactly what turns a slow job into a failed one.
     * Below the threshold nothing changes; above it, compressing is the
     * difference between a file and no file.
     */
    maxStoredBytes: 512 * 1024 * 1024,       // 512MB

    /**
     * The size a large download is brought to, per upscale level: about its
     * targetOutputBytes, give or take outputBandFraction of it (2× 15–19 MB,
     * 4× 46–56 MB, 8× 90–110 MB).
     *
     * A file that is already no bigger than the top of the band is left
     * exactly as it was — stored uncompressed. A bigger one is reduced only
     * as far as it takes to land near the target, losslessly wherever that
     * can be done (see encodeOutput). A file smaller than the band stays its
     * real size: nothing is ever padded to look bigger.
     */
    targetOutputBytes: {
      2: 17 * 1024 * 1024,                   // 17MB
      4: 51 * 1024 * 1024,                   // 51MB
      8: 100 * 1024 * 1024                   // 100MB
    },
    outputBandFraction: 0.1,                 // ± 10%

    /** Above this estimate the button warns before starting a long job. */
    slowJobSeconds: 45,

    /** Upscale level selected when an image is first loaded. */
    defaultScale: 2
  };

  /**
   * The format the app saves.
   *
   * PNG is lossless, so the file holds exactly the pixels the network
   * produced, and every browser and image tool reads it. A JPEG is written
   * only when no PNG of the result can come down to the size band.
   */
  const OUTPUT_MIME = 'image/png';

  /** The largest payload a single deflate "stored" block can carry. */
  const DEFLATE_BLOCK_MAX = 65535;
  const OUTPUT_EXTENSION = 'png';

  /** JPEG qualities tried, best first, when only a lossy file will fit. */
  const JPEG_QUALITIES = [1, 0.95, 0.9, 0.85, 0.8, 0.72, 0.64, 0.55, 0.45, 0.35];

  /** Framing every PNG here carries: signature, IHDR, IDAT header and CRC, zlib header, Adler-32, IEND. */
  const PNG_FRAMING_BYTES = 63;

  /** An upscale level's size band, in bytes and as people read it ("about 51 MB", "46–56 MB"). */
  const outputBand = (scale) => {
    const target = CONFIG.targetOutputBytes[scale];
    const spread = Math.round(target * CONFIG.outputBandFraction);
    return { target: target, low: target - spread, high: target + spread };
  };
  const megabytes = (bytes) => Math.round(bytes / (1024 * 1024));
  const targetLabel = (scale) => megabytes(outputBand(scale).target) + ' MB';
  const bandLabel = (scale) => megabytes(outputBand(scale).low) + '–' + megabytes(outputBand(scale).high) + ' MB';

  const ZOOM_STEPS = [1, 1.25, 1.5, 2, 3, 4];

  /**
   * Largest side of the canvas used to *show* the result.
   *
   * The result itself can be far larger than any canvas, so the comparison
   * view is given its own copy. Even at full zoom the frame is only a few
   * thousand pixels across, so past this point there is nothing more to see —
   * and handing the compositor a gigabyte of pixels to squeeze into a box a
   * few hundred pixels wide costs memory for no visible gain.
   *
   * The download is written from the full-size pixels either way.
   */
  const PREVIEW_MAX_SIDE = 4096;


  /* ========================================================================
     02. DOM ELEMENTS
     ======================================================================== */

  const $ = (id) => document.getElementById(id);

  const el = {
    // Header / navigation
    header:        $('siteHeader'),

    // Tool shell
    tool:          $('tool'),

    // Panels
    panelUpload:     $('panelUpload'),
    panelEditor:     $('panelEditor'),
    panelProcessing: $('panelProcessing'),
    panelResult:     $('panelResult'),

    // Upload
    fileInput:     $('fileInput'),
    dropzone:      $('dropzone'),

    // Preview
    previewImg:    $('previewImg'),
    fileName:      $('fileName'),
    fileDims:      $('fileDims'),
    fileSize:      $('fileSize'),
    fileFormat:    $('fileFormat'),
    changeBtn:     $('changeBtn'),
    removeBtn:     $('removeBtn'),

    // Scale picker
    scaleGroup:    $('scaleGroup'),
    scaleButtons:  Array.prototype.slice.call(document.querySelectorAll('.scale')),
    outputRes:     $('outputRes'),
    outputEstimate:$('outputEstimate'),
    upscaleBtn:    $('upscaleBtn'),

    // Processing
    procMessage:   $('procMessage'),
    procDetail:    $('procDetail'),
    progress:      $('progress'),
    progressBar:   $('progressBar'),
    cancelBtn:     $('cancelBtn'),

    // Comparison
    baFrame:       $('baFrame'),
    baBefore:      $('baBefore'),
    baAfter:       $('baAfter'),
    baTagAfter:    $('baTagAfter'),
    baHandle:      $('baHandle'),

    // Zoom
    zoomIn:        $('zoomIn'),
    zoomOut:       $('zoomOut'),
    zoomReset:     $('zoomReset'),

    // Result info
    infoOrigDims:  $('infoOrigDims'),
    infoOrigSize:  $('infoOrigSize'),
    infoOutDims:   $('infoOutDims'),
    infoOutSize:   $('infoOutSize'),
    infoScale:     $('infoScale'),
    infoEngine:    $('infoEngine'),
    infoFormat:    $('infoFormat'),
    infoQuality:   $('infoQuality'),

    // Download
    downloadSizeValue: $('downloadSizeValue'),
    filesizeHint:  $('filesizeHint'),
    downloadBtn:   $('downloadBtn'),
    resetBtn:      $('resetBtn'),

    // Toasts
    toasts:        $('toasts')
  };


  /* ========================================================================
     03. APPLICATION STATE
     ======================================================================== */

  const state = {
    file: null,              // the File the user picked
    originalImage: null,     // ImageBitmap | HTMLImageElement used as draw source
    originalUrl: null,       // object URL for the preview / "before" layer
    resultPixels: null,      // Uint8ClampedArray, the upscaled result itself
    resultImage: null,       // <canvas> showing it, downscaled if it is huge
    previewFactor: 1,        // how much resultImage was reduced by, 1 = not at all
    originalWidth: 0,
    originalHeight: 0,
    resultWidth: 0,
    resultHeight: 0,
    scale: CONFIG.defaultScale,
    processing: false
  };

  /** Non-user-facing runtime bits kept out of the state object above. */
  const runtime = {
    comparePos: 50,          // 0-100, position of the comparison divider
    zoomIndex: 0,            // index into ZOOM_STEPS
    dragging: false,
    cancelled: false,        // set by the Cancel button mid-run
    startedAt: 0,            // when the current job began, for the elapsed clock
    elapsedTimer: null,
    procDetail: '',          // detail line without the clock appended
    procMessage: '',         // headline, tracked so it only animates on change
    outputBlob: null,        // the encoded result, kept so downloading is free
    outputKind: 'stored',    // how it was written: 'stored' | 'hybrid' | 'compressed' | 'jpeg'
    outputChannels: 4,       // 3 when the unused alpha channel was left out
    outputQuality: 1,        // JPEG quality, when it is one
    outputOverCap: false,    // true only if no file of it could come under the band's top
    sizeToken: 0,            // identifies the newest measurement, so a slow
                             // encode cannot overwrite a fresher one
    panelMorph: null,        // the running height animation between panels
  };


  /* ========================================================================
     04. SMALL UTILITIES
     ======================================================================== */

  const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

  /** Script-driven animations are not covered by the CSS motion override. */
  const prefersReducedMotion = () =>
    !!window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /** Hands the thread back long enough for the page to draw. */
  const nextFrame = () => new Promise((resolve) => {
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(() => resolve());
    else setTimeout(resolve, 0);
  });

  /**
   * Gives the page a turn — to draw, and to answer a drag or a click — in the
   * middle of long work, then carries on straight away.
   *
   * Not nextFrame(): a hidden tab stops animation frames and slows timers to
   * once a second, so work that yields through them all but stops the moment
   * the user switches tabs. A message to ourselves is not held back like that.
   */
  const yieldToPage = () => {
    if (window.scheduler && typeof window.scheduler.yield === 'function') {
      return window.scheduler.yield();
    }
    return new Promise((resolve) => {
      const channel = new MessageChannel();
      channel.port1.onmessage = () => {
        channel.port1.close();
        resolve();
      };
      channel.port2.postMessage(null);
    });
  };

  /**
   * Splits a long loop into slices short enough that the page never stutters.
   * Call tick() once per step: it yields only when the current slice has run
   * past its budget, so fast steps are not slowed by yielding after each one.
   */
  function timeSlicer(budgetMs) {
    const budget = budgetMs || 8;
    let sliceStart = performance.now();
    return async function tick() {
      if (performance.now() - sliceStart < budget) return;
      await yieldToPage();
      sliceStart = performance.now();
    };
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) return '0 KB';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(bytes < 10240 ? 1 : 0) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
  }

  const formatDimensions = (w, h) => w + ' × ' + h;

  function getExtension(name) {
    const match = /\.([a-z0-9]+)$/i.exec(name || '');
    return match ? match[1].toLowerCase() : '';
  }

  /** "photo.final.jpg" → "photo.final" */
  function getBaseName(name) {
    const clean = (name || 'image').replace(/[\\/:*?"<>|]+/g, '-').trim();
    const dot = clean.lastIndexOf('.');
    const base = dot > 0 ? clean.slice(0, dot) : clean;
    return base.replace(/\s+/g, '-').slice(0, 60) || 'image';
  }

  /** Human label for a file: JPEG / PNG / WEBP */
  function formatLabel(file) {
    const type = (file && file.type ? file.type : '').toLowerCase();
    if (type === 'image/jpeg' || type === 'image/jpg') return 'JPEG';
    if (type === 'image/png') return 'PNG';
    if (type === 'image/webp') return 'WEBP';
    const ext = getExtension(file && file.name);
    if (ext === 'jpg' || ext === 'jpeg') return 'JPEG';
    return ext ? ext.toUpperCase() : 'IMAGE';
  }

  /** Frees a canvas' memory: setting the size to 0 drops the backing store. */
  function releaseCanvas(canvas) {
    if (!canvas || canvas.tagName !== 'CANVAS') return;
    canvas.width = 0;
    canvas.height = 0;
  }

  function closeImage(image) {
    if (image && typeof image.close === 'function') image.close(); // ImageBitmap
  }

  function revoke(url) {
    if (url) URL.revokeObjectURL(url);
  }


  /* ========================================================================
     05. TOASTS / ERROR HANDLING
     ======================================================================== */

  const TOAST_ICONS = { error: '!', success: '✓', info: 'i' };

  /**
   * @param {string} title   short headline, e.g. "File Too Large"
   * @param {string} message friendly explanation
   * @param {'error'|'success'|'info'} [type]
   * @param {number} [duration] ms before auto-dismiss (0 keeps it until closed)
   */
  function showToast(title, message, type, duration) {
    const kind = type || 'error';
    const life = typeof duration === 'number' ? duration : (kind === 'error' ? 6500 : 4500);

    // Never stack more than three at once.
    while (el.toasts.children.length >= 3) {
      el.toasts.removeChild(el.toasts.firstElementChild);
    }

    const toast = document.createElement('div');
    toast.className = 'toast toast--' + kind;
    toast.setAttribute('role', kind === 'error' ? 'alert' : 'status');

    const icon = document.createElement('span');
    icon.className = 'toast__icon';
    icon.setAttribute('aria-hidden', 'true');
    icon.textContent = TOAST_ICONS[kind] || 'i';

    const content = document.createElement('div');
    content.className = 'toast__content';

    const heading = document.createElement('p');
    heading.className = 'toast__title';
    heading.textContent = title;
    content.appendChild(heading);

    if (message) {
      const body = document.createElement('p');
      body.className = 'toast__msg';
      body.textContent = message;
      content.appendChild(body);
    }

    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'toast__close';
    close.setAttribute('aria-label', 'Dismiss notification');
    close.innerHTML = '&times;';

    let timer = null;
    const dismiss = () => {
      if (timer) clearTimeout(timer);
      if (!toast.parentNode) return;
      if (toast.classList.contains('is-leaving')) return;
      toast.classList.add('is-leaving');

      const remove = () => { if (toast.parentNode) toast.parentNode.removeChild(toast); };
      if (typeof toast.animate !== 'function' || prefersReducedMotion()) {
        setTimeout(remove, 280);
        return;
      }

      // After fading out, the toast folds its height (and the stack gap)
      // away, so the ones below slide up instead of jumping.
      const height = toast.getBoundingClientRect().height;
      const gap = parseFloat(getComputedStyle(el.toasts).rowGap) || 0;
      setTimeout(() => {
        const fold = toast.animate([
          { height: height + 'px', marginBottom: '0px' },
          { height: '0px', marginBottom: -gap + 'px', paddingTop: '0px', paddingBottom: '0px', borderWidth: '0px' }
        ], { duration: 220, easing: 'cubic-bezier(0.22, 1, 0.36, 1)', fill: 'forwards' });
        fold.onfinish = remove;
        fold.oncancel = remove;
      }, 240);
    };

    close.addEventListener('click', dismiss);
    toast.appendChild(icon);
    toast.appendChild(content);
    toast.appendChild(close);
    el.toasts.appendChild(toast);

    if (life > 0) timer = setTimeout(dismiss, life);
    return dismiss;
  }


  /* ========================================================================
     06. SMOOTH SCROLLING
     ======================================================================== */

  /**
   * Smooth scrolling and focus management for in-page links.
   *
   * Two remain: the skip link that keyboard and screen-reader users land on
   * first, and the logo.
   */
  function initSmoothScrolling() {
    document.addEventListener('click', (event) => {
      const link = event.target.closest('a[href^="#"]');
      if (!link) return;

      const id = link.getAttribute('href').slice(1);
      const target = id ? document.getElementById(id) : null;
      if (!target) return;

      event.preventDefault();

      const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      target.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });

      // Move keyboard focus to the destination without scrolling twice.
      if (!target.hasAttribute('tabindex')) target.setAttribute('tabindex', '-1');
      target.focus({ preventScroll: true });

      if (history.replaceState) history.replaceState(null, '', '#' + id);
    });
  }


  /* ========================================================================
     07. PANELS (UI STATE MACHINE)
     ======================================================================== */

  const PANELS = {
    upload:     () => el.panelUpload,
    editor:     () => el.panelEditor,
    processing: () => el.panelProcessing,
    result:     () => el.panelResult
  };

  function showPanel(name) {
    // Measured before the swap, including any morph still in flight, so a
    // quick second change carries on from where the card actually is.
    const from = el.tool.getBoundingClientRect().height;
    if (runtime.panelMorph) runtime.panelMorph.cancel();

    Object.keys(PANELS).forEach((key) => {
      PANELS[key]().hidden = key !== name;
    });
    el.tool.setAttribute('data-busy', String(name === 'processing'));
    el.tool.setAttribute('aria-busy', String(name === 'processing'));

    morphToolHeight(from);
  }

  /**
   * The card glides from its old height to its new one instead of snapping,
   * while the incoming panel runs its own CSS rise. Both heights are measured,
   * so this only works because every panel is filled before it is shown.
   */
  function morphToolHeight(from) {
    if (!from || typeof el.tool.animate !== 'function' || prefersReducedMotion()) return;

    const to = el.tool.getBoundingClientRect().height;
    if (Math.abs(to - from) < 2) return;

    el.tool.classList.add('is-morphing');
    const morph = el.tool.animate(
      [{ height: from + 'px' }, { height: to + 'px' }],
      { duration: 420, easing: 'cubic-bezier(0.22, 1, 0.36, 1)' }
    );
    runtime.panelMorph = morph;

    const done = () => {
      if (runtime.panelMorph !== morph) return;
      runtime.panelMorph = null;
      el.tool.classList.remove('is-morphing');
    };
    morph.onfinish = done;
    morph.oncancel = done;
  }

  /** Scrolls the tool into view when the layout changes a lot (mobile). */
  function scrollToolIntoView() {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const rect = el.tool.getBoundingClientRect();
    if (rect.top < 0 || rect.top > window.innerHeight * 0.5) {
      el.tool.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
    }
  }


  /* ========================================================================
     08. FILE UPLOAD
     ======================================================================== */

  function openFilePicker() {
    if (state.processing) return;
    el.fileInput.click();
  }

  function initUpload() {
    el.dropzone.addEventListener('click', openFilePicker);

    el.dropzone.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ' || event.key === 'Spacebar') {
        event.preventDefault();
        openFilePicker();
      }
    });

    el.fileInput.addEventListener('change', (event) => {
      const files = event.target.files;
      // Reset so picking the same file twice still fires "change".
      handleFiles(files);
      el.fileInput.value = '';
    });

    el.changeBtn.addEventListener('click', openFilePicker);
    el.removeBtn.addEventListener('click', resetAll);
  }


  /* ========================================================================
     09. DRAG & DROP
     ======================================================================== */

  function initDragAndDrop() {
    // Stop the browser from navigating away when a file is dropped anywhere.
    ['dragover', 'drop'].forEach((type) => {
      window.addEventListener(type, (event) => event.preventDefault());
    });

    const zone = el.dropzone;

    zone.addEventListener('dragenter', (event) => {
      event.preventDefault();
      if (!state.processing) zone.classList.add('is-dragover');
    });

    zone.addEventListener('dragover', (event) => {
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
      if (!state.processing) zone.classList.add('is-dragover');
    });

    zone.addEventListener('dragleave', (event) => {
      // Ignore events fired while moving over child elements.
      if (event.relatedTarget && zone.contains(event.relatedTarget)) return;
      zone.classList.remove('is-dragover');
    });

    zone.addEventListener('drop', (event) => {
      event.preventDefault();
      zone.classList.remove('is-dragover');
      if (!event.dataTransfer) return;
      handleFiles(event.dataTransfer.files);
    });
  }


  /* ========================================================================
     10. CLIPBOARD PASTE
     ======================================================================== */

  function initPaste() {
    document.addEventListener('paste', (event) => {
      if (state.processing || !event.clipboardData) return;

      const items = event.clipboardData.items || [];
      for (let i = 0; i < items.length; i += 1) {
        if (items[i].kind !== 'file') continue;
        const file = items[i].getAsFile();
        if (file && file.type.indexOf('image/') === 0) {
          event.preventDefault();
          handleFiles([file]);
          return;
        }
      }
    });
  }


  /* ========================================================================
     11. IMAGE VALIDATION
     ======================================================================== */

  /** @returns {{ok: boolean, title?: string, message?: string}} */
  function validateFile(file) {
    if (!file) {
      return {
        ok: false,
        title: 'No Image Selected',
        message: 'Choose a JPG, PNG or WEBP file to continue.'
      };
    }

    const type = (file.type || '').toLowerCase();
    const ext = getExtension(file.name);
    const typeOk = CONFIG.acceptedTypes.indexOf(type) !== -1;
    const extOk = CONFIG.acceptedExtensions.indexOf(ext) !== -1;

    // Some sources (clipboard, odd file systems) report no MIME type at all,
    // so fall back to the extension before rejecting.
    if (!typeOk && !(type === '' && extOk)) {
      return {
        ok: false,
        title: 'Unsupported Image',
        message: 'Please upload JPG, PNG or WEBP.'
      };
    }

    if (file.size === 0) {
      return {
        ok: false,
        title: 'Invalid Image',
        message: 'The selected file could not be loaded as an image.'
      };
    }

    if (file.size > CONFIG.maxFileSize) {
      return {
        ok: false,
        title: 'File Too Large',
        message: 'Maximum upload size is ' + Math.round(CONFIG.maxFileSize / (1024 * 1024)) + 'MB. This file is ' + formatBytes(file.size) + '.'
      };
    }

    return { ok: true };
  }


  /* ========================================================================
     12. IMAGE METADATA & DECODING
     ======================================================================== */

  /**
   * Decodes a file into something drawable. Prefers createImageBitmap (fast,
   * honours EXIF orientation) and falls back to an <img> element.
   * @returns {Promise<{image: ImageBitmap|HTMLImageElement, width: number, height: number}>}
   */
  async function decodeImage(file, objectUrl) {
    if (typeof window.createImageBitmap === 'function') {
      try {
        const bitmap = await window.createImageBitmap(file, { imageOrientation: 'from-image' });
        if (bitmap && bitmap.width > 0 && bitmap.height > 0) {
          return { image: bitmap, width: bitmap.width, height: bitmap.height };
        }
      } catch (err) {
        // Older browsers reject the options argument — fall through to <img>.
      }
    }

    return new Promise((resolve, reject) => {
      const img = new Image();
      img.decoding = 'async';
      img.onload = () => {
        const width = img.naturalWidth || img.width;
        const height = img.naturalHeight || img.height;
        if (!width || !height) {
          reject(new Error('Image has no dimensions'));
          return;
        }
        resolve({ image: img, width: width, height: height });
      };
      img.onerror = () => reject(new Error('Image could not be decoded'));
      img.src = objectUrl;
    });
  }

  /** Validates, decodes and stores the chosen file, then shows the editor. */
  async function handleFiles(files) {
    if (state.processing) return;

    const file = files && files.length ? files[0] : null;
    const check = validateFile(file);
    if (!check.ok) {
      showToast(check.title, check.message, 'error');
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    let decoded;

    try {
      decoded = await decodeImage(file, objectUrl);
    } catch (err) {
      revoke(objectUrl);
      showToast('Invalid Image', 'The selected file could not be loaded as an image.', 'error');
      return;
    }

    // Everything decoded — release whatever the previous run was holding.
    releaseResultResources();
    releaseImageResources();

    state.file = file;
    state.originalUrl = objectUrl;
    state.originalImage = decoded.image;
    state.originalWidth = decoded.width;
    state.originalHeight = decoded.height;

    renderPreview();
    setScale(CONFIG.defaultScale, { silent: true });
    showPanel('editor');

    // Start fetching the model now, so pressing Upscale does not wait on it.
    preloadEngine();
  }

  function renderPreview() {
    el.previewImg.src = state.originalUrl;
    el.previewImg.alt = 'Preview of ' + state.file.name;
    el.fileName.textContent = state.file.name;
    el.fileDims.textContent = formatDimensions(state.originalWidth, state.originalHeight);
    el.fileSize.textContent = formatBytes(state.file.size);
    el.fileFormat.textContent = formatLabel(state.file);
    el.upscaleBtn.disabled = false;
  }


  /* ========================================================================
     13. UPSCALE SELECTION
     ======================================================================== */

  function setScale(scale, options) {
    const opts = options || {};
    state.scale = scale;

    el.scaleButtons.forEach((button) => {
      const isActive = Number(button.dataset.scale) === scale;
      button.setAttribute('aria-checked', String(isActive));
      button.tabIndex = isActive ? 0 : -1;
    });

    updateOutputResolution(!opts.silent);
  }

  function initScaleSelection() {
    el.scaleGroup.addEventListener('click', (event) => {
      const button = event.target.closest('.scale');
      if (!button) return;
      setScale(Number(button.dataset.scale));
    });

    // Arrow-key support, as expected from a radiogroup.
    el.scaleGroup.addEventListener('keydown', (event) => {
      const keys = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'];
      if (keys.indexOf(event.key) === -1) return;

      event.preventDefault();
      const current = el.scaleButtons.findIndex((b) => b.getAttribute('aria-checked') === 'true');
      const last = el.scaleButtons.length - 1;
      let next = current;

      if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = current <= 0 ? last : current - 1;
      if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = current >= last ? 0 : current + 1;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = last;

      setScale(Number(el.scaleButtons[next].dataset.scale));
      el.scaleButtons[next].focus();
    });
  }


  /* ========================================================================
     14. OUTPUT RESOLUTION
     ======================================================================== */

  function getOutputSize(scale) {
    const factor = scale || state.scale;
    return {
      width: Math.round(state.originalWidth * factor),
      height: Math.round(state.originalHeight * factor)
    };
  }

  /**
   * How much memory one job is allowed to hold.
   *
   * navigator.deviceMemory is Chrome-only and rounded to a power of two, but
   * that is enough to tell a 4GB laptop from a 16GB desktop, and promising a
   * small machine something it cannot deliver is worse than refusing. Where
   * the browser does not say, the fixed ceiling stands — the lower iOS one
   * on an iPhone or iPad.
   */
  function peakMemoryLimit() {
    const gigabytes = navigator.deviceMemory;
    if (!(gigabytes > 0)) return isAppleMobile() ? CONFIG.iosMaxPeakBytes : CONFIG.maxPeakBytes;
    return Math.min(CONFIG.maxPeakBytes, gigabytes * 1024 * 1024 * 1024 * 0.5);
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
   * Everything the job will hold at its heaviest moment, in bytes.
   *
   * The pixels are the bulk of it, but the finished file sits alongside them
   * and has to be counted too. A stored file weighs as much as the pixels, and
   * so does the canvas a JPEG is drawn on; a compressed PNG is smaller. The
   * JPEG is only tried up to maxStoredBytes (see encodeOutput), so that is
   * the most the file side can ever need.
   */
  function estimateJobBytes(width, height, scale) {
    const pixels = ImageUpAI.estimatePeakBytes(
      state.originalWidth, state.originalHeight, scale || state.scale
    );
    return pixels + Math.min(pngByteLength(width, height), CONFIG.maxStoredBytes);
  }

  /**
   * Whether a given output can actually be produced.
   *
   * Only states what is wrong, never what to do about it. The advice is added
   * by describeLimit(), which works out which scale would fit — and doing that
   * means calling this function again, so the two have to stay separate or
   * they would call each other forever.
   */
  function checkOutputLimits(width, height, scale) {
    if (CONFIG.engine !== 'local-ai' || !state.file) return { ok: true };

    const needed = estimateJobBytes(width, height, scale);
    const ceiling = peakMemoryLimit();

    if (needed > ceiling) {
      return {
        ok: false,
        reason: formatDimensions(width, height) + ' would need about ' + formatBytes(needed) +
          ' of memory to build, more than this device can be relied on for.'
      };
    }

    return { ok: true };
  }

  /** Every upscale level the interface offers, largest first. */
  function offeredScales() {
    return el.scaleButtons
      .map((button) => Number(button.dataset.scale))
      .filter((value) => value > 0)
      .sort((a, b) => b - a);
  }

  /**
   * Turns a refusal into something the user can act on.
   *
   * "Try a lower upscale level" left people guessing which one, and on a large
   * photo two of the three can be out of reach. This names the level that
   * fits, with the resolution it produces, so the next click is the right one.
   */
  function describeLimit(limit) {
    const scales = offeredScales();

    for (let i = 0; i < scales.length; i += 1) {
      if (scales[i] >= state.scale) continue;         // only smaller ones help
      const size = getOutputSize(scales[i]);
      if (checkOutputLimits(size.width, size.height, scales[i]).ok) {
        return limit.reason + ' ' + scales[i] + '× (' +
          formatDimensions(size.width, size.height) + ') will work.';
      }
    }

    return limit.reason + ' This image is too large to upscale here at any level — ' +
      'try a smaller one.';
  }

  function updateOutputResolution(announce) {
    if (!state.file) {
      el.outputRes.textContent = '0 × 0';
      el.outputEstimate.textContent = '';
      el.upscaleBtn.disabled = true;
      return;
    }

    const size = getOutputSize();
    const dims = formatDimensions(size.width, size.height);
    if (el.outputRes.textContent !== dims) {
      el.outputRes.textContent = dims;
      restartAnimation(el.outputRes);
      restartAnimation(el.outputEstimate);
    }

    const limit = checkOutputLimits(size.width, size.height, state.scale);
    el.upscaleBtn.disabled = !limit.ok;

    if (limit.ok) {
      el.upscaleBtn.removeAttribute('title');
      // A stored file's size is arithmetic, so it can be shown before the job
      // runs at all. Past the top of the size band the file is brought down
      // to about the target instead, so the target is what is shown.
      const bytes = pngByteLength(size.width, size.height);
      const fileNote = bytes <= outputBand(state.scale).high
        ? formatBytes(bytes) + ' PNG'
        : 'file about ' + targetLabel(state.scale);
      el.outputEstimate.textContent = [describeJob(), fileNote].filter(Boolean).join(' · ');
      el.outputEstimate.classList.toggle('is-slow', isSlowJob());
    } else {
      const message = describeLimit(limit);
      el.upscaleBtn.title = message;
      el.outputEstimate.textContent = '';
      if (announce) showToast('Output Too Large', message, 'error');
    }
  }

  /** The network is real work, so say roughly how long it will take. */
  function estimateJobSeconds() {
    if (CONFIG.engine !== 'local-ai' || !state.file) return null;
    return ImageUpAI.estimateSeconds(state.originalWidth, state.originalHeight, state.scale);
  }

  function isSlowJob() {
    const seconds = estimateJobSeconds();
    return seconds !== null && seconds > CONFIG.slowJobSeconds;
  }

  function describeJob() {
    const seconds = estimateJobSeconds();
    if (seconds === null) return '';
    const passes = ImageUpAI.passesFor(state.scale);
    const passLabel = passes > 1 ? '2 AI passes · ' : '';
    return passLabel + (seconds < 2 ? 'about a second' : 'about ' + formatDuration(seconds));
  }


  /* ========================================================================
     15. UPSCALE PROCESSING (UI FLOW)
     ======================================================================== */

  /**
   * Plays an element's CSS animation again from the start.
   *
   * Only ever call this when the content has actually changed. Restarting on
   * every update of a value that changes many times a second leaves the
   * element permanently mid-fade, which reads as invisible rather than
   * animated — a bug this codebase has already had once.
   */
  function restartAnimation(node) {
    if (!node) return;
    node.style.animation = 'none';
    void node.offsetWidth;           // forces the style change to take effect
    node.style.animation = '';
  }

  function setProcessingMessage(text, detail) {
    runtime.procDetail = detail || '';
    paintProcessingDetail();

    if (text === runtime.procMessage) return;
    runtime.procMessage = text;
    el.procMessage.textContent = text;
    restartAnimation(el.procMessage);
  }

  /**
   * The detail line always ends with a running clock. Long jobs otherwise look
   * stalled at exactly the moments when there is least to report — a slow
   * first download, or a large tile on a slow device.
   */
  function paintProcessingDetail() {
    const seconds = Math.round((Date.now() - runtime.startedAt) / 1000);
    const clock = runtime.startedAt && seconds > 2 ? seconds + 's elapsed' : '';
    el.procDetail.textContent = runtime.procDetail && clock
      ? runtime.procDetail + ' · ' + clock
      : (runtime.procDetail || clock);
  }

  function startElapsedClock() {
    runtime.startedAt = Date.now();
    stopElapsedClock();
    runtime.elapsedTimer = setInterval(paintProcessingDetail, 1000);
  }

  function stopElapsedClock() {
    if (runtime.elapsedTimer) {
      clearInterval(runtime.elapsedTimer);
      runtime.elapsedTimer = null;
    }
  }

  /** ratio null = indeterminate (we genuinely do not know yet). */
  function setProgress(ratio) {
    if (ratio === null || ratio === undefined) {
      el.progress.classList.add('is-indeterminate');
      el.progress.removeAttribute('aria-valuenow');
      el.progressBar.style.width = '';
      return;
    }
    const percent = clamp(Math.round(ratio * 100), 0, 100);
    el.progress.classList.remove('is-indeterminate');
    el.progress.setAttribute('aria-valuenow', String(percent));
    el.progressBar.style.width = percent + '%';
  }

  function formatDuration(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return '';
    if (seconds < 1) return 'a second';
    if (seconds < 60) {
      const whole = Math.round(seconds);
      return whole + (whole === 1 ? ' second' : ' seconds');
    }
    const minutes = Math.floor(seconds / 60);
    const rest = Math.round(seconds % 60);
    return minutes + ' min' + (rest ? ' ' + rest + ' s' : '');
  }

  function setProcessing(isProcessing) {
    state.processing = isProcessing;
    el.upscaleBtn.disabled = isProcessing || !state.file;
    el.downloadBtn.disabled = isProcessing;
  }

  async function handleUpscale() {
    // Guard against double clicks / Enter being held down.
    if (state.processing || !state.file || !state.originalImage) return;

    const size = getOutputSize();
    const limit = checkOutputLimits(size.width, size.height, state.scale);
    if (!limit.ok) {
      showToast('Output Too Large', describeLimit(limit), 'error');
      return;
    }

    setProcessing(true);
    runtime.cancelled = false;
    el.cancelBtn.disabled = false;
    runtime.procMessage = '';
    startElapsedClock();
    showPanel('processing');
    setProgress(ImageUpAI.isLoaded() ? 0 : null);
    setProcessingMessage(
      ImageUpAI.isLoaded() ? 'Preparing image…' : 'Loading the AI model…',
      ImageUpAI.isLoaded() ? '' : 'One-time download, then it is cached.'
    );
    scrollToolIntoView();

    try {
      const result = await upscaleImage(state.file, state.scale, {
        image: state.originalImage,
        width: state.originalWidth,
        height: state.originalHeight,

        // First run only: downloading the runtime and the model.
        onLoadProgress: (info) => {
          if (info.phase === 'runtime') {
            setProgress(info.ratio);
            setProcessingMessage(
              'Downloading the AI engine…',
              Math.round(info.ratio * 100) + '% of 5.5 MB · one-time download, then cached'
            );
          } else if (info.phase === 'model') {
            setProgress(info.ratio);
            setProcessingMessage(
              'Downloading the AI model…',
              Math.round(info.ratio * 100) + '% of 4.6 MB · cached for next time'
            );
          } else {
            // Compiling the model — short, and there is nothing to count.
            setProgress(null);
            setProcessingMessage('Starting the model…', 'Almost there.');
          }
        },

        // Real per-tile progress, so there is nothing to fake.
        onProgress: (info) => {
          setProgress(info.ratio);
          const passLabel = info.passes > 1 ? 'Pass ' + info.pass + ' of ' + info.passes + ' · ' : '';
          const left = info.secondsLeft !== null && info.secondsLeft > 1
            ? ' · about ' + formatDuration(info.secondsLeft) + ' left'
            : '';
          setProcessingMessage(
            'Reconstructing detail…',
            passLabel + 'tile ' + info.tilesDone + ' of ' + info.tilesTotal + left
          );
        },

        isCancelled: () => runtime.cancelled
      });

      if (runtime.cancelled) throw new Error('Cancelled by user');

      setProgress(1);
      setProcessingMessage('Finalizing image…', '');

      // Replace any previous result before storing the new one.
      releaseResultResources();
      state.resultPixels = result.rgba;
      state.resultWidth = result.width;
      state.resultHeight = result.height;

      // The result lives as raw bytes; the comparison view gets its own copy,
      // reduced if the full size is more than a canvas can take.
      setProcessingMessage('Preparing the preview…', '');
      const preview = await buildPreview(result.rgba, result.width, result.height);
      state.resultImage = preview.canvas;
      state.previewFactor = preview.factor;

      // Encode once up-front so the result card can show a real file size.
      setProcessingMessage('Writing the file…', '');
      keepOutput(await encodeOutput(result.rgba, result.width, result.height, state.scale));

      renderResult(result.engine);
      showPanel('result');
      scrollToolIntoView();
      showToast(
        'Upscale Complete',
        'Your image is now ' + formatDimensions(result.width, result.height) + '.',
        'success'
      );
    } catch (error) {
      if (runtime.cancelled) {
        showToast('Upscale Cancelled', 'Nothing was changed. Your image is still loaded.', 'info');
      } else {
        console.error('[ImageUp AI] Upscaling failed:', error);
        showToast(
          'Upscaling Failed',
          error && error.userMessage
            ? error.userMessage
            : 'Something went wrong while processing your image.',
          'error'
        );
      }
      showPanel('editor');
    } finally {
      stopElapsedClock();
      runtime.cancelled = false;
      setProcessing(false);
    }
  }

  function handleCancel() {
    if (!state.processing) return;
    runtime.cancelled = true;
    el.cancelBtn.disabled = true;
    setProcessingMessage('Stopping…', '');
  }



  /* ========================================================================
     16. UPSCALING ENGINE
     ------------------------------------------------------------------------
     The only place that knows how pixels are produced. Everything above talks
     to upscaleImage() and nothing else, so the engine can be swapped without
     touching the interface.

     'local-ai' runs a real super-resolution network on the user's device
     (js/ai-upscaler.js). 'api' posts the file to a backend you control.
     ======================================================================== */

  /**
   * @param {File}   file   the original file (sent as-is in 'api' mode)
   * @param {number} scale  2, 4 or 8
   * @param {object} [options] image/width/height plus progress callbacks
   * @returns {Promise<{canvas: HTMLCanvasElement, width: number, height: number, engine: string}>}
   */
  async function upscaleImage(file, scale, options) {
    const opts = options || {};

    if (CONFIG.engine === 'api') {
      return upscaleWithApi(file, scale);
    }

    let source = opts.image;
    let width = opts.width;
    let height = opts.height;

    if (!source) {
      const url = URL.createObjectURL(file);
      try {
        const decoded = await decodeImage(file, url);
        source = decoded.image;
        width = decoded.width;
        height = decoded.height;
      } finally {
        revoke(url);
      }
    }

    const output = await ImageUpAI.upscale(source, width, height, scale, {
      onLoadProgress: opts.onLoadProgress,
      onProgress: opts.onProgress,
      isCancelled: opts.isCancelled
    });

    return {
      rgba: output.rgba,
      width: output.width,
      height: output.height,
      engine: 'local-ai'
    };
  }

  /**
   * Alternative engine — talks to *your own* backend, not a third-party API.
   *
   * Never place a private production API key in this file: anything shipped to
   * the browser can be read by anyone. Put the key on a small server/serverless
   * proxy that forwards the request, and point CONFIG.apiEndpoint at it.
   * The README has a worked example.
   */
  async function upscaleWithApi(file, scale) {
    const form = new FormData();
    form.append('image', file, file.name);
    form.append('scale', String(scale));

    let response;
    try {
      response = await fetch(CONFIG.apiEndpoint, { method: 'POST', body: form });
    } catch (networkError) {
      const error = new Error('Network request failed');
      error.userMessage = 'Could not reach the upscaling service. Check your connection and try again.';
      throw error;
    }

    if (!response.ok) {
      const error = new Error('Upscale API returned ' + response.status);
      error.userMessage = 'The upscaling service returned an error (' + response.status + ').';
      throw error;
    }

    const blob = await response.blob();
    if (!blob.type || blob.type.indexOf('image/') !== 0) {
      const error = new Error('Upscale API did not return an image');
      error.userMessage = 'The upscaling service did not return a valid image.';
      throw error;
    }

    const url = URL.createObjectURL(blob);
    try {
      const decoded = await decodeImage(blob, url);
      const canvas = document.createElement('canvas');
      canvas.width = decoded.width;
      canvas.height = decoded.height;

      const context = canvas.getContext('2d', { willReadFrequently: true });

      // The rest of the app works in raw pixels, so the decoded image is read
      // out and the canvas let go of straight away. Drawn and read in strips,
      // with the page given a turn between them: all at once, a large result
      // holds the page still for as long as the copy takes.
      const rgba = new Uint8ClampedArray(canvas.width * canvas.height * 4);
      const stripRows = Math.max(1, Math.floor((2 * 1024 * 1024) / (canvas.width * 4)));
      const tick = timeSlicer();
      for (let y = 0; y < canvas.height; y += stripRows) {
        const rows = Math.min(stripRows, canvas.height - y);
        context.drawImage(decoded.image, 0, y, canvas.width, rows, 0, y, canvas.width, rows);
        rgba.set(context.getImageData(0, y, canvas.width, rows).data, y * canvas.width * 4);
        await tick();
      }
      closeImage(decoded.image);

      const result = {
        rgba: rgba,
        width: canvas.width,
        height: canvas.height,
        engine: 'api'
      };
      releaseCanvas(canvas);
      return result;
    } finally {
      revoke(url);
    }
  }

  /**
   * Starts fetching the model as soon as an image is chosen, so pressing
   * Upscale does not begin with a download. Failures are ignored here — they
   * surface properly when the user actually asks for an upscale.
   */
  /**
   * Starts fetching the engine before it is asked for.
   *
   * This is the single biggest thing the page can do for how fast it feels.
   * The download is ~10MB the first time; starting it when the page goes idle
   * means it usually finishes while the visitor is still choosing a file, so
   * pressing Upscale begins working immediately instead of downloading.
   * Failures are ignored here — they surface properly on actual use.
   */
  function preloadEngine() {
    if (CONFIG.engine !== 'local-ai' || ImageUpAI.isLoaded()) return;
    ImageUpAI.load().catch(() => { /* reported on use */ });
  }

  function preloadEngineWhenIdle() {
    if (CONFIG.engine !== 'local-ai') return;
    if (typeof window.requestIdleCallback === 'function') {
      window.requestIdleCallback(preloadEngine, { timeout: 2500 });
    } else {
      setTimeout(preloadEngine, 1200);
    }
  }


  /* ========================================================================
     17. CANVAS ENCODING HELPERS
     ======================================================================== */

  /**
   * Exactly how many bytes the PNG will occupy.
   *
   * Because nothing is compressed, this is arithmetic rather than a guess,
   * and it can be shown before the job has even run. A normal PNG's size
   * depends on the content of the picture and cannot be known in advance.
   */
  function pngByteLength(width, height, channels) {
    const raw = (1 + width * (channels || 4)) * height;   // filter byte per scanline
    const blocks = Math.ceil(raw / DEFLATE_BLOCK_MAX);
    //  signature + IHDR + IDAT framing + IEND, then the zlib stream itself
    return 57 + 2 + blocks * 5 + raw + 4;
  }

  /**
   * CRC-32, which PNG requires on every chunk.
   *
   * Kept resumable because a compressed chunk stream arrives in pieces: the
   * checksum has to be carried across them rather than computed over a whole
   * buffer that is never assembled.
   */
  let crcTable = null;
  function crcUpdate(crc, bytes, start, end) {
    if (!crcTable) {
      crcTable = new Uint32Array(256);
      for (let n = 0; n < 256; n += 1) {
        let c = n;
        for (let k = 0; k < 8; k += 1) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
        crcTable[n] = c >>> 0;
      }
    }
    for (let i = start; i < end; i += 1) {
      crc = crcTable[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
    }
    return crc;
  }

  const crcStart = () => 0xffffffff;
  const crcFinish = (crc) => (crc ^ 0xffffffff) >>> 0;
  const crc32 = (bytes, start, end) => crcFinish(crcUpdate(crcStart(), bytes, start, end));

  /** Writes a big-endian 32-bit number, as every length and CRC in PNG is. */
  function put32(bytes, at, value) {
    bytes[at] = (value >>> 24) & 0xff;
    bytes[at + 1] = (value >>> 16) & 0xff;
    bytes[at + 2] = (value >>> 8) & 0xff;
    bytes[at + 3] = value & 0xff;
  }

  const IDAT_TAG = new Uint8Array([0x49, 0x44, 0x41, 0x54]);
  const PNG_IEND = new Uint8Array([0, 0, 0, 0, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82]);

  /**
   * Signature and IHDR: 8 bits a channel, no interlacing, and colour type 6
   * (RGBA) unless told otherwise — 2 (RGB) for an image with no transparency.
   */
  function pngHeader(width, height, colorType) {
    const out = new Uint8Array(33);
    out.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a], 0);
    put32(out, 8, 13);
    out.set([0x49, 0x48, 0x44, 0x52], 12);       // IHDR
    put32(out, 16, width);
    put32(out, 20, height);
    out[24] = 8;
    out[25] = colorType || 6;
    out[26] = 0; out[27] = 0; out[28] = 0;
    put32(out, 29, crc32(out, 12, 29));
    return out;
  }

  function encodingFailure(bytes) {
    const error = new Error('Could not build a ' + bytes + ' byte PNG');
    error.userMessage = 'This image needs about ' + formatBytes(bytes) +
      ' to save, which is more than this browser can build in one piece. ' +
      'Try a lower upscale level.';
    return error;
  }

  /**
   * Writes the result as an uncompressed PNG.
   *
   * `canvas.toBlob` gives no control over compression, and a compressed PNG of
   * a 4000 x 3000 result came out around 15 MB against 46 MB of actual pixels.
   * Writing the file here instead makes the size predictable and maximal.
   *
   * The pixels are stored through deflate's "stored" block type, which copies
   * bytes verbatim. That is an ordinary part of the zlib format, so the result
   * is a perfectly normal PNG: every decoder reads it, and the pixels are
   * byte-identical to a compressed one. It is simply not squeezed.
   *
   * Throws if the file cannot be allocated in one piece; encodeOutput() then
   * falls back to compressing, which needs far less room.
   *
   * @param {Uint8ClampedArray} pixels  RGBA, width * height * 4 bytes
   * @returns {Blob}
   */
  /**
   * Stored PNGs can run to hundreds of megabytes, so this works in slices
   * (timeSlicer) instead of one long loop: the result screen stays live —
   * slider, zoom, animations — while the file is being measured. The copy
   * is done by the browser (TypedArray.set); only the two checksums walk the
   * bytes in script, and the CRC is carried along row by row rather than run
   * over the whole buffer again at the end.
   */
  async function encodePngStored(pixels, width, height) {
    const out = new Uint8Array(pngByteLength(width, height));

    const header = pngHeader(width, height);
    out.set(header, 0);
    let pos = header.length;

    const u32 = (value) => { put32(out, pos, value); pos += 4; };

    // IDAT, holding the whole zlib stream
    const rawLength = (1 + width * 4) * height;
    const blocks = Math.ceil(rawLength / DEFLATE_BLOCK_MAX);
    u32(2 + blocks * 5 + rawLength + 4);
    const idatStart = pos;
    out.set(IDAT_TAG, pos);
    pos += 4;
    out[pos] = 0x78; out[pos + 1] = 0x01;        // zlib header, no compression
    pos += 2;

    // Deflate stored blocks, with Adler-32 running over the raw bytes.
    let blockLeft = 0;
    let rawLeft = rawLength;
    let s1 = 1;
    let s2 = 0;

    const openBlock = () => {
      const length = Math.min(DEFLATE_BLOCK_MAX, rawLeft);
      out[pos] = (rawLeft === length) ? 1 : 0;   // BFINAL on the last one
      out[pos + 1] = length & 0xff;
      out[pos + 2] = (length >>> 8) & 0xff;
      out[pos + 3] = (~length) & 0xff;
      out[pos + 4] = ((~length) >>> 8) & 0xff;
      pos += 5;
      blockLeft = length;
    };

    /** Copies count bytes of the raw stream, splitting across blocks. */
    const writeRaw = (source, from, count) => {
      let left = count;
      let at = from;

      while (left > 0) {
        if (blockLeft === 0) openBlock();
        const run = Math.min(blockLeft, left);

        out.set(source.subarray(at, at + run), pos);

        // Adler-32 over the bytes just copied, reduced every 2048 bytes.
        // That keeps both sums below 2^30, inside the small-integer range the
        // JIT runs at full speed; left to grow for a whole block they spill
        // into floating point and the loop runs several times slower.
        for (let i = at, end = at + run; i < end;) {
          const stop = Math.min(end, i + 2048);
          for (; i < stop; i += 1) {
            s1 += source[i];
            s2 += s1;
          }
          s1 %= 65521;
          s2 %= 65521;
        }

        pos += run;
        at += run;
        left -= run;
        blockLeft -= run;
        rawLeft -= run;
      }
    };

    const filter = new Uint8Array(1);             // filter type 0: none
    const rowBytes = width * 4;

    // The IDAT CRC covers everything from its tag on; it is brought up to
    // date after each row, over just the bytes that row wrote.
    let crc = crcStart();
    let crcDone = idatStart;
    const tick = timeSlicer();

    for (let row = 0; row < height; row += 1) {
      writeRaw(filter, 0, 1);
      writeRaw(pixels, row * rowBytes, rowBytes);
      crc = crcUpdate(crc, out, crcDone, pos);
      crcDone = pos;
      await tick();
    }

    s1 %= 65521;
    s2 %= 65521;
    u32(((s2 << 16) | s1) >>> 0);
    crc = crcUpdate(crc, out, crcDone, pos);      // the Adler-32 just written
    u32(crcFinish(crc));

    out.set(PNG_IEND, pos);
    return new Blob([out], { type: OUTPUT_MIME });
  }

  /**
   * Whether every pixel is fully opaque. When it is, the alpha channel says
   * nothing, and a file without it holds a quarter fewer bytes for exactly
   * the same picture. Checked in slices, so a huge result never stalls the page.
   */
  async function isOpaque(pixels) {
    const tick = timeSlicer();
    const step = 1 << 20;                        // a multiple of 4: stays on alpha
    for (let start = 3; start < pixels.length; start += step) {
      const end = Math.min(pixels.length, start + step);
      for (let i = start; i < end; i += 4) {
        if (pixels[i] !== 255) return false;
      }
      await tick();
    }
    return true;
  }

  /** Copies scanline y of the RGBA result into `into`, as RGB when channels is 3. */
  function readScanline(pixels, width, y, channels, into) {
    const from = y * width * 4;
    if (channels === 4) {
      into.set(pixels.subarray(from, from + width * 4));
      return;
    }
    for (let i = from, o = 0, end = from + width * 4; i < end; i += 4, o += 3) {
      into[o] = pixels[i];
      into[o + 1] = pixels[i + 1];
      into[o + 2] = pixels[i + 2];
    }
  }

  /**
   * PNG's Paeth filter over one scanline: each byte is written as its
   * difference from whichever neighbour — left, above or above-left — best
   * predicts it. On photographs that turns most bytes into small numbers,
   * which deflate packs far tighter than the raw values. The PNG decoder
   * undoes it exactly, so the file stays lossless.
   *
   * @param {Uint8Array} row    this scanline, unfiltered
   * @param {Uint8Array} above  the previous scanline, unfiltered (zeros for the first)
   * @param {number}     bpp    bytes per pixel
   */
  function paethFilter(row, above, bpp, out, at) {
    // No left neighbour for the first pixel: Paeth reduces to "above".
    for (let i = 0; i < bpp; i += 1) out[at + i] = (row[i] - above[i]) & 0xff;

    for (let i = bpp, n = row.length; i < n; i += 1) {
      const a = row[i - bpp];
      const b = above[i];
      const c = above[i - bpp];
      const pa = Math.abs(b - c);                // |p - a| for p = a + b - c
      const pb = Math.abs(a - c);                // |p - b|
      const pc = Math.abs(a + b - c - c);        // |p - c|
      const predicted = (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
      out[at + i] = (row[i] - predicted) & 0xff;
    }
  }

  /** Whether the browser can deflate without a zlib wrapper, which the hybrid PNG needs. */
  function supportsDeflateRaw() {
    try {
      new CompressionStream('deflate-raw');
      return true;
    } catch (err) {
      return false;
    }
  }

  /**
   * A PNG whose first `storedRows` scanlines are stored as they are and whose
   * remaining ones are compressed — all in one ordinary zlib stream, since
   * deflate allows the two kinds of block side by side. Moving that one
   * boundary sets the file's size anywhere between "fully compressed" and
   * "fully stored", while every pixel stays exactly as the AI produced it.
   * With storedRows equal to the height it is simply a stored PNG, whose
   * size pngByteLength() gives to the byte.
   *
   * The stored rows go in 65535-byte stored blocks, none marked final unless
   * nothing follows. The rest go through the browser's raw deflate
   * (CompressionStream 'deflate-raw'), whose last block ends the stream.
   * Stored rows use no filter; compressed ones use Paeth, which — like every
   * PNG filter — reads the row above unfiltered, whatever that row used.
   *
   * @param {number} channels    3 (RGB, alpha left out) or 4 (RGBA)
   * @param {number} storedRows  scanlines, from the top, stored uncompressed
   * @returns {Promise<{blob: Blob, compressedBytes: number}>}
   */
  async function encodePngHybrid(pixels, width, height, channels, storedRows) {
    const rowBytes = width * channels;
    const lineBytes = rowBytes + 1;              // filter byte + samples
    const tick = timeSlicer();

    // Adler-32 of everything the stream inflates to, reduced every 2048
    // bytes so both sums stay in the JIT's fast small-integer range.
    let s1 = 1;
    let s2 = 0;
    const adler = (bytes, from, to) => {
      for (let i = from; i < to;) {
        const stop = Math.min(to, i + 2048);
        for (; i < stop; i += 1) {
          s1 += bytes[i];
          s2 += s1;
        }
        s1 %= 65521;
        s2 %= 65521;
      }
    };

    const zlibHeader = new Uint8Array([0x78, 0x01]);
    let crc = crcUpdate(crcUpdate(crcStart(), IDAT_TAG, 0, 4), zlibHeader, 0, 2);
    const parts = [zlibHeader];
    let length = 2;

    let above = new Uint8Array(rowBytes);        // zeros: nothing above row 0
    let row = new Uint8Array(rowBytes);

    // 1. The stored rows, verbatim.
    if (storedRows > 0) {
      const raw = storedRows * lineBytes;
      const out = new Uint8Array(raw + Math.ceil(raw / DEFLATE_BLOCK_MAX) * 5);
      const endsStream = storedRows === height;
      const filterNone = new Uint8Array(1);
      let pos = 0;
      let blockLeft = 0;
      let rawLeft = raw;
      let crcDone = 0;

      const put = (source, from, count) => {
        while (count > 0) {
          if (blockLeft === 0) {
            const len = Math.min(DEFLATE_BLOCK_MAX, rawLeft);
            out[pos] = (endsStream && rawLeft === len) ? 1 : 0;   // BFINAL
            out[pos + 1] = len & 0xff;
            out[pos + 2] = (len >>> 8) & 0xff;
            out[pos + 3] = (~len) & 0xff;
            out[pos + 4] = ((~len) >>> 8) & 0xff;
            pos += 5;
            blockLeft = len;
          }
          const run = Math.min(blockLeft, count);
          out.set(source.subarray(from, from + run), pos);
          adler(source, from, from + run);
          pos += run;
          from += run;
          count -= run;
          blockLeft -= run;
          rawLeft -= run;
        }
      };

      for (let y = 0; y < storedRows; y += 1) {
        readScanline(pixels, width, y, channels, row);
        put(filterNone, 0, 1);
        put(row, 0, rowBytes);
        const done = above;                      // this row is next row's "above"
        above = row;
        row = done;
        crc = crcUpdate(crc, out, crcDone, pos);
        crcDone = pos;
        await tick();
      }
      parts.push(out);
      length += out.length;
    }

    // 2. The rest, Paeth-filtered and deflated.
    let compressedBytes = 0;
    if (storedRows < height) {
      const stream = new CompressionStream('deflate-raw');
      const writer = stream.writable.getWriter();
      const reader = stream.readable.getReader();

      // Output is gathered into Blobs, which the browser may keep on disk.
      let batch = [];
      let batched = 0;
      const flush = () => {
        if (!batch.length) return;
        parts.push(new Blob(batch));
        batch = [];
        batched = 0;
      };

      // Draining runs alongside the writing: a stream nobody reads stops
      // accepting input, and both halves would wait on each other.
      const drain = (async () => {
        for (;;) {
          const step = await reader.read();
          if (step.done) break;
          batch.push(step.value);
          batched += step.value.length;
          compressedBytes += step.value.length;
          crc = crcUpdate(crc, step.value, 0, step.value.length);
          if (batched >= 32 * 1024 * 1024) flush();
        }
        flush();
      })();

      // The browser deflates each piece in one go on this thread, so the
      // pieces stay small: half a megabyte takes a few milliseconds.
      const rowsPerBlock = Math.max(1, Math.floor((512 * 1024) / lineBytes));

      const feed = (async () => {
        for (let top = storedRows; top < height; top += rowsPerBlock) {
          const rows = Math.min(rowsPerBlock, height - top);
          const block = new Uint8Array(rows * lineBytes);

          for (let r = 0; r < rows; r += 1) {
            readScanline(pixels, width, top + r, channels, row);
            const at = r * lineBytes;
            block[at] = 4;                        // filter type 4: Paeth
            paethFilter(row, above, channels, block, at + 1);
            const done = above;
            above = row;
            row = done;
            await tick();
          }

          adler(block, 0, block.length);
          await writer.write(block);
        }
        await writer.close();
      })();

      await Promise.all([feed, drain]);
      length += compressedBytes;
    }

    // 3. Adler-32 closes the zlib stream; the IDAT chunk wraps it all.
    const adlerBytes = new Uint8Array(4);
    put32(adlerBytes, 0, ((s2 << 16) | s1) >>> 0);
    crc = crcUpdate(crc, adlerBytes, 0, 4);
    length += 4;

    const idatHead = new Uint8Array(8);
    put32(idatHead, 0, length);
    idatHead.set(IDAT_TAG, 4);

    const idatTail = new Uint8Array(4 + PNG_IEND.length);
    put32(idatTail, 0, crcFinish(crc));
    idatTail.set(PNG_IEND, 4);

    return {
      blob: new Blob(
        [pngHeader(width, height, channels === 3 ? 2 : 6), idatHead].concat(parts, [adlerBytes, idatTail]),
        { type: OUTPUT_MIME }
      ),
      compressedBytes: compressedBytes
    };
  }

  /**
   * How well the result compresses, measured on a sample: about 4% of its
   * rows, in up to 32 bands spread down the image, Paeth-filtered and
   * deflated just as encodePngHybrid() would. Compressed bytes per raw byte.
   */
  async function sampleCompressionRatio(pixels, width, height, channels) {
    const rowBytes = width * channels;
    const lineBytes = rowBytes + 1;
    const bands = Math.min(32, height);
    const bandRows = Math.max(1, Math.round((height * 0.04) / bands));
    const tick = timeSlicer();

    const stream = new CompressionStream('deflate-raw');
    const writer = stream.writable.getWriter();
    const reader = stream.readable.getReader();
    let compressed = 0;
    let raw = 0;
    const drain = (async () => {
      for (;;) {
        const step = await reader.read();
        if (step.done) break;
        compressed += step.value.length;
      }
    })();

    for (let b = 0; b < bands; b += 1) {
      const centre = Math.floor(((b + 0.5) * height) / bands);
      const first = clamp(centre - Math.floor(bandRows / 2), 0, height - bandRows);
      let above = new Uint8Array(rowBytes);
      let row = new Uint8Array(rowBytes);
      if (first > 0) readScanline(pixels, width, first - 1, channels, above);

      const block = new Uint8Array(bandRows * lineBytes);
      for (let r = 0; r < bandRows; r += 1) {
        readScanline(pixels, width, first + r, channels, row);
        block[r * lineBytes] = 4;
        paethFilter(row, above, channels, block, r * lineBytes + 1);
        const done = above;
        above = row;
        row = done;
      }
      raw += block.length;
      await writer.write(block);
      await tick();
    }
    await writer.close();
    await drain;
    return raw ? compressed / raw : 1;
  }

  /**
   * How many scanlines to store uncompressed so a hybrid PNG comes out at
   * about `target` bytes, when the compressed ones shrink to `ratio` of
   * their size. Solves
   *   framing + stored × line × (1 + 5/65535) + (height − stored) × line × ratio = target
   * — the 5/65535 being the stored-block headers.
   */
  function storedRowsFor(target, ratio, lineBytes, height) {
    const perStored = lineBytes * (1 + 5 / DEFLATE_BLOCK_MAX);
    const perCompressed = lineBytes * ratio;
    if (perStored <= perCompressed) return height;
    const rows = (target - PNG_FRAMING_BYTES - perCompressed * height) / (perStored - perCompressed);
    return clamp(Math.round(rows), 0, height);
  }

  /** JPEG has no transparency: blends a strip over white, so clear areas do not turn black. */
  function flattenOnWhite(rgba) {
    const out = new Uint8ClampedArray(rgba.length);
    for (let i = 0; i < rgba.length; i += 4) {
      const alpha = rgba[i + 3];
      const white = 255 * (255 - alpha);
      out[i] = (rgba[i] * alpha + white) / 255;
      out[i + 1] = (rgba[i + 1] * alpha + white) / 255;
      out[i + 2] = (rgba[i + 2] * alpha + white) / 255;
      out[i + 3] = 255;
    }
    return out;
  }

  /**
   * The best JPEG of the result that fits within maxBytes: the highest
   * quality from JPEG_QUALITIES whose file is small enough. Full resolution
   * always — only the compression changes.
   *
   * Needs a canvas as big as the image. When the browser cannot make one
   * that size, or cannot write JPEG, this returns null and the caller keeps
   * a PNG instead.
   *
   * @returns {Promise<{blob: Blob, quality: number} | null>}
   */
  async function encodeJpegToFit(pixels, width, height, maxBytes) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');
    // An over-sized canvas comes back without a context, or quietly at 0 × 0.
    if (!context || canvas.width !== width || canvas.height !== height) {
      releaseCanvas(canvas);
      return null;
    }

    try {
      const opaque = await isOpaque(pixels);
      const tick = timeSlicer();
      const stripRows = Math.max(1, Math.floor((2 * 1024 * 1024) / (width * 4)));

      for (let y = 0; y < height; y += stripRows) {
        const rows = Math.min(stripRows, height - y);
        const from = y * width * 4;
        let strip = pixels.subarray(from, from + rows * width * 4);
        if (!opaque) strip = flattenOnWhite(strip);
        context.putImageData(new ImageData(strip, width, rows), 0, y);
        await tick();
      }

      // A browser that cannot write JPEG hands back a PNG, or nothing.
      const jpegAt = async (quality) => {
        const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', quality));
        return blob && blob.type === 'image/jpeg' ? blob : null;
      };

      let best = null;
      let tooBig = null;                          // the lowest quality known not to fit
      for (const quality of JPEG_QUALITIES) {
        const blob = await jpegAt(quality);
        if (!blob) return null;
        if (blob.size <= maxBytes) {
          best = { blob: blob, quality: quality };
          break;
        }
        tooBig = quality;
      }
      if (!best || tooBig === null) return best;

      // The steps are coarse at the top — 95% can be half the size of 100% —
      // so search the whole percentages in between for the highest that fits.
      let fits = Math.round(best.quality * 100);
      let over = Math.round(tooBig * 100);
      while (over - fits > 1) {
        const mid = Math.floor((fits + over) / 2);
        const blob = await jpegAt(mid / 100);
        if (!blob) break;
        if (blob.size <= maxBytes) {
          fits = mid;
          best = { blob: blob, quality: mid / 100 };
        } else {
          over = mid;
        }
      }
      return best;
    } finally {
      releaseCanvas(canvas);
    }
  }

  /**
   * Turns the result into the file to download, brought to about the
   * upscale level's CONFIG.targetOutputBytes (within its ± 10% band) when
   * it would otherwise be bigger — and never reduced further than that takes:
   *
   *   1. No bigger than the band's top as it is: PNG stored whole, exactly
   *      as before. Its size was shown before the job ran.
   *   2. Without the alpha channel an opaque image carries for nothing, it
   *      lands in the band: that, still stored whole.
   *   3. Otherwise a hybrid PNG — some rows stored, the rest compressed —
   *      with the split chosen to land near the target. Lossless all the
   *      same. The split comes from a sample of how well the image
   *      compresses; if the file misses the band, the real figure from the
   *      rows actually compressed sets a second, closer split.
   *   4. Only when no PNG can come under the band's top: a JPEG at the
   *      highest quality that does.
   *
   * A result smaller than the band is left its real size (step 1); nothing
   * is padded. The JPEG needs a canvas as big as the pixels, so it is only
   * tried within the memory the job was planned with (maxStoredBytes). If
   * nothing gets under the top, the fully compressed PNG is kept and marked
   * overCap, and the result card says so.
   *
   * @returns {Promise<{blob: Blob, kind: string, channels: number, quality: number, overCap: boolean}>}
   */
  async function encodeOutput(pixels, width, height, scale) {
    const band = outputBand(scale);
    const whole = pngByteLength(width, height, 4);
    const file = (blob, kind, channels, quality, overCap) =>
      ({ blob: blob, kind: kind, channels: channels, quality: quality, overCap: overCap });

    // 1. Already no bigger than the band's top.
    if (whole <= band.high) {
      return file(await encodePngStored(pixels, width, height), 'stored', 4, 1, false);
    }

    const hybridOk = supportsDeflateRaw();
    const opaque = await isOpaque(pixels);

    // 2. The alpha channel was all that pushed it over.
    const rgbWhole = pngByteLength(width, height, 3);
    if (opaque && rgbWhole <= band.high && rgbWhole >= band.low) {
      return file((await encodePngHybrid(pixels, width, height, 3, height)).blob, 'stored', 3, 1, false);
    }

    // 3. Part stored, part compressed, split to land near the target. RGBA
    //    when dropping alpha would already undershoot the band.
    if (hybridOk) {
      const channels = opaque && rgbWhole > band.high ? 3 : 4;
      const lineBytes = width * channels + 1;
      const ratio = await sampleCompressionRatio(pixels, width, height, channels);
      const fullyCompressed = PNG_FRAMING_BYTES + ratio * lineBytes * height;

      if (fullyCompressed <= band.high) {
        let rows = storedRowsFor(band.target, ratio, lineBytes, height);
        let out = await encodePngHybrid(pixels, width, height, channels, rows);

        if ((out.blob.size < band.low || out.blob.size > band.high) && rows < height) {
          const measured = out.compressedBytes / ((height - rows) * lineBytes);
          const retry = storedRowsFor(band.target, measured, lineBytes, height);
          if (retry !== rows) {
            out = null;                           // let the first attempt go first
            rows = retry;
            out = await encodePngHybrid(pixels, width, height, channels, rows);
          }
        }

        const kind = rows === height ? 'stored' : rows === 0 ? 'compressed' : 'hybrid';
        return file(out.blob, kind, channels, 1, out.blob.size > band.high);
      }
    }

    // 4. No lossless file comes under the top: the best JPEG that does.
    if (whole <= CONFIG.maxStoredBytes) {
      const jpeg = await encodeJpegToFit(pixels, width, height, band.high);
      if (jpeg) return file(jpeg.blob, 'jpeg', 3, jpeg.quality, false);
    }

    // Nothing fits: the smallest lossless file there is, marked as over.
    if (hybridOk) {
      const channels = opaque ? 3 : 4;
      return file((await encodePngHybrid(pixels, width, height, channels, 0)).blob, 'compressed', channels, 1, true);
    }
    if (whole <= CONFIG.maxStoredBytes) {
      return file(await encodePngStored(pixels, width, height), 'stored', 4, 1, true);
    }
    throw encodingFailure(whole);
  }

  /** Keeps an encoded file, and what kind it is, for the result card and the download. */
  function keepOutput(file) {
    runtime.outputBlob = file.blob;
    runtime.outputKind = file.kind;
    runtime.outputChannels = file.channels;
    runtime.outputQuality = file.quality;
    runtime.outputOverCap = file.overCap;
  }

  /**
   * Builds the canvas the comparison view shows.
   *
   * A result that fits is used exactly as it is, pixel for pixel. A larger one
   * is averaged down by a whole number: every source pixel is read once and
   * counts the same, so what appears is a fair picture of the result rather
   * than a sample of it. The file that gets downloaded is unaffected — it is
   * written from the full-size pixels.
   *
   * @returns {Promise<{canvas: HTMLCanvasElement, factor: number}>}
   */
  /**
   * putImageData in horizontal strips, giving the page a turn between them.
   * One call for a whole large image copies tens of megabytes in a single
   * go and holds the page still while it does.
   */
  async function putImageDataInStrips(context, image, tick) {
    const stripRows = Math.max(1, Math.floor((2 * 1024 * 1024) / (image.width * 4)));
    for (let y = 0; y < image.height; y += stripRows) {
      context.putImageData(image, 0, 0, 0, y, image.width, Math.min(stripRows, image.height - y));
      await tick();
    }
  }

  async function buildPreview(pixels, width, height) {
    const factor = Math.max(1, Math.ceil(Math.max(width, height) / PREVIEW_MAX_SIDE));
    const outWidth = Math.ceil(width / factor);
    const outHeight = Math.ceil(height / factor);

    const canvas = document.createElement('canvas');
    canvas.width = outWidth;
    canvas.height = outHeight;
    const context = canvas.getContext('2d');
    const tick = timeSlicer();

    if (factor === 1) {
      await putImageDataInStrips(context, new ImageData(pixels, width, height), tick);
      return { canvas: canvas, factor: 1 };
    }

    const image = context.createImageData(outWidth, outHeight);
    const out = image.data;

    for (let oy = 0; oy < outHeight; oy += 1) {
      const lastRow = Math.min((oy + 1) * factor, height);

      for (let ox = 0; ox < outWidth; ox += 1) {
        const firstColumn = ox * factor;
        const lastColumn = Math.min(firstColumn + factor, width);
        let r = 0, g = 0, b = 0, a = 0, counted = 0;

        for (let y = oy * factor; y < lastRow; y += 1) {
          let at = (y * width + firstColumn) * 4;
          for (let x = firstColumn; x < lastColumn; x += 1) {
            r += pixels[at];
            g += pixels[at + 1];
            b += pixels[at + 2];
            a += pixels[at + 3];
            at += 4;
            counted += 1;
          }
        }

        const to = (oy * outWidth + ox) * 4;
        out[to] = r / counted;
        out[to + 1] = g / counted;
        out[to + 2] = b / counted;
        out[to + 3] = a / counted;
      }

      // Averaging a huge result takes a moment. Yielding by time rather than
      // every 64 rows keeps each pause short however wide the image is.
      await tick();
    }

    await putImageDataInStrips(context, image, tick);
    return { canvas: canvas, factor: factor };
  }


  /* ========================================================================
     18. BEFORE / AFTER COMPARISON
     ======================================================================== */

  function renderResult(engine) {
    const ratio = state.originalWidth + '/' + state.originalHeight;

    el.baFrame.style.setProperty('--ar', ratio);
    el.baBefore.src = state.originalUrl;
    el.baBefore.alt = 'Original image, ' + formatDimensions(state.originalWidth, state.originalHeight);

    // Show the preview canvas directly instead of decoding a second copy.
    el.baAfter.textContent = '';
    state.resultImage.setAttribute('role', 'img');
    state.resultImage.setAttribute(
      'aria-label',
      'Upscaled image, ' + formatDimensions(state.resultWidth, state.resultHeight)
    );
    el.baAfter.appendChild(state.resultImage);

    // A reduced preview should say so rather than pass itself off as the
    // result: the file holds every pixel, this view does not.
    el.baTagAfter.textContent = state.previewFactor > 1
      ? 'AFTER · preview at 1/' + state.previewFactor + ' size'
      : 'AFTER';

    setComparePosition(50);
    setZoomIndex(0);

    // Result information card
    el.infoOrigDims.textContent = formatDimensions(state.originalWidth, state.originalHeight);
    el.infoOrigSize.textContent = formatBytes(state.file.size) + ' · ' + formatLabel(state.file);
    el.infoOutDims.textContent = formatDimensions(state.resultWidth, state.resultHeight);
    el.infoOutSize.textContent = runtime.outputBlob ? formatBytes(runtime.outputBlob.size) : '—';
    el.infoScale.textContent = state.scale + '×';
    el.infoEngine.textContent = describeEngine(engine);

    // Download options start from a clean slate every run.
    const target = targetLabel(state.scale);
    const band = bandLabel(state.scale);
    const quality = Math.round(runtime.outputQuality * 100);
    const noAlpha = runtime.outputChannels === 3
      ? ', with the unused transparency channel left out'
      : '';
    el.infoFormat.textContent = runtime.outputKind === 'jpeg' ? 'JPG' : OUTPUT_EXTENSION.toUpperCase();
    el.infoQuality.textContent =
      runtime.outputKind === 'jpeg' ? 'JPEG · ' + quality + '% quality'
      : runtime.outputKind === 'hybrid' ? 'Lossless · partly compressed'
      : runtime.outputKind === 'compressed' ? 'Lossless · compressed'
      : 'Lossless · uncompressed';

    let hint;
    if (runtime.outputOverCap) {
      hint = 'Lossless PNG, but bigger than ' + band + ': a file that size of this image ' +
        'needs more memory than this browser can give. A lower upscale level makes a smaller file.';
    } else if (runtime.outputKind === 'jpeg') {
      hint = 'JPEG at ' + quality + '% quality, full resolution — no lossless file of this image ' +
        'comes down to ' + band + ', so this is the highest quality that does.';
    } else if (runtime.outputKind === 'hybrid' || runtime.outputKind === 'compressed') {
      hint = 'PNG, lossless — ' + (runtime.outputKind === 'hybrid' ? 'partly ' : '') +
        'compressed to land at about ' + target + noAlpha + '. Every pixel is exactly as the AI produced it.';
    } else {
      hint = 'PNG, stored without compression' + noAlpha + ' — every pixel exactly as the AI ' +
        'produced it. A bigger file, not a better image.';
    }
    el.filesizeHint.textContent = hint;
    refreshOutputSize();
  }

  function setComparePosition(percent) {
    runtime.comparePos = clamp(percent, 0, 100);
    const rounded = Math.round(runtime.comparePos);
    el.baFrame.style.setProperty('--pos', runtime.comparePos + '%');
    el.baHandle.setAttribute('aria-valuenow', String(rounded));
    el.baHandle.setAttribute('aria-valuetext', rounded + '%');
  }

  /**
   * Converts a pointer position into a 0-100 divider position.
   * The stage is scaled from its centre when zoomed in, so the screen position
   * has to be mapped back into the image's own coordinate space.
   */
  function positionFromPointer(clientX) {
    const rect = el.baFrame.getBoundingClientRect();
    if (!rect.width) return runtime.comparePos;

    const screenFraction = (clientX - rect.left) / rect.width;
    const zoom = ZOOM_STEPS[runtime.zoomIndex] || 1;
    const imageFraction = 0.5 + (screenFraction - 0.5) / zoom;

    return clamp(imageFraction * 100, 0, 100);
  }

  function initComparison() {
    const frame = el.baFrame;

    const onPointerDown = (event) => {
      if (event.button !== undefined && event.button !== 0) return;
      runtime.dragging = true;
      frame.classList.add('is-dragging');
      if (frame.setPointerCapture) {
        try { frame.setPointerCapture(event.pointerId); } catch (err) { /* ignore */ }
      }
      setComparePosition(positionFromPointer(event.clientX));
      event.preventDefault();
    };

    const onPointerMove = (event) => {
      if (!runtime.dragging) return;
      setComparePosition(positionFromPointer(event.clientX));
      event.preventDefault();
    };

    const onPointerUp = (event) => {
      if (!runtime.dragging) return;
      runtime.dragging = false;
      frame.classList.remove('is-dragging');
      if (frame.releasePointerCapture && event.pointerId !== undefined) {
        try { frame.releasePointerCapture(event.pointerId); } catch (err) { /* ignore */ }
      }
    };

    frame.addEventListener('pointerdown', onPointerDown);
    frame.addEventListener('pointermove', onPointerMove);
    // Pointer capture keeps the moves coming even outside the frame; the window
    // listeners are the safety net for a release that happens elsewhere.
    frame.addEventListener('pointerup', onPointerUp);
    window.addEventListener('pointerup', onPointerUp);
    window.addEventListener('pointercancel', onPointerUp);

    // Keyboard control of the divider.
    el.baHandle.addEventListener('keydown', (event) => {
      const step = event.shiftKey ? 10 : 2;
      let next = runtime.comparePos;

      switch (event.key) {
        case 'ArrowLeft':  next -= step; break;
        case 'ArrowRight': next += step; break;
        case 'Home':       next = 0; break;
        case 'End':        next = 100; break;
        default: return;
      }

      event.preventDefault();
      event.stopPropagation();
      setComparePosition(next);
    });

    // A click on the handle should not also start a drag on the frame twice.
    el.baHandle.addEventListener('click', (event) => event.stopPropagation());
  }


  /* ========================================================================
     19. ZOOM
     ======================================================================== */

  function setZoomIndex(index) {
    runtime.zoomIndex = clamp(index, 0, ZOOM_STEPS.length - 1);
    const zoom = ZOOM_STEPS[runtime.zoomIndex];

    el.baFrame.style.setProperty('--zoom', String(zoom));
    const label = Math.round(zoom * 100) + '%';
    if (el.zoomReset.textContent !== label) {
      el.zoomReset.textContent = label;
      restartAnimation(el.zoomReset);
    }
    el.zoomOut.disabled = runtime.zoomIndex === 0;
    el.zoomIn.disabled = runtime.zoomIndex === ZOOM_STEPS.length - 1;
  }

  function initZoom() {
    el.zoomIn.addEventListener('click', () => setZoomIndex(runtime.zoomIndex + 1));
    el.zoomOut.addEventListener('click', () => setZoomIndex(runtime.zoomIndex - 1));
    el.zoomReset.addEventListener('click', () => setZoomIndex(0));
  }


  /* ========================================================================
     20. DOWNLOAD
     ======================================================================== */

  function buildFileName() {
    const extension = runtime.outputKind === 'jpeg' ? 'jpg' : OUTPUT_EXTENSION;
    return getBaseName(state.file.name) + '-' + state.scale + 'x-upscaled.' + extension;
  }

  /** Encodes once and keeps it, so downloading costs nothing extra. */
  async function getOutputBlob() {
    if (runtime.outputBlob) return runtime.outputBlob;
    keepOutput(await encodeOutput(state.resultPixels, state.resultWidth, state.resultHeight, state.scale));
    return runtime.outputBlob;
  }

  /**
   * Shows what the file actually weighs, before downloading.
   *
   * The size used to appear only once the file had been saved. A PNG's size
   * depends on the picture's content, so it cannot be predicted — it has to
   * be encoded and measured, which is what happens here. The encode is kept,
   * so the download itself is immediate.
   */
  function refreshOutputSize() {
    if (!state.resultPixels) return;

    const token = ++runtime.sizeToken;
    const show = (text) => {
      // Re-run the fade only on a genuinely new figure, so that repeated
      // measurements of the same value do not restart it mid-flight.
      if (el.downloadSizeValue.textContent !== text) {
        el.downloadSizeValue.textContent = text;
        restartAnimation(el.downloadSizeValue);
      }
      el.infoOutSize.textContent = text;
    };

    show('measuring…');

    getOutputBlob().then(
      (blob) => { if (token === runtime.sizeToken) show(formatBytes(blob.size)); },
      () => { if (token === runtime.sizeToken) show('—'); }
    );
  }

  async function handleDownload() {
    if (!state.resultPixels || state.processing) return;

    const label = el.downloadBtn.innerHTML;
    el.downloadBtn.disabled = true;
    el.downloadBtn.textContent = 'Preparing file…';

    try {
      const blob = await getOutputBlob();
      const fileName = buildFileName();
      const url = URL.createObjectURL(blob);

      const link = document.createElement('a');
      link.href = url;
      link.download = fileName;
      link.rel = 'noopener';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      // Give the browser a moment to start the download before releasing it.
      setTimeout(() => revoke(url), 2000);

      el.infoOutSize.textContent = formatBytes(blob.size);
      showToast('Download Started', fileName + ' · ' + formatBytes(blob.size), 'success');
    } catch (error) {
      console.error('[ImageUp AI] Download failed:', error);
      showToast(
        'Download Failed',
        error && error.userMessage ? error.userMessage : 'The image could not be prepared for download.',
        'error'
      );
    } finally {
      el.downloadBtn.disabled = false;
      el.downloadBtn.innerHTML = label;
    }
  }

  function initDownload() {
    el.downloadBtn.addEventListener('click', handleDownload);
  }


  /* ========================================================================
     21. RESET
     ======================================================================== */

  function releaseResultResources() {
    if (state.resultImage && state.resultImage.parentNode) {
      state.resultImage.parentNode.removeChild(state.resultImage);
    }
    releaseCanvas(state.resultImage);
    state.resultImage = null;
    state.resultPixels = null;      // the big one: up to a gigabyte of pixels
    state.previewFactor = 1;
    state.resultWidth = 0;
    state.resultHeight = 0;
    runtime.outputBlob = null;
    runtime.outputKind = 'stored';
    runtime.outputChannels = 4;
    runtime.outputQuality = 1;
    runtime.outputOverCap = false;
    // Invalidates any measurement still in flight for the old result.
    runtime.sizeToken += 1;
    el.downloadSizeValue.textContent = '—';
  }

  function releaseImageResources() {
    closeImage(state.originalImage);
    revoke(state.originalUrl);
    state.originalImage = null;
    state.originalUrl = null;
  }

  /** Full reset — back to the empty upload screen. */
  function resetAll() {
    releaseResultResources();
    releaseImageResources();

    state.file = null;
    state.originalWidth = 0;
    state.originalHeight = 0;
    state.scale = CONFIG.defaultScale;
    state.processing = false;

    runtime.comparePos = 50;
    runtime.zoomIndex = 0;
    runtime.dragging = false;

    // Clear every image reference so nothing large stays in memory.
    el.previewImg.removeAttribute('src');
    el.previewImg.alt = 'Preview of the image you selected';
    el.baBefore.removeAttribute('src');
    el.baAfter.textContent = '';
    el.fileInput.value = '';

    el.fileName.textContent = '';
    el.fileDims.textContent = '';
    el.fileSize.textContent = '';
    el.fileFormat.textContent = '';
    el.outputRes.textContent = '0 × 0';
    el.outputEstimate.textContent = '';
    el.infoOutSize.textContent = '—';

    el.downloadBtn.disabled = false;

    el.baFrame.style.setProperty('--pos', '50%');
    el.baFrame.style.setProperty('--zoom', '1');
    setZoomIndex(0);
    setScale(CONFIG.defaultScale, { silent: true });

    el.upscaleBtn.disabled = true;
    el.upscaleBtn.removeAttribute('title');

    showPanel('upload');
    scrollToolIntoView();
  }


  /* ========================================================================
     22. INIT
     ======================================================================== */

  /** Short label for the engine that produced a result. */
  function describeEngine(engine) {
    if (engine === 'api') return 'AI API';
    const info = ImageUpAI.getInfo();
    const backend = info.backend === 'webgpu'
      ? 'WebGPU'
      : 'WASM' + (info.threads > 1 ? ' ×' + info.threads : '');
    return 'Real-ESRGAN ×4 · ' + backend;
  }

  function init() {
    initSmoothScrolling();
    initUpload();
    initDragAndDrop();
    initPaste();
    initScaleSelection();
    initComparison();
    initZoom();
    initDownload();

    el.upscaleBtn.addEventListener('click', handleUpscale);
    el.cancelBtn.addEventListener('click', handleCancel);
    el.resetBtn.addEventListener('click', resetAll);

    // Release object URLs if the page is closed mid-session.
    window.addEventListener('pagehide', () => {
      revoke(state.originalUrl);
    });

    setScale(CONFIG.defaultScale, { silent: true });
    setZoomIndex(0);
    showPanel('upload');
    preloadEngineWhenIdle();
  }

  init();
})();
