/* VAE Rhythm project page: link config, copy buttons, the main-table toggle and three small SVG charts.
   Plain JavaScript, no dependencies; data come from assets/data/page-data.js (window.VR_DATA). */
(function () {
  'use strict';

  var SVGNS = 'http://www.w3.org/2000/svg';
  var reduceMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

  /* ---------------------------------------------------------------- helpers */
  function $(id) { return document.getElementById(id); }

  function svgEl(tag, attrs, parent) {
    var el = document.createElementNS(SVGNS, tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (attrs[k] !== undefined && attrs[k] !== null) el.setAttribute(k, attrs[k]);
      });
    }
    if (parent) parent.appendChild(el);
    return el;
  }

  function htmlEl(tag, cls, text, parent) {
    var el = document.createElement(tag);
    if (cls) el.className = cls;
    if (text !== undefined && text !== null) el.textContent = text;
    if (parent) parent.appendChild(el);
    return el;
  }

  function minus(s) { return String(s).replace(/-/g, '\u2212'); }
  function fmt(v, d) { return minus(v.toFixed(d)); }
  function signed(v, d) {
    if (v > 0) return '+' + v.toFixed(d);
    if (v < 0) return '\u2212' + Math.abs(v).toFixed(d);
    return (0).toFixed(d);
  }
  function scale(d0, d1, r0, r1) {
    return function (v) { return r0 + (v - d0) / (d1 - d0) * (r1 - r0); };
  }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function linePath(xs, ys) {
    var d = '';
    for (var i = 0; i < xs.length; i++) d += (i ? 'L' : 'M') + xs[i].toFixed(1) + ',' + ys[i].toFixed(1);
    return d;
  }
  function bandPath(xs, lo, hi) {
    var d = '';
    for (var i = 0; i < xs.length; i++) d += (i ? 'L' : 'M') + xs[i].toFixed(1) + ',' + hi[i].toFixed(1);
    for (var j = xs.length - 1; j >= 0; j--) d += 'L' + xs[j].toFixed(1) + ',' + lo[j].toFixed(1);
    return d + 'Z';
  }
  function range(n) { var a = []; for (var i = 0; i < n; i++) a.push(i); return a; }

  /* Re-render a chart when its container width changes. */
  function observeWidth(el, fn) {
    var last = 0;
    function run() {
      var w = Math.floor(el.clientWidth);
      if (w > 0 && w !== last) { last = w; fn(w); }
    }
    if ('ResizeObserver' in window) new ResizeObserver(run).observe(el);
    else window.addEventListener('resize', run);
    run();
    return function () { return last; };
  }

  /* Truncate an SVG text node to a maximum width, keeping the full text in a <title>. */
  function fitText(node, maxW) {
    if (!node.getComputedTextLength) return;
    var full = node.textContent;
    if (node.getComputedTextLength() <= maxW) return;
    var t = full;
    while (t.length > 3 && node.getComputedTextLength() > maxW) {
      t = t.slice(0, -1);
      node.textContent = t.replace(/[\s,]+$/, '') + '\u2026';
    }
    svgEl('title', null, node).textContent = full;
  }

  /* "C4" -> C with subscript 4, "Ppk" -> P with subscript pk */
  function statName(stat, parent) {
    var base = stat.charAt(0), sub = stat.slice(1);
    var wrap = htmlEl('span', null, base, parent);
    htmlEl('sub', null, sub, wrap);
    return wrap;
  }

  /* ---------------------------------------------------------------- project links */
  function initLinks() {
    var cfg = window.PROJECT_LINKS || {};
    var links = document.querySelectorAll('a[data-link]');
    Array.prototype.forEach.call(links, function (a) {
      var url = String(cfg[a.getAttribute('data-link')] || '').trim();
      if (url && url !== '#') {
        a.setAttribute('href', url);
        a.classList.remove('is-pending');
        a.removeAttribute('aria-disabled');
      } else {
        a.setAttribute('href', '#');
        a.classList.add('is-pending');
        a.setAttribute('aria-disabled', 'true');
        a.setAttribute('title', 'Not public yet');
      }
    });
    document.addEventListener('click', function (e) {
      var a = e.target && e.target.closest ? e.target.closest('a.is-pending') : null;
      if (a) e.preventDefault();
    });
  }

  /* ---------------------------------------------------------------- copy buttons */
  function legacyCopy(text) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.top = '0';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    var ok = false;
    try { ok = document.execCommand('copy'); } catch (err) { ok = false; }
    document.body.removeChild(ta);
    return ok;
  }

  /* Clipboard API first, then the older execCommand route (some embedded or file:// contexts block the API). */
  function copyText(text) {
    return new Promise(function (resolve, reject) {
      function fallback() { if (legacyCopy(text)) resolve(); else reject(new Error('copy failed')); }
      if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(resolve, fallback);
      else fallback();
    });
  }

  function initCopy() {
    var status = $('copy-status');
    Array.prototype.forEach.call(document.querySelectorAll('[data-copy]'), function (btn) {
      btn.addEventListener('click', function () {
        var src = $(btn.getAttribute('data-copy'));
        if (!src) return;
        var label = btn.textContent;
        copyText(src.textContent).then(function () {
          btn.textContent = 'Copied';
          if (status) status.textContent = 'Copied to the clipboard.';
        }, function () {
          var sel = window.getSelection(), r = document.createRange();
          r.selectNodeContents(src);
          sel.removeAllRanges();
          sel.addRange(r);
          btn.textContent = 'Selected';
          if (status) status.textContent = 'Text selected; press Control or Command and C to copy.';
        });
        setTimeout(function () { btn.textContent = label; }, 1800);
      });
    });
  }

  /* ---------------------------------------------------------------- main results table */
  function initTable() {
    var wrap = $('main-table-wrap'), btn = $('table-toggle');
    if (!wrap || !btn) return;
    wrap.classList.add('is-compact');
    btn.hidden = false;
    btn.addEventListener('click', function () {
      var compact = wrap.classList.toggle('is-compact');
      btn.textContent = compact ? 'Show all 19 methods' : 'Show fewer methods';
      btn.setAttribute('aria-expanded', String(!compact));
    });
  }

  /* ---------------------------------------------------------------- section highlight in the nav */
  function initNav() {
    var links = document.querySelectorAll('.site-nav a[href^="#"]');
    if (!links.length || !('IntersectionObserver' in window)) return;
    var map = {};
    Array.prototype.forEach.call(links, function (a) { map[a.getAttribute('href').slice(1)] = a; });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        Array.prototype.forEach.call(links, function (a) { a.removeAttribute('aria-current'); });
        var a = map[en.target.id];
        if (a) a.setAttribute('aria-current', 'true');
      });
    }, { rootMargin: '-40% 0px -55% 0px' });
    Object.keys(map).forEach(function (id) { var s = $(id); if (s) io.observe(s); });
    // the hero and the overview block above the first section clear the highlight
    Array.prototype.forEach.call(document.querySelectorAll('.hero, .intro-block'), function (el) { io.observe(el); });
  }

  /* ---------------------------------------------------------------- hero strip */
  function initHeroStrip(byId) {
    var host = $('hero-strip');
    var gen = byId['sora-w13'], real = byId['aigvdbench-real-clips-w13'];
    if (!host || !gen || !real) return;
    var animated = false;
    observeWidth(host, function (w) {
      host.textContent = '';
      var h = host.clientHeight || 140;
      var narrow = w < 520;
      var m = { l: 30, r: narrow ? 50 : 70, t: 6, b: 24 };
      var svg = svgEl('svg', { width: w, height: h, viewBox: '0 0 ' + w + ' ' + h, 'aria-hidden': 'true', focusable: 'false' }, host);
      var x = scale(0, 23, m.l, w - m.r);
      var L = gen.ylim;
      var y = scale(-L, L, h - m.b, m.t);
      var step = x(1) - x(0);
      for (var k = gen.bandPhase; k < 24; k += gen.period) {
        svgEl('rect', { 'class': 'band', x: (x(k) - step / 2).toFixed(1), y: m.t, width: step.toFixed(1), height: h - m.b - m.t }, svg);
      }
      svgEl('line', { 'class': 'zero', x1: m.l, x2: w - m.r, y1: y(0), y2: y(0) }, svg);
      [L, 0, -L].forEach(function (t) {
        var lab = t === 0 ? '0' : (t > 0 ? '+' + t + '%' : '\u2212' + Math.abs(t) + '%');
        svgEl('text', { 'class': 'tick-label', x: m.l - 6, y: y(t) + 4, 'text-anchor': 'end' }, svg).textContent = lab;
      });
      var xt = narrow ? [0, 8, 16] : [0, 4, 8, 12, 16, 20];
      xt.forEach(function (t) {
        svgEl('text', { 'class': 'tick-label', x: x(t), y: h - 6, 'text-anchor': 'middle' }, svg).textContent = String(t);
      });
      svgEl('text', { 'class': 'tick-label', x: x(23), y: h - 6, 'text-anchor': 'end' }, svg).textContent = narrow ? 'k' : 'offset k';
      var ks = range(24);
      var xs = ks.map(x);
      svgEl('path', { 'class': 'line-real', d: linePath(xs, real.mean.map(y)) }, svg);
      var pg = svgEl('path', { 'class': 'line-gen', d: linePath(xs, gen.mean.map(y)) }, svg);
      // direct labels at the line ends, pushed apart if they would collide
      var yg = y(gen.mean[23]), yr = y(real.mean[23]);
      if (Math.abs(yg - yr) < 15) {
        var mid = (yg + yr) / 2, up = yg <= yr;
        yg = mid + (up ? -8 : 8);
        yr = mid + (up ? 8 : -8);
      }
      svgEl('text', { 'class': 'direct-label', x: x(23) + 8, y: yg + 4 }, svg).textContent = 'Sora';
      svgEl('text', { 'class': 'direct-label secondary', x: x(23) + 8, y: yr + 4 }, svg).textContent = 'Real';
      if (!animated && !reduceMotion && pg.getTotalLength) {
        animated = true;
        var len = pg.getTotalLength();
        pg.style.strokeDasharray = len + ' ' + len;
        pg.style.strokeDashoffset = String(len);
        pg.getBoundingClientRect();
        pg.style.transition = 'stroke-dashoffset 1.8s cubic-bezier(.45,0,.2,1)';
        requestAnimationFrame(function () { pg.style.strokeDashoffset = '0'; });
      }
    });
  }

  /* ---------------------------------------------------------------- explorer */
  function initExplorer(panels, byId) {
    var select = $('ex-select'), host = $('ex-chart'), stats = $('ex-stats'), gridHost = $('ex-grid');
    var tableHost = $('ex-table'), readout = $('ex-readout'), browse = $('ex-browse');
    if (!select || !host) return;

    function optionLabel(p) {
      if (p.kind === 'real') return p.name;
      if (p.window === 9) return p.name + ' (9-frame windows)';
      if (p.group === 'Short-window scans') return p.name + ' (13-frame windows, ' + p.n + ' clips)';
      return p.name;
    }
    function tileValue(p) {
      if (p.kind === 'real') return 'n = ' + p.n;
      if (p.auroc !== undefined) return p.auroc.toFixed(1);
      if (p.window === 9) return 'W = 9';
      return p.n + ' clips';
    }
    function tileAria(p) {
      var s = optionLabel(p) + ', ' + p.group;
      if (p.auroc !== undefined) s += ', zero-shot AUROC ' + p.auroc.toFixed(1);
      return s;
    }

    // select with one optgroup per group
    var group = null, og = null, groups = [];
    panels.forEach(function (p) {
      if (p.group !== group) {
        group = p.group;
        og = document.createElement('optgroup');
        og.label = group;
        select.appendChild(og);
        groups.push({ name: group, items: [] });
      }
      var o = document.createElement('option');
      o.value = p.id;
      o.textContent = optionLabel(p);
      og.appendChild(o);
      groups[groups.length - 1].items.push(p);
    });

    // tiles: small multiples that double as the picker (a radio group with roving focus)
    var tiles = [];
    groups.forEach(function (g, gi) {
      var box = htmlEl('div', null, null, gridHost);
      box.setAttribute('role', 'group');
      var title = htmlEl('p', 'ex-group-title', g.name, box);
      title.id = 'ex-group-' + gi;
      box.setAttribute('aria-labelledby', title.id);
      var wrap = htmlEl('div', 'ex-tiles', null, box);
      g.items.forEach(function (p) {
        var b = htmlEl('button', 'tile', null, wrap);
        b.type = 'button';
        b.setAttribute('role', 'radio');
        b.setAttribute('aria-checked', 'false');
        b.setAttribute('aria-label', tileAria(p));
        b.tabIndex = -1;
        b.dataset.id = p.id;
        var head = htmlEl('span', 'tile-head', null, b);
        htmlEl('span', 'tile-name', p.kind === 'real' ? p.bench : p.name, head);
        htmlEl('span', 'tile-val', tileValue(p), head);
        var sp = svgEl('svg', { viewBox: '0 0 23 20', preserveAspectRatio: 'none', 'aria-hidden': 'true', focusable: 'false' }, b);
        var sy = function (v) { return 10 - v / p.ylim * 10; };
        svgEl('line', { 'class': 'sp-zero', x1: 0, x2: 23, y1: 10, y2: 10 }, sp);
        if (p.realRef && byId[p.realRef]) svgEl('path', { 'class': 'sp-real', d: linePath(range(24), byId[p.realRef].mean.map(sy)) }, sp);
        svgEl('path', { 'class': p.kind === 'real' ? 'sp-real' : 'sp-gen', d: linePath(range(24), p.mean.map(sy)) }, sp);
        tiles.push(b);
      });
    });

    var current = Math.max(0, panels.findIndex(function (p) { return p.id === 'sora-w13'; }));
    var activeK = null;
    var tooltip = htmlEl('div', 'tooltip', null, host);
    tooltip.hidden = true;
    var chart = null;   // { x, y, svg, cross, dotG, dotR }

    function render(w) {
      var p = panels[current];
      var ref = p.realRef ? byId[p.realRef] : null;
      var isReal = p.kind === 'real';
      Array.prototype.slice.call(host.querySelectorAll('svg')).forEach(function (s) { s.remove(); });
      var h = Math.round(clamp(w * 0.46, 230, 330));
      var m = { l: 44, r: 12, t: 22, b: 40 };
      var svg = svgEl('svg', { width: w, height: h, viewBox: '0 0 ' + w + ' ' + h, 'aria-hidden': 'true', focusable: 'false' });
      host.insertBefore(svg, tooltip);
      var x = scale(0, 23, m.l + 6, w - m.r - 6);
      var L = p.ylim;
      var y = scale(-L, L, h - m.b, m.t);
      var step = x(1) - x(0);
      if (p.period) {
        for (var k = p.bandPhase; k < 24; k += p.period) {
          svgEl('rect', { 'class': 'band', x: (x(k) - step / 2).toFixed(1), y: m.t, width: step.toFixed(1), height: h - m.b - m.t }, svg);
        }
      }
      [-L, -L / 2, L / 2, L].forEach(function (t) {
        svgEl('line', { 'class': 'grid', x1: m.l, x2: w - m.r, y1: y(t), y2: y(t) }, svg);
      });
      svgEl('line', { 'class': 'zero', x1: m.l, x2: w - m.r, y1: y(0), y2: y(0) }, svg);
      [-L, -L / 2, 0, L / 2, L].forEach(function (t) {
        var lab = t === 0 ? '0' : (t > 0 ? '+' : '\u2212') + Math.abs(t);
        svgEl('text', { 'class': 'tick-label', x: m.l - 8, y: y(t) + 4, 'text-anchor': 'end' }, svg).textContent = lab;
      });
      svgEl('text', { 'class': 'axis-title', x: 0, y: 12 }, svg).textContent = 'r(k), %';
      (w < 420 ? [0, 8, 16] : [0, 4, 8, 12, 16, 20]).forEach(function (t) {
        svgEl('text', { 'class': 'tick-label', x: x(t), y: h - m.b + 16, 'text-anchor': 'middle' }, svg).textContent = String(t);
      });
      svgEl('text', { 'class': 'axis-title', x: (m.l + w - m.r) / 2, y: h - 4, 'text-anchor': 'middle' }, svg).textContent = 'Start offset k (frames)';

      var ks = range(24), xs = ks.map(x);
      var hi = p.mean.map(function (v, i) { return y(v + p.sem[i]); });
      var lo = p.mean.map(function (v, i) { return y(v - p.sem[i]); });
      svgEl('path', { 'class': isReal ? 'sem-real' : 'sem', d: bandPath(xs, lo, hi) }, svg);
      if (ref) svgEl('path', { 'class': 'line-real', d: linePath(xs, ref.mean.map(y)) }, svg);
      svgEl('path', { 'class': isReal ? 'line-real is-main' : 'line-gen', d: linePath(xs, p.mean.map(y)) }, svg);

      var cross = svgEl('line', { 'class': 'cross', y1: m.t, y2: h - m.b, visibility: 'hidden' }, svg);
      var dotR = ref ? svgEl('circle', { 'class': 'dot-real', r: 4, visibility: 'hidden' }, svg) : null;
      var dotG = svgEl('circle', { 'class': isReal ? 'dot-real' : 'dot-gen', r: 4.5, visibility: 'hidden' }, svg);
      var hit = svgEl('rect', { x: m.l, y: 0, width: w - m.l - m.r, height: h, fill: 'transparent' }, svg);
      hit.style.touchAction = 'pan-y';
      chart = { p: p, ref: ref, x: x, y: y, w: w, h: h, m: m, cross: cross, dotG: dotG, dotR: dotR, svg: svg };

      function kFromEvent(e) {
        var r = svg.getBoundingClientRect();
        var px = (e.clientX - r.left) * (w / r.width);
        return clamp(Math.round((px - x(0)) / step), 0, 23);
      }
      hit.addEventListener('pointermove', function (e) { showAt(kFromEvent(e), false); });
      hit.addEventListener('pointerdown', function (e) { showAt(kFromEvent(e), false); });
      hit.addEventListener('pointerleave', function () { if (document.activeElement !== host) hide(); });
      if (activeK !== null) showAt(activeK, false);
    }

    function showAt(k, announce) {
      if (!chart) return;
      activeK = k;
      var c = chart, p = c.p;
      var cx = c.x(k);
      c.cross.setAttribute('x1', cx);
      c.cross.setAttribute('x2', cx);
      c.cross.setAttribute('visibility', 'visible');
      c.dotG.setAttribute('cx', cx);
      c.dotG.setAttribute('cy', c.y(clamp(p.mean[k], -p.ylim, p.ylim)));
      c.dotG.setAttribute('visibility', 'visible');
      if (c.dotR) {
        c.dotR.setAttribute('cx', cx);
        c.dotR.setAttribute('cy', c.y(clamp(c.ref.mean[k], -p.ylim, p.ylim)));
        c.dotR.setAttribute('visibility', 'visible');
      }
      tooltip.textContent = '';
      htmlEl('div', 'tt-head', 'Offset k = ' + k, tooltip);
      var row = htmlEl('div', 'tt-row', null, tooltip);
      htmlEl('span', 'key-line ' + (p.kind === 'real' ? 'real' : 'gen'), null, row);
      htmlEl('b', null, fmt(p.mean[k], 2) + '%', row);
      htmlEl('span', null, '\u00b1' + p.sem[k].toFixed(2) + (p.kind === 'real' ? ' real' : ' generated'), row);
      if (c.ref) {
        var row2 = htmlEl('div', 'tt-row', null, tooltip);
        htmlEl('span', 'key-line real', null, row2);
        htmlEl('b', null, fmt(c.ref.mean[k], 2) + '%', row2);
        htmlEl('span', null, 'real', row2);
      }
      tooltip.hidden = false;
      var tw = tooltip.offsetWidth, sw = c.svg.getBoundingClientRect().width / c.w;
      var left = cx * sw + 12;
      if (left + tw > host.clientWidth) left = cx * sw - tw - 12;
      tooltip.style.left = Math.max(0, left) + 'px';
      tooltip.style.top = (c.m.t + 2) + 'px';
      if (announce) {
        var msg = 'Offset ' + k + ': ' + (p.kind === 'real' ? 'real ' : 'generated ') + fmt(p.mean[k], 2) + ' percent';
        if (c.ref) msg += ', real ' + fmt(c.ref.mean[k], 2) + ' percent';
        readout.textContent = msg;
      }
    }
    function hide() {
      activeK = null;
      tooltip.hidden = true;
      if (!chart) return;
      ['cross', 'dotG', 'dotR'].forEach(function (key) { if (chart[key]) chart[key].setAttribute('visibility', 'hidden'); });
    }

    host.addEventListener('keydown', function (e) {
      var k = activeK === null ? 0 : activeK;
      if (e.key === 'ArrowRight') k = Math.min(23, activeK === null ? 0 : k + 1);
      else if (e.key === 'ArrowLeft') k = Math.max(0, activeK === null ? 0 : k - 1);
      else if (e.key === 'Home') k = 0;
      else if (e.key === 'End') k = 23;
      else if (e.key === 'Escape') { hide(); return; }
      else return;
      e.preventDefault();
      showAt(k, true);
    });
    host.addEventListener('blur', hide);

    function statRow(dl, label, value, ref) {
      var item = htmlEl('div', null, null, dl);
      var dt = htmlEl('dt', null, null, item);
      if (typeof label === 'string') dt.textContent = label; else dt.appendChild(label);
      var dd = htmlEl('dd', null, null, item);
      htmlEl('span', 'value', value, dd);
      if (ref) htmlEl('span', 'ref', ref, dd);
    }

    function renderStats() {
      var p = panels[current];
      stats.textContent = '';
      htmlEl('h4', null, optionLabel(p), stats);
      var meta = htmlEl('p', 'meta', null, stats);
      var bits = [p.group, p.n + ' clips, ' + p.window + '-frame windows'];
      if (p.vae) bits.push('source VAE: ' + p.vae);
      bits.forEach(function (b, i) { if (i) htmlEl('br', null, null, meta); meta.appendChild(document.createTextNode(b)); });
      var dl = htmlEl('dl', null, null, stats);
      if (p.kind === 'real') {
        var lab = document.createDocumentFragment();
        lab.appendChild(document.createTextNode('Median '));
        statName('C4', lab);
        statRow(dl, lab, p.medianC4.toFixed(2));
        statRow(dl, 'Phase concentration R, period 4', p.realR4.toFixed(2), 'Near 0 means scattered phases.');
        htmlEl('p', 'caveat', 'Each generated variant is compared with the real clips of its own benchmark.', stats);
        return;
      }
      if (p.auroc !== undefined) {
        statRow(dl, 'Zero-shot AUROC', p.auroc.toFixed(1),
          p.auroc720 !== null && p.auroc720 !== undefined ? String(p.auroc720) + ' at native 720p, on a smaller subset' : null);
        statRow(dl, 'Clips above the real 95th percentile', p.above + '%', 'Real clips: 5%, by definition');
        statRow(dl, 'Phase concentration R', p.R.toFixed(2), p.realR !== null && p.realR !== undefined ? 'Real clips: ' + p.realR.toFixed(2) : null);
        var lab2 = document.createDocumentFragment();
        lab2.appendChild(document.createTextNode('Median '));
        statName(p.stat, lab2);
        lab2.appendChild(document.createTextNode(', period ' + p.period));
        statRow(dl, lab2, p.median.toFixed(2), 'Real clips: ' + p.realMedian.toFixed(2));
        statRow(dl, 'Versus real clips', p.sig ? 'Significant' : 'Not significant',
          'Holm-corrected p\u00a0' + (p.p.charAt(0) === '<' ? p.p.replace(' ', '\u00a0') : '=\u00a0' + p.p));
        if (p.group === '2026 models') htmlEl('p', 'caveat', 'Compared with the DF26 real clips; this collection has no real cohort of its own.', stats);
      } else if (p.window === 9) {
        statRow(dl, 'Zero-shot AUROC', p.censusAuroc.toFixed(1), 'Not a matched comparison');
        htmlEl('p', 'caveat', 'This clip length only allows 9-frame windows, and no real clips were scanned that way, so the AUROC compares against 13-frame reals and the variant is left out of the significance tests.', stats);
      } else {
        htmlEl('p', 'caveat', 'Ten of this generator\u2019s 50 clips were scanned with 13-frame windows and are shown here; the other 40 are 33 frames long and use 9-frame windows. No statistics are reported for this panel.', stats);
      }
    }

    function renderTable() {
      var p = panels[current], ref = p.realRef ? byId[p.realRef] : null;
      tableHost.textContent = '';
      var t = htmlEl('table', 'mini', null, tableHost);
      htmlEl('caption', 'visually-hidden', 'Mean residual r(k) in percent for ' + optionLabel(p), t);
      var tr = htmlEl('tr', null, null, htmlEl('thead', null, null, t));
      ['k', p.kind === 'real' ? 'Real mean' : 'Generated mean', 'Standard error'].concat(ref ? ['Real mean'] : []).forEach(function (s) {
        htmlEl('th', null, s, tr).setAttribute('scope', 'col');
      });
      var tb = htmlEl('tbody', null, null, t);
      for (var k = 0; k < 24; k++) {
        var r = htmlEl('tr', null, null, tb);
        htmlEl('th', null, String(k), r).setAttribute('scope', 'row');
        htmlEl('td', null, fmt(p.mean[k], 3), r);
        htmlEl('td', null, p.sem[k].toFixed(3), r);
        if (ref) htmlEl('td', null, fmt(ref.mean[k], 3), r);
      }
    }

    var getWidth = observeWidth(host, render);

    function setCurrent(i, focusTile) {
      current = (i + panels.length) % panels.length;
      var p = panels[current];
      select.value = p.id;
      tiles.forEach(function (b, j) {
        b.setAttribute('aria-checked', j === current ? 'true' : 'false');
        b.tabIndex = j === current ? 0 : -1;
      });
      if (focusTile) tiles[current].focus();
      activeK = null;
      tooltip.hidden = true;
      if (getWidth()) render(getWidth());
      renderStats();
      renderTable();
    }

    select.addEventListener('change', function () {
      var i = panels.findIndex(function (p) { return p.id === select.value; });
      if (i >= 0) setCurrent(i, false);
    });
    $('ex-prev').addEventListener('click', function () { setCurrent(current - 1, false); });
    $('ex-next').addEventListener('click', function () { setCurrent(current + 1, false); });
    tiles.forEach(function (b, i) {
      b.addEventListener('click', function () { setCurrent(i, false); });
      b.addEventListener('keydown', function (e) {
        var d = 0;
        if (e.key === 'ArrowRight' || e.key === 'ArrowDown') d = 1;
        else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') d = -1;
        else if (e.key === 'Home') { e.preventDefault(); setCurrent(0, true); return; }
        else if (e.key === 'End') { e.preventDefault(); setCurrent(panels.length - 1, true); return; }
        if (!d) return;
        e.preventDefault();
        setCurrent(i + d, true);
      });
    });
    if (browse && window.matchMedia && window.matchMedia('(max-width: 40rem)').matches) browse.open = false;
    setCurrent(current, false);
  }

  /* ---------------------------------------------------------------- plug-in chart */
  function initPlugin(plugin) {
    var host = $('plugin-chart'), seg = $('plugin-seg'), tableHost = $('plugin-table'), readout = $('plugin-readout');
    if (!host || !seg) return;
    var BENCH = [
      ['df26All', 'DF26, all'],
      ['df26Closed', 'DF26, closed-source'],
      ['aigvdOpen', 'AIGVDBench, open-source'],
      ['aigvdClosed', 'AIGVDBench, closed-source'],
      ['closed2026', '2026 models']
    ];
    var bench = 'df26All', active = -1, rows = [], chart = null;
    var tooltip = htmlEl('div', 'tooltip', null, host);
    tooltip.hidden = true;

    var buttons = BENCH.map(function (b, i) {
      var btn = htmlEl('button', null, b[1], seg);
      btn.type = 'button';
      btn.setAttribute('role', 'radio');
      btn.setAttribute('aria-checked', b[0] === bench ? 'true' : 'false');
      btn.tabIndex = b[0] === bench ? 0 : -1;
      btn.addEventListener('click', function () { choose(i, false); });
      btn.addEventListener('keydown', function (e) {
        var d = (e.key === 'ArrowRight' || e.key === 'ArrowDown') ? 1 : (e.key === 'ArrowLeft' || e.key === 'ArrowUp') ? -1 : 0;
        if (!d) return;
        e.preventDefault();
        choose((i + d + BENCH.length) % BENCH.length, true);
      });
      return btn;
    });

    function choose(i, focus) {
      bench = BENCH[i][0];
      buttons.forEach(function (b, j) {
        b.setAttribute('aria-checked', j === i ? 'true' : 'false');
        b.tabIndex = j === i ? 0 : -1;
      });
      if (focus) buttons[i].focus();
      active = -1;
      tooltip.hidden = true;
      if (getWidth()) render(getWidth());
      renderTable();
    }

    function sorted() {
      return plugin.detectors.slice().sort(function (a, b) { return b.base[bench] - a.base[bench]; });
    }

    function render(w) {
      rows = sorted();
      Array.prototype.slice.call(host.querySelectorAll('svg')).forEach(function (s) { s.remove(); });
      var narrow = w < 560;
      var labelW = w >= 760 ? 250 : (narrow ? 118 : 178);
      var deltaW = narrow ? 46 : 58;
      var rowH = 26, top = 46, n = rows.length;
      var h = top + n * rowH + 6;
      var vals = [plugin.alone[bench]];
      rows.forEach(function (d) { vals.push(d.base[bench], d.plus[bench]); });
      var lo = Math.floor((Math.min.apply(null, vals) - 2) / 10) * 10;
      var hi = Math.min(100, Math.ceil((Math.max.apply(null, vals) + 2) / 10) * 10);
      var x0 = labelW + 12, x1 = w - deltaW - 10;
      var x = scale(lo, hi, x0, x1);
      var svg = svgEl('svg', { width: w, height: h, viewBox: '0 0 ' + w + ' ' + h, 'aria-hidden': 'true', focusable: 'false' });
      host.insertBefore(svg, tooltip);

      var hits = rows.map(function (d, i) {
        return svgEl('rect', { 'class': 'row-hit', x: 0, y: top + i * rowH, width: w, height: rowH, rx: 4 }, svg);
      });
      var stepT = (x1 - x0) < 220 ? 20 : ((hi - lo) > 40 ? 10 : 5);
      if (stepT === 20 && lo % 20) lo -= 10;
      x = scale(lo, hi, x0, x1);
      for (var t = lo; t <= hi + 1e-9; t += stepT) {
        svgEl('line', { 'class': 'grid', x1: x(t), x2: x(t), y1: top - 6, y2: h - 4 }, svg);
        svgEl('text', { 'class': 'tick-label', x: x(t), y: top - 12, 'text-anchor': 'middle' }, svg).textContent = String(t);
      }
      svgEl('text', { 'class': 'col-head', x: w, y: top - 12, 'text-anchor': 'end' }, svg).textContent = 'Change';
      svgEl('text', { 'class': 'col-head', x: 0, y: top - 12 }, svg).textContent = 'AUROC, %';
      var xa = x(plugin.alone[bench]);
      svgEl('line', { 'class': 'ref-line', x1: xa, x2: xa, y1: top - 4, y2: h - 4 }, svg);
      var refLabel = svgEl('text', { 'class': 'ref-label', x: xa, y: 11, 'text-anchor': 'middle' }, svg);
      refLabel.textContent = 'Rhythm alone ' + plugin.alone[bench].toFixed(1);
      var rl = refLabel.getComputedTextLength ? refLabel.getComputedTextLength() : 100;
      if (xa + rl / 2 > w) { refLabel.setAttribute('text-anchor', 'end'); refLabel.setAttribute('x', w); }
      else if (xa - rl / 2 < x0) { refLabel.setAttribute('text-anchor', 'start'); refLabel.setAttribute('x', Math.max(x0, xa - 4)); }

      rows.forEach(function (d, i) {
        var cy = top + i * rowH + rowH / 2;
        var b = d.base[bench], p = d.plus[bench], dv = d.delta[bench];
        var lab = svgEl('text', { 'class': 'row-label' + (d.ours ? ' ours' : ''), x: 0, y: cy + 4 }, svg);
        lab.textContent = labelW >= 250 ? d.name : d.short;
        fitText(lab, labelW);
        svgEl('line', { 'class': 'db-line' + (dv < 0 ? ' neg' : ''), x1: x(b), x2: x(p), y1: cy, y2: cy }, svg);
        svgEl('circle', { 'class': 'db-base', cx: x(b), cy: cy, r: 4.5 }, svg);
        svgEl('circle', { 'class': 'db-plus', cx: x(p), cy: cy, r: 5 }, svg);
        svgEl('text', { 'class': 'row-delta', x: w, y: cy + 4, 'text-anchor': 'end' }, svg).textContent = signed(dv, 1);
      });
      chart = { svg: svg, w: w, top: top, rowH: rowH, hits: hits, x: x };
      svg.style.touchAction = 'pan-y';
      svg.addEventListener('pointermove', function (e) { setActive(rowFromEvent(e), false); });
      svg.addEventListener('pointerdown', function (e) { setActive(rowFromEvent(e), false); });
      svg.addEventListener('pointerleave', function () { if (document.activeElement !== host) setActive(-1, false); });
      if (active >= 0) setActive(active, false);
    }

    function rowFromEvent(e) {
      var r = chart.svg.getBoundingClientRect();
      var py = (e.clientY - r.top) * (chart.svg.viewBox.baseVal.height / r.height);
      var i = Math.floor((py - chart.top) / chart.rowH);
      return (i >= 0 && i < rows.length) ? i : -1;
    }

    function setActive(i, announce) {
      if (!chart) return;
      chart.hits.forEach(function (h, j) { h.setAttribute('class', 'row-hit' + (j === i ? ' is-active' : '')); });
      active = i;
      if (i < 0) { tooltip.hidden = true; return; }
      var d = rows[i];
      tooltip.textContent = '';
      htmlEl('div', 'tt-head', d.name, tooltip);
      var r1 = htmlEl('div', 'tt-row', null, tooltip);
      htmlEl('span', 'key-dot with', null, r1);
      htmlEl('b', null, d.plus[bench].toFixed(1), r1);
      htmlEl('span', null, 'with plug-in', r1);
      var r2 = htmlEl('div', 'tt-row', null, tooltip);
      htmlEl('span', 'key-dot without', null, r2);
      htmlEl('b', null, d.base[bench].toFixed(1), r2);
      htmlEl('span', null, 'without', r2);
      tooltip.hidden = false;
      var scaleX = chart.svg.getBoundingClientRect().width / chart.w;
      var right = Math.max(chart.x(d.plus[bench]), chart.x(d.base[bench])) * scaleX;
      var left = right + 14;
      if (left + tooltip.offsetWidth > host.clientWidth) left = Math.min(chart.x(d.plus[bench]), chart.x(d.base[bench])) * scaleX - tooltip.offsetWidth - 14;
      tooltip.style.left = Math.max(0, left) + 'px';
      var top = chart.top + i * chart.rowH + chart.rowH / 2 - tooltip.offsetHeight / 2;
      tooltip.style.top = Math.max(0, top) + 'px';
      if (announce) {
        readout.textContent = d.name + ': ' + d.base[bench].toFixed(1) + ' without the plug-in, ' +
          d.plus[bench].toFixed(1) + ' with it, change ' + minus(signed(d.delta[bench], 1)) + '.';
      }
    }

    host.addEventListener('keydown', function (e) {
      var i = active;
      if (e.key === 'ArrowDown') i = active < 0 ? 0 : Math.min(rows.length - 1, active + 1);
      else if (e.key === 'ArrowUp') i = active < 0 ? 0 : Math.max(0, active - 1);
      else if (e.key === 'Home') i = 0;
      else if (e.key === 'End') i = rows.length - 1;
      else if (e.key === 'Escape') { setActive(-1, false); return; }
      else return;
      e.preventDefault();
      setActive(i, true);
    });
    host.addEventListener('blur', function () { setActive(-1, false); });

    function renderTable() {
      tableHost.textContent = '';
      var t = htmlEl('table', 'mini', null, tableHost);
      var label = BENCH.filter(function (b) { return b[0] === bench; })[0][1];
      htmlEl('caption', 'visually-hidden', 'AUROC with and without the plug-in, ' + label, t);
      var tr = htmlEl('tr', null, null, htmlEl('thead', null, null, t));
      ['Detector', 'Without', 'With plug-in', 'Change'].forEach(function (s) { htmlEl('th', null, s, tr).setAttribute('scope', 'col'); });
      var tb = htmlEl('tbody', null, null, t);
      sorted().forEach(function (d) {
        var r = htmlEl('tr', null, null, tb);
        htmlEl('th', null, d.name, r).setAttribute('scope', 'row');
        htmlEl('td', null, d.base[bench].toFixed(1), r);
        htmlEl('td', null, d.plus[bench].toFixed(1), r);
        htmlEl('td', null, signed(d.delta[bench], 1), r);
      });
      var ra = htmlEl('tr', null, null, tb);
      htmlEl('th', null, 'Rhythm alone, zero-shot', ra).setAttribute('scope', 'row');
      htmlEl('td', null, '\u2013', ra);
      htmlEl('td', null, plugin.alone[bench].toFixed(1), ra);
      htmlEl('td', null, '', ra);
    }

    var getWidth = observeWidth(host, render);
    renderTable();
  }

  /* ---------------------------------------------------------------- start */
  function start() {
    initLinks();
    initCopy();
    initTable();
    initNav();
    var data = window.VR_DATA;
    if (!data) return;
    var byId = {};
    data.panels.forEach(function (p) { byId[p.id] = p; });
    initHeroStrip(byId);
    initExplorer(data.panels, byId);
    initPlugin(data.plugin);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
