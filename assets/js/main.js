/* VAE Rhythm project page: link config, copy buttons, the main-table toggle and the section highlight in the nav.
   Plain JavaScript, no dependencies. */
(function () {
  'use strict';

  function $(id) { return document.getElementById(id); }

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
    // the hero above the first tracked section clears the highlight
    Array.prototype.forEach.call(document.querySelectorAll('.hero'), function (el) { io.observe(el); });
  }

  function start() {
    initLinks();
    initCopy();
    initTable();
    initNav();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
