/* FaceHub docs — tech hero effects
 * 1. Particle-network canvas background (mouse-reactive)
 * 2. Typing-effect tagline (phrases from data-phrases)
 * 3. Count-up stats on scroll into view
 */
(function () {
  "use strict";

  /* ── 1. Particle network ─────────────────────────────────── */
  function initParticles() {
    var canvas = document.getElementById("fh-particles");
    if (!canvas) return;
    var hero = canvas.parentElement;
    var ctx = canvas.getContext("2d");
    var particles = [];
    var mouse = { x: -9999, y: -9999 };
    var raf = null;
    var DPR = Math.min(window.devicePixelRatio || 1, 2);

    function resize() {
      var r = hero.getBoundingClientRect();
      canvas.width = r.width * DPR;
      canvas.height = r.height * DPR;
      canvas.style.width = r.width + "px";
      canvas.style.height = r.height + "px";
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      seed();
    }

    function seed() {
      var area = (canvas.width * canvas.height) / (DPR * DPR);
      var n = Math.max(30, Math.min(90, Math.floor(area / 16000)));
      particles = [];
      for (var i = 0; i < n; i++) {
        particles.push({
          x: Math.random() * canvas.width / DPR,
          y: Math.random() * canvas.height / DPR,
          vx: (Math.random() - 0.5) * 0.35,
          vy: (Math.random() - 0.5) * 0.35,
          r: Math.random() * 1.6 + 0.6
        });
      }
    }

    function accent() {
      var s = getComputedStyle(document.body);
      return (s.getPropertyValue("--md-accent-fg-color") || "#22d3ee").trim();
    }

    function tick() {
      var w = canvas.width / DPR, h = canvas.height / DPR;
      ctx.clearRect(0, 0, w, h);
      var color = accent();
      var LINK = 110;

      for (var i = 0; i < particles.length; i++) {
        var p = particles[i];
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0 || p.x > w) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;

        // gentle attraction towards the mouse
        var dxm = mouse.x - p.x, dym = mouse.y - p.y;
        var dm = Math.sqrt(dxm * dxm + dym * dym);
        if (dm < 140 && dm > 0.01) {
          p.x += (dxm / dm) * 0.25;
          p.y += (dym / dm) * 0.25;
        }

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.75;
        ctx.fill();

        for (var j = i + 1; j < particles.length; j++) {
          var q = particles[j];
          var dx = p.x - q.x, dy = p.y - q.y;
          var d = Math.sqrt(dx * dx + dy * dy);
          if (d < LINK) {
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(q.x, q.y);
            ctx.strokeStyle = color;
            ctx.globalAlpha = (1 - d / LINK) * 0.35;
            ctx.lineWidth = 0.6;
            ctx.stroke();
          }
        }
      }
      ctx.globalAlpha = 1;
      raf = requestAnimationFrame(tick);
    }

    hero.addEventListener("mousemove", function (e) {
      var r = hero.getBoundingClientRect();
      mouse.x = e.clientX - r.left;
      mouse.y = e.clientY - r.top;
    });
    hero.addEventListener("mouseleave", function () {
      mouse.x = -9999; mouse.y = -9999;
    });

    // pause when off-screen to save CPU
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { if (!raf) tick(); }
        else if (raf) { cancelAnimationFrame(raf); raf = null; }
      });
    });
    io.observe(hero);

    window.addEventListener("resize", resize);
    resize();
  }

  /* ── 2. Typing effect ────────────────────────────────────── */
  function initTyper() {
    var el = document.getElementById("fh-typer");
    if (!el) return;
    var phrases;
    try { phrases = JSON.parse(el.getAttribute("data-phrases") || "[]"); }
    catch (e) { phrases = []; }
    if (!phrases.length) return;

    var pi = 0, ci = 0, deleting = false;
    function step() {
      var text = phrases[pi];
      ci += deleting ? -1 : 1;
      el.textContent = text.slice(0, ci);
      var delay = deleting ? 32 : 68;
      if (!deleting && ci === text.length) { delay = 2100; deleting = true; }
      else if (deleting && ci === 0) {
        deleting = false;
        pi = (pi + 1) % phrases.length;
        delay = 350;
      }
      setTimeout(step, delay);
    }
    step();
  }

  /* ── 3. Count-up stats ───────────────────────────────────── */
  function initCounters() {
    var nums = document.querySelectorAll(".fh-stat-num[data-target]");
    if (!nums.length) return;
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (!en.isIntersecting) return;
        io.unobserve(en.target);
        var el = en.target;
        var target = parseFloat(el.getAttribute("data-target"));
        var suffix = el.getAttribute("data-suffix") || "";
        var decimals = (String(target).split(".")[1] || "").length;
        var t0 = null, DUR = 1400;
        function frame(ts) {
          if (!t0) t0 = ts;
          var k = Math.min((ts - t0) / DUR, 1);
          var eased = 1 - Math.pow(1 - k, 3);
          el.textContent = (target * eased).toFixed(decimals) + suffix;
          if (k < 1) requestAnimationFrame(frame);
        }
        requestAnimationFrame(frame);
      });
    }, { threshold: 0.4 });
    nums.forEach(function (n) { io.observe(n); });
  }

  function init() {
    initParticles();
    initTyper();
    initCounters();
  }

  // mkdocs-material instant loading fires navigation events
  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(init);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
