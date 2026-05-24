// Animated network loading screen — draws nodes + connecting edges, then fades
// out once the page has loaded. Only runs when the override injected #graph-loader
// (first hard load of a session); a no-op otherwise.
(function () {
  var loader = document.getElementById("graph-loader");
  if (!loader) return;
  var cv = document.getElementById("gl-canvas");
  var ctx = cv.getContext("2d");
  var DPR = window.devicePixelRatio || 1, W = 320, H = 240;
  cv.width = W * DPR; cv.height = H * DPR;
  cv.style.width = W + "px"; cv.style.height = H + "px";
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);

  var N = 16, nodes = [];
  for (var i = 0; i < N; i++) {
    nodes.push({
      x: 30 + Math.random() * (W - 60), y: 30 + Math.random() * (H - 60),
      vx: (Math.random() - 0.5) * 0.3, vy: (Math.random() - 0.5) * 0.3,
      born: Math.random() * 0.5
    });
  }

  var t0 = performance.now(), MIN = 1100, raf;
  function draw(now) {
    var t = (now - t0) / 1000;
    ctx.clearRect(0, 0, W, H);
    for (var i = 0; i < N; i++) {
      var n = nodes[i]; n.x += n.vx; n.y += n.vy;
      if (n.x < 18 || n.x > W - 18) n.vx *= -1;
      if (n.y < 18 || n.y > H - 18) n.vy *= -1;
    }
    for (var i = 0; i < N; i++) for (var j = i + 1; j < N; j++) {
      var a = nodes[i], b = nodes[j], dx = a.x - b.x, dy = a.y - b.y;
      var d = Math.sqrt(dx * dx + dy * dy);
      if (d < 96) {
        var app = Math.min(1, Math.max(0, (t - Math.max(a.born, b.born)) * 2));
        ctx.globalAlpha = app * (1 - d / 96) * 0.6;
        ctx.strokeStyle = "#7c6cff"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    for (var i = 0; i < N; i++) {
      var n = nodes[i], app = Math.min(1, Math.max(0, (t - n.born) * 2.5));
      var r = (2.4 + 1.8 * Math.sin(t * 2 + i)) * app + 1.6;
      ctx.fillStyle = "#a89cff";
      ctx.beginPath(); ctx.arc(n.x, n.y, Math.max(0.5, r), 0, 7); ctx.fill();
    }
    raf = requestAnimationFrame(draw);
  }
  raf = requestAnimationFrame(draw);

  function remove() {
    loader.classList.add("gl-done");
    setTimeout(function () {
      cancelAnimationFrame(raf);
      if (loader.parentNode) loader.parentNode.removeChild(loader);
      document.documentElement.classList.remove("gl-loading");
    }, 450);
  }
  function finish() { setTimeout(remove, Math.max(0, MIN - (performance.now() - t0))); }
  if (document.readyState === "complete") finish();
  else window.addEventListener("load", finish);
  // safety: never block the page for more than 4s
  setTimeout(remove, 4000);
})();
