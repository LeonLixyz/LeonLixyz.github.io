// Existing DeltaNet / TTT figure animation, migrated from learning-agent-blog.
document.addEventListener("DOMContentLoaded", function () {
  (function () {
    var svg = document.getElementById("dctx-svg");
    if (!svg) return;
    var ns = "http://www.w3.org/2000/svg";
    function el(t, a) {
      var e = document.createElementNS(ns, t);
      for (var k in a) e.setAttribute(k, a[k]);
      return e;
    }
    function txt(x, y, s, cls, an) {
      var e = el("text", { x: x, y: y, class: cls });
      if (an) e.setAttribute("text-anchor", an);
      e.textContent = s;
      svg.appendChild(e);
      return e;
    }
    var N = 5,
      cs = 26,
      gp = 4;
    function grid(ox, oy, w) {
      var c = [];
      for (var r = 0; r < N; r++) {
        c[r] = [];
        for (var k = 0; k < N; k++) {
          var rc = el("rect", {
            x: ox + k * (cs + gp),
            y: oy + r * (cs + gp),
            width: cs,
            height: cs,
            rx: 4,
            class: "dctx-cell" + (w ? " dctx-cellw" : ""),
          });
          rc.style.fillOpacity = "0.06";
          svg.appendChild(rc);
          c[r][k] = rc;
        }
      }
      return c;
    }
    txt(107, 42, "DeltaNet S  \u00b7  context state", "dctx-lbl", "middle");
    txt(553, 42, "TTT-Linear W  \u00b7  fast weights", "dctx-lbl", "middle");
    var L = grid(34, 66, false),
      R = grid(480, 66, true);
    txt(107, 250, "S \u2190 S + \u03b2(v\u209c \u2212 S k\u209c) k\u209c\u1d40", "dctx-eq", "middle");
    txt(553, 250, "W \u2190 W + \u03b7(v\u209c \u2212 W k\u209c) k\u209c\u1d40", "dctx-eq", "middle");
    txt(330, 150, "\u2248", "dctx-sign", "middle");
    var tok = txt(330, 118, "token 1", "dctx-tok", "middle");
    txt(330, 182, "same rank-1 write", "dctx-sub", "middle");
    var T = [
        { r: 1, c: 3 },
        { r: 3, c: 1 },
        { r: 0, c: 4 },
        { r: 4, c: 2 },
      ],
      sL = [],
      sR = [];
    for (var i = 0; i < N; i++) {
      sL.push([0, 0, 0, 0, 0]);
      sR.push([0, 0, 0, 0, 0]);
    }
    var step = 0,
      timer = null,
      paused = false;
    function settle(c, s) {
      for (var r = 0; r < N; r++) for (var k = 0; k < N; k++) c[r][k].style.fillOpacity = (0.06 + s[r][k]).toFixed(3);
    }
    function wr(c, s, t) {
      for (var j = 0; j < N; j++) c[t.r][j].style.fillOpacity = "0.55";
      for (var i = 0; i < N; i++) c[i][t.c].style.fillOpacity = "0.55";
      c[t.r][t.c].style.fillOpacity = "0.95";
      for (var j2 = 0; j2 < N; j2++) s[t.r][j2] = Math.min(0.5, s[t.r][j2] + 0.1);
      for (var i2 = 0; i2 < N; i2++) s[i2][t.c] = Math.min(0.5, s[i2][t.c] + 0.1);
      s[t.r][t.c] = Math.min(0.62, s[t.r][t.c] + 0.18);
      setTimeout(function () {
        settle(c, s);
      }, 560);
    }
    function reset() {
      for (var r = 0; r < N; r++)
        for (var k = 0; k < N; k++) {
          sL[r][k] = 0;
          sR[r][k] = 0;
        }
      settle(L, sL);
      settle(R, sR);
    }
    function tick() {
      var i = step % (T.length + 1);
      if (i < T.length) {
        wr(L, sL, T[i]);
        wr(R, sR, T[i]);
        tok.textContent = "token " + (i + 1) + " \u2192 same correction";
      } else {
        reset();
        tok.textContent = "sequence ends \u2192 wiped";
      }
      step++;
    }
    function start() {
      if (!timer) timer = setInterval(tick, 1350);
    }
    function stop() {
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
    }
    var btn = document.getElementById("dctx-toggle"),
      st = document.getElementById("dctx-status");
    btn.onclick = function () {
      paused = !paused;
      if (paused) {
        stop();
        btn.textContent = "\u25b6 play";
        st.textContent = "paused";
      } else {
        start();
        btn.textContent = "\u23f8 pause";
        st.textContent = "running\u2026";
      }
    };
    var red = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (red) {
      paused = true;
      btn.textContent = "\u25b6 play";
      st.textContent = "paused";
      [
        { r: 1, c: 3 },
        { r: 3, c: 1 },
      ].forEach(function (t) {
        for (var j = 0; j < N; j++) {
          sL[t.r][j] = 0.3;
          sR[t.r][j] = 0.3;
        }
        for (var i = 0; i < N; i++) {
          sL[i][t.c] = 0.3;
          sR[i][t.c] = 0.3;
        }
      });
      settle(L, sL);
      settle(R, sR);
    } else {
      tick();
      start();
    }
  })();
});
