/*
 * Reveal animations for the three places a computed result actually
 * appears: the headline score, the per-component breakdown bars, and the
 * history trend line. Deliberately not used anywhere else (no page-load
 * fades, no hover transitions, no pulsing panels) — motion here is meant
 * to read as "the result arriving," not decoration.
 *
 * Every animation is skipped (final state shown immediately) when the
 * visitor has prefers-reduced-motion set.
 */
(function () {
  "use strict";

  var REDUCE_MOTION = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function easeOutExpo(t) {
    return t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
  }

  // ---- 1. Score count-up ----
  // The server already rendered the correct final value (so a no-JS visitor
  // sees the right number, just without the animation) — this resets to 0
  // and counts back up, rather than assuming a blank starting state.
  function animateScoreValue(el) {
    var target = parseFloat(el.getAttribute("data-target"));
    if (isNaN(target) || REDUCE_MOTION) return;

    el.textContent = "0.0";
    var duration = 700;
    var start = null;
    function step(timestamp) {
      if (start === null) start = timestamp;
      var progress = Math.min((timestamp - start) / duration, 1);
      var value = target * easeOutExpo(progress);
      el.textContent = value.toFixed(1);
      if (progress < 1) window.requestAnimationFrame(step);
      else el.textContent = target.toFixed(1);
    }
    window.requestAnimationFrame(step);
  }

  // ---- 2. Breakdown bars filling in ----
  // Same principle: the server-rendered width is already correct; this
  // drops each bar to 0% and lets the CSS transition (static/style.css)
  // carry it back up to that same width, staggered per row.
  function animateBreakdownBars(fills) {
    if (REDUCE_MOTION) return;
    fills.forEach(function (fill, i) {
      var target = fill.getAttribute("data-target-width");
      if (target === null) return;
      fill.style.width = "0%";
      window.setTimeout(function () {
        fill.style.width = target + "%";
      }, i * 90 + 30);
    });
  }

  // ---- 3. Trend line drawing itself in ----
  function animateTrendLine(svg) {
    var line = svg.querySelector(".trend-line");
    var area = svg.querySelector(".trend-area");
    if (!line) return;
    var d = line.getAttribute("d");
    if (!d) return; // single-point chart has no line to draw

    if (REDUCE_MOTION) {
      if (area) area.style.opacity = 1;
      return;
    }

    var length = line.getTotalLength();
    line.style.strokeDasharray = length;
    line.style.strokeDashoffset = length;
    if (area) area.style.opacity = 0;

    // Force a reflow so the browser registers the starting offset before
    // the transition to 0 is applied — otherwise it can skip straight to
    // the end state instead of animating.
    line.getBoundingClientRect();

    var lineDuration = 900;
    line.style.transition = "stroke-dashoffset " + lineDuration + "ms cubic-bezier(0.16, 1, 0.3, 1)";
    line.style.strokeDashoffset = "0";

    if (area) {
      window.setTimeout(function () {
        area.style.transition = "opacity 500ms ease";
        area.style.opacity = 1;
      }, lineDuration * 0.7);
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".score-value[data-target]").forEach(animateScoreValue);

    var breakdownFills = document.querySelectorAll(".breakdown-fill[data-target-width]");
    if (breakdownFills.length) animateBreakdownBars(Array.prototype.slice.call(breakdownFills));

    document.querySelectorAll(".trend-card svg").forEach(function (svg) {
      if (svg.querySelector(".trend-line")) animateTrendLine(svg);
    });
  });
})();
