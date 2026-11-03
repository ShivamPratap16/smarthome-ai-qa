/* Hawa Home AI quality dashboard.
 * Renders window.DASHBOARD_DATA (built by scripts/build_dashboard.py) into the page.
 * Vanilla JS + inline SVG, no libraries. */
(function () {
  "use strict";
  const D = window.DASHBOARD_DATA;
  if (!D) { document.querySelector("main").innerHTML = "<p>No data embedded. Run scripts/build_dashboard.py.</p>"; return; }

  const SERIES_COLORS = { evals: "#2a78d6", redteam: "#eb6834", tool_calling: "#1baf7a" };
  const LOWER_BETTER = new Set(["hallucination_rate", "p95_latency_ms", "p50_latency_ms", "cost_usd", "error_rate"]);
  const NICE = {
    pass_rate: "Pass rate", critical_pass_rate: "Critical attacks blocked", hallucination_rate: "Hallucination rate",
    faithfulness: "Faithfulness", answer_relevancy: "Answer relevancy", context_precision: "Context precision",
    context_recall: "Context recall", p95_latency_ms: "p95 latency", p50_latency_ms: "p50 latency",
    cost_usd: "Cost (run)", attack_block_rate: "Attack block rate", rps: "Requests / s", error_rate: "Error rate",
  };
  const OPS = { ">=": "≥", "<=": "≤", "==": "=" };
  const ICON = {
    pass: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7" fill="currentColor" opacity=".15"/><path d="M4.8 8.3l2.1 2.1 4.3-4.6" stroke="currentColor" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    fail: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7" fill="currentColor" opacity=".15"/><path d="M5.5 5.5l5 5m0-5l-5 5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
    warn: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.8l6.6 11.7H1.4z" fill="currentColor" opacity=".18"/><path d="M8 6v3.6M8 11.6v.1" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
    skipped: '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7" fill="currentColor" opacity=".15"/><path d="M5 8h6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
  };
  const STATUS_TEXT = { pass: "Pass", fail: "Fail", warn: "Warning", skipped: "Not run" };

  // ------------------------------------------------------------ helpers
  const $ = (sel) => document.querySelector(sel);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const isRate = (m) => /(_rate|faithfulness|relevancy|precision|recall)$/.test(m);
  const pct = (v, d = 1) => (v == null ? "—" : (v * 100).toFixed(d).replace(/\.0+$/, "") + "%");
  function fmt(metric, v) {
    if (v == null) return "—";
    if (isRate(metric)) return pct(v);
    if (metric.endsWith("_ms")) return Math.round(v).toLocaleString("en-US") + " ms";
    if (metric.startsWith("cost")) return "$" + v.toFixed(v < 0.01 ? 4 : 2);
    return Number.isInteger(v) ? v.toLocaleString("en-US") : v.toFixed(2);
  }
  function limitText(metric, op, value) { return (OPS[op] || op) + " " + fmt(metric, value); }
  const pill = (status, text) => `<span class="pill ${status}">${ICON[status] || ""}${esc(text || STATUS_TEXT[status])}</span>`;
  function fmtDate(iso, withTime = true) {
    if (!iso) return "—";
    const d = new Date(iso);
    if (isNaN(d)) return iso;
    const opts = withTime ? { month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit" } : { month: "short", day: "numeric" };
    return d.toLocaleString("en-US", opts);
  }
  const rulesFor = (suite) => (D.thresholds[suite] && D.thresholds[suite].rules) || [];
  const ruleFor = (suite, metric) => rulesFor(suite).find((r) => r.metric === metric);
  const suiteByKey = Object.fromEntries(D.suites.map((s) => [s.key, s]));

  // ------------------------------------------------------------ header
  function renderHeader() {
    $("#sample-banner").hidden = !D.sample;
    const checks = D.gate.checks.filter((c) => c.status !== "skipped");
    const passed = checks.filter((c) => c.status === "pass").length;
    const skipped = D.suites.filter((s) => !s.present).length;
    const anyReport = D.suites.some((s) => s.present);
    const badge = $("#gate-badge");
    const cls = !anyReport ? "none" : D.gate.passed ? "pass" : "fail";
    badge.className = "gate-badge " + cls;
    badge.innerHTML = `<div class="label">CI quality gate</div>
      <div class="verdict">${!anyReport ? "No reports yet" : D.gate.passed ? ICON.pass.replace("<svg", '<svg width="24" height="24"') + "Passing" : ICON.fail.replace("<svg", '<svg width="24" height="24"') + "Failing"}</div>
      <div class="sub">${passed} of ${checks.length} checks passed${skipped ? ` · ${skipped} suite${skipped > 1 ? "s" : ""} not run` : ""}</div>`;

    const repo = D.repo_url;
    const sha = D.git_sha ? (repo ? `<a href="${esc(repo)}/commit/${esc(D.git_sha)}">${esc(D.git_sha.slice(0, 7))}</a>` : esc(D.git_sha.slice(0, 7))) : "—";
    const items = [
      ["Last run", fmtDate(D.last_run_at)], ["Commit", sha], ["Model", esc(D.model || "—")],
      ["LLM mode", `<span title="replay = recorded responses (deterministic, no API key); live = real Gemini calls">${esc(D.llm_mode || "—")}</span>`],
      ["Regression baseline", D.gate.has_baseline ? "compared" : "not set"],
    ];
    $("#run-meta").innerHTML = items.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
    $("#generated-at").textContent = fmtDate(D.generated_at);
    const links = [];
    if (repo) {
      links.push(`<a href="${esc(repo)}">Source code on GitHub</a>`);
      links.push(`<a href="${esc(repo)}/actions">CI runs</a>`);
      links.push(`<a href="${esc(repo)}/blob/main/qa/thresholds.yaml">Gate thresholds</a>`);
    }
    links.push(`<a href="data.json">Raw data (JSON)</a>`);
    $("#footer-links").innerHTML = links.join("");
  }

  // ------------------------------------------------------------ KPIs
  function kpiStatus(suite, metric, value) {
    const r = ruleFor(suite, metric);
    if (!r || value == null) return "";
    const ok = r.op === ">=" ? value >= r.value - 1e-9 : r.op === "<=" ? value <= r.value + 1e-9 : Math.abs(value - r.value) < 1e-9;
    return `<span class="${ok ? "dot-ok" : "dot-bad"}" style="display:inline-flex;width:14px">${ok ? ICON.pass : ICON.fail}</span>Gate ${limitText(metric, r.op, r.value)}`;
  }
  function renderKpis() {
    const k = D.kpis;
    const present = D.suites.filter((s) => s.present).length;
    const tiles = [
      { label: "Overall pass rate", value: pct(k.overall_pass_rate), na: k.overall_pass_rate == null,
        foot: k.total_cases ? `${k.passed_cases.toLocaleString()} of ${k.total_cases.toLocaleString()} cases · ${present} suite${present === 1 ? "" : "s"}` : "No reports yet" },
      { label: "Hallucination rate", value: pct(k.hallucination_rate), na: k.hallucination_rate == null,
        foot: kpiStatus("evals", "hallucination_rate", k.hallucination_rate) || "Needs eval report" },
      { label: "Red-team block rate", value: pct(k.redteam_block_rate), na: k.redteam_block_rate == null,
        foot: kpiStatus("redteam", "pass_rate", k.redteam_block_rate) || "Needs red-team report" },
      { label: k.p95_source === "load" ? "API p95 latency (LLM stubbed)" : "p95 latency", value: fmt("p95_latency_ms", k.p95_latency_ms), na: k.p95_latency_ms == null,
        foot: k.p95_source === "load" ? kpiStatus("load", "p95_latency_ms", k.p95_latency_ms) : k.p95_source ? `From ${k.p95_source} run` : "Needs load report" },
      { label: "Cost per 100 chats", value: k.cost_per_100_chats == null ? "—" : "$" + k.cost_per_100_chats.toFixed(k.cost_per_100_chats < 0.1 ? 3 : 2), na: k.cost_per_100_chats == null,
        foot: k.cost_per_100_chats == null ? (suiteByKey.evals?.present ? "No cost in eval report" : "Needs eval report") : "LLM token cost, from eval run" },
    ];
    $("#kpis").innerHTML = tiles.map((t) => `<div class="card kpi"><div class="k-label">${t.label}</div>
      <div class="k-value${t.na ? " na" : ""}">${t.na ? "Not measured" : t.value}</div><div class="k-foot">${t.foot}</div></div>`).join("");
  }

  // ------------------------------------------------------------ suites
  function checkLabel(c) {
    if (c.check === "report present") return "Report present";
    const reg = c.check.startsWith("regression: ");
    const metric = reg ? c.check.slice(12) : c.check.split(" ")[0];
    return (reg ? "vs baseline: " : "") + (NICE[metric] || metric);
  }
  function checkMetric(c) { return c.check.startsWith("regression: ") ? c.check.slice(12) : c.check.split(" ")[0]; }
  function checkLimit(c) {
    const m = checkMetric(c);
    const [op, raw] = c.limit.split(" ");
    const n = Number(raw);
    return isNaN(n) ? c.limit : limitText(m, op, n);
  }

  function renderSuites() {
    $("#suite-grid").innerHTML = D.suites.map((s) => {
      const head = `<div class="suite-head"><div><h3>${esc(s.title)}</h3><div class="tool">${esc(s.tool)}</div></div>${pill(s.status)}</div>`;
      if (!s.present) {
        return `<article class="card suite is-skipped">${head}<p class="suite-empty">No report yet — this suite has not written <code>reports/${esc(s.key)}/latest.json</code>. The gate shows it as skipped, not passed.</p></article>`;
      }
      const sm = s.summary;
      const rate = sm.pass_rate;
      const metrics = s.checks.filter((c) => c.check !== "report present").map((c) => {
        const m = checkMetric(c);
        return `<li><span class="m-name">${esc(checkLabel(c))} <span class="muted">(${esc(checkLimit(c))})</span></span><span class="m-val ${c.status === "fail" ? "dot-bad" : ""}">${c.actual == null ? "—" : esc(fmt(m, c.actual))}</span></li>`;
      }).join("");
      const meta = [fmtDate(s.started_at), s.prompt_version && s.key === "evals" ? "prompt " + s.prompt_version : null, s.llm_mode].filter(Boolean).join(" · ");
      return `<article class="card suite ${s.status === "fail" ? "is-fail" : ""}">${head}
        <div class="suite-rate"><span class="big">${pct(rate)}</span><span class="of">${sm.passed} / ${sm.total} passed</span></div>
        <div class="bar ${s.status === "fail" ? "fail" : ""}"><span style="width:${(rate || 0) * 100}%"></span></div>
        <ul class="metric-list">${metrics}</ul>
        <div class="suite-foot">${esc(meta)}</div></article>`;
    }).join("");

    const rows = D.gate.checks.map((c) => `<tr><td>${esc(suiteByKey[c.suite]?.title || c.suite)}</td><td>${esc(checkLabel(c))}</td>
      <td class="num">${c.actual == null ? "—" : esc(fmt(checkMetric(c), c.actual))}</td><td class="num">${esc(checkLimit(c))}</td>
      <td>${pill(c.status)}</td><td class="muted">${esc(c.note)}</td></tr>`).join("");
    $("#checks-table").innerHTML = `<thead><tr><th>Suite</th><th>Check</th><th class="num">Actual</th><th class="num">Limit</th><th>Status</th><th>Note</th></tr></thead><tbody>${rows}</tbody>`;
  }

  // ------------------------------------------------------------ charts
  const tip = $("#tooltip");
  function showTip(html, x, y) {
    tip.innerHTML = html; tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let left = x + 14, top = y - h - 10;
    if (left + w > window.innerWidth - 8) left = x - w - 14;
    if (top < 8) top = y + 14;
    tip.style.left = left + "px"; tip.style.top = top + "px";
  }
  const hideTip = () => { tip.hidden = true; };

  function promptBands() {
    const runs = (D.history.evals || []).filter((r) => r.prompt_version);
    const bands = [];
    runs.forEach((r) => {
      const t = Date.parse(r.started_at);
      const last = bands[bands.length - 1];
      if (last && last.version === r.prompt_version) last.end = t; else bands.push({ version: r.prompt_version, start: t, end: t });
    });
    return new Set(bands.map((b) => b.version)).size > 1 ? bands : [];
  }

  const bandsTop = (opts) => (opts.bands || []).length > 0;
  /** Pick a round tick step (at most ~5 intervals) and stretch the top of the domain to a multiple of it. */
  function niceDomain(lo, hi) {
    const step = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25].find((s) => (hi - lo) / s <= 5) || 0.25;
    return { yDomain: [Math.floor(lo / step + 1e-9) * step, Math.ceil(hi / step - 1e-9) * step], yStep: step };
  }

  function lineChart(el, opts) {
    const series = opts.series.filter((s) => s.points.length);
    if (!series.length) { el.innerHTML = `<div class="chart-empty">${esc(opts.empty)}</div>`; return; }
    const W = Math.max(280, el.clientWidth || 520), H = W < 420 ? 210 : 240;
    const m = { l: 44, r: 14, t: bandsTop(opts) ? 26 : 14, b: 28 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const ts = [...new Set(series.flatMap((s) => s.points.map((p) => p.t)))].sort((a, b) => a - b);
    let t0 = ts[0], t1 = ts[ts.length - 1];
    if (t0 === t1) { t0 -= 864e5; t1 += 864e5; }
    const pad = (t1 - t0) * 0.04; t0 -= pad; t1 += pad;
    const [y0, y1] = opts.yDomain;
    const X = (t) => m.l + ((t - t0) / (t1 - t0)) * iw;
    const Y = (v) => m.t + (1 - (v - y0) / (y1 - y0)) * ih;
    let svg = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(opts.label)}">`;

    // prompt-version bands (midpoints between runs mark the switch)
    const bands = opts.bands || [];
    bands.forEach((b, i) => {
      const prev = bands[i - 1], next = bands[i + 1];
      const xs = prev ? X((prev.end + b.start) / 2) : m.l;
      const xe = next ? X((b.end + next.start) / 2) : m.l + iw;
      svg += `<rect x="${xs}" y="${m.t}" width="${Math.max(0, xe - xs)}" height="${ih}" fill="${i % 2 ? "#f3f6fb" : "#f8f9fb"}"/>`;
      if (xe - xs > 50) svg += `<text class="band-text" x="${xs + 6}" y="${m.t - 8}">prompt ${esc(b.version)}</text>`;
      if (prev) svg += `<line x1="${xs}" x2="${xs}" y1="${m.t}" y2="${m.t + ih}" stroke="#c3c7cf" stroke-dasharray="2 3"/>`;
    });

    // y grid + ticks
    const nTicks = Math.round((y1 - y0) / opts.yStep);
    for (let i = 0; i <= nTicks; i++) {
      const v = y0 + opts.yStep * i;
      svg += `<line x1="${m.l}" x2="${m.l + iw}" y1="${Y(v)}" y2="${Y(v)}" stroke="${i === 0 ? "#c3c7cf" : "#eceef1"}"/>`;
      svg += `<text class="axis-text" x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end">${opts.yFormat(v)}</text>`;
    }
    // x ticks: up to ~6 run dates
    const maxTicks = Math.max(2, Math.floor(iw / 80));
    const step = Math.max(1, Math.ceil(ts.length / maxTicks));
    ts.forEach((t, i) => {
      if (i % step && i !== ts.length - 1) return;
      if (i === ts.length - 1 && i % step && X(t) - X(ts[i - (i % step)]) < 60) return;
      svg += `<text class="axis-text" x="${X(t)}" y="${H - 8}" text-anchor="middle">${fmtDate(new Date(t).toISOString(), false)}</text>`;
    });
    // threshold
    if (opts.threshold != null) {
      const ty = Y(opts.threshold);
      svg += `<line x1="${m.l}" x2="${m.l + iw}" y1="${ty}" y2="${ty}" stroke="#d03b3b" stroke-width="1.5" stroke-dasharray="5 4"/>`;
      svg += `<text class="axis-text" x="${m.l + iw}" y="${ty - 5}" text-anchor="end" style="fill:#b42828">gate ${opts.yFormat(opts.threshold)}</text>`;
    }
    // lines + markers
    series.forEach((s) => {
      const d = s.points.map((p, i) => `${i ? "L" : "M"}${X(p.t).toFixed(1)},${Y(p.y).toFixed(1)}`).join("");
      svg += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
      s.points.forEach((p) => { svg += `<circle cx="${X(p.t)}" cy="${Y(p.y)}" r="3.5" fill="${s.color}" stroke="#fff" stroke-width="2"/>`; });
    });
    svg += `<line class="xhair" x1="0" x2="0" y1="${m.t}" y2="${m.t + ih}" stroke="#7a8394" stroke-width="1" visibility="hidden"/>`;
    svg += `<rect class="hit" x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent"/></svg>`;

    const showNames = series.length > 1 || opts.alwaysLegend;
    const legend = showNames || opts.threshold != null
      ? `<div class="legend">${showNames ? series.map((s) => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("") : ""}${opts.threshold != null ? `<span><i class="dash"></i>Gate threshold</span>` : ""}</div>` : "";
    const note = ts.length === 1 ? `<div class="muted" style="font-size:12.5px;margin-top:2px">Only one run recorded so far — the trend line grows with every CI run.</div>` : "";
    el.innerHTML = legend + svg + note;

    const svgEl = el.querySelector("svg"), xhair = svgEl.querySelector(".xhair");
    function onMove(ev) {
      const rect = svgEl.getBoundingClientRect();
      const px = (ev.touches ? ev.touches[0].clientX : ev.clientX) - rect.left;
      const t = ts.reduce((best, cur) => (Math.abs(X(cur) - px) < Math.abs(X(best) - px) ? cur : best), ts[0]);
      xhair.setAttribute("x1", X(t)); xhair.setAttribute("x2", X(t)); xhair.setAttribute("visibility", "visible");
      const rows = series.map((s) => {
        const p = s.points.reduce((b, c) => (Math.abs(c.t - t) < Math.abs(b.t - t) ? c : b));
        return Math.abs(p.t - t) <= 36e5 ? { s, p } : null;
      }).filter(Boolean);
      const meta = rows[0] && rows[0].p.meta ? `<div class="muted" style="color:#94a3b8">${esc(rows[0].p.meta)}</div>` : "";
      showTip(`<div class="t-title">${fmtDate(new Date(t).toISOString())}</div>` +
        rows.map(({ s, p }) => `<div class="t-row"><i style="background:${s.color}"></i>${esc(s.name)}<b>${opts.yFormat(p.y, true)}</b></div>`).join("") + meta,
        ev.clientX ?? ev.touches[0].clientX, ev.clientY ?? ev.touches[0].clientY);
    }
    const hit = svgEl.querySelector(".hit");
    hit.addEventListener("mousemove", onMove);
    hit.addEventListener("touchstart", onMove, { passive: true });
    hit.addEventListener("mouseleave", () => { hideTip(); xhair.setAttribute("visibility", "hidden"); });
  }

  function seriesFor(suite, getY, name) {
    const runs = D.history[suite] || [];
    return {
      name: name || suiteByKey[suite]?.title || suite, color: SERIES_COLORS[suite] || "#2a78d6",
      points: runs.map((r) => ({ t: Date.parse(r.started_at), y: getY(r), meta: [r.prompt_version && suite === "evals" ? "prompt " + r.prompt_version : null, r.git_sha ? "commit " + r.git_sha.slice(0, 7) : null].filter(Boolean).join(" · ") }))
        .filter((p) => !isNaN(p.t) && p.y != null),
    };
  }

  function renderCharts() {
    const bands = promptBands();
    const passSeries = ["evals", "redteam", "tool_calling"].map((s) => seriesFor(s, (r) => r.pass_rate));
    const minY = Math.min(1, ...passSeries.flatMap((s) => s.points.map((p) => p.y)));
    lineChart($("#chart-pass"), {
      series: passSeries, bands, alwaysLegend: true, label: "Pass rate by suite over runs",
      ...niceDomain(Math.min(0.6, Math.floor(minY * 10) / 10), 1), yFormat: (v, precise) => pct(v, precise ? 1 : 0),
      empty: "Trends appear once eval, red-team or tool-calling reports exist.",
    });
    const hSeries = [seriesFor("evals", (r) => r.metrics.hallucination_rate, "Hallucination rate")];
    const hRule = ruleFor("evals", "hallucination_rate");
    const maxH = Math.max(0.2, ...(hSeries[0].points.map((p) => p.y)), hRule ? hRule.value : 0);
    lineChart($("#chart-halluc"), {
      series: hSeries, bands, label: "Hallucination rate over eval runs", threshold: hRule ? hRule.value : null,
      ...niceDomain(0, maxH * 1.15), yFormat: (v, precise) => pct(v, precise ? 1 : 0),
      empty: "Trend appears once eval reports with a hallucination_rate exist.",
    });
  }

  // ------------------------------------------------------------ prompt compare
  function renderCompare() {
    const pc = D.prompt_compare, el = $("#prompt-compare");
    if (!pc) {
      el.innerHTML = `<div class="card note-card">The comparison appears when eval reports exist for at least two prompt versions (run <code>python -m qa.evals.run</code> with prompt v1 and v2).</div>`;
      return;
    }
    const [a, b] = pc.versions.slice(-2);
    const rows = pc.rows.map((r) => {
      const va = r.values[a], vb = r.values[b];
      let delta = "—", cls = "delta-flat";
      if (va != null && vb != null) {
        const diff = vb - va;
        const better = LOWER_BETTER.has(r.metric) ? diff < 0 : diff > 0;
        cls = Math.abs(diff) < 1e-9 ? "delta-flat" : better ? "delta-good" : "delta-bad";
        const size = isRate(r.metric) ? (Math.abs(diff) * 100).toFixed(1) + " pts" : fmt(r.metric, Math.abs(diff));
        delta = Math.abs(diff) < 1e-9 ? "no change" : (diff > 0 ? "▲ +" : "▼ −") + size;
      }
      return `<tr><td>${esc(NICE[r.metric] || r.metric)}${LOWER_BETTER.has(r.metric) ? ' <span class="muted">(lower is better)</span>' : ""}</td>
        <td class="num">${fmt(r.metric, va)}</td><td class="num"><b>${fmt(r.metric, vb)}</b></td><td class="num ${cls}">${delta}</td></tr>`;
    }).join("");
    const BAR_METRICS = ["pass_rate", "faithfulness", "answer_relevancy", "hallucination_rate"];
    const bars = pc.rows.filter((r) => BAR_METRICS.includes(r.metric)).map((r) => {
      const line = (v, ver, color) => `<div class="cb-line"><span class="v">${esc(ver)}</span><div class="cb-track"><span style="width:${Math.max(0, Math.min(1, v || 0)) * 100}%;background:${color}"></span></div><span class="val">${pct(v)}</span></div>`;
      return `<div class="cb-row"><div class="cb-name">${esc(NICE[r.metric] || r.metric)}${LOWER_BETTER.has(r.metric) ? " · lower is better" : ""}</div>${line(r.values[a], a, "#a3acb9")}${line(r.values[b], b, "#2a78d6")}</div>`;
    }).join("");
    const runInfo = (v) => pc.runs[v] ? `${pc.runs[v].total} cases · ${fmtDate(pc.runs[v].started_at, false)}` : "";
    el.innerHTML = `<div class="compare">
      <div class="card"><div class="table-wrap"><table class="table"><thead><tr><th>Metric</th><th class="num">Prompt ${esc(a)}</th><th class="num">Prompt ${esc(b)}</th><th class="num">Change</th></tr></thead><tbody>${rows}</tbody></table></div>
      <p class="muted" style="font-size:12.5px;margin:0;padding:10px 16px;border-top:1px solid var(--grid)">Latest eval run per version — ${esc(a)}: ${runInfo(a)}; ${esc(b)}: ${runInfo(b)}.</p></div>
      <div class="card compare-bars"><h3>Side by side</h3>${bars}</div></div>`;
  }

  // ------------------------------------------------------------ failures
  function renderFailures() {
    const rows = D.failures;
    const fSuite = $("#f-suite"), fCat = $("#f-cat"), fQ = $("#f-q");
    const suites = [...new Set(rows.map((r) => r.suite))];
    fSuite.innerHTML = `<option value="">All suites</option>` + suites.map((s) => `<option value="${esc(s)}">${esc(suiteByKey[s]?.title || s)}</option>`).join("");
    function fillCats() {
      const cats = [...new Set(rows.filter((r) => !fSuite.value || r.suite === fSuite.value).map((r) => r.category))].sort();
      const keep = fCat.value;
      fCat.innerHTML = `<option value="">All categories</option>` + cats.map((c) => `<option>${esc(c)}</option>`).join("");
      fCat.value = cats.includes(keep) ? keep : "";
    }
    function draw() {
      const q = fQ.value.trim().toLowerCase();
      const shown = rows.filter((r) => (!fSuite.value || r.suite === fSuite.value) && (!fCat.value || r.category === fCat.value) &&
        (!q || (r.id + " " + r.reason).toLowerCase().includes(q)));
      $("#f-count").textContent = `${shown.length} of ${rows.length} failing case${rows.length === 1 ? "" : "s"}`;
      const body = shown.map((r) => {
        const scores = Object.entries(r.scores || {}).map(([k, v]) => `${esc(NICE[k] || k)} ${typeof v === "number" ? (v <= 1 ? v.toFixed(2) : Math.round(v)) : esc(v)}`).join("<br>");
        return `<tr><td>${esc(suiteByKey[r.suite]?.title || r.suite)}</td><td class="mono">${esc(r.id)}${r.critical ? '<span class="tag crit">critical</span>' : ""}</td>
          <td><span class="tag">${esc(r.category)}</span></td><td class="reason">${esc(r.reason) || '<span class="muted">No reason given</span>'}</td><td class="num muted">${scores || "—"}</td></tr>`;
      }).join("");
      const empty = rows.length ? "No cases match these filters." : D.suites.some((s) => s.present) ? "No failing cases in the latest runs." : "No reports yet.";
      $("#fail-table").innerHTML = `<thead><tr><th>Suite</th><th>Case</th><th>Category</th><th>Reason</th><th class="num">Scores</th></tr></thead>
        <tbody>${body || `<tr class="empty-row"><td colspan="5">${empty}</td></tr>`}</tbody>`;
    }
    fSuite.addEventListener("change", () => { fillCats(); draw(); });
    fCat.addEventListener("change", draw);
    fQ.addEventListener("input", draw);
    fillCats(); draw();
  }

  // ------------------------------------------------------------ how it's tested
  function renderHow() {
    $("#how-grid").innerHTML = D.suites.map((s) => {
      const rules = rulesFor(s.key).map((r) => `${NICE[r.metric] || r.metric} ${limitText(r.metric, r.op, r.value)}`).join(", ");
      return `<article class="card how"><h3>${esc(s.title)} ${pill(s.status)}</h3><div class="tool">${esc(s.tool)}</div>
        <p>${esc(s.what)}</p>${rules ? `<div class="gate-rule"><b>Gate:</b> ${esc(rules)}</div>` : ""}</article>`;
    }).join("");
  }

  renderHeader(); renderKpis(); renderSuites(); renderCharts(); renderCompare(); renderFailures(); renderHow();
  let lastW = window.innerWidth;
  window.addEventListener("resize", () => { if (Math.abs(window.innerWidth - lastW) > 30) { lastW = window.innerWidth; renderCharts(); } });
})();
