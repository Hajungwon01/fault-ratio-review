/* 교통사고 과실비율 확인 — 화면 동작
   흐름: 입력 → (대기) → 질문 → 결과. API 는 docs/USER_API_DESIGN.md 의 /v1/assess 계약을 따른다.

   지키는 규칙
   - 서버가 준 문자열은 항상 textContent 로만 넣는다 (HTML 로 해석하지 않는다). innerHTML 은 이 파일 안의 고정 SVG 에만 쓴다.
   - 질문 화면은 보기를 서버가 준 순서 그대로 보여 주고(섞기는 서버가 한다), 처음에는 아무것도 선택하지 않는다.
   - 질문 화면에는 과실·도표 정보가 없다 (서버가 주지 않는다). */
(() => {
  "use strict";

  // 파일로 열어도(file://) 붙도록 기본은 로컬 백엔드. ?api=https://... 로 바꿀 수 있다.
  const API = new URLSearchParams(location.search).get("api") || "http://localhost:8000";
  const MIN_LEN = 10;
  const UNKNOWN_ID = -1, OTHER_ID = -2;
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const $ = (id) => document.getElementById(id);
  const state = { sessionId: null, seed: null, timer: null, rafId: 0 };

  // ---------- 고정 아이콘 (신뢰할 수 있는 상수만 innerHTML 로 넣는다) ----------
  const ICON = {
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
    check: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6L9 17l-5-5"/></svg>',
    alert: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="flex:none;margin-top:3px"><circle cx="12" cy="12" r="9"/><path d="M12 8v4M12 16h.01"/></svg>',
  };

  const EXAMPLES = [
    ["차로 변경", "제가(A) 2차로에서 직진 중이었는데 옆 차로의 상대 차량(B)이 방향지시등 없이 제 앞으로 차로를 변경하면서 충돌했어요."],
    ["직진 vs 좌회전", "신호가 있는 교차로에서 제가(A) 직진하는데 맞은편 상대 차량(B)이 좌회전하면서 충돌했어요."],
    ["후방 추돌", "제가(A) 신호 대기로 정차해 있었는데 뒤에서 오던 상대 차량(B)이 제 차를 들이받았어요."],
    ["주차장 후진", "주차장에서 제가(A) 후진으로 나오다가 통로를 지나가던 상대 차량(B)과 부딪혔어요."],
  ];

  // ---------- 테마 ----------
  function effectiveTheme() {
    const set = document.documentElement.dataset.theme;
    return set || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  }
  function paintThemeIcon() {
    $("theme-icon").innerHTML = effectiveTheme() === "dark" ? ICON.sun : ICON.moon;       // 다크일 때는 '밝게' 아이콘
    $("theme-btn").setAttribute("aria-label", effectiveTheme() === "dark" ? "밝은 화면으로 바꾸기" : "어두운 화면으로 바꾸기");
  }
  function initTheme() {
    try { const saved = localStorage.getItem("theme"); if (saved === "light" || saved === "dark") document.documentElement.dataset.theme = saved; } catch (_) { /* 저장소를 못 쓰면 시스템 설정을 따른다 */ }
    paintThemeIcon();
    $("theme-btn").addEventListener("click", () => {
      const next = effectiveTheme() === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("theme", next); } catch (_) { /* 무시 */ }
      paintThemeIcon();
    });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", paintThemeIcon);
  }

  // ---------- 단계 · 화면 전환 ----------
  const VIEWS = ["input", "wait", "question", "result"];
  const STEP_OF = { input: 0, wait: 0, question: 1, result: 2 };
  function setSteps(view) {
    const cur = STEP_OF[view];
    [...$("steps").children].forEach((li, i) => {
      li.classList.toggle("done", i < cur);
      if (i === cur) li.setAttribute("aria-current", "step"); else li.removeAttribute("aria-current");
    });
  }
  function show(view) {
    VIEWS.forEach((v) => $("view-" + v).classList.toggle("hidden", v !== view));
    setSteps(view);
    const target = $("view-" + view);
    target.style.animation = "none"; void target.offsetWidth; target.style.animation = "";       // 전환 애니메이션 다시 재생
    const heading = target.querySelector("[tabindex='-1']");
    if (heading) heading.focus({ preventScroll: true });
    $("steps").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
  }
  function setError(msg) {
    const box = $("error");
    if (!msg) { box.classList.add("hidden"); box.replaceChildren(); return; }
    const close = document.createElement("button");
    close.type = "button"; close.setAttribute("aria-label", "닫기"); close.textContent = "×";
    close.addEventListener("click", () => setError(""));
    const text = document.createElement("span");
    text.textContent = msg;
    box.innerHTML = ICON.alert;                    // 고정 아이콘
    box.append(text, close);
    box.classList.remove("hidden");
  }

  // ---------- 입력 화면 ----------
  function updateInputState() {
    const len = $("description").value.trim().length;
    $("counter").textContent = `${$("description").value.length} / 2000`;
    $("counter").classList.toggle("low", len > 0 && len < MIN_LEN);
    $("submit-btn").disabled = len < MIN_LEN;
  }
  function initInput() {
    const list = $("examples");
    EXAMPLES.forEach(([label, text]) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "example"; b.textContent = label;
      b.addEventListener("click", () => { $("description").value = text; updateInputState(); $("description").focus(); });
      list.append(b);
    });
    $("description").addEventListener("input", updateInputState);
    updateInputState();
    $("accident-form").addEventListener("submit", onSubmitAccident);
  }

  // ---------- 통신 ----------
  async function post(path, body) {
    const res = await fetch(API + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    let data = null;
    try { data = await res.json(); } catch (_) { /* 본문 없음 */ }
    return { res, data };
  }
  function errorMessage(res, data) {
    const d = data && data.detail;
    if (res.status === 422 && d && d.code === "out_of_scope") return "이 서비스는 자동차 대 자동차 사고만 확인할 수 있어요. (" + d.message + ")";
    if (res.status === 422 && d && d.code === "invalid_option") return "선택한 답변이 올바르지 않아요. 다시 선택해 주세요.";
    if (res.status === 422) return "입력을 확인해 주세요: " + (Array.isArray(d) ? d.map((x) => x.msg).join(", ") : "형식이 올바르지 않아요.");
    if (res.status === 404) return "질문 시간이 지났거나 찾을 수 없는 요청이에요. 처음부터 다시 시도해 주세요.";
    if (res.status === 409) return "이미 답변한 요청이에요. 처음부터 다시 시도해 주세요.";
    if (res.status === 503) return "요청이 많아요. 잠시 후 다시 시도해 주세요.";
    return "오류가 발생했어요 (HTTP " + res.status + ").";
  }

  // ---------- 대기 ----------
  function startWait(text) {
    $("wait-text").textContent = text;
    $("elapsed").textContent = "0";
    const t0 = Date.now();
    clearInterval(state.timer);
    state.timer = setInterval(() => ($("elapsed").textContent = String(Math.floor((Date.now() - t0) / 1000))), 500);
    show("wait");
  }
  const stopWait = () => clearInterval(state.timer);

  function handle(res, data, fallbackView) {
    stopWait();
    if (!res.ok) { setError(errorMessage(res, data)); show(fallbackView); return; }
    if (data.status === "needs_answer") renderQuestion(data);
    else { state.sessionId = null; renderResult(data.result); }
  }

  async function onSubmitAccident(e) {
    e.preventDefault();
    if ($("submit-btn").disabled) return;
    setError("");
    state.sessionId = null;
    const num = (id) => { const v = parseFloat($(id).value); return Number.isNaN(v) ? null : v; };
    const body = {
      description: $("description").value,
      lane_info: $("lane_info").value || null,
      speed_a_kmh: num("speed_a"),
      speed_b_kmh: num("speed_b"),
    };
    startWait("사고 상황을 분석하고 있어요");
    try { const { res, data } = await post("/v1/assess", body); handle(res, data, "input"); }
    catch (_) { stopWait(); show("input"); setError("서버에 연결하지 못했어요. 백엔드가 실행 중인지 확인해 주세요."); }
  }

  // ---------- 질문 ----------
  function renderQuestion(data) {
    state.sessionId = data.session_id;
    state.seed = data.question.seed;
    $("q-text").textContent = data.question.text;
    $("q-notice-text").textContent = data.notice || "";
    $("q-notice").classList.toggle("hidden", !data.notice);

    const ul = $("q-options");
    ul.replaceChildren();
    let separated = false;
    data.question.options.forEach((o) => {
      const special = o.id === UNKNOWN_ID || o.id === OTHER_ID;
      if (special && !separated) {
        separated = true;
        const sep = document.createElement("li");
        sep.className = "sep"; sep.setAttribute("aria-hidden", "true"); sep.textContent = "잘 모르겠거나 해당하는 보기가 없다면";
        ul.append(sep);
      }
      const li = document.createElement("li");
      const label = document.createElement("label");
      label.className = "option" + (special ? " option--special" : "");
      const input = document.createElement("input");
      input.type = "radio"; input.name = "option"; input.value = String(o.id);
      input.addEventListener("change", () => ($("answer-btn").disabled = false));
      const body = document.createElement("span");
      body.className = "option__body";
      const dot = document.createElement("span"); dot.className = "option__dot"; dot.setAttribute("aria-hidden", "true");
      const text = document.createElement("span"); text.textContent = o.label;
      body.append(dot, text);
      label.append(input, body);
      li.append(label);
      ul.append(li);
    });
    $("answer-btn").disabled = true;                 // 처음에는 아무것도 선택하지 않는다
    show("question");
  }

  async function onSubmitAnswer(e) {
    e.preventDefault();
    setError("");
    const picked = document.querySelector('input[name="option"]:checked');
    if (!picked) return;
    const id = state.sessionId;
    startWait("답변을 반영하고 있어요");
    try {
      const { res, data } = await post("/v1/assess/" + encodeURIComponent(id) + "/answer", { option_id: Number(picked.value), seed: state.seed });
      // 잘못된 선택(422)은 세션이 열려 있으므로 질문으로 돌아가고, 나머지 오류는 처음 화면으로
      handle(res, data, res.status === 422 && data && data.detail && data.detail.code === "invalid_option" ? "question" : "input");
    } catch (_) { stopWait(); show("question"); setError("서버에 연결하지 못했어요."); }
  }

  // ---------- 결과 ----------
  function countUp(node, to, render) {
    cancelAnimationFrame(state.rafId);
    if (reduceMotion || to === 0) { node.textContent = render(to); return; }
    const t0 = performance.now(), dur = 900;
    // 탭이 백그라운드이면 rAF 가 멈출 수 있으므로 끝값은 타이머로도 보장한다
    setTimeout(() => { cancelAnimationFrame(state.rafId); node.textContent = render(to); }, dur + 150);
    const tick = (now) => {
      const p = Math.min(1, (now - t0) / dur), eased = 1 - Math.pow(1 - p, 3);
      node.textContent = render(Math.round(to * eased));
      if (p < 1) state.rafId = requestAnimationFrame(tick);
    };
    state.rafId = requestAnimationFrame(tick);
  }

  function renderChartBasis(items) {
    const box = $("r-charts");
    box.replaceChildren(...items.map((c) => {
      const card = document.createElement("div");
      card.className = "chart";
      const head = document.createElement("div"); head.className = "chart__head";
      const no = document.createElement("span"); no.className = "chart__no"; no.textContent = c.chart_no;
      const label = document.createElement("span"); label.className = "chart__label"; label.textContent = c.label;
      head.append(no, label);
      card.append(head);
      const row = (text) => { const d = document.createElement("div"); d.className = "chart__row"; d.textContent = text; return d; };
      card.append(row("도표가 다루는 상황: " + c.situation));
      if (c.laws && c.laws.length) card.append(row("관련 법규: " + c.laws.join(" / ")));
      const base = document.createElement("div"); base.className = "chart__base";
      const me = document.createElement("span"); me.style.width = c.my_base + "%";
      base.append(me);
      const legend = document.createElement("div"); legend.className = "chart__legend";
      const l1 = document.createElement("span"); l1.textContent = "기본 과실비율 · 나 " + c.my_base;
      const l2 = document.createElement("span"); l2.textContent = "상대 " + c.other_base;
      legend.append(l1, l2);
      card.append(base, legend);
      return card;
    }));
    $("r-charts-wrap").classList.toggle("hidden", items.length === 0);
  }

  function renderResult(r) {
    state.lastResult = r;
    const same = r.my_fault_low === r.my_fault_high;
    const range = (a, b) => (a === b ? `${a}%` : `${a} ~ ${b}%`);

    // 큰 숫자: 한 값이면 한 번에 올라가고, 범위면 하한·상한이 함께 올라간다
    const num = document.createElement("span");
    const small = document.createElement("small");
    small.textContent = same ? "" : "범위";
    $("r-mine").replaceChildren(num, small);
    countUp(num, r.my_fault_high, (v) => {
      const lo = same ? v : Math.round(v * (r.my_fault_low / (r.my_fault_high || 1)));
      return same ? `${v}%` : `${lo} ~ ${v}%`;
    });

    $("bar-me").style.width = "0%"; $("bar-range").style.width = "0%";
    requestAnimationFrame(() => requestAnimationFrame(() => {
      $("bar-me").style.width = r.my_fault_low + "%";
      $("bar-range").style.width = r.my_fault_high - r.my_fault_low + "%";
    }));
    $("leg-me").textContent = range(r.my_fault_low, r.my_fault_high);
    $("leg-other").textContent = range(r.other_fault_low, r.other_fault_high);

    $("r-assumption").textContent = r.assumption;
    $("r-assumption").classList.toggle("hidden", !r.assumption);
    $("r-uncertain").classList.toggle("hidden", !r.uncertain);
    $("r-summary").textContent = r.summary;
    renderChartBasis(r.chart_basis || []);

    const ev = $("r-evidence");
    ev.replaceChildren(...(r.evidence_to_check || []).map((t) => {
      const li = document.createElement("li");
      li.innerHTML = ICON.check;                       // 고정 아이콘
      const span = document.createElement("span"); span.textContent = t;
      li.append(span);
      return li;
    }));
    $("r-evidence-card").classList.toggle("hidden", !(r.evidence_to_check || []).length);

    $("r-other-text").textContent = r.other_possibilities;
    $("r-other-card").classList.toggle("hidden", !r.other_possibilities);
    $("r-disclaimer").textContent = r.disclaimer;
    $("copy-btn").textContent = "결과 복사하기";
    show("result");
  }

  async function copyResult() {
    const r = state.lastResult;
    if (!r) return;
    const range = (a, b) => (a === b ? `${a}%` : `${a}~${b}%`);
    const lines = [
      `나의 과실 ${range(r.my_fault_low, r.my_fault_high)} / 상대 ${range(r.other_fault_low, r.other_fault_high)}`,
      r.assumption, "", r.summary,
      ...(r.chart_basis || []).map((c) => `· ${c.chart_no} ${c.label} (기본 나 ${c.my_base} : 상대 ${c.other_base})`),
      ...((r.evidence_to_check || []).length ? ["", "확인할 증거: " + r.evidence_to_check.join(", ")] : []),
      "", r.disclaimer,
    ].filter((x, i, a) => !(x === "" && a[i - 1] === ""));
    const text = lines.join("\n").trim();
    let ok = false;
    try { await navigator.clipboard.writeText(text); ok = true; } catch (_) {
      const ta = document.createElement("textarea"); ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
      document.body.append(ta); ta.select();
      try { ok = document.execCommand("copy"); } catch (_) { ok = false; }
      ta.remove();
    }
    $("copy-btn").textContent = ok ? "복사했어요 ✓" : "복사하지 못했어요";
    setTimeout(() => ($("copy-btn").textContent = "결과 복사하기"), 1800);
  }

  // ---------- 시작 ----------
  initTheme();
  initInput();
  $("answer-form").addEventListener("submit", onSubmitAnswer);
  $("copy-btn").addEventListener("click", copyResult);
  $("restart-btn").addEventListener("click", () => { setError(""); state.sessionId = null; show("input"); $("description").focus(); });
  setSteps("input");
})();
