/* 앱 셸 — 뷰 전환(제품 선택 / 채팅 / 대화 기록 / 내 제품) + 모달 */
requireLogin();
const S = { appliances: [], conversations: [], conv: null, picked: null, streaming: false };
const $ = (sel) => document.querySelector(sel);
const ROUTE_LABEL = {
  error_code: ['⚠️ 오류코드 안내', 'ok mono-tag'], feature_absent: ['이 모델에 없는 기능', 'warn'], vector: ['설명서 근거', 'accent'],
  none: ['설명서에 없는 내용', ''], unregistered: ['미등록 제품', 'warn'], sibling: ['다른 모델 설명서 기준', 'warn'],
  followup: ['이전 답변에 이어서', 'accent'],
};
// 첫 화면 기본 질문 — 제품군의 모든 모델에서 근거가 나오는 것만 (web/tools/make_suggest.py 로 전수 검사해서 고름)
const SUGGEST = {
  '에어컨': ['필터 청소는 어떻게 해요?', '찬바람이 안 나와요', '예약 기능 설정하는 법', '이상한 냄새가 나요'],
  '냉장고': ['냉장실 음식이 얼어요', '냉장고에서 소리가 나요', '냉장고가 시원하지 않아요', '냉장고 청소는 어떻게 해요?'],
  '김치냉장고': ['김치 숙성은 어떻게 해요?', '김치가 빨리 시어요', '김치냉장고에서 소리가 나요', '김치냉장고가 시원하지 않아요'],
  '세탁기': ['탈수가 안 돼요', 'UE 떴어요', '세제는 어디에 넣어요?', '세탁통 청소 방법'],
  '세탁건조기': ['건조가 잘 안 돼요', '문이 안 열려요', '탈수가 안 돼요', '세제는 어디에 넣어요?'],
  '건조기': ['건조가 너무 오래 걸려요', '필터 청소는 어떻게 해요?', '옷에 보풀이 묻어나요', '이불 건조 코스 있어요?'],
  '스타일러': ['물통에 물 채우는 법', '옷에서 냄새가 나요', '연기가 나와요', '예약 설정하는 법'],
  '슈드레서': ['신발 관리 코스 뭐 있어요?', '물통 비우는 법', '냄새가 안 빠져요', '소음이 심해요'],
};

/* ── 뷰 전환 ─────────────────────────────────────────────────── */
function show(view) {
  document.querySelectorAll('.view').forEach(v => v.classList.add('hidden'));
  $('#view-' + view).classList.remove('hidden');
  document.querySelectorAll('.nav-item').forEach(b => b.classList.toggle('active', b.dataset.view === view));
  document.querySelectorAll('.recent-item').forEach(b => b.classList.toggle('active', view === 'chat' && S.conv && +b.dataset.cid === S.conv.id));
  $('#topbarRight').innerHTML = '';
  if (view === 'history') $('#topbarLeft').innerHTML = '대화 기록';
  if (view === 'products') $('#topbarLeft').innerHTML = '내 제품';
  if (view === 'picker') $('#topbarLeft').innerHTML = '새 대화';
}

/* ── 데이터 ──────────────────────────────────────────────────── */
async function loadAppliances() { S.appliances = await API.get('/api/appliances'); }
async function loadConversations() {
  S.conversations = await API.get('/api/conversations');
  $('#recentList').innerHTML = S.conversations.slice(0, 20).map(c =>
    `<button class="recent-item" data-cid="${c.id}" title="${esc(c.title)}">${esc(c.title)}</button>`).join('') || '<div class="muted small" style="padding:6px 10px">아직 대화가 없어요</div>';
}

/* ── 제품 선택 (새 대화) ─────────────────────────────────────── */
function renderPicker() {
  S.picked = null; $('#pickerGo').disabled = true;
  $('#pickerList').innerHTML = S.appliances.map(a => `<button class="picker-option" data-aid="${a.id}">
    <div class="radio"></div>${thumbHTML(a.manual_id, a.product_type)}
    <div><div class="p-name">${esc(a.nickname)}</div><div class="p-model"><span class="nameplate">${esc(a.model)}</span>${brandTag(a.brand, a.brand_name)}${a.location ? `<span class="tag loc-tag">📍 ${esc(a.location)}</span>` : ''}</div></div></button>`).join('')
    || '<div class="empty-note">등록된 제품이 없어요. 아래에서 먼저 등록해 주세요.</div>';
  show('picker');
}
$('#pickerList').addEventListener('click', (e) => {
  const opt = e.target.closest('.picker-option'); if (!opt) return;
  document.querySelectorAll('.picker-option').forEach(o => o.classList.remove('selected')); opt.classList.add('selected');
  S.picked = +opt.dataset.aid; $('#pickerGo').disabled = false;
});
$('#pickerGo').onclick = async () => { const c = await API.post('/api/conversations', { appliance_id: S.picked }); openConversation(c); };
$('#pickerAdd').onclick = () => openAddModal();

/* ── 채팅 ────────────────────────────────────────────────────── */
function openConversation(c) {
  S.conv = c; show('chat');
  const a = c.appliance;
  $('#topbarLeft').innerHTML = `<div class="context-label">${thumbHTML(a.manual_id, a.product_type, 'context-thumb')}<b>${esc(a.nickname)}</b>${brandTag(a.brand, a.brand_name)}<span class="nameplate">${esc(a.model)}</span><span class="muted">기준으로 답변 중</span></div>`;
  $('#topbarRight').innerHTML = `<button class="btn ghost sm" id="supportBtn"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/></svg> 고객센터 연결</button>`;
  $('#supportBtn').onclick = async () => { const s = await API.get('/api/catalog/support/' + a.brand); if (confirm(`${s.name} ${s.phone}\n홈페이지로 이동할까요?`)) window.open(s.url, '_blank'); };
  const conv = $('#conv'); conv.innerHTML = '';
  if (!c.messages.length) {
    const sug = SUGGEST[a.product_type] || SUGGEST['세탁기'];
    conv.innerHTML = `<div class="chat-empty"><div class="avatar-wrap">${thumbHTML(a.manual_id, a.product_type, 'avatar')}<span class="avatar-badge">${CHARACTER_MARK}</span></div>
      <h4>${esc(a.nickname)}에 대해 무엇이든 물어보세요</h4><p class="small">${esc(a.brand_name)} ${esc(a.model)} 설명서를 기반으로 답변해드려요</p>
      <div class="suggest-grid">${sug.map(s => `<button class="suggest-chip" data-q="${esc(s)}">"${esc(s)}"</button>`).join('')}</div></div>`;
    renderSources([], null);
  } else {
    c.messages.forEach(m => appendMessage(m));
    const last = [...c.messages].reverse().find(m => m.role === 'assistant');
    renderSources(last ? last.sources : [], last ? last.route : null);
  }
  $('#askInput').focus();
}
$('#conv').addEventListener('click', (e) => { const chip = e.target.closest('[data-q]'); if (chip) ask(chip.dataset.q); const reg = e.target.closest('[data-register]'); if (reg) openAddModal(); });

// 서버(manual_figures)가 줄 번호로 매칭한 설명서 그림 [{line, figures:[{url, caption, credit?}]}] 의 HTML. 제조사 사이트 사진이면 출처를 표시한다
function figsHtml(figs) {
  if (!figs || !figs.length) return '';
  const credit = (figs.find(f => f.credit) || {}).credit;
  return `<div class="ans-figs">${figs.map(f =>
    `<a class="ans-fig" href="${f.url}" target="_blank" rel="noopener" title="${esc(f.caption)} - 클릭하면 크게 볼 수 있어요"><img src="${f.url}" alt="" loading="lazy" onerror="this.closest('.ans-fig').remove()"></a>`).join('')}</div>`
    + (credit ? `<div class="ans-credit">이미지 출처: ${credit.url ? `<a href="${esc(credit.url)}" target="_blank" rel="noopener">${esc(credit.text)}</a>` : esc(credit.text)}</div>` : '');
}

// 라이브러리 없이 답변에 자주 나오는 서식만 처리 - 번호목록/불릿/굵게, 나머지는 문단.
// 그림은 서버가 매긴 줄 번호(split('\n') 기준 - manual_figures.split_lines 와 같은 규칙) 뒤에 끼운다: 목록 항목이면 그 항목 안에, 문단이면 문단 뒤에.
function renderAnswer(text, figures) {
  const figsByLine = {}; (figures || []).forEach(f => { figsByLine[f.line] = f.figures; });
  const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  let html = '', listType = null, liOpen = false, para = [];
  const flushPara = () => { if (para.length) { html += `<p>${para.join('<br>')}</p>`; para = []; } };
  const closeLi = () => { if (liOpen) { html += '</li>'; liOpen = false; } };
  const closeList = () => { closeLi(); if (listType) { html += `</${listType}>`; listType = null; } };
  String(text ?? '').split('\n').forEach((raw, i) => {
    const line = raw.trim();
    const figs = figsHtml(figsByLine[i]);
    // 빈 줄은 목록을 닫지 않는다 - 번호 항목 사이에 빈 줄이 있어도 다음이 같은 종류의 항목이면 한 목록으로 이어야 번호가 1,2,3으로 이어진다
    if (!line) { flushPara(); if (figs) { closeList(); html += figs; } return; }
    const ol = line.match(/^(\d+)[.)]\s+(.*)/);
    const ul = line.match(/^[-*·]\s+(.*)/);
    if (ol) { flushPara(); closeLi(); if (listType !== 'ol') { closeList(); html += '<ol>'; listType = 'ol'; } html += `<li>${inline(ol[2])}`; liOpen = true; html += figs; }
    else if (ul) { flushPara(); closeLi(); if (listType !== 'ul') { closeList(); html += '<ul>'; listType = 'ul'; } html += `<li>${inline(ul[1])}`; liOpen = true; html += figs; }
    else { closeList(); para.push(inline(line)); if (figs) { flushPara(); html += figs; } }
  });
  closeList(); flushPara();
  return html;
}

function appendMessage(m) {
  const conv = $('#conv'); conv.querySelector('.chat-empty')?.remove();
  if (m.role === 'user') { conv.insertAdjacentHTML('beforeend', `<div class="bubble-user">${esc(m.content)}</div>`); }
  else {
    const el = document.createElement('div'); el.className = 'answer-block'; el.dataset.mid = m.id || '';
    el.innerHTML = `<div class="bot-avatar">${BRAND_MARK}</div><div style="flex:1;min-width:0"><div class="answer-body"></div><div class="answer-meta"></div></div>`;
    conv.appendChild(el); setAnswer(el, m);
  }
  conv.scrollTop = conv.scrollHeight;
}
function setAnswer(el, m) {
  const body = el.querySelector('.answer-body'), meta = el.querySelector('.answer-meta');
  if (m.route === 'unregistered') {
    body.innerHTML = `<div class="register-prompt"><div class="msg">${esc(m.content)}</div><button class="btn sm" data-register="1">이 제품 등록하기</button></div>`;
  } else body.innerHTML = renderAnswer(m.content, m.figures);
  const [label, cls] = ROUTE_LABEL[m.route] || ['', ''];
  meta.innerHTML = label ? `<span class="tag ${cls}">${label}</span>` : '';
}
const ROUTE_ICON = { error_code: '⚠️', feature_absent: '🚫', vector: '📄', sibling: '📎', followup: '↪️' };
// LG/삼성 CI 컬러를 옅게 우려서 "어느 회사 설명서 기준인지"만 표시 — 서비스 자체가 특정 브랜드 소속처럼 보이지 않도록 배지 하나로만 씀
function brandTag(brand, brandName) { return brand ? `<span class="tag brand-${esc(brand)}">${esc(brandName)}</span>` : ''; }
// 거리값을 관련도 %로 바꾸고, 구간별로 배지 색을 다르게 (80%+ 초록 / 50~80% 노랑 / 그 미만 무채색)
function relevanceBadge(distance) {
  const pct = Math.round((1 - distance) * 100);
  const cls = pct >= 80 ? 'ok' : pct >= 50 ? 'warn' : '';
  return `<span class="tag ${cls}">관련도 ${pct}%</span>`;
}
function renderSources(sources, route) {
  const p = $('#sources');
  if (route === 'unregistered') { p.innerHTML = `<h4>참고한 설명서 / 출처</h4><div class="sources-empty">등록된 제품이 아니라서<br>연결된 설명서가 없어요</div>`; return; }
  if (!sources || !sources.length) {
    p.innerHTML = `<h4>참고한 설명서 / 출처</h4><div class="sources-empty">${route === 'none'
      ? `<div class="empty-character">${CHARACTER_MARK}</div>이 질문과 관련된 설명서 내용을<br>찾지 못했어요`
      : '질문하면 참고한 설명서 조각이<br>여기 표시돼요'}</div>`;
    return;
  }
  const a = S.conv?.appliance;
  const seenPages = new Set();
  p.innerHTML = `<h4>${ROUTE_ICON[route] || '📄'} 참고한 설명서 / 출처 ${sources.length}건 ${a ? brandTag(a.brand, a.brand_name) : ''}</h4>` + sources.map((s, i) => `<div class="source-card clickable" data-src="${i}" title="클릭하면 전문과 PDF 페이지를 볼 수 있어요">
    <div class="top-row"><div class="s-title">${esc(s.title)}</div><span class="tag">${esc(s.tag)}</span></div>
    <div class="s-snippet">${esc(s.snippet)}</div>
    ${thumbHtml(s, seenPages)}
    <div class="s-doc"><span>${esc(s.doc_id)}${pageLabel(s) ? ` <b>${pageLabel(s)}</b>` : ''}</span>${relevanceBadge(s.distance)}</div></div>`).join('');
  p.querySelectorAll('[data-src]').forEach(el => el.onclick = () => openSource(sources[+el.dataset.src]));
}
// 청크가 실린 PDF 페이지의 렌더링 이미지(GET /api/manuals/{doc_id}/page/{n}.png). 같은 페이지의 출처가 여럿이면 첫 카드에만 보여준다
function thumbHtml(s, seen) {
  if (!s.pdf_url || !s.page_start) return '';
  const src = `${s.pdf_url.replace(/\/pdf$/, '')}/page/${s.page_start}.png`;
  if (seen.has(src)) return '';
  seen.add(src);
  return `<img class="s-thumb" src="${src}" alt="설명서 ${pageLabel(s)}" loading="lazy" onerror="this.remove()">`;
}
// 청크가 실린 PDF 페이지 (assign_pages). 에러코드 청크나 예전 대화(페이지 정보 없이 저장된 것)는 빈 문자열
function pageLabel(s) {
  if (!s.page_start) return '';
  return s.page_end && s.page_end !== s.page_start ? `p.${s.page_start}–${s.page_end}` : `p.${s.page_start}`;
}
// 출처 카드 클릭 → 조각 전문 + (PDF가 있으면) 브라우저 PDF 뷰어를 해당 페이지로 연 iframe. #page=N 은 Chrome/Edge 내장 뷰어가 지원
function openSource(s) {
  const page = s.page_start || 1;
  const pdfHref = s.pdf_url ? `${s.pdf_url}#page=${page}` : null;
  $('#srcTitle').textContent = s.title;
  $('#modalSourceBody').innerHTML = `
    <div class="src-meta"><span class="tag">${esc(s.tag)}</span> <span>${esc(s.doc_id)}</span>${pageLabel(s) ? ` <span class="muted">${pageLabel(s)}</span>` : ''}
      ${pdfHref ? ` <a class="src-open" href="${pdfHref}" target="_blank" rel="noopener">새 탭에서 PDF 열기 ↗</a>` : ''}</div>
    <pre class="src-body">${esc(s.body || s.snippet)}</pre>
    ${pdfHref ? `<iframe class="src-pdf" src="${pdfHref}" title="설명서 PDF"></iframe>` : `<div class="sources-empty">이 출처는 PDF 설명서가 아니라 제조사 에러코드 표에서 가져온 내용이에요${(() => { const pg = (s.images || []).find(i => i.page); return pg ? `<br><a href="${esc(pg.page)}" target="_blank" rel="noopener">${esc(pg.credit || '제조사')} 원문 보기 ↗</a>` : ''; })()}</div>`}`;
  $('#modalSource').classList.remove('hidden');
}
function renderSourcesLoading() { $('#sources').innerHTML = `<h4>참고한 설명서 / 출처</h4>` + [1, 2].map(() => `<div class="source-card"><div class="skeleton" style="width:60%"></div><div class="skeleton"></div><div class="skeleton" style="width:80%"></div></div>`).join(''); }

async function ask(text) {
  text = (text || $('#askInput').value).trim(); if (!text || S.streaming || !S.conv) return;
  S.streaming = true; $('#askInput').value = ''; $('#askInput').disabled = true; $('#askBtn').disabled = true;
  $('#askInput').placeholder = '답변을 생성하는 동안에는 잠시만 기다려주세요';
  appendMessage({ role: 'user', content: text });
  const el = document.createElement('div'); el.className = 'answer-block';
  el.innerHTML = `<div class="bot-avatar thinking">${BRAND_MARK}</div><div style="flex:1;min-width:0"><div class="answer-body"><span class="typing-dots"><span></span><span></span><span></span></span> <span class="muted small">관련 설명서를 찾고 답변을 준비하고 있어요…</span></div><div class="answer-meta"></div></div>`;
  $('#conv').appendChild(el); $('#conv').scrollTop = $('#conv').scrollHeight; renderSourcesLoading();
  const body = el.querySelector('.answer-body'); let acc = '', route = 'vector', figures = [];
  // 첫 질문이면(초안, id=null) /start 가 대화 행을 만들고 스트림 첫 줄(conv)로 알려준다 — 질문 없이 나간 대화는 기록에 안 남는다
  const url = S.conv.id ? `/api/conversations/${S.conv.id}/messages` : '/api/conversations/start';
  const payload = S.conv.id ? { content: text } : { appliance_id: S.conv.appliance.id, content: text };
  try {
    await API.stream(url, payload, (ev) => {
      if (ev.type === 'conv') { S.conv = { ...ev.conversation, messages: [] }; }
      else if (ev.type === 'status') { body.innerHTML = `<span class="typing-dots"><span></span><span></span><span></span></span> <span class="muted small">${esc(ev.text)}</span>`; }
      else if (ev.type === 'sources') { route = ev.route; renderSources(ev.sources, ev.route); }
      else if (ev.type === 'token') { if (acc === '') el.querySelector('.bot-avatar')?.classList.remove('thinking'); acc += ev.text; if (route !== 'unregistered') body.innerHTML = renderAnswer(acc); $('#conv').scrollTop = $('#conv').scrollHeight; }
      else if (ev.type === 'figures') { figures = ev.items; }
      else if (ev.type === 'done') { el.querySelector('.bot-avatar')?.classList.remove('thinking'); setAnswer(el, { content: acc, route, figures }); el.dataset.mid = ev.message_id; }
    });
  } catch (ex) { body.textContent = '오류: ' + ex.message; }
  S.streaming = false; $('#askInput').disabled = false; $('#askBtn').disabled = false; $('#askInput').placeholder = '추가로 궁금한 점을 입력하세요'; $('#askInput').focus();
  loadConversations().then(() => show('chat'));
}
$('#askForm').addEventListener('submit', (e) => { e.preventDefault(); ask(); });

/* ── 대화 기록 ───────────────────────────────────────────────── */
// 시간(오늘 14:20) / 날짜(오늘·어제는 라벨, 그 외는 M월 D일 요일) 두 가지 표기 — 카드에선 시간, 그룹 헤더엔 날짜만
const histTimeFmt = (iso) => new Date(iso).toLocaleTimeString('ko-KR', { hour: 'numeric', minute: '2-digit' });
function histDateBucket(iso) {
  const d = new Date(iso), today = new Date(); const startOf = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate());
  const days = Math.round((startOf(today) - startOf(d)) / 86400000);
  if (days === 0) return '오늘';
  if (days === 1) return '어제';
  if (days <= 7) return '지난 7일';
  if (days <= 30) return '지난 30일';
  return d.toLocaleDateString('ko-KR', { year: startOf(d).getFullYear() !== startOf(today).getFullYear() ? 'numeric' : undefined, month: 'long' });
}
let histFilter = null; // appliance_id 또는 null(전체)
function historyCard(c) {
  return `<div class="history-card" data-open="${c.id}">
      ${thumbHTML(c.appliance.manual_id, c.appliance.product_type, 'thumb-xs')}
      <div class="h-body">
        <div class="h-title">${esc(c.title)}</div>
        <div class="h-meta"><span class="tag">${esc(c.appliance.nickname)}</span><span class="muted small">${c.message_count}개 메시지</span><span class="muted small h-time">${histTimeFmt(c.updated_at)}</span></div>
      </div>
      <button class="btn ghost sm" data-open="${c.id}">이어보기 ›</button>
      <button class="remove-btn" data-delconv="${c.id}" title="삭제">🗑</button></div>`;
}
async function renderHistory() {
  await loadConversations(); show('history');
  const main = $('#historyMain');
  if (!S.conversations.length) {
    main.innerHTML = `<div class="empty-state"><div class="icon-circle">🗂</div><h4>아직 대화 기록이 없어요</h4><p>"+ 새 대화 시작"에서 제품을 선택하고 질문해 보세요</p></div>`;
    return;
  }
  // 제품별 필터 chip — 등장한 제품만, 대화 많은 순
  const byAppliance = new Map();
  S.conversations.forEach(c => { const a = c.appliance; if (!byAppliance.has(a.id)) byAppliance.set(a.id, { a, n: 0 }); byAppliance.get(a.id).n++; });
  const chips = [...byAppliance.values()].sort((x, y) => y.n - x.n);
  if (histFilter && !byAppliance.has(histFilter)) histFilter = null;
  const list = histFilter ? S.conversations.filter(c => c.appliance.id === histFilter) : S.conversations;

  const filterHTML = chips.length > 1 ? `<div class="hist-filters">
      <button class="filter-chip ${histFilter === null ? 'active' : ''}" data-filter="">전체 <span class="muted">${S.conversations.length}</span></button>
      ${chips.map(({ a, n }) => `<button class="filter-chip ${histFilter === a.id ? 'active' : ''}" data-filter="${a.id}">${esc(a.nickname)} <span class="muted">${n}</span></button>`).join('')}
    </div>` : '';

  // 날짜 그룹 헤더 — updated_at 내림차순은 이미 API가 정렬해서 옴
  const groups = [];
  list.forEach(c => { const label = histDateBucket(c.updated_at); let g = groups[groups.length - 1]; if (!g || g.label !== label) { g = { label, items: [] }; groups.push(g); } g.items.push(c); });

  main.innerHTML = filterHTML + (list.length
    ? groups.map(g => `<section class="hist-group"><div class="hist-group-head">${g.label}</div>${g.items.map(historyCard).join('')}</section>`).join('')
    : `<div class="empty-note">이 제품과의 대화가 아직 없어요</div>`);
}
$('#historyMain').addEventListener('click', async (e) => {
  const f = e.target.closest('[data-filter]');
  if (f) { histFilter = f.dataset.filter ? +f.dataset.filter : null; return renderHistory(); }
  const d = e.target.closest('[data-delconv]');
  if (d) { if (confirm('이 대화를 삭제할까요?')) { await API.del('/api/conversations/' + d.dataset.delconv); renderHistory(); } return; }
  const o = e.target.closest('[data-open]');
  if (o) openConversationById(o.dataset.open, o);
});
// 대화를 여는 동안(설명서 그림·쪽 보정 계산으로 몇 초 걸릴 수 있음) 클릭한 항목에 진행 표시를 하고, 실패하면 알린다. 마지막으로 누른 항목이 이긴다.
let openSeq = 0;
async function openConversationById(id, btn) {
  const seq = ++openSeq;
  document.querySelectorAll('[aria-busy="true"]').forEach(x => x.removeAttribute('aria-busy'));
  btn.setAttribute('aria-busy', 'true');
  try {
    const conv = await API.get('/api/conversations/' + id);
    if (seq === openSeq) openConversation(conv);
  } catch (e) {
    if (e.message !== 'unauthorized' && seq === openSeq) toast('대화를 불러오지 못했어요: ' + e.message, 'danger');
  } finally {
    if (seq === openSeq) btn.removeAttribute('aria-busy');
  }
}
$('#recentList').addEventListener('click', (e) => { const b = e.target.closest('[data-cid]'); if (b) openConversationById(b.dataset.cid, b); });

/* ── 내 제품 ─────────────────────────────────────────────────── */
// 브랜드(LG/삼성)별로 묶어서 보여준다 — LG, 삼성 순, 그 외는 뒤에
function groupByBrand(list) {
  const order = { lg: 0, samsung: 1 };
  const groups = new Map();
  list.forEach(a => { const key = a.brand || 'etc'; if (!groups.has(key)) groups.set(key, { brand: a.brand, brand_name: a.brand_name, items: [] }); groups.get(key).items.push(a); });
  return [...groups.values()].sort((x, y) => (order[x.brand] ?? 9) - (order[y.brand] ?? 9));
}
function productCard(a) {
  return `<div class="product-grid-card" data-brand="${esc(a.brand)}">
      <div class="thumb-wrap">${thumbHTML(a.manual_id, a.product_type, 'thumb')}<span class="tag type-badge">${esc(a.product_type)}</span></div>
      <div class="name">${esc(a.nickname)}</div>
      <div class="model"><span class="nameplate">${esc(a.model)}</span>${brandTag(a.brand, a.brand_name)}${a.location ? `<span class="tag loc-tag">📍 ${esc(a.location)}</span>` : ''}</div>
      <div class="row-btns"><button class="btn outline sm" data-edit="${a.id}">수정</button><button class="btn danger sm" data-del="${a.id}">삭제</button></div></div>`;
}
async function renderProducts() {
  await loadAppliances(); show('products');
  const groups = groupByBrand(S.appliances);
  $('#productsMain').innerHTML = groups.map(g => `<section class="brand-group">
      <div class="brand-group-head">${brandTag(g.brand, g.brand_name) || '<span class="tag">기타</span>'}<span class="muted small">${g.items.length}개</span></div>
      <div class="product-grid">${g.items.map(productCard).join('')}</div></section>`).join('')
    + `<div class="product-grid"><button class="add-product-card" id="addProduct">+ 새 제품 등록하기</button></div>`;
}
$('#productsMain').addEventListener('click', async (e) => {
  if (e.target.closest('#addProduct')) return openAddModal();
  const ed = e.target.closest('[data-edit]'), del = e.target.closest('[data-del]');
  if (ed) openEditModal(S.appliances.find(a => a.id === +ed.dataset.edit));
  if (del && confirm('이 제품과 관련 대화를 삭제할까요?')) { await API.del('/api/appliances/' + del.dataset.del); await renderProducts(); await loadConversations(); toast('삭제했어요', 'warn'); }
});

/* ── 모달 ────────────────────────────────────────────────────── */
function openAddModal() {
  $('#modalAdd').classList.remove('hidden');
  RegisterFlow.mount($('#modalAddBody'), {
    doneLabel: '확인', onCancel: () => $('#modalAdd').classList.add('hidden'),
    onDone: async () => { $('#modalAdd').classList.add('hidden'); await loadAppliances(); toast('내 제품 목록에 추가됐어요', 'ok'); if (!$('#view-products').classList.contains('hidden')) renderProducts(); else renderPicker(); },
  });
}
let editing = null;
async function openEditModal(a) {
  editing = a; const opts = await API.get('/api/catalog/options');
  $('#modalEditBody').innerHTML = `<div class="product-card flat">${thumbHTML(a.manual_id, a.product_type, 'product-thumb lg')}
    <div class="meta"><div class="name">${esc(a.model)}</div><div class="model">${esc(a.manual_id)}</div></div><span class="tag">${esc(a.product_type)}</span>${brandTag(a.brand, a.brand_name)}</div>
    <div class="row-2"><div class="field"><label>제품 별칭</label><input class="input" id="ed-nick" value="${esc(a.nickname)}"></div>
      <div class="field"><label>설치 위치</label><select class="select" id="ed-loc">${['', '거실', '안방', '주방', '베란다', '세탁실', '작은방'].map(l => `<option value="${l}" ${l === a.location ? 'selected' : ''}>${l || '선택 안 함'}</option>`).join('')}</select></div></div>
    <div class="field"><label>모델 다시 찾기</label><div style="display:flex;gap:8px"><input class="input" id="ed-q" placeholder="모델명 검색 (예: FQ25)" style="flex:1"><button class="btn outline" id="ed-search" style="width:88px">검색</button></div>
      <div id="ed-results" style="margin-top:8px"></div><input type="hidden" id="ed-manual" value="${esc(a.manual_id)}"></div>
    <button class="link-btn danger" id="ed-delete">이 제품 삭제하기</button>`;
  $('#modalEdit').classList.remove('hidden');
  const search = async () => { const r = await API.get('/api/catalog/models?q=' + encodeURIComponent($('#ed-q').value.trim()));
    $('#ed-results').innerHTML = r.slice(0, 6).map(m => `<div class="product-card compact">${thumbHTML(m.manual_id, m.product_type, 'thumb-xs')}<div class="meta"><div class="name" style="font-size:13px">${esc(m.brand_name)} ${esc(m.product_type)} ${esc(m.model)}</div></div><button class="btn ghost sm" data-pick="${m.manual_id}" data-label="${esc(m.model)}">선택</button></div>`).join('') || '<div class="empty-note">결과 없음</div>'; };
  $('#ed-search').onclick = search; $('#ed-q').onkeydown = (e) => { if (e.key === 'Enter') { e.preventDefault(); search(); } };
  $('#ed-results').onclick = (e) => { const p = e.target.closest('[data-pick]'); if (p) { $('#ed-manual').value = p.dataset.pick; $('#ed-results').innerHTML = `<div class="tag accent">변경할 모델: ${esc(p.dataset.label)}</div>`; } };
  $('#ed-delete').onclick = async () => { if (confirm('이 제품과 관련 대화를 삭제할까요?')) { await API.del('/api/appliances/' + a.id); $('#modalEdit').classList.add('hidden'); renderProducts(); loadConversations(); } };
}
$('#editSave').onclick = async () => {
  try { await API.patch('/api/appliances/' + editing.id, { nickname: $('#ed-nick').value, location: $('#ed-loc').value, manual_id: $('#ed-manual').value });
    $('#modalEdit').classList.add('hidden'); toast('저장했어요', 'ok'); renderProducts(); } catch (ex) { toast(ex.message, 'danger'); }
};
document.querySelectorAll('[data-close]').forEach(b => b.onclick = () => $('#' + b.dataset.close).classList.add('hidden'));

/* ── 네비 ────────────────────────────────────────────────────── */
document.querySelectorAll('.nav-item').forEach(b => b.onclick = () => b.dataset.view === 'history' ? renderHistory() : renderProducts());
$('#newChat').onclick = async () => { await loadAppliances(); renderPicker(); };
$('#logout').onclick = async () => { await API.post('/api/auth/logout').catch(() => {}); localStorage.removeItem('token'); location.href = '/'; };

/* ── 사이드바 / 출처 패널 너비 드래그 조절 (localStorage에 저장, UI 전용) ───── */
function makeResizable(handle, panel, { min, max, storageKey, sign = 1 }) {
  if (!handle || !panel) return;
  const saved = +localStorage.getItem(storageKey);
  if (saved) panel.style.width = Math.min(max, Math.max(min, saved)) + 'px';
  handle.addEventListener('mousedown', (e) => {
    e.preventDefault();
    const startX = e.clientX, startW = panel.getBoundingClientRect().width;
    handle.classList.add('active'); document.body.style.cursor = 'col-resize'; document.body.style.userSelect = 'none';
    const onMove = (ev) => { const w = Math.min(max, Math.max(min, startW + sign * (ev.clientX - startX))); panel.style.width = w + 'px'; };
    const onUp = () => {
      document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp);
      handle.classList.remove('active'); document.body.style.cursor = ''; document.body.style.userSelect = '';
      localStorage.setItem(storageKey, parseInt(panel.style.width, 10));
    };
    document.addEventListener('mousemove', onMove); document.addEventListener('mouseup', onUp);
  });
}
makeResizable($('#sidebarResize'), document.querySelector('.sidebar'), { min: 200, max: 340, storageKey: 'ui.sidebarWidth', sign: 1 });
makeResizable($('#sourcesResize'), $('#sources'), { min: 260, max: 460, storageKey: 'ui.sourcesWidth', sign: -1 });

(async function init() {
  $('#whoami').textContent = localStorage.getItem('username') || 'admin1'; $('#avatar').textContent = ($('#whoami').textContent[0] || 'A').toUpperCase();
  await Promise.all([loadAppliances(), loadConversations()]);
  if (!S.appliances.length) { location.href = '/onboarding'; return; }
  renderPicker();
  watchReady();
})();

/* 검색기 준비 상태 — 준비될 때까지 입력창 위에 배너 */
async function watchReady() {
  const bar = document.querySelector('.input-bar');
  let banner = document.getElementById('readyBanner');
  if (!banner) { banner = document.createElement('div'); banner.id = 'readyBanner'; banner.className = 'ready-banner'; bar.parentNode.insertBefore(banner, bar); }
  for (;;) {
    let ok = false;
    try { ok = (await API.get('/api/health?warm=1')).ready; } catch (e) {}
    if (ok) { banner.remove(); $('#askInput').placeholder = '질문을 자유롭게 입력하세요'; return; }
    banner.innerHTML = `<span class="typing-dots"><span></span><span></span><span></span></span> 설명서 검색 엔진을 준비하고 있어요 (처음 한 번, 1~2분). 질문은 지금 입력해도 준비되는 대로 답해드려요.`;
    await new Promise(r => setTimeout(r, 3000));
  }
}
