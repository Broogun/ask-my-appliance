/* API 래퍼 — 토큰 붙이기, 401이면 로그인으로 */
const API = {
  token() { return localStorage.getItem('token') || ''; },
  async req(method, url, body) {
    const res = await fetch(url, {
      method, headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + this.token() },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (res.status === 401) { localStorage.removeItem('token'); location.href = '/'; throw new Error('unauthorized'); }
    if (res.status === 204) return null;
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || ('요청 실패 (' + res.status + ')'));
    return data;
  },
  get(url) { return this.req('GET', url); },
  post(url, body) { return this.req('POST', url, body); },
  patch(url, body) { return this.req('PATCH', url, body); },
  del(url) { return this.req('DELETE', url); },

  /* 스트리밍(NDJSON): onEvent(obj) 를 줄마다 호출 */
  async stream(url, body, onEvent) {
    const res = await fetch(url, {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + this.token() },
      body: JSON.stringify(body),
    });
    if (res.status === 401) { localStorage.removeItem('token'); location.href = '/'; return; }
    if (!res.ok) { const d = await res.json().catch(() => ({})); throw new Error(d.detail || '전송 실패'); }
    const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
        if (line) onEvent(JSON.parse(line));
      }
    }
  },
};

// 브랜드 마크(렌치+말풍선) — 이모지 대신 쓰는 인라인 SVG, currentColor라 배경에 맞춰 색이 따라감
const BRAND_MARK = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H17.5A2.5 2.5 0 0 1 20 5.5V13.5A2.5 2.5 0 0 1 17.5 16H10.5L6.5 19V16A2.5 2.5 0 0 1 4 13.5V5.5Z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/><g transform="translate(6,3) scale(0.58)"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" fill="currentColor"/></g></svg>`;
// 브랜드 마크를 살짝 캐릭터화(눈+미소) — 감정적으로 의미있는 순간(등록 완료, 답변 못 찾음)에만 제한적으로 사용
const CHARACTER_MARK = `<svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H17.5A2.5 2.5 0 0 1 20 5.5V13.5A2.5 2.5 0 0 1 17.5 16H10.5L6.5 19V16A2.5 2.5 0 0 1 4 13.5V5.5Z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/><circle cx="9" cy="8.3" r="1" fill="currentColor"/><circle cx="15" cy="8.3" r="1" fill="currentColor"/><path d="M9 11.6c1 1.1 5 1.1 6 0" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" fill="none"/><g transform="translate(13.3,10.3) scale(0.32)"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" fill="currentColor"/></g></svg>`;
const ICON = { '에어컨': '❄️', '냉장고': '🧊', '김치냉장고': '🥬', '세탁기': '🫧', '세탁건조기': '🫧', '건조기': '🌬️', '스타일러': '👔', '슈드레서': '👟' };
const iconOf = (t) => ICON[t] || '🔌';
// 제품군마다 다른 톤을 줘서 아이콘 자리(사진 없을 때)를 더 눈에 띄고 재밌게 — 사진이 있으면 has-img가 흰 배경으로 덮어써서 안 보임
const TYPE_TINT = {
  '에어컨': 'linear-gradient(160deg,#dff1f5,#c3e6ee)', '냉장고': 'linear-gradient(160deg,#e2eefc,#c9def7)',
  '김치냉장고': 'linear-gradient(160deg,#e8f5e0,#d3ecc2)', '세탁기': 'linear-gradient(160deg,#eee5fb,#ddceF5)',
  '세탁건조기': 'linear-gradient(160deg,#eee5fb,#ddceF5)', '건조기': 'linear-gradient(160deg,#fdedd9,#fad9b0)',
  '스타일러': 'linear-gradient(160deg,#efe6ee,#ddc9db)', '슈드레서': 'linear-gradient(160deg,#f5ecda,#e9d7ae)',
};
const tintOf = (t) => TYPE_TINT[t] || '';
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
// type: '' (기본, 어두운 톤) | 'ok' (성공) | 'warn' (삭제 등 주의) | 'danger' (에러)
function toast(msg, type = '') {
  let t = document.querySelector('.toast');
  if (!t) { t = document.createElement('div'); t.className = 'toast'; document.body.appendChild(t); }
  t.textContent = msg; t.className = 'toast show' + (type ? ' toast-' + type : '');
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove('show'), 2200);
}
function requireLogin() { if (!API.token()) location.href = '/'; }

/* 제품 썸네일 — 3단 폴백: 실제 사진(web/static/img/models/{id}.jpg|png, 팀이 넣음) → 설명서 선화(auto/{id}.png, web/tools/make_thumbs.py) → 제품군 아이콘.
   <img>가 404면 onerror 체인으로 다음 후보를 시도한다. cls: 감쌀 div 클래스 (product-thumb / thumb-xs / p-thumb …) */
function thumbHTML(manualId, productType, cls = 'product-thumb') {
  const id = encodeURIComponent(manualId || '');
  const icon = iconOf(productType);
  const tintStyle = tintOf(productType) ? ` style="--thumb-tint:${tintOf(productType)}"` : '';
  if (!id) return `<div class="${cls}"${tintStyle}>${icon}</div>`;
  const chain = [`/static/img/models/${id}.png`, `/static/img/models/${id}.jpg`, `/static/img/models/auto/${id}.png`];   // png 우선 (LG 사진은 png로 통일)
  return `<div class="${cls} has-img"${tintStyle}><img src="${chain[0]}" alt="" loading="lazy" data-next="${chain.slice(1).join('|')}" data-icon="${icon}"
    onerror="const n=(this.dataset.next||'').split('|').filter(Boolean); if(n.length){this.dataset.next=n.slice(1).join('|'); this.src=n[0];} else {const d=this.parentElement; d.classList.remove('has-img'); d.textContent=this.dataset.icon;}"></div>`;
}
