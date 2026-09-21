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

const ICON = { '에어컨': '❄️', '냉장고': '🧊', '김치냉장고': '🥬', '세탁기': '🫧', '세탁건조기': '🫧', '건조기': '🌬️', '스타일러': '👔', '슈드레서': '👟' };
const iconOf = (t) => ICON[t] || '🔌';
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
function toast(msg) {
  let t = document.querySelector('.toast');
  if (!t) { t = document.createElement('div'); t.className = 'toast'; document.body.appendChild(t); }
  t.textContent = msg; t.classList.add('show'); clearTimeout(t._h); t._h = setTimeout(() => t.classList.remove('show'), 2200);
}
function requireLogin() { if (!API.token()) location.href = '/'; }

/* 제품 썸네일 — 3단 폴백: 실제 사진(web/static/img/models/{id}.jpg|png, 팀이 넣음) → 설명서 선화(auto/{id}.png, web/tools/make_thumbs.py) → 제품군 아이콘.
   <img>가 404면 onerror 체인으로 다음 후보를 시도한다. cls: 감쌀 div 클래스 (product-thumb / thumb-xs / p-thumb …) */
function thumbHTML(manualId, productType, cls = 'product-thumb') {
  const id = encodeURIComponent(manualId || '');
  const icon = iconOf(productType);
  if (!id) return `<div class="${cls}">${icon}</div>`;
  const chain = [`/static/img/models/${id}.png`, `/static/img/models/${id}.jpg`, `/static/img/models/auto/${id}.png`];   // png 우선 (LG 사진은 png로 통일)
  return `<div class="${cls} has-img"><img src="${chain[0]}" alt="" loading="lazy" data-next="${chain.slice(1).join('|')}" data-icon="${icon}"
    onerror="const n=(this.dataset.next||'').split('|').filter(Boolean); if(n.length){this.dataset.next=n.slice(1).join('|'); this.src=n[0];} else {const d=this.parentElement; d.classList.remove('has-img'); d.textContent=this.dataset.icon;}"></div>`;
}
