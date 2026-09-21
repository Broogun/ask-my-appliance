/* 제품 등록 3단계 (찾기 → 확인 → 완료). 온보딩 페이지와 "내 제품" 모달이 같은 코드를 쓴다.
   RegisterFlow.mount(container, { onDone(created), onCancel })                                   */
const RegisterFlow = {
  mount(el, opts = {}) {
    const st = { step: 1, cart: [], options: null, results: [], created: [] };
    const LOCATIONS = ['', '거실', '안방', '주방', '베란다', '세탁실', '작은방'];

    const stepTrack = () => `<div class="step-track">${[1, 2, 3].map((n, i) =>
      `<div class="step-dot ${n === st.step ? 'current' : n < st.step ? 'done' : ''}">${n < st.step ? '✓' : n}</div>${i < 2 ? '<div class="step-line"></div>' : ''}`).join('')}</div>`;

    const render = () => {
      if (st.step === 1) return renderSearch();
      if (st.step === 2) return renderConfirm();
      return renderDone();
    };

    /* ── 1/3 제품 찾기 ─────────────────────────────────────────── */
    const renderSearch = async () => {
      if (!st.options) st.options = await API.get('/api/catalog/options');
      el.innerHTML = `${stepTrack()}
        <h3>내 제품을 등록해주세요</h3>
        <div class="sub">여러 개를 한 번에 등록할 수 있어요.</div>
        <div class="row-2">
          <div class="field"><label>제품 카테고리</label>
            <select class="select" id="rf-type"><option value="">전체</option>${st.options.product_types.map(t => `<option>${t}</option>`).join('')}</select></div>
          <div class="field"><label>제조사</label>
            <select class="select" id="rf-brand"><option value="">전체</option>${st.options.brands.map(b => `<option value="${b.id}">${b.name}</option>`).join('')}</select></div>
        </div>
        <div class="field"><label>모델명 검색</label>
          <div style="display:flex;gap:8px;"><input class="input" id="rf-q" placeholder="모델명 일부를 입력 (예: FQ18, WA19)" style="flex:1"><button class="btn" id="rf-search" style="width:72px">검색</button></div></div>
        <div class="field"><label>검색 결과 <span class="muted small" id="rf-count"></span></label><div id="rf-results"><div class="empty-note">카테고리·제조사를 고르거나 모델명을 입력해 검색하세요</div></div></div>
        <div class="cart"><div class="head"><b>담긴 제품 (<span id="rf-cartn">${st.cart.length}</span>)</b><span class="muted">계속 검색해서 추가할 수 있어요</span></div><div id="rf-cart"></div></div>
        <div class="onboard-actions" style="justify-content:${opts.onCancel ? 'space-between' : 'flex-end'}">
          ${opts.onCancel ? '<button class="link-btn" id="rf-cancel">닫으면 담은 내용은 저장되지 않아요 · 취소</button>' : ''}
          <button class="btn" id="rf-next" ${st.cart.length ? '' : 'disabled'}>담은 제품 확인하러 가기 (${st.cart.length})</button></div>`;
      const search = async () => {
        const p = new URLSearchParams({ product_type: q('#rf-type').value, brand: q('#rf-brand').value, q: q('#rf-q').value.trim() });
        st.results = await API.get('/api/catalog/models?' + p);
        q('#rf-count').textContent = st.results.length ? `${st.results.length}개` : '';
        q('#rf-results').innerHTML = st.results.length ? st.results.map(m => `
          <div class="product-card">${thumbHTML(m.manual_id, m.product_type)}
            <div class="meta"><div class="name">${esc(m.brand_name)} ${esc(m.product_type)} ${esc(m.model)}</div><div class="model">${esc(m.manual_id)}</div></div>
            <button class="btn ghost sm" data-add="${m.manual_id}" ${st.cart.some(c => c.manual_id === m.manual_id) ? 'disabled' : ''}>${st.cart.some(c => c.manual_id === m.manual_id) ? '담김' : '+ 추가'}</button></div>`).join('')
          : '<div class="empty-note">검색 결과가 없어요. 모델명을 다시 확인해 주세요</div>';
      };
      const renderCart = () => {
        q('#rf-cartn').textContent = st.cart.length;
        q('#rf-next').disabled = !st.cart.length; q('#rf-next').textContent = `담은 제품 확인하러 가기 (${st.cart.length})`;
        q('#rf-cart').innerHTML = st.cart.map(c => `<div class="added-item-row">${thumbHTML(c.manual_id, c.product_type, 'thumb-xs')}
          <div class="info">${esc(c.product_type)} · ${esc(c.model)} <span class="muted">(${esc(c.brand_name)})</span></div><button class="remove-btn" data-rm="${c.manual_id}">✕</button></div>`).join('');
      };
      renderCart();
      q('#rf-search').onclick = search; q('#rf-q').onkeydown = (e) => { if (e.key === 'Enter') { e.preventDefault(); search(); } };
      q('#rf-type').onchange = search; q('#rf-brand').onchange = search;
      el.onclick = (e) => {
        const add = e.target.closest('[data-add]'), rm = e.target.closest('[data-rm]');
        if (add) { const m = st.results.find(x => x.manual_id === add.dataset.add); if (m) st.cart.push({ ...m, nickname: '', location: '' }); renderCart(); search(); }
        if (rm) { st.cart = st.cart.filter(c => c.manual_id !== rm.dataset.rm); renderCart(); search(); }
      };
      q('#rf-next').onclick = () => { st.step = 2; render(); };
      if (opts.onCancel) q('#rf-cancel').onclick = opts.onCancel;
    };

    /* ── 2/3 정보 확인 ─────────────────────────────────────────── */
    const renderConfirm = () => {
      el.innerHTML = `${stepTrack()}
        <h3>이 제품들이 맞는지 확인해주세요</h3>
        <div class="sub">담은 제품마다 구분할 별칭을 정해주세요.</div>
        ${st.cart.map((c, i) => `<div class="confirm-card">
          <div class="product-card" style="border:none;padding:0;margin-bottom:14px;">${thumbHTML(c.manual_id, c.product_type, 'product-thumb lg')}
            <div class="meta"><div class="name">${esc(c.model)}</div><div class="model">${esc(c.manual_id)}</div></div><span class="tag">${esc(c.product_type)} · ${esc(c.brand_name)}</span></div>
          <div class="row-2">
            <div class="field" style="margin:0"><label>제품 별칭</label><input class="input" data-nick="${i}" value="${esc(c.nickname)}" placeholder="예: 거실 에어컨"></div>
            <div class="field" style="margin:0"><label>설치 위치 (선택)</label><select class="select" data-loc="${i}">${LOCATIONS.map(l => `<option value="${l}" ${l === c.location ? 'selected' : ''}>${l || '선택 안 함'}</option>`).join('')}</select></div>
          </div></div>`).join('')}
        <div class="onboard-actions"><button class="link-btn" id="rf-back">← 이전으로 (제품 더 찾기)</button><button class="btn" id="rf-submit">확인하고 등록하기</button></div>
        <div class="error-text" id="rf-err"></div>`;
      el.oninput = (e) => { const n = e.target.dataset.nick, l = e.target.dataset.loc; if (n !== undefined) st.cart[n].nickname = e.target.value; if (l !== undefined) st.cart[l].location = e.target.value; };
      el.onclick = null;
      q('#rf-back').onclick = () => { st.step = 1; render(); };
      q('#rf-submit').onclick = async () => {
        q('#rf-submit').disabled = true;
        try {
          st.created = await API.post('/api/appliances', st.cart.map(c => ({ manual_id: c.manual_id, nickname: c.nickname, location: c.location })));
          st.step = 3; render();
        } catch (ex) { q('#rf-err').textContent = ex.message; q('#rf-submit').disabled = false; }
      };
    };

    /* ── 3/3 완료 ──────────────────────────────────────────────── */
    const renderDone = () => {
      el.innerHTML = `${stepTrack()}
        <div style="text-align:center;padding:10px 0 26px;"><div style="width:56px;height:56px;border-radius:50%;background:#eefaf2;color:var(--ok);margin:0 auto 16px;display:flex;align-items:center;justify-content:center;font-size:24px;">✓</div>
          <h3 style="margin-bottom:6px">등록이 완료됐어요</h3><div class="sub" style="margin:0">등록한 제품으로 바로 질문할 수 있어요</div></div>
        <div class="field"><label>등록된 제품 (${st.created.length})</label>
          ${st.created.map(a => `<div class="product-card">${thumbHTML(a.manual_id, a.product_type)}<div class="meta"><div class="name">${esc(a.nickname)}</div><div class="model">${esc(a.model)} · ${esc(a.brand_name)}${a.location ? ' · ' + esc(a.location) : ''}</div></div><span class="tag">${esc(a.product_type)}</span></div>`).join('')}</div>
        <div class="onboard-actions"><button class="btn outline" id="rf-more">제품 더 등록하기</button><button class="btn" id="rf-done">${opts.doneLabel || '질문하러 가기'}</button></div>`;
      el.oninput = null; el.onclick = null;
      q('#rf-more').onclick = () => { st.step = 1; st.cart = []; render(); };
      q('#rf-done').onclick = () => opts.onDone && opts.onDone(st.created);
    };

    const q = (sel) => el.querySelector(sel);
    render();
  },
};
