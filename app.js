/* No runtime dependencies. README.md remains the source of the bibliography. */
(() => {
  'use strict';
  const normalize = value => String(value).normalize('NFKD').replace(/[\u0300-\u036f]/g, '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
  const escapeHTML = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const linkHasResource = (link, kind) => link.label.toLowerCase().split('&').some(part => part.trim() === kind);
  const hasResource = (paper, kind) => paper.links.some(link => linkHasResource(link, kind));
  const defaults = { q: '', track: 'all', year: 'all', resource: 'all', sort: 'newest', collection: 'all' };
  function parseState(params, data) {
    const result = { ...defaults };
    result.q = (params.get('q') || '').slice(0, 240);
    const track = params.get('track');
    if (data.tracks.some(t => t.id === track)) result.track = track;
    const year = params.get('year');
    if (data.papers.some(p => String(p.year) === year)) result.year = year;
    if (['code', 'data'].includes(params.get('resource'))) result.resource = params.get('resource');
    if (['oldest', 'title'].includes(params.get('sort'))) result.sort = params.get('sort');
    if (params.get('view') === 'saved') result.collection = 'saved';
    return result;
  }
  function selectPapers(data, state, saved = new Set()) {
    const tokens = normalize(state.q).split(' ').filter(Boolean);
    return data.papers.filter(p => {
      if (state.collection === 'saved' && !saved.has(p.id)) return false;
      if (state.track !== 'all' && !p.tracks.includes(state.track)) return false;
      if (state.year !== 'all' && String(p.year) !== state.year) return false;
      if (state.resource !== 'all' && !hasResource(p, state.resource)) return false;
      const haystack = normalize([p.title, ...p.aliases, p.venue, ...p.tracks.map(id => data.tracks.find(t => t.id === id)?.name || '')].join(' '));
      return tokens.every(token => haystack.includes(token));
    }).sort((a, b) => (state.sort === 'title' ? 0 : state.sort === 'oldest' ? a.year - b.year : b.year - a.year) || a.title.localeCompare(b.title));
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = { normalize, escapeHTML, hasResource, parseState, selectPapers, defaults };
  if (typeof document === 'undefined') return;

  const data = window.ATLAS_DATA;
  const $ = selector => document.querySelector(selector);
  const $$ = selector => [...document.querySelectorAll(selector)];
  const icon = (name, cls = '') => `<svg class="icon ${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
  const trackById = Object.fromEntries(data.tracks.map(t => [t.id, t]));
  const paperIds = new Set(data.papers.map(p => p.id));
  const countFor = id => data.papers.filter(p => p.tracks.includes(id)).length;
  const readLocal = (key, fallback) => { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } };
  const writeLocal = (key, value) => { try { localStorage.setItem(key, JSON.stringify(value)); return true; } catch { return false; } };
  const storedSaved = readLocal('adverse-weather-reading-list', []);
  let saved = new Set((Array.isArray(storedSaved) ? storedSaved : []).filter(id => paperIds.has(id)));
  let state = parseState(new URLSearchParams(location.search), data);
  let visible = 8;
  let toastTimer;
  function toast(message) { $('#toast').textContent = message; $('#toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 3800); }
  function syncUrl(mode = 'replace') {
    const url = new URL(location.href);
    for (const key of ['q', 'track', 'year', 'resource', 'sort', 'view']) url.searchParams.delete(key);
    for (const key of ['q', 'track', 'year', 'resource', 'sort']) if (state[key] !== defaults[key]) url.searchParams.set(key, state[key]);
    if (state.collection === 'saved') url.searchParams.set('view', 'saved');
    if (url.href !== location.href) history[mode === 'push' ? 'pushState' : 'replaceState']({}, '', url);
  }
  function goToLibrary() { $('#library').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start' }); }
  function update(patch, { push = true, scroll = false } = {}) {
    state = { ...state, ...patch };
    visible = 8;
    syncUrl(push ? 'push' : 'replace');
    render();
    if (scroll) goToLibrary();
  }
  function reset() { update({ ...defaults, collection: state.collection }); }
  function titleMarkup(title) {
    const query = state.q.trim();
    const index = query ? title.toLowerCase().indexOf(query.toLowerCase()) : -1;
    if (index < 0) return escapeHTML(title);
    return escapeHTML(title.slice(0, index)) + '<mark>' + escapeHTML(title.slice(index, index + query.length)) + '</mark>' + escapeHTML(title.slice(index + query.length));
  }
  function paperCard(paper, index) {
    const tags = paper.tracks.map(id => trackById[id]);
    const primary = paper.links.find(l => l.label === 'paper') || paper.links[0];
    const safeLinks = paper.links.filter(l => /^https?:\/\//i.test(l.url));
    const links = safeLinks.map(l => {
      const label = l.label === 'paper' ? 'Paper' : l.label === 'code&data' ? 'Code & data' : l.label === 'data' ? 'Data' : l.label === 'code' ? 'Code' : l.label;
      return `<a href="${escapeHTML(l.url)}" target="_blank" rel="noopener noreferrer">${icon(linkHasResource(l, 'code') ? 'code' : linkHasResource(l, 'data') ? 'database' : 'external', 'small')}${escapeHTML(label)}<span class="sr-only">: ${escapeHTML(paper.title)} (opens in a new tab)</span></a>`;
    }).join('');
    return `<article class="paper-card" data-paper="${paper.id}"><span class="paper-number" aria-hidden="true">${String(index + 1).padStart(2, '0')}</span><div class="paper-content"><div class="paper-heading">${tags.map(t => `<span class="tag ${t.color}">${escapeHTML(t.short)}</span>`).join('')}<span class="paper-year">${paper.year}</span></div><h3><a href="${escapeHTML(primary.url)}" target="_blank" rel="noopener noreferrer">${titleMarkup(paper.title)}<span class="sr-only"> (opens in a new tab)</span></a></h3><div class="paper-meta"><span class="paper-venue">${escapeHTML(paper.venue)}</span><div class="paper-links">${links}</div></div></div><button class="save-paper" data-save="${paper.id}" aria-pressed="${saved.has(paper.id)}" aria-label="${saved.has(paper.id) ? 'Remove from' : 'Add to'} reading list: ${escapeHTML(paper.title)}" title="${saved.has(paper.id) ? 'Remove from reading list' : 'Save to reading list'}">${icon('bookmark')}</button></article>`;
  }
  function render() {
    const results = selectPapers(data, state, saved);
    const filtered = state.q || state.track !== 'all' || state.year !== 'all' || state.resource !== 'all';
    $('#paper-search').value = state.q;
    $('#year-filter').value = state.year;
    $('#resource-filter').value = state.resource;
    $('#sort-filter').value = state.sort;
    $('#mobile-track-filter').value = state.track;
    $('#clear-search').hidden = !state.q;
    $('.search-box kbd').hidden = !!state.q;
    $$('#track-filters [data-track]').forEach(el => el.setAttribute('aria-pressed', String(el.dataset.track === state.track)));
    $$('[data-collection]').forEach(el => el.setAttribute('aria-pressed', String(el.dataset.collection === state.collection)));
    $('#saved-count').textContent = saved.size;
    $('#saved-note').hidden = state.collection !== 'saved';
    $('#export-list').hidden = state.collection !== 'saved' || saved.size === 0;
    $('#reset-filters').hidden = !filtered;
    $('#share-filters').innerHTML = state.collection === 'saved' ? icon('download') + 'Export reading list' : icon('share') + 'Share this view';
    $('#share-filters').setAttribute('aria-label', state.collection === 'saved' ? 'Export reading list' : 'Share this view');
    $('#results-status').innerHTML = `<strong>${results.length}</strong> ${results.length === 1 ? 'resource' : 'resources'}${state.track !== 'all' ? ' in ' + escapeHTML(trackById[state.track].name) : state.collection === 'saved' ? ' in your reading list' : ' to explore'}`;
    const chips = [];
    if (state.q) chips.push(['q', 'Search: ' + state.q]);
    if (state.track !== 'all') chips.push(['track', trackById[state.track].name]);
    if (state.year !== 'all') chips.push(['year', state.year]);
    if (state.resource !== 'all') chips.push(['resource', 'With ' + state.resource]);
    $('#active-filters').hidden = chips.length === 0;
    $('#active-filters').innerHTML = chips.map(([key, label]) => `<button class="filter-chip" data-remove-filter="${key}" aria-label="Remove filter: ${escapeHTML(label)}"><span>${escapeHTML(label)}</span>${icon('close')}</button>`).join('');
    $('#paper-list').innerHTML = results.slice(0, visible).map(paperCard).join('');
    $('#empty-state').hidden = results.length > 0;
    const emptySaved = state.collection === 'saved' && saved.size === 0;
    $('#empty-title').textContent = emptySaved ? 'A little space for your next big idea.' : 'No matches yet.';
    $('#empty-description').textContent = emptySaved ? 'Tap the bookmark beside any paper to keep it here.' : 'Try a broader search or clear a filter to see more papers.';
    $('#empty-action').textContent = emptySaved ? 'Explore the library' : 'Clear filters';
    $('#load-more').hidden = visible >= results.length;
    $('#visible-count').textContent = results.length ? `Showing ${Math.min(visible, results.length)} of ${results.length} resources` : '';
  }
  function exportList() {
    const papers = data.papers.filter(p => saved.has(p.id));
    if (!papers.length) { toast('Save a paper first to build your reading list.'); return; }
    const text = ['# Adverse Weather · Reading List', '', `Source: ${data.source}`, `Bibliography snapshot: ${data.updated}`, '', ...papers.flatMap(p => [`## ${p.title}`, p.venue, `Tracks: ${p.tracks.map(id => trackById[id].name).join(', ')}`, ...p.links.map(l => `- [${l.label}](${l.url})`), ''])].join('\n');
    const url = URL.createObjectURL(new Blob([text], { type: 'text/markdown;charset=utf-8' }));
    const link = document.createElement('a'); link.href = url; link.download = 'adverse-weather-reading-list.md'; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 20000);
    toast(`Exported ${papers.length} ${papers.length === 1 ? 'resource' : 'resources'}.`);
  }
  $('#stat-papers').textContent = data.papers.length;
  $('#stat-code').textContent = data.papers.filter(p => hasResource(p, 'code') || hasResource(p, 'data')).length;
  $('#all-count').textContent = data.papers.length;
  const dateFormat = value => { const [y, m, d] = value.split('.').map(Number); return new Date(y, m - 1, d).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }); };
  $('#source-update').textContent = 'Source updated ' + dateFormat(data.updated);
  $('#track-filters').innerHTML = `<button class="track-filter" data-track="all" aria-pressed="true">${icon('grid')}<span>All research</span><span class="track-count">${data.papers.length}</span></button>` + data.tracks.map(t => `<button class="track-filter" data-track="${t.id}" aria-pressed="false">${icon(t.icon)}<span>${escapeHTML(t.name)}</span><span class="track-count">${countFor(t.id)}</span></button>`).join('');
  const years = [...new Set(data.papers.map(p => p.year))].sort((a, b) => b - a);
  $('#year-filter').insertAdjacentHTML('beforeend', years.map(y => `<option value="${y}">${y}</option>`).join(''));
  $('#mobile-track-filter').insertAdjacentHTML('beforeend', data.tracks.map(t => `<option value="${t.id}">${escapeHTML(t.name)}</option>`).join(''));
  $('#research-grid').innerHTML = data.tracks.map((t, i) => `<button class="research-card" data-track="${t.id}" aria-label="Explore ${escapeHTML(t.name)}: ${countFor(t.id)} resources"><span class="research-card-top">${icon(t.icon)}<span>${String(i + 1).padStart(2, '0')}</span></span><h3>${escapeHTML(t.name)}</h3><p>${escapeHTML(t.description)}</p><span class="research-card-bottom">${countFor(t.id)} resources ${icon('arrow')}</span></button>`).join('');
  const updateEntries = [...data.updates].reverse();
  $('#update-list').innerHTML = [updateEntries[0], updateEntries[1], data.updates[0]].filter(Boolean).map(u => `<li><time datetime="${u.date.split('.').map((v,i) => i ? v.padStart(2,'0') : v).join('-')}">${dateFormat(u.date)}</time><p>${escapeHTML(u.text)}</p></li>`).join('');
  const sensors = {
    lidar: { title: 'Precise geometry. A changing signal.', text: 'LiDAR captures detailed 3D structure. Fog, rain, and snow introduce backscatter, attenuation, and clutter, making restoration and weather-aware detection important research directions.', track: 'lidar', link: 'Explore LiDAR & camera detection' },
    camera: { title: 'Rich semantics. Limited visibility.', text: 'Cameras capture texture and visual context. Fog, glare, darkness, and wet lenses reduce contrast and visibility. Multimodal methods combine these visual cues with complementary measurements.', track: 'lidar', link: 'Explore multimodal detection' },
    radar: { title: 'Range and motion, through the weather.', text: 'Radar provides weather-resilient ranging and Doppler cues. Sparsity, noise, clutter, and elevation uncertainty still matter. 4D radar and camera fusion explore the complementary strengths of both sensors.', track: 'radar', link: 'Explore 4D radar & camera' },
    v2x: { title: 'A wider view, shared between agents.', text: 'Nearby vehicles and infrastructure bring spatial diversity to limited-visibility scenes. Cooperative perception must also account for calibration, latency, bandwidth, and the reliability of shared observations.', track: 'icp', link: 'Explore cooperative perception' }
  };
  function showSensor(key) {
    const s = sensors[key];
    $$('[data-sensor]').forEach(el => { const selected = el.dataset.sensor === key; el.setAttribute('aria-selected', String(selected)); el.tabIndex = selected ? 0 : -1; });
    $('#sensor-content').setAttribute('aria-labelledby', 'sensor-' + key);
    $('#sensor-content').innerHTML = `<h3>${s.title}</h3><p>${s.text}</p><button data-track="${s.track}" class="text-link">${s.link} ${icon('arrow', 'small')}</button>`;
  }
  document.addEventListener('click', event => {
    const track = event.target.closest('[data-track]');
    if (track) { update({ track: track.dataset.track, collection: 'all' }, { scroll: !track.closest('#track-filters') }); return; }
    const collection = event.target.closest('[data-collection]');
    if (collection) { update({ collection: collection.dataset.collection }); return; }
    const remove = event.target.closest('[data-remove-filter]');
    if (remove) { update({ [remove.dataset.removeFilter]: defaults[remove.dataset.removeFilter] }); $('#paper-search').focus({ preventScroll: true }); return; }
    const save = event.target.closest('[data-save]');
    if (save) {
      const id = save.dataset.save;
      if (saved.has(id)) saved.delete(id); else saved.add(id);
      const persisted = writeLocal('adverse-weather-reading-list', [...saved]);
      const isSaved = saved.has(id);
      render();
      const newButton = $(`[data-save="${id}"]`);
      (newButton || $('[data-collection="saved"]')).focus({ preventScroll: true });
      toast(persisted ? isSaved ? 'Added to your reading list.' : 'Removed from your reading list.' : 'Saved for this session. Export your list to keep a copy.');
      return;
    }
    const sensor = event.target.closest('[data-sensor]');
    if (sensor) showSensor(sensor.dataset.sensor);
  });
  $('#paper-search').maxLength = 240;
  $('#paper-search').addEventListener('input', event => update({ q: event.target.value }, { push: false }));
  $('#clear-search').addEventListener('click', () => { update({ q: '' }, { push: false }); $('#paper-search').focus(); });
  $('#year-filter').addEventListener('change', event => update({ year: event.target.value }));
  $('#resource-filter').addEventListener('change', event => update({ resource: event.target.value }));
  $('#sort-filter').addEventListener('change', event => update({ sort: event.target.value }));
  $('#mobile-track-filter').addEventListener('change', event => update({ track: event.target.value }));
  $('#reset-filters').addEventListener('click', reset);
  $('#empty-action').addEventListener('click', () => { if (state.collection === 'saved' && !saved.size) update({ ...defaults }); else reset(); });
  $('#export-list').addEventListener('click', exportList);
  $('#load-more').addEventListener('click', () => { const previous = visible; visible += 8; render(); const item = $$('.paper-card')[previous]; if (item) item.querySelector('h3 a').focus({ preventScroll: true }); });
  $('#share-filters').addEventListener('click', () => {
    if (state.collection === 'saved') { exportList(); return; }
    const url = new URL(location.href); url.hash = 'library';
    $('#share-url').value = url.href; $('#share-dialog').showModal(); $('#share-url').select();
  });
  $('#close-share').addEventListener('click', () => $('#share-dialog').close());
  $('#share-dialog').addEventListener('click', event => { if (event.target === $('#share-dialog')) { const rect = event.target.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) event.target.close(); } });
  $('#copy-share').addEventListener('click', async () => { try { await navigator.clipboard.writeText($('#share-url').value); $('#share-dialog').close(); toast('Link copied. Your filters are included.'); } catch { $('#share-url').focus(); $('#share-url').select(); toast('Select and copy the link above.'); } });
  $('.sensor-tabs').addEventListener('keydown', event => {
    const keys = Object.keys(sensors); const index = keys.indexOf(event.target.dataset.sensor);
    if (index < 0 || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? 3 : (index + (event.key === 'ArrowRight' ? 1 : 3)) % 4;
    showSensor(keys[next]); $(`[data-sensor="${keys[next]}"]`).focus();
  });
  function applyTheme(theme) { document.documentElement.dataset.theme = theme; $('.theme-button').setAttribute('aria-label', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'); $('.theme-button').innerHTML = icon(theme === 'dark' ? 'sun' : 'moon'); }
  const storedTheme = readLocal('adverse-weather-theme', 'light'); applyTheme(storedTheme === 'dark' ? 'dark' : 'light');
  $('.theme-button').addEventListener('click', () => { const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'; applyTheme(theme); writeLocal('adverse-weather-theme', theme); });
  function closeMenu() { $('#mobile-nav').hidden = true; $('.menu-button').setAttribute('aria-expanded', 'false'); $('.menu-button').setAttribute('aria-label', 'Open navigation'); }
  $('.menu-button').addEventListener('click', () => { const open = $('#mobile-nav').hidden; $('#mobile-nav').hidden = !open; $('.menu-button').setAttribute('aria-expanded', String(open)); $('.menu-button').setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation'); });
  $$('#mobile-nav a').forEach(el => el.addEventListener('click', closeMenu));
  document.addEventListener('keydown', event => {
    const editing = /INPUT|TEXTAREA|SELECT/.test(event.target.tagName) || event.target.isContentEditable;
    if (event.key === '/' && !editing && !event.ctrlKey && !event.metaKey && !event.altKey && !$('#share-dialog').open) { event.preventDefault(); goToLibrary(); $('#paper-search').focus({ preventScroll: true }); }
    if (event.key === 'Escape') { closeMenu(); if (event.target === $('#paper-search') && state.q) update({ q: '' }, { push: false }); }
  });
  window.addEventListener('popstate', () => { state = parseState(new URLSearchParams(location.search), data); visible = 8; render(); });
  window.addEventListener('storage', event => { if (event.key === 'adverse-weather-reading-list') { const list = readLocal(event.key, []); saved = new Set((Array.isArray(list) ? list : []).filter(id => paperIds.has(id))); render(); } });
  const sectionObserver = new IntersectionObserver(entries => { for (const entry of entries) if (entry.isIntersecting) $$('.nav-link').forEach(link => { const active = link.hash === '#' + entry.target.id; link.classList.toggle('active', active); if (active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current'); }); }, { rootMargin: '-15% 0px -60% 0px', threshold: 0 });
  ['research-map', 'field-notes', 'library'].forEach(id => sectionObserver.observe(document.getElementById(id)));
  showSensor('lidar'); render();
  const originalAnchor = location.hash.slice(1);
  if (trackById[originalAnchor]) { update({ track: originalAnchor }, { push: false, scroll: true }); }
  const redirects = { map: 'research-map', signals: 'field-notes', papers: 'library', updates: 'field-notes' };
  if (redirects[originalAnchor]) document.getElementById(redirects[originalAnchor]).scrollIntoView();
})();
