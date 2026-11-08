/* Geographic annual map UI. No synthetic predictions or map-year substitution:
 * at the chosen place the map shows the timeline's year, or nothing. A year
 * whose HR-VPP comes from another year (the API's vpp_substitutions) is
 * labelled with the API's wording wherever the year is named. */
'use strict';
const $ = id => document.getElementById(id);
// A browser may still hold the page from before the timeline: ask for a
// reload instead of failing half-initialised against the old controls.
if (!$('year-ticks')) {
  $('status').textContent = 'Sidan har uppdaterats. Ladda om sidan för att fortsätta.';
  throw new Error('lclu.html predates lclu.js');
}
const apiBase = new URL((window.IMINT_LCLU_CONFIG?.apiBase || window.location.origin).replace(/\/?$/, '/'));
const apiUrl = path => new URL(path.replace(/^\/+/, ''), apiBase).href;
const map = L.map('map', {minZoom: 4, maxZoom: 18, zoomAnimation: false, fadeAnimation: false}).setView([62.1, 16.5], 5);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 18, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
}).addTo(map);
// In the showcase the map sits in a sub-tab: redraw it when the tab is shown again.
new ResizeObserver(([entry]) => { if (entry.contentRect.width) map.invalidateSize(); }).observe($('map'));
// place: the chosen window {key, lat, lon, years}; key is the server's grid centre.
let place = null, footprint = null, generation = 0, busy = false, ready = false, inspection = 0, clicks = 0;
let latestYear = 0, candidateYears = [], areaLayers = [], areaNames = [], substitutions = new Map();
// Maps made this browser session, bottom to top. Results persist on the server,
// so a reload restores the list from sessionStorage and reuses the artifacts.
const maps = [];
let pendingYear = null, created = 0, warmTimer, createTimer;
const STORE = 'imint-lclu-session-maps-v1';
const phaseLabels = {queued:'I kö …', preparing:'Förbereder årsdata …', classifying:'Klassificerar …', exporting:'Skapar kartfiler …', complete:'Årskartan är klar.'};
// The place's other years, made one at a time in the background once the user
// has made a year there: nearest year first, the later one first at equal
// distance. A full queue or a passing outage is retried after these waits.
let fill = null, filling = false, fillTimer;
const RETRY_SECONDS = [15, 30, 60];
function status(text, error=false) { $('status').textContent = text; $('status').classList.toggle('error', error); }
async function request(url, options) {
  const response = await fetch(apiUrl(url), options);
  const fail = message => Object.assign(new Error(message), {status: response.status});
  if (!response.headers.get('content-type')?.includes('application/json')) {
    throw fail('Årskartans tjänst är inte tillgänglig just nu. Försök igen senare.');
  }
  const body = await response.json();
  if (!response.ok) throw fail(typeof body.detail === 'string' ? body.detail : `Begäran misslyckades (${response.status}).`);
  return body;
}
const timelineYear = () => Number($('year').value);
// "<year> · <the API's label>" for a substituted year, else the year.
const yearLabel = year => substitutions.has(year) ? `${year} · ${substitutions.get(year).label}` : String(year);
const available = year => Boolean(place?.years.includes(year));
const find = (key, year) => maps.find(entry => entry.key === key && entry.year === year);
const shownAt = key => maps.find(entry => entry.key === key && entry.visible);
const opacity = () => Number($('opacity').value) / 100;
function placeLabel(lat, lon) {
  const index = areaLayers.findIndex(layer => layer.getBounds().contains([lat, lon]));
  return `${index >= 0 ? areaNames[index] + ' · ' : ''}${lat.toFixed(3)}° N, ${lon.toFixed(3)}° E`;
}
function closeInspection() { ++inspection; map.closePopup(); }
function save() {
  try { sessionStorage.setItem(STORE, JSON.stringify(maps.map(({overlay, ...entry}) => entry))); } catch { /* storage off: list lives in memory only */ }
}
function addMap(entry, restored=false) {
  entry.overlay = L.imageOverlay(apiUrl(entry.result.artifacts['overlay.png']), entry.result.overlay_bounds,
    {opacity: opacity(), interactive: false, className: 'lclu-classification'});
  entry.overlay.on('error', () => restored
    ? drop(entry, 'En sparad karta finns inte längre på servern och togs bort ur listan.')
    : status('Kartlagret kunde inte visas. GeoTIFF-filen kan fortfarande hämtas.', true));
  maps.push(entry);
  if (entry.visible) entry.overlay.addTo(map);
  return entry;
}
function setVisible(entry, visible) { entry.visible = visible; if (visible) entry.overlay.addTo(map); else entry.overlay.remove(); }
// One year per place is visible; the shown map goes on top for pixel inspection.
function show(entry) {
  for (const other of maps) if (other !== entry && other.key === entry.key && other.visible) setVisible(other, false);
  setVisible(entry, true);
  maps.splice(maps.indexOf(entry), 1); maps.push(entry);
  entry.overlay.bringToFront();
  render();
}
function hidePlace(key) { for (const entry of maps) if (entry.key === key && entry.visible) setVisible(entry, false); }
function drop(entry, message) {
  const index = maps.indexOf(entry);
  if (index < 0) return;
  entry.overlay.remove(); maps.splice(index, 1); render(); status(message, true);
}
function removePlace(key) {
  closeInspection();
  for (const entry of maps.filter(item => item.key === key)) { entry.overlay.remove(); maps.splice(maps.indexOf(entry), 1); }
  render(); status('Kartorna togs bort från sessionen. De kan skapas igen.');
}
function controls() {
  const year = timelineYear(), entry = place && find(place.key, year);
  $('area').disabled = busy;
  $('year').disabled = !place || place.years.length === 0;
  $('create').disabled = !ready || !place || !available(year) || Boolean(entry?.visible) || (busy && !entry);
  $('create-label').textContent = !place ? 'Skapa årskarta'
    : entry ? (entry.visible ? `Årskarta ${year} visas` : `Visa årskarta ${year}`) : `Skapa årskarta ${year}`;
  $('progress').hidden = !busy;
}
function describeYear(year) {
  if (!place) return 'Välj plats först';
  if (!available(year)) return `${year} – ej verifierade årsdata`;
  return find(place.key, year) ? `${yearLabel(year)} · skapad` : yearLabel(year);
}
function renderTimeline() {
  const year = timelineYear(), first = candidateYears[0], span = candidateYears.at(-1) - first;
  $('year-ticks').replaceChildren(...candidateYears.map(candidate => {
    const tick = document.createElement('button');
    tick.type = 'button'; tick.tabIndex = -1; tick.className = 'year-tick'; tick.textContent = String(candidate);
    tick.style.left = `calc(var(--thumb) / 2 + (100% - var(--thumb)) * ${span ? (candidate - first) / span : 0})`;
    tick.disabled = !available(candidate);
    tick.classList.toggle('substituted', substitutions.has(candidate));
    tick.title = substitutions.has(candidate) ? yearLabel(candidate) : '';
    tick.classList.toggle('unavailable', Boolean(place) && !available(candidate));
    tick.classList.toggle('made', Boolean(place && find(place.key, candidate)));
    tick.classList.toggle('current', Boolean(place) && candidate === year);
    tick.addEventListener('click', () => { $('year').value = String(candidate); onTimeline(); onRelease(); });
    return tick;
  }));
  $('year-value').value = place ? describeYear(year) : '–';
  $('year').setAttribute('aria-valuetext', describeYear(year));
}
function renderResult() {
  const entry = place && shownAt(place.key);
  $('result').hidden = !entry;
  if (!entry) return;
  const r = entry.result;
  $('download-tif').href = apiUrl(r.artifacts['classification.tif']);
  $('download-provenance').href = apiUrl(r.artifacts['provenance.json']);
  $('result-title').textContent = `Årskarta ${yearLabel(entry.year)}`;
  $('vpp-note').textContent = substitutions.get(entry.year)?.note ?? '';
  $('vpp-note').hidden = !substitutions.has(entry.year);
  $('coverage').textContent = `${(100*r.valid_pixels/(504*504)).toFixed(1)} % giltiga pixlar · ${r.nodata_pixels.toLocaleString('sv-SE')} utan data. Stödlager kan ha luckor; se proveniens.`;
  $('timing').textContent = entry.seconds == null ? '' : `Svarstid: ${entry.seconds.toFixed(1)} s`;
}
function renderList() {
  const groups = new Map();
  for (const entry of [...maps].sort((a, b) => a.order - b.order)) {
    if (!groups.has(entry.key)) groups.set(entry.key, []);
    groups.get(entry.key).push(entry);
  }
  $('session').hidden = maps.length === 0;
  $('session-list').replaceChildren(...[...groups.values()].map(entries => {
    const first = entries[0], item = document.createElement('li'), head = document.createElement('div');
    item.className = 'session-place'; item.classList.toggle('current', first.key === place?.key);
    head.className = 'session-place-head';
    const name = document.createElement('button');
    name.type = 'button'; name.className = 'session-place-name'; name.textContent = placeLabel(first.lat, first.lon);
    name.addEventListener('click', () => focusPlace(first));
    const remove = document.createElement('button');
    remove.type = 'button'; remove.className = 'session-remove'; remove.textContent = '×';
    remove.setAttribute('aria-label', `Ta bort kartorna för ${name.textContent}`);
    remove.addEventListener('click', () => removePlace(first.key));
    head.append(name, remove);
    const years = document.createElement('div');
    years.className = 'session-years';
    for (const entry of entries.sort((a, b) => a.year - b.year)) {
      const chip = document.createElement('button');
      chip.type = 'button'; chip.className = 'year-chip'; chip.textContent = String(entry.year);
      chip.setAttribute('aria-pressed', String(entry.visible));
      chip.title = entry.visible ? 'Dölj kartan' : 'Visa kartan';
      chip.addEventListener('click', () => toggle(entry));
      years.append(chip);
    }
    item.append(head, years);
    return item;
  }));
}
function renderFill() {
  const text = !fill || fill.key !== place?.key ? ''
    : fill.years.length ? `Övriga år: ${fill.done} av ${fill.total} klara …`
    : fill.failed ? `Övriga år klara · ${fill.failed} kunde inte skapas.` : 'Alla år för platsen är klara.';
  $('fill-status').textContent = text; $('fill-status').hidden = !text;
}
function render() { renderTimeline(); renderResult(); renderList(); renderFill(); controls(); save(); }
// Start preparing the chosen place and year while the user decides. Best
// effort: creating the map reports any real error, so a failed warm-up is silent.
function warm(year) {
  if (!place || !available(year) || find(place.key, year)) return;
  request('/v1/lclu/prepare', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({lat:place.lat,lon:place.lon,year})}).catch(() => {});
}
// Warm remote TESSERA for a box (a municipality or the view). Also best
// effort: staged areas have nothing to fetch, and the server bounds the box.
function prefetch(bounds, year) {
  if (!year) return;
  const box = {west:bounds.getWest(), south:bounds.getSouth(), east:bounds.getEast(), north:bounds.getNorth()};
  request('/v1/lclu/prefetch', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...box,year})}).catch(() => {});
}
let prefetchTimer;
map.on('moveend', () => {
  clearTimeout(prefetchTimer);
  if (map.getZoom() < 12) return;
  prefetchTimer = setTimeout(() => prefetch(map.getBounds(), available(timelineYear()) ? timelineYear() : latestYear), 600);
});
// Resolves true when this call's selection became the current place.
async function select(lat, lon, {pan=true, year}={}) {
  if (busy) return false;
  clearTimeout(createTimer); clearTimeout(warmTimer);
  const current = ++generation, previous = place ? timelineYear() : null;
  closeInspection(); place = null; render();
  $('lat').value = lat.toFixed(5); $('lon').value = lon.toFixed(5);
  $('location-title').textContent = `${lat.toFixed(4)}° N, ${lon.toFixed(4)}° E`;
  status('Kontrollerar tillgängliga år …');
  try {
    const data = await request(`/v1/lclu/availability?lon=${lon}&lat=${lat}`);
    if (current !== generation) return false;
    if (footprint) map.removeLayer(footprint);
    footprint = L.geoJSON(data.geometry.footprint, {interactive:false,style:{color:'#1a4338',weight:2,fillOpacity:.06}}).addTo(map);
    // Keep the user's detail zoom when choosing another nearby point.
    if (pan && map.getZoom() < 12) map.fitBounds(footprint.getBounds(), {maxZoom:12,padding:[25,25]});
    else if (pan) map.panTo([lat,lon]);
    place = {key: data.geometry.center.join(','), lat, lon, years: data.years};
    if (fill && fill.key !== place.key) stopFill();
    if (!place.years.length) { status(data.message); return true; }
    // Keep the year being compared across places when this place has it.
    const pick = year ?? shownAt(place.key)?.year ?? (available(previous) ? previous : Math.max(...place.years));
    $('year').value = String(pick);
    const entry = find(place.key, pick);
    if (entry && year !== undefined) show(entry);
    if (entry) status(`Årskarta ${yearLabel(pick)}. Dra i tidsaxeln för att byta år.`);
    else { status('Välj år på tidsaxeln och skapa din karta. Pixlarnas datatäckning kontrolleras vid körning.'); warm(pick); }
    return true;
  } catch (error) { if (current === generation) status(error.message,true); return false; }
  finally { if (current === generation) render(); }
}
function focusPlace(entry) {
  map.fitBounds(entry.overlay.getBounds(), {maxZoom:13,padding:[25,25]});
  if (entry.key !== place?.key) select(entry.lat, entry.lon, {pan:false, year: shownAt(entry.key)?.year});
}
function toggle(entry) {
  closeInspection();
  if (entry.visible) { setVisible(entry, false); render(); return; }
  if (entry.key === place?.key) $('year').value = String(entry.year);
  show(entry);
  if (entry.key !== place?.key) select(entry.lat, entry.lon, {pan:false, year: entry.year});
}
// Dragging swaps to a map made this session at once; any other year hides
// the place's map, so no year stands in for another, and warms that year.
function onTimeline() {
  if (!place) return;
  const year = timelineYear(), entry = find(place.key, year);
  closeInspection();
  if (entry) { show(entry); if (!busy) status(`Årskarta ${yearLabel(year)}.`); return; }
  hidePlace(place.key); render();
  if (!available(year)) { status(`${year} – ej verifierade årsdata för denna plats.`); return; }
  if (!busy) status(`${yearLabel(year)}: ingen karta än. Släpp tidsaxeln för att skapa den.`);
  const key = place.key;
  clearTimeout(warmTimer);
  warmTimer = setTimeout(() => { if (place?.key === key && timelineYear() === year) warm(year); }, 350);
}
// A release (or the last of several arrow-key steps) creates the year.
function onRelease() {
  clearTimeout(createTimer);
  if (!place) return;
  const key = place.key, year = timelineYear();
  createTimer = setTimeout(() => {
    if (place?.key === key && timelineYear() === year && available(year) && !find(key, year)) create(year);
  }, 450);
}
// Submit one year at one place and wait for its map; adds it to the session hidden.
async function makeMap(target, year, onPhase) {
  const started = performance.now();
  const submission = await request('/v1/lclu', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({lat:target.lat,lon:target.lon,year})});
  let job;
  do {
    job = await request(submission.poll_url);
    onPhase(job);
    if (!['success','failed'].includes(job.status)) await new Promise(resolve=>setTimeout(resolve,900));
  } while (!['success','failed'].includes(job.status));
  if (job.status === 'failed') throw Object.assign(new Error(job.error || 'Årskartan kunde inte skapas.'), {jobFailed: true});
  const r = job.result;
  const entry = addMap({id: submission.job_id, key: target.key, lat: target.lat, lon: target.lon, year, visible: false,
    order: ++created, seconds: (performance.now()-started)/1000,
    result: {artifacts: r.artifacts, overlay_bounds: r.overlay_bounds, valid_pixels: r.valid_pixels, nodata_pixels: r.nodata_pixels}});
  return {entry, cacheHit: r.cache_hit};
}
async function create(year) {
  if (!place || !available(year)) return;
  const existing = find(place.key, year);
  if (existing) return show(existing);
  // One job at a time per visitor: the user's year goes before the next background year.
  if (busy || filling) { pendingYear = {key: place.key, year}; if (!busy) status(`${yearLabel(year)} står på tur …`); return; }
  const target = place;
  let made = false;
  busy = true; controls(); status(`${yearLabel(year)}: skickar begäran …`);
  try {
    const {entry, cacheHit} = await makeMap(target, year, job => {
      const queued = pendingYear && pendingYear.year !== year ? ` · ${pendingYear.year} står på tur` : '';
      status(`${yearLabel(year)}: ${phaseLabels[job.phase] || phaseLabels[job.status] || 'Bearbetar …'}${queued}`);
    });
    made = true;
    if (place?.key === target.key && timelineYear() === year) {
      show(entry); status(cacheHit ? `Årskarta ${yearLabel(year)} är klar · tidigare verifierat resultat.` : `Årskarta ${yearLabel(year)} är klar.`);
    } else { render(); status(`Årskarta ${yearLabel(year)} är klar och finns under Dina kartor.`); }
  } catch(error) { status(`${yearLabel(year)}: ${error.message}`,true); }
  finally {
    busy = false; controls();
    if (made && place?.key === target.key && fill?.key !== target.key) startFill(target, year);
    afterJob();
  }
}
function afterJob() {
  const next = pendingYear; pendingYear = null;
  if (next && next.key === place?.key && next.year === timelineYear() && !find(next.key, next.year)) return create(next.year);
  runFill();
}
function startFill(target, year) {
  const years = target.years.filter(other => other !== year && !find(target.key, other))
    .sort((a, b) => Math.abs(a - year) - Math.abs(b - year) || b - a);
  fill = years.length ? {key: target.key, target, years, total: years.length, done: 0, failed: 0, attempt: 0} : null;
  renderFill();
}
function stopFill() { fill = null; clearTimeout(fillTimer); renderFill(); }
function runFill() {
  if (!fill || busy || filling) return;
  if (fill.key !== place?.key) return stopFill();
  while (fill.years.length && find(fill.key, fill.years[0])) { fill.years.shift(); fill.done++; }
  renderFill();
  if (fill.years.length) fillOne(fill, fill.years[0]);
}
async function fillOne(job, year) {
  filling = true; controls();
  let wait = 0;
  try {
    const {entry} = await makeMap(job.target, year, () => {});
    job.done++;
    if (place?.key === job.key && timelineYear() === year) { show(entry); status(`Årskarta ${yearLabel(year)} är klar.`); }
    else render();
  } catch (error) {
    // The service refused or was unreachable: try this year again later, a few times.
    const passing = !error.jobFailed && (error.status === undefined || error.status === 429 || error.status >= 500);
    if (passing && job.attempt < RETRY_SECONDS.length) wait = RETRY_SECONDS[job.attempt++] * 1000;
    else job.failed++;
  }
  if (!wait) { job.years.shift(); job.attempt = 0; }
  filling = false; controls(); renderFill();
  if (wait) { fillTimer = setTimeout(() => { if (fill === job) afterJob(); }, wait); return; }
  afterJob();
}
function topmostAt(latlng) {
  for (let i = maps.length - 1; i >= 0; i--) if (maps[i].visible && maps[i].overlay.getBounds().contains(latlng)) return maps[i];
  return null;
}
async function mapClick(latlng) {
  const click = ++clicks, entry = topmostAt(latlng);
  if (!entry) return select(latlng.lat,latlng.lng);
  if (entry.key !== place?.key && !busy && !(await select(entry.lat, entry.lon, {pan:false, year: entry.year}))) return;
  if (click !== clicks) return;
  const current=++inspection;
  const message=document.createElement('div'); message.textContent='Läser markklass …';
  const popup=L.popup().setLatLng(latlng).setContent(message).openOn(map);
  try {
    const pixel=await request(`/v1/lclu/jobs/${entry.id}/pixel?lon=${latlng.lng}&lat=${latlng.lat}`);
    if(current!==inspection || !entry.visible) return;
    if(!pixel.inside) { map.closePopup(); return select(latlng.lat,latlng.lng); }
    const content=document.createElement('div'); content.className='pixel-info';
    const title=document.createElement('div'); title.className='pixel-info-year'; title.textContent=`Markklass · ${pixel.year}`;
    const row=document.createElement('div'); row.className='pixel-info-class';
    const swatch=document.createElement('span'); swatch.className='swatch';
    if(pixel.color) swatch.style.backgroundColor=pixel.color; else swatch.classList.add('nodata');
    const name=document.createElement('strong'); name.textContent=pixel.name;
    row.append(swatch,name); content.append(title,row);
    if(pixel.note) { const note=document.createElement('p'); note.className='pixel-info-note'; note.textContent=pixel.note; content.append(note); }
    popup.setContent(content);
  } catch(error) { if(current===inspection) message.textContent='Markklassen kunde inte läsas. Försök igen.'; }
}
function restore() {
  let saved;
  try { saved = JSON.parse(sessionStorage.getItem(STORE) || '[]'); } catch { saved = []; }
  if (!Array.isArray(saved)) return;
  // Only this service's own artifact paths; anything else in storage is ignored.
  const own = path => typeof path === 'string' && path.startsWith('/v1/lclu/jobs/');
  for (const entry of saved) {
    const artifacts = entry?.result?.artifacts;
    if (!artifacts || !['overlay.png','classification.tif','provenance.json'].every(name => own(artifacts[name]))
        || !Array.isArray(entry.result.overlay_bounds) || find(entry.key, entry.year)) continue;
    addMap(entry, true);
    created = Math.max(created, entry.order || 0);
  }
}
map.on('click', event => mapClick(event.latlng));
$('location-form').addEventListener('submit', event => { event.preventDefault(); select(Number($('lat').value),Number($('lon').value)); });
$('year').addEventListener('input', onTimeline);
$('year').addEventListener('change', onRelease);
$('create').addEventListener('click', () => create(timelineYear()));
$('opacity').addEventListener('input', () => { $('opacity-value').value=`${$('opacity').value} %`; for (const entry of maps) entry.overlay.setOpacity(opacity()); });
restore(); render();
request('/v1/lclu').then(info=>{
  ready=info.ready; status(info.message); $('historical-note').textContent=info.historical_note;
  candidateYears=info.candidate_years;
  // An API without the list substitutes no year.
  substitutions=new Map((info.vpp_substitutions ?? []).map(item=>[item.year,item]));
  $('year').min=String(candidateYears[0]); $('year').max=String(candidateYears.at(-1));
  if(info.areas?.length){
    const areas=L.geoJSON(info.areas,{interactive:false,style:{color:'#245045',weight:2,dashArray:'5 5',fillOpacity:.02}}).addTo(map);
    areaLayers=areas.getLayers(); areaNames=info.areas.map(area=>area.properties.name);
    map.fitBounds(areaLayers[0].getBounds(),{padding:[25,25]});
    $('area-title').textContent=(info.areas.length > 2 ? `${info.areas.length} demoområden` : areaNames.join(' / '))+' / DEMO';
    $('map-caption').textContent='Välj en punkt inom ett streckat demoområde. Årskartan är 5,04 × 5,04 km och kan sträcka sig utanför området.';
    info.areas.forEach((area,index)=>$('area').add(new Option(area.properties.name,String(index))));
    $('area-control').hidden=info.areas.length<2;
    const offered=info.areas.flatMap(area=>area.properties.years||[]), firstYear=Math.min(...offered);
    latestYear=Math.max(0,...offered);
    if(offered.length) $('year-span').textContent=firstYear===latestYear ? String(firstYear) : `${firstYear}\u2060–\u2060${latestYear}`;
    $('year').value=String(latestYear || candidateYears.at(-1));
    $('area').addEventListener('change',()=>{
      const index=Number($('area').value), point=info.locations[index];
      map.fitBounds(areaLayers[index].getBounds(),{padding:[25,25],maxZoom:13});
      prefetch(areaLayers[index].getBounds(), Math.max(0,...(info.areas[index].properties.years||[])));
      if(point) select(point.lat,point.lon);
    });
    if(info.locations.length){ $('lat').value=info.locations[0].lat.toFixed(5); $('lon').value=info.locations[0].lon.toFixed(5); }
  }
  for(const item of info.legend){
    const row=document.createElement('div'); row.className='legend-item';
    if(item.note) row.title=item.note;
    const swatch=document.createElement('span'); swatch.className='swatch';
    if(item.color) swatch.style.backgroundColor=item.color; else swatch.classList.add('nodata');
    row.append(swatch,document.createTextNode(item.name)); $('legend').append(row);
    if(item.note) { const detail=document.createElement('li'); const name=document.createElement('strong'); name.textContent=item.name+': '; detail.append(name,document.createTextNode(item.note)); $('legend-details').append(detail); }
  }
  for(const point of info.locations){ L.circleMarker([point.lat,point.lon],{radius:6,bubblingMouseEvents:false,color:'#1a4338',fillColor:'#cff8e4',fillOpacity:1}).addTo(map).on('click', event=>{L.DomEvent.stopPropagation(event);mapClick(event.latlng);}); }
  render();
  // Return to the session's maps after a reload.
  const top = maps.filter(entry => entry.visible).at(-1) || maps.at(-1);
  if (top) { map.fitBounds(top.overlay.getBounds(), {maxZoom:13,padding:[25,25]}); select(top.lat, top.lon, {pan:false, year: top.visible ? top.year : undefined}); }
}).catch(()=>status('Årskartans tjänst är inte tillgänglig just nu. Försök igen senare.',true));
