// Runs docs/js/lclu.js against a stub DOM, Leaflet and annual-map API with
// controlled responses and timers, and prints one JSON object of scenario
// outcomes for tests/lclu/test_page_script.py. Dependency-free (Node only).
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const SCRIPT = fs.readFileSync(path.join(root, 'docs/js/lclu.js'), 'utf8');
const HTML = fs.readFileSync(path.join(root, 'docs/lclu.html'), 'utf8');
const YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];
// As GET /v1/lclu sends it (imint.inference.lclu.vpp_substitutions).
const SUBSTITUTIONS = [{year: 2025, vpp_year: 2024, label: 'växtfenologi från 2024',
  note: 'Uppskattningsvis ~7 % av pixlarna kan avvika jämfört med en karta med årets växtfenologi.'}];
const PLACES = {A: {lat: 59.5, lon: 17.5}, B: {lat: 59.6, lon: 17.6}, C: {lat: 59.7, lon: 17.7}};
const box = (lat, lon) => [[lat - 0.01, lon - 0.01], [lat + 0.01, lon + 0.01]];

class Element {
  constructor(tag, id = '', value = '') {
    Object.assign(this, {tagName: tag.toUpperCase(), id, value, disabled: false, hidden: false,
      textContent: '', children: [], attributes: {}, listeners: {}, style: {}, options: [], href: ''});
    const classes = new Set();
    this.classList = {toggle: (name, on) => (on ? classes.add(name) : classes.delete(name)),
      add: name => classes.add(name), contains: name => classes.has(name)};
  }
  addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); }
  dispatch(type) { for (const listener of this.listeners[type] || []) listener({preventDefault() {}}); }
  click() { this.dispatch('click'); }
  replaceChildren(...children) { this.children = children; }
  append(...children) { this.children.push(...children); }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name]; }
  add(option) { this.options.push(option); }
}

const bounds = ([[south, west], [north, east]]) => ({
  contains: point => {
    const [lat, lng] = Array.isArray(point) ? point : [point.lat, point.lng];
    return south <= lat && lat <= north && west <= lng && lng <= east;
  },
  getWest: () => west, getSouth: () => south, getEast: () => east, getNorth: () => north,
});

// years: what the place and its area offer; substitutions: null for an API without the list.
async function boot({omit = [], storage = {}, years = [2023, 2024], substitutions = SUBSTITUTIONS, refuse = []} = {}) {
  const elements = {};
  for (const [, tag, attributes] of HTML.matchAll(/<(\w+)([^>]*\bid="[\w-]+"[^>]*)>/g)) {
    const id = attributes.match(/\bid="([\w-]+)"/)[1];
    if (!omit.includes(id)) elements[id] = new Element(tag, id, attributes.match(/\bvalue="([^"]*)"/)?.[1] ?? '');
  }
  let now = 0, timerId = 0;
  const timers = [], requests = [], popups = [], overlays = [], hold = new Set();
  const flush = async () => { for (let i = 0; i < 30; i++) await new Promise(resolve => setImmediate(resolve)); };
  async function advance(ms) {
    const end = now + ms;
    for (;;) {
      await flush();
      const due = timers.filter(timer => timer.at <= end).sort((a, b) => a.at - b.at || a.id - b.id)[0];
      if (!due) break;
      timers.splice(timers.indexOf(due), 1); now = due.at; due.run();
    }
    now = end; await flush();
  }
  const mapListeners = {};
  const map = {
    setView() { return this; }, on(type, listener) { (mapListeners[type] ||= []).push(listener); return this; },
    getZoom: () => 13, fitBounds() {}, panTo() {}, closePopup() {}, removeLayer(layer) { layer.remove?.(); },
    getBounds: () => bounds([[59, 17], [60, 18]]),
  };
  const geoJSON = data => {
    const layers = (Array.isArray(data) ? data : [data]).map(feature => {
      const ring = (feature.geometry || feature).coordinates[0];
      const lats = ring.map(point => point[1]), lons = ring.map(point => point[0]);
      return {getBounds: () => bounds([[Math.min(...lats), Math.min(...lons)], [Math.max(...lats), Math.max(...lons)]])};
    });
    return {addTo() { return this; }, getLayers: () => layers, getBounds: () => layers[0].getBounds()};
  };
  const L = {
    map: () => map, tileLayer: () => ({addTo() { return this; }}), geoJSON,
    imageOverlay(url, area) {
      const overlay = {url, onMap: false, on() { return this; }, addTo() { this.onMap = true; return this; },
        remove() { this.onMap = false; return this; }, bringToFront() {}, setOpacity() {}, getBounds: () => bounds(area)};
      overlays.push(overlay); return overlay;
    },
    popup() {
      const popup = {setLatLng(point) { this.point = point; return this; }, setContent(content) { this.content = content; return this; },
        openOn() { popups.push(this); return this; }};
      return popup;
    },
    circleMarker: () => ({addTo() { return this; }, on() { return this; }}),
    DomEvent: {stopPropagation() {}},
  };
  let jobs = 0;
  function answer(request) {
    const url = new URL(request.url), route = url.pathname;
    if (route === '/v1/lclu' && request.method === 'GET') {
      return {ready: true, message: 'Välj en plats.', historical_note: '', candidate_years: YEARS, legend: [],
        ...(substitutions ? {vpp_substitutions: substitutions} : {}),
        locations: [{lat: 59.5, lon: 17.5}],
        areas: [{type: 'Feature', properties: {name: 'Testområde', years},
          geometry: {type: 'Polygon', coordinates: [[[17, 59], [18, 59], [18, 60], [17, 60], [17, 59]]]}}]};
    }
    if (route === '/v1/lclu/availability') {
      const lat = Number(url.searchParams.get('lat')), lon = Number(url.searchParams.get('lon'));
      const [[south, west], [north, east]] = box(lat, lon);
      return {years, unavailable_years: YEARS.filter(year => !years.includes(year)), message: '',
        geometry: {center: [Math.round(lon * 1000), Math.round(lat * 1000)],
          footprint: {type: 'Polygon', coordinates: [[[west, south], [east, south], [east, north], [west, north], [west, south]]]}}};
    }
    if (route === '/v1/lclu' && request.method === 'POST') {
      const id = `job-${++jobs}`;
      request.job = {id, ...request.body};
      return {job_id: id, poll_url: `/v1/lclu/jobs/${id}`};
    }
    const job = requests.find(item => item.job && route.startsWith(`/v1/lclu/jobs/${item.job.id}`))?.job;
    if (route.endsWith('/pixel')) return {inside: true, year: job.year, name: `klass ${job.id}`, color: '#123456'};
    if (job) {
      const files = Object.fromEntries(['overlay.png', 'classification.tif', 'provenance.json']
        .map(name => [name, `/v1/lclu/jobs/${job.id}/files/${name}`]));
      return {status: 'success', phase: 'complete', result: {artifacts: files, overlay_bounds: box(job.lat, job.lon),
        valid_pixels: 504 * 504, nodata_pixels: 0, cache_hit: false}};
    }
    return {prepared: true};
  }
  let posts = 0;
  function fetch(url, options = {}) {
    return new Promise(resolve => {
      const request = {url: String(url), method: options.method || 'GET', at: now,
        body: options.body ? JSON.parse(options.body) : null,
        respond: () => resolve({ok: true, status: 200, headers: {get: () => 'application/json'},
          json: async () => answer(request)})};
      if (request.method === 'POST' && new URL(request.url).pathname === '/v1/lclu' && refuse.includes(++posts)) {
        request.refused = true;
        request.respond = () => resolve({ok: false, status: 429, headers: {get: () => 'application/json'},
          json: async () => ({detail: 'Job queue is full; retry later'})});
      }
      requests.push(request);
      if (!hold.has(new URL(request.url).pathname)) { request.done = true; request.respond(); }
    });
  }
  const store = new Map(Object.entries(storage));
  const document = {getElementById: id => elements[id] || null, createElement: tag => new Element(tag),
    createTextNode: text => ({textContent: text})};
  const context = vm.createContext({
    window: {IMINT_LCLU_CONFIG: {apiBase: 'https://api.test'}, location: {origin: 'https://page.test'}},
    document, L, fetch, URL, console,
    Option: class { constructor(text, value) { this.text = text; this.value = value; } },
    ResizeObserver: class { observe() {} },
    setTimeout: (run, ms) => { timers.push({id: ++timerId, at: now + ms, run}); return timerId; },
    clearTimeout: id => { const index = timers.findIndex(timer => timer.id === id); if (index >= 0) timers.splice(index, 1); },
    sessionStorage: {getItem: key => store.get(key) ?? null, setItem: (key, value) => store.set(key, value)},
    performance: {now: () => now},
  });
  let error = null;
  try { vm.runInContext(SCRIPT, context); } catch (caught) { error = caught.message; }
  await advance(0);
  const page = {
    elements, requests, popups, overlays, hold, advance, error, store,
    async select(name) {
      elements.lat.value = String(PLACES[name].lat); elements.lon.value = String(PLACES[name].lon);
      elements['location-form'].dispatch('submit'); await advance(0);
    },
    async slide(year, release = true) {
      elements.year.value = String(year); elements.year.dispatch('input');
      if (release) elements.year.dispatch('change');
      await advance(0);
    },
    async create() { elements.create.click(); await advance(0); },
    click(name) { for (const listener of mapListeners.click) listener({latlng: {lat: PLACES[name].lat, lng: PLACES[name].lon}}); },
    respond(pathname, name) {
      const request = requests.find(item => !item.done && new URL(item.url).pathname === pathname
        && (!name || Number(new URL(item.url).searchParams.get('lat')) === PLACES[name].lat));
      request.done = true; request.respond();
    },
    creations: () => requests.filter(item => item.method === 'POST' && new URL(item.url).pathname === '/v1/lclu' && !item.refused)
      .map(item => ({place: Object.keys(PLACES).find(name => PLACES[name].lat === item.body.lat), year: item.body.year})),
    posts: () => requests.filter(item => item.method === 'POST' && new URL(item.url).pathname === '/v1/lclu')
      .map(item => ({year: item.body.year, at: item.at, refused: Boolean(item.refused)})),
    fillStatus: () => (elements['fill-status'].hidden ? null : elements['fill-status'].textContent),
    pixelJobs: () => requests.filter(item => new URL(item.url).pathname.endsWith('/pixel'))
      .map(item => new URL(item.url).pathname.split('/')[4]),
    jobAt: name => requests.find(item => item.job && item.job.lat === PLACES[name].lat)?.job.id,
  };
  return page;
}

const results = {};

// A release creates the released year at the released place.
{
  const page = await boot();
  await page.select('A'); await page.slide(2023); await page.advance(1000);
  results.release = {creations: page.creations()};
}

// Release on A, then choose B before the release delay ends: nothing is created.
{
  const page = await boot();
  await page.select('A'); await page.slide(2023); await page.advance(100);
  await page.select('B'); await page.advance(1000);
  results.release_then_place = {creations: page.creations()};
}

// Maps at A (2024) and B (2023), place C current. Click B, then A, while both
// availability answers are pending; answer A first and B last.
{
  const page = await boot();
  await page.select('A'); await page.create();
  await page.select('B'); await page.slide(2023, false); await page.create();
  await page.select('C');
  page.hold.add('/v1/lclu/availability');
  page.click('B'); await page.advance(0);
  page.click('A'); await page.advance(0);
  page.respond('/v1/lclu/availability', 'A'); await page.advance(0);
  page.respond('/v1/lclu/availability', 'B'); await page.advance(0);
  results.stale_click = {pixel_jobs: page.pixelJobs(), job_a: page.jobAt('A'), job_b: page.jobAt('B'),
    title: page.elements['location-title'].textContent, year: page.elements.year.value};
}

// Restore keeps only this service's own artifact paths.
{
  const own = name => ({'overlay.png': `/v1/lclu/jobs/${name}/files/overlay.png`,
    'classification.tif': `/v1/lclu/jobs/${name}/files/classification.tif`,
    'provenance.json': `/v1/lclu/jobs/${name}/files/provenance.json`});
  const entry = (id, key, artifacts) => ({id, key, lat: 59.5, lon: 17.5, year: 2024, visible: true, order: 1,
    result: {artifacts, overlay_bounds: box(59.5, 17.5), valid_pixels: 1, nodata_pixels: 0}});
  const saved = [entry('own', '1', own('own')),
    entry('foreign', '2', {...own('foreign'), 'overlay.png': 'https://elsewhere.example/x.png'})];
  const page = await boot({storage: {'imint-lclu-session-maps-v1': JSON.stringify(saved)}});
  results.restore = {overlays: page.overlays.map(overlay => overlay.url)};
}

// A cached page from before the timeline gets a reload message, not a broken page.
{
  const page = await boot({omit: ['year-ticks']});
  results.outdated_page = {error: page.error, status: page.elements.status.textContent,
    requests: page.requests.length};
}

// Where the page names the year: readout, tick, status and result panel.
function labels(page) {
  const ticks = page.elements['year-ticks'].children, note = page.elements['vpp-note'];
  return {readout: page.elements['year-value'].value, valuetext: page.elements.year.getAttribute('aria-valuetext'),
    status: page.elements.status.textContent, result_shown: !page.elements.result.hidden,
    title: page.elements['result-title'].textContent, note: note.hidden ? null : note.textContent,
    substituted_ticks: ticks.filter(tick => tick.classList.contains('substituted')).map(tick => [tick.textContent, tick.title])};
}

// 2025 offered with HR-VPP 2024: chosen, created, then 2024 beside it.
{
  const page = await boot({years: [2023, 2024, 2025]});
  await page.select('A');
  const chosen = labels(page);
  page.hold.add('/v1/lclu/jobs/job-2');  // the background 2024 map stays in progress
  await page.create();
  const made = labels(page);
  await page.slide(2024, false);
  const other = labels(page);
  await page.create();
  page.respond('/v1/lclu/jobs/job-2'); await page.advance(0);
  results.substituted = {error: page.error, chosen, made, other, plain: labels(page),
    span: page.elements['year-span'].textContent, creations: page.creations()};
}

// An API without the list (a release before the table) labels no year.
{
  const page = await boot({substitutions: null});
  await page.select('A'); await page.create();
  results.without_list = {error: page.error, ...labels(page)};
}

// A catalogue offering a single year names that year, not a range.
{
  const page = await boot({years: [2024]});
  results.single_year = {span: page.elements['year-span'].textContent};
}

// After the user's year, the place's other years follow, nearest first, the later first at a tie.
{
  const page = await boot({years: YEARS});
  await page.select('A'); await page.slide(2020); await page.advance(1000);
  results.fill_order = {years: page.creations().map(item => item.year), fill: page.fillStatus(),
    status: page.elements.status.textContent};
}

// One job at a time: while a background map is in progress nothing else is submitted.
{
  const page = await boot({years: YEARS});
  page.hold.add('/v1/lclu/jobs/job-2');
  await page.select('A'); await page.slide(2020); await page.advance(1000);
  const during = page.creations().map(item => item.year), fill = page.fillStatus();
  page.respond('/v1/lclu/jobs/job-2'); await page.advance(1000);
  results.one_at_a_time = {during, fill, after: page.creations().length};
}

// A full queue (429) is retried after 15 s, not at once.
{
  const page = await boot({refuse: [2]});
  await page.select('A'); await page.create(); await page.advance(14000);
  const before = page.posts().length;
  await page.advance(2000);
  results.retry = {posts: page.posts(), before, years: page.creations().map(item => item.year)};
}

// Choosing another place stops the old place's background years.
{
  const page = await boot({years: YEARS});
  page.hold.add('/v1/lclu/jobs/job-2');
  await page.select('A'); await page.slide(2020); await page.advance(1000);
  await page.select('B');
  page.respond('/v1/lclu/jobs/job-2'); await page.advance(60000);
  results.place_change = {creations: page.creations(), fill: page.fillStatus()};
}

// A map already in the session is not made again.
{
  const files = id => Object.fromEntries(['overlay.png', 'classification.tif', 'provenance.json']
    .map(name => [name, `/v1/lclu/jobs/${id}/files/${name}`]));
  const saved = [{id: 'old', key: '17500,59500', lat: 59.5, lon: 17.5, year: 2021, visible: false, order: 1,
    result: {artifacts: files('old'), overlay_bounds: box(59.5, 17.5), valid_pixels: 1, nodata_pixels: 0}}];
  const page = await boot({years: [2019, 2020, 2021, 2022], storage: {'imint-lclu-session-maps-v1': JSON.stringify(saved)}});
  await page.select('A'); await page.slide(2020); await page.advance(1000);
  results.existing = {years: page.creations().map(item => item.year)};
}

// Restoring a session after a reload starts no background years.
{
  const files = id => Object.fromEntries(['overlay.png', 'classification.tif', 'provenance.json']
    .map(name => [name, `/v1/lclu/jobs/${id}/files/${name}`]));
  const saved = [{id: 'kept', key: '17500,59500', lat: 59.5, lon: 17.5, year: 2024, visible: true, order: 1,
    result: {artifacts: files('kept'), overlay_bounds: box(59.5, 17.5), valid_pixels: 1, nodata_pixels: 0}}];
  const page = await boot({years: YEARS, storage: {'imint-lclu-session-maps-v1': JSON.stringify(saved)}});
  await page.advance(60000);
  results.restore_no_fill = {posts: page.posts().length, fill: page.fillStatus()};
}

// A year the user asks for during the background years goes first.
{
  const page = await boot({years: YEARS});
  page.hold.add('/v1/lclu/jobs/job-2');
  await page.select('A'); await page.slide(2020); await page.advance(1000);
  await page.slide(2025); await page.advance(1000);
  const waiting = page.elements.status.textContent;
  page.respond('/v1/lclu/jobs/job-2'); await page.advance(1000);
  results.user_first = {years: page.creations().map(item => item.year), waiting,
    status: page.elements.status.textContent};
}

// Removing a place stops its background years; a result that arrives afterwards is dropped.
{
  const page = await boot({years: YEARS});
  page.hold.add('/v1/lclu/jobs/job-2');
  await page.select('A'); await page.slide(2020); await page.advance(1000);
  page.elements['session-list'].children[0].children[0].children[1].click();  // × on place A
  page.respond('/v1/lclu/jobs/job-2'); await page.advance(60000);
  results.remove_place = {creations: page.creations().map(item => item.year),
    listed: page.elements['session-list'].children.length, fill: page.fillStatus()};
}

// Choosing the same place again while a background year finishes resumes the queue.
{
  const page = await boot({years: YEARS});
  page.hold.add('/v1/lclu/jobs/job-2');
  await page.select('A'); await page.slide(2020); await page.advance(1000);
  page.hold.add('/v1/lclu/availability');
  page.elements.lat.value = String(PLACES.A.lat); page.elements.lon.value = String(PLACES.A.lon);
  page.elements['location-form'].dispatch('submit'); await page.advance(0);
  page.respond('/v1/lclu/jobs/job-2'); await page.advance(0);
  page.respond('/v1/lclu/availability', 'A'); await page.advance(60000);
  results.reselect = {years: page.creations().map(item => item.year), fill: page.fillStatus()};
}

process.stdout.write(JSON.stringify(results));
