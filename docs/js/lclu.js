/* Geographic annual map UI. No synthetic predictions or year substitution. */
'use strict';
const $ = id => document.getElementById(id);
const map = L.map('map', {minZoom: 4, maxZoom: 18, zoomAnimation: false, fadeAnimation: false}).setView([62.1, 16.5], 5);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 18, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
}).addTo(map);
let selection = null, footprint = null, overlay = null, generation = 0, busy = false, ready = false;
let resultJob = null, inspection = 0;
const phaseLabels = {queued:'I kö …', preparing:'Förbereder årsdata …', classifying:'Klassificerar …', exporting:'Skapar kartfiler …', complete:'Årskartan är klar.'};
function status(text, error=false) { $('status').textContent = text; $('status').classList.toggle('error', error); }
function clearResult() { ++inspection; resultJob=null; map.closePopup(); if (overlay) { map.removeLayer(overlay); overlay = null; } $('result').hidden = true; }
async function request(url, options) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `Begäran misslyckades (${response.status}).`);
  return body;
}
function controls() {
  $('area').disabled = busy;
  $('create').disabled = !ready || !selection || !$('year').value || busy;
  $('year').disabled = busy || $('year').options.length === 0 || !$('year').options[0].value;
  $('progress').hidden = !busy;
}
async function select(lat, lon) {
  if (busy) return;
  const current = ++generation;
  clearResult(); selection = null; $('year').replaceChildren(); controls();
  $('lat').value = lat.toFixed(5); $('lon').value = lon.toFixed(5);
  $('location-title').textContent = `${lat.toFixed(4)}° N, ${lon.toFixed(4)}° E`;
  status('Kontrollerar tillgängliga år …');
  try {
    const data = await request(`/v1/lclu/availability?lon=${lon}&lat=${lat}`);
    if (current !== generation) return;
    if (footprint) map.removeLayer(footprint);
    footprint = L.geoJSON(data.geometry.footprint, {interactive:false,style:{color:'#1a4338',weight:2,fillOpacity:.06}}).addTo(map);
    // Keep the user's detail zoom when choosing another nearby point.
    if (map.getZoom() < 12) map.fitBounds(footprint.getBounds(), {maxZoom:12,padding:[25,25]});
    else map.panTo([lat,lon]);
    selection = {lat,lon};
    if (data.years.length) {
      data.years.slice().reverse().forEach(year => $('year').add(new Option(String(year), String(year))));
      status('Välj år och skapa din karta. Pixelernas datatäckning kontrolleras vid körning.');
    } else { $('year').add(new Option('Inga tillgängliga år','')); status(data.message); }
  } catch (error) { if (current === generation) status(error.message,true); }
  if (current === generation) controls();
}
async function mapClick(latlng) {
  if (busy) return;
  if (!overlay || !resultJob || !overlay.getBounds().contains(latlng)) {
    return select(latlng.lat,latlng.lng);
  }
  const current=++inspection, job=resultJob;
  const message=document.createElement('div'); message.textContent='Läser markklass …';
  const popup=L.popup().setLatLng(latlng).setContent(message).openOn(map);
  try {
    const pixel=await request(`/v1/lclu/jobs/${job}/pixel?lon=${latlng.lng}&lat=${latlng.lat}`);
    if(current!==inspection || job!==resultJob) return;
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
  } catch(error) { if(current===inspection && job===resultJob) message.textContent='Markklassen kunde inte läsas. Försök igen.'; }
}
map.on('click', event => mapClick(event.latlng));
$('location-form').addEventListener('submit', event => { event.preventDefault(); select(Number($('lat').value),Number($('lon').value)); });
$('year').addEventListener('change', () => { clearResult(); controls(); });
$('opacity').addEventListener('input', () => { const value=Number($('opacity').value); $('opacity-value').value=`${value} %`; if(overlay) overlay.setOpacity(value/100); });
$('create').addEventListener('click', async () => {
  if (busy || !selection) return;
  const current = ++generation, year = Number($('year').value), started = performance.now();
  busy=true; clearResult(); controls(); status('Skickar begäran …');
  try {
    const submission = await request('/v1/lclu', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...selection,year})});
    let job;
    do {
      job = await request(submission.poll_url);
      if (current !== generation) return;
      status(phaseLabels[job.phase] || phaseLabels[job.status] || 'Bearbetar …');
      if (!['success','failed'].includes(job.status)) await new Promise(resolve=>setTimeout(resolve,900));
    } while (!['success','failed'].includes(job.status));
    if(job.status === 'failed') throw new Error(job.error || 'Årskartan kunde inte skapas.');
    const r=job.result;
    resultJob=submission.job_id;
    overlay=L.imageOverlay(r.artifacts['overlay.png'],r.overlay_bounds,{opacity:Number($('opacity').value)/100,interactive:false,className:'lclu-classification'}).addTo(map);
    overlay.on('error',()=>status('Kartlagret kunde inte visas. GeoTIFF-filen kan fortfarande hämtas.',true));
    $('download-tif').href=r.artifacts['classification.tif'];
    $('download-provenance').href=r.artifacts['provenance.json'];
    $('result-title').textContent=`Årskarta ${year}`;
    $('coverage').textContent=`${(100*r.valid_pixels/(504*504)).toFixed(1)} % giltiga pixlar · ${r.nodata_pixels.toLocaleString('sv-SE')} utan data. Stödlager kan ha luckor; se proveniens.`;
    $('timing').textContent=`Svarstid: ${((performance.now()-started)/1000).toFixed(1)} s`;
    $('result').hidden=false;
    status(r.cache_hit ? 'Årskartan är klar · tidigare verifierat resultat.' : 'Årskartan är klar.');
  } catch(error) { status(error.message,true); }
  finally { busy=false; controls(); }
});
request('/v1/lclu').then(info=>{
  ready=info.ready; status(info.message); $('historical-note').textContent=info.historical_note;
  if(info.areas?.length){
    const areas=L.geoJSON(info.areas,{interactive:false,style:{color:'#245045',weight:2,dashArray:'5 5',fillOpacity:.02}}).addTo(map);
    const layers=areas.getLayers();
    map.fitBounds(layers[0].getBounds(),{padding:[25,25]});
    $('area-title').textContent=info.areas.map(area=>area.properties.name).join(' / ')+' / DEMO';
    $('map-caption').textContent='Välj en punkt inom ett streckat demoområde. Årskartan är 5,04 × 5,04 km och kan sträcka sig utanför området.';
    info.areas.forEach((area,index)=>$('area').add(new Option(area.properties.name,String(index))));
    $('area-control').hidden=info.areas.length<2;
    $('area').addEventListener('change',()=>{
      const index=Number($('area').value), point=info.locations[index];
      map.fitBounds(layers[index].getBounds(),{padding:[25,25],maxZoom:13});
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
  controls();
}).catch(error=>status(`Tjänsten kunde inte nås: ${error.message}`,true));
