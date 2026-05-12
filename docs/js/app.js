'use strict';

(function() {

    var allMaps = {};
    var allOverlays = {};
    var allBgLayers = {};

    // ── Reusable HTML components ─────────────────────────────────────

    function renderLegend(legendKey) {
        var items = LEGENDS[legendKey];
        if (!items || !items.length) return '';
        var html = '<div class="legend-strip">';
        items.forEach(function(item) {
            html += '<span class="legend-item">' +
                '<span class="legend-swatch" style="background:' + item.color + '"></span>' +
                item.label + '</span>';
        });
        html += '</div>';
        return html;
    }

    function renderSummaryCards(cards) {
        var html = '<div class="summary-section">';
        cards.forEach(function(c) {
            html += '<div class="summary-card">' +
                '<h4>' + c.title + '</h4>' +
                '<div class="value">' + c.value + '</div>' +
                '<div class="detail">' + c.detail + '</div>' +
                '</div>';
        });
        html += '</div>';
        return html;
    }

    function renderPanelToolbar(panels) {
        var html = '<div class="panel-toolbar">' +
            '<span class="panel-toolbar-label">Paneler:</span>';
        panels.forEach(function(p) {
            html += '<button class="panel-chip active" data-panel-id="' + p.id + '">' +
                p.title + '</button>';
        });
        html += '</div>';
        return html;
    }

    function renderBgToggle(panel) {
        if (!panel.bgToggle) return '';
        var html = '<div class="bg-toggle" data-map-id="' + panel.id + '">' +
            '<span class="bg-label">Visa:</span>';
        panel.bgToggle.forEach(function(btn) {
            html += '<button class="bg-btn' + (btn.active ? ' active' : '') +
                '" data-bg="' + btn.key + '">' + btn.label + '</button>';
        });
        html += '</div>';
        return html;
    }

    function renderMapCell(panel) {
        var html = '<div class="map-cell" data-panel-id="' + panel.id + '">' +
            '<div class="map-cell-header">' +
            '<h3>' + panel.title + '</h3>' +
            '<div class="header-controls">' +
            renderBgToggle(panel) +
            '<div class="opacity-control">' +
            '<label for="opacity-' + panel.id + '">Opacitet</label>' +
            '<input type="range" id="opacity-' + panel.id + '" min="0" max="100" ' +
            'value="100" data-map-id="' + panel.id + '">' +
            '<span class="opacity-value" id="opacity-val-' + panel.id + '">100%</span>' +
            '</div>' +
            '<button class="hide-panel-btn" data-panel-id="' + panel.id +
            '" title="Dölj panel">&#x2715;</button>' +
            '</div></div>' +
            '<div id="' + panel.id + '" class="map-container"></div>' +
            (panel.legend ? renderLegend(panel.legend) : '') +
            '</div>';
        return html;
    }

    function renderMapGrid(panels) {
        var html = '<div class="map-grid">';
        panels.forEach(function(p) { html += renderMapCell(p); });
        html += '</div>';
        return html;
    }

    // ── Copy block (lede + hint + collapsible methods) ──────────────
    // New schema introduced 2026-05-11. Tabs that have migrated set
    // config.copy = { lede, hint?, methods? }. Untouched tabs still set
    // config.intro and fall through to the legacy <p> render below.
    function renderCopy(copy) {
        var html = '<div class="tab-intro">';
        html += '<p class="lede">' + copy.lede + '</p>';
        if (copy.hint) {
            html += '<p class="hint">' + copy.hint + '</p>';
        }
        if (copy.methods && copy.methods.length) {
            html += '<details class="methods"><summary>Visa metod och datakällor</summary>';
            copy.methods.forEach(function(s) {
                html += '<h4>' + s.heading + '</h4>' + s.body;
            });
            html += '</details>';
        }
        html += '</div>';
        return html;
    }

    function renderTabDynamic(config) {
        var html = '<div class="section-header"><h2>' + config.title + '</h2></div>';
        html += renderSummaryCards(config.summary);
        if (config.copy) {
            html += renderCopy(config.copy);
        } else {
            html += '<div class="tab-intro"><p>' + config.intro + '</p></div>';
        }
        html += renderPanelToolbar(config.panels);
        html += renderMapGrid(config.panels);
        return html;
    }

    // ── Glossary tooltip decorator ──────────────────────────────────
    // For each tab, walk the rendered DOM and wrap the first occurrence of
    // every GLOSSARY term in a <span class="gloss"> with a dotted underline.
    // A single shared #gloss-card in <body> handles hover display — avoids
    // creating N DOM nodes per acronym and survives the hover gap.
    var GLOSS_SKIP_SELECTOR = 'A, CODE, .gloss, .legend-item, .summary-card, ' +
        '.panel-chip, .bg-btn, .opacity-control, H2, H3, H4, SUMMARY';

    function ensureGlossCard() {
        if (document.getElementById('gloss-card')) return;
        var card = document.createElement('div');
        card.id = 'gloss-card';
        document.body.appendChild(card);
    }

    function decorateGlossary(rootEl) {
        if (!rootEl || typeof GLOSSARY === 'undefined') return;
        var terms = Object.keys(GLOSSARY);
        if (!terms.length) return;
        // Sort longest-first so 'NDCI' wins over a hypothetical shorter prefix.
        terms.sort(function(a, b) { return b.length - a.length; });
        // Escape regex special chars defensively (none today, but cheap).
        var escaped = terms.map(function(t) {
            return t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        });
        var re = new RegExp('\\b(' + escaped.join('|') + ')\\b');
        var seen = {};

        // Snapshot text nodes first; mutating during traversal corrupts the walker.
        var walker = document.createTreeWalker(rootEl, NodeFilter.SHOW_TEXT, null, false);
        var nodes = [];
        var n;
        while ((n = walker.nextNode())) {
            if (!n.nodeValue || !n.nodeValue.trim()) continue;
            if (n.parentNode && n.parentNode.closest && n.parentNode.closest(GLOSS_SKIP_SELECTOR)) continue;
            nodes.push(n);
        }

        nodes.forEach(function(node) {
            var text = node.nodeValue;
            var pieces = [];
            var idx = 0;
            while (idx < text.length) {
                re.lastIndex = 0;
                var sub = text.substring(idx);
                var m = sub.match(re);
                if (!m || seen[m[1]]) {
                    pieces.push(document.createTextNode(text.substring(idx)));
                    break;
                }
                var matchPos = idx + m.index;
                if (matchPos > idx) {
                    pieces.push(document.createTextNode(text.substring(idx, matchPos)));
                }
                var span = document.createElement('span');
                span.className = 'gloss';
                span.setAttribute('data-term', m[1]);
                span.textContent = m[1];
                pieces.push(span);
                seen[m[1]] = true;
                idx = matchPos + m[1].length;
            }
            if (pieces.length === 1 && pieces[0].nodeType === 3) return; // no-op
            var parent = node.parentNode;
            pieces.forEach(function(p) { parent.insertBefore(p, node); });
            parent.removeChild(node);
        });
    }

    function bindGlossHover() {
        var card = document.getElementById('gloss-card');
        if (!card) return;
        document.addEventListener('mouseover', function(e) {
            var t = e.target;
            if (!t || !t.classList || !t.classList.contains('gloss')) return;
            var term = t.getAttribute('data-term');
            var entry = GLOSSARY[term];
            if (!entry) return;
            card.innerHTML = '<strong>' + term + '</strong> — ' + entry.full +
                '<br><span class="gloss-card-sv">' + entry.sv + '</span>';
            var r = t.getBoundingClientRect();
            var top = r.bottom + window.scrollY + 6;
            var left = r.left + window.scrollX;
            // Keep card on-screen horizontally
            var maxLeft = window.scrollX + window.innerWidth - 300;
            if (left > maxLeft) left = maxLeft;
            if (left < window.scrollX + 8) left = window.scrollX + 8;
            card.style.top = top + 'px';
            card.style.left = left + 'px';
            card.classList.add('visible');
        });
        document.addEventListener('mouseout', function(e) {
            var t = e.target;
            if (!t || !t.classList || !t.classList.contains('gloss')) return;
            card.classList.remove('visible');
        });
    }

    // ── GeoJSON loading ──────────────────────────────────────────────

    function loadGeoJSON(filename) {
        var path = GEOJSON_FILES[filename];
        if (!path) return Promise.resolve(null);
        return fetch(path)
            .then(function(r) { return r.json(); })
            .catch(function() { return null; });
    }

    // ── GeoJSON styling ──────────────────────────────────────────────

    function gjStyle(feature) {
        var p = feature.properties || {};
        if (p.year) {
            var yc = {2018:'#e41a1c',2019:'#ff7f00',2020:'#ffd700',2021:'#4daf4a',
                       2022:'#00ced1',2023:'#377eb8',2024:'#984ea3',2025:'#e41a9d'};
            return {color: yc[p.year] || '#fff', weight: 2.5, opacity: 0.9, fill: false};
        }
        if (p.change_type) {
            var cc = {'erosion':'#d73027','stable':'#ffffbf','accumulation':'#1a9850'};
            return {color: cc[p.change_type] || '#fff', weight: 3.5, opacity: 0.95, fill: false};
        }
        // AI2 vessel with speed attribute — color by speed
        if (p.label === 'vessel' && typeof p.speed_knots === 'number') {
            var spd = p.speed_knots;
            var vc = spd <= 0.5 ? '#50A0FF' : spd < 5 ? '#FFE600' : spd < 15 ? '#FF6600' : '#FF0000';
            return {color: vc, weight: 2.5, fillColor: vc, fillOpacity: 0.25, opacity: 1};
        }
        // YOLO vessel (no speed attribute)
        if (p.label === 'vessel') {
            return {color: '#E6119D', weight: 2, fillOpacity: 0.15, opacity: 1};
        }
        var cls = p.predicted_class;
        var color = '#888888';
        if (cls === 1) color = '#E6119D';
        else if (cls === 0) color = '#111111';
        return {color: color, weight: 2, fillOpacity: 0.15, opacity: 1};
    }

    function gjPopup(feature, layer) {
        var p = feature.properties || {};
        if (p.year) {
            layer.bindPopup('<b>' + p.year + '</b><br>Längd: ' + Math.round(p.length_m) + ' m');
            return;
        }
        if (p.change_type) {
            var labels = {'erosion':'Erosion','stable':'Stabil','accumulation':'Ackumulation'};
            layer.bindPopup('<b>' + (labels[p.change_type]||p.change_type) + '</b><br>Förändring: ' + p.change_m + ' m');
            return;
        }
        if (p.label === 'vessel') {
            var html = '<b>Fartyg</b><br>Konfidens: ' + Math.round((p.confidence||p.score||0)*100) + '%';
            if (p.vessel_type) html += '<br>Typ: ' + p.vessel_type;
            if (typeof p.speed_knots === 'number') html += '<br>Fart: ' + p.speed_knots.toFixed(1) + ' kn';
            if (typeof p.heading_deg === 'number') html += '<br>Kurs: ' + Math.round(p.heading_deg) + '°';
            if (typeof p.length_m === 'number') html += '<br>Längd: ' + Math.round(p.length_m) + ' m';
            layer.bindPopup(html);
            return;
        }
        if (p.class_label) {
            layer.bindPopup(
                '<b>Block ' + (p.blockid || '') + '</b><br>' +
                p.class_label + ' (' + Math.round((p.confidence||0)*100) + '%)'
            );
        }
    }

    /**
     * Convert pixel-space GeoJSON (y-down) to Leaflet CRS.Simple (y-up).
     * All showcase GeoJSON uses image pixel coordinates where row 0 is
     * the top of the image, but Leaflet CRS.Simple has lat 0 at the
     * bottom. The coordsToLatLng callback flips y: lat = imgH - row.
     */
    function makeGeoJSON(gjData, map, imgH) {
        var gjLayer = L.geoJSON(gjData, {
            style: gjStyle,
            onEachFeature: gjPopup,
            coordsToLatLng: function(coords) {
                return L.latLng(imgH - coords[1], coords[0]);
            }
        }).addTo(map);

        return gjLayer;
    }

    // ── Map initialization ───────────────────────────────────────────

    function initMaps(panels, images, imgH, imgW, hasBgToggle, geojsonMap, nativeZoom) {
        var bounds = [[0, 0], [imgH, imgW]];
        var maps = [];

        panels.forEach(function(panel) {
            var container = document.getElementById(panel.id);
            var isVector = panel.vector || false;
            if (!container || (!images[panel.id] && !isVector)) return;

            // Native-zoom panels lock at zoom 0 and disable all zoom
            // interactions (scroll, dblclick, pinch, +/- buttons). At zoom 0
            // each bound-unit equals one screen pixel, so combined with
            // image-rendering: pixelated the browser never bilinear- or
            // bicubic-interpolates the image data. Pan is kept so the user
            // can inspect different parts of the tile.
            var mapOpts = {
                crs: L.CRS.Simple,
                minZoom: nativeZoom ? 0 : -5,
                maxZoom: nativeZoom ? 0 : 5,
                attributionControl: false,
                zoomSnap: 0.25,
                zoomControl: !nativeZoom,
                scrollWheelZoom: !nativeZoom,
                doubleClickZoom: !nativeZoom,
                touchZoom: !nativeZoom,
                boxZoom: !nativeZoom,
                keyboard: !nativeZoom,
            };
            var map = L.map(panel.id, mapOpts);

            var panelHasBg = hasBgToggle || !!panel.bgToggle;
            var prefix = panel.id.split('-')[0];
            var rgbId = prefix + '-rgb';

            // Background layers
            if (panelHasBg) {
                allBgLayers[panel.id] = {};
                var rgbUrl = (panel.key === 'rgb') ? images[panel.id] : images[rgbId];
                if (rgbUrl) {
                    allBgLayers[panel.id].rgb = L.imageOverlay(rgbUrl, bounds, {zIndex:0, opacity:1}).addTo(map);
                }
                var extraBgs = {sjokort: prefix+'-sjokort', baseline: prefix+'-baseline', nmd: prefix+'-nmd'};
                Object.keys(extraBgs).forEach(function(key) {
                    var imgId = extraBgs[key];
                    if (images[imgId] && (key !== 'nmd' || panel.key !== 'nmd')) {
                        allBgLayers[panel.id][key] = L.imageOverlay(images[imgId], bounds, {zIndex:0, opacity:0}).addTo(map);
                    }
                });
            } else {
                if (panel.key !== 'rgb' && images[rgbId]) {
                    L.imageOverlay(images[rgbId], bounds, {zIndex:0}).addTo(map);
                }
            }

            // Overlay (content) layer
            if (panel.key === 'rgb' && panelHasBg) {
                allOverlays[panel.id] = allBgLayers[panel.id].rgb;
            } else if (isVector) {
                var gjData = geojsonMap ? geojsonMap[panel.geojsonFile || '_default'] : null;
                if (gjData) {
                    allOverlays[panel.id] = makeGeoJSON(gjData, map, imgH);
                }
            } else {
                allOverlays[panel.id] = L.imageOverlay(images[panel.id], bounds, {zIndex:1}).addTo(map);
            }

            // Native-zoom mode: pin the map at zoom 0 (1 bound-unit = 1
            // screen pixel) instead of fitting the whole tile to the cell.
            // Used by the SR showcase so pixel-level differences between
            // methods are not erased by browser bicubic downsampling.
            if (nativeZoom) {
                map.setView([imgH / 2, imgW / 2], 0);
            } else {
                map.fitBounds(bounds);
            }
            map._imgBounds = bounds;
            map._nativeZoom = !!nativeZoom;
            maps.push(map);
            allMaps[panel.id] = map;
        });

        // Sync maps within tab
        for (var i = 0; i < maps.length; i++) {
            for (var j = 0; j < maps.length; j++) {
                if (i !== j) maps[i].sync(maps[j]);
            }
        }
    }

    // ── Event handlers (bound once) ──────────────────────────────────

    function bindOpacitySliders() {
        document.querySelectorAll('.opacity-control input[type="range"]').forEach(function(slider) {
            slider.addEventListener('input', function() {
                var mapId = this.dataset.mapId;
                var val = parseInt(this.value);
                var valEl = document.getElementById('opacity-val-' + mapId);
                if (valEl) valEl.textContent = val + '%';
                var ov = allOverlays[mapId];
                if (ov) {
                    if (ov.setOpacity) ov.setOpacity(val / 100);
                    else if (ov.setStyle) ov.setStyle({opacity: val / 100});
                }
            });
        });
    }

    function bindBgToggles() {
        document.querySelectorAll('.bg-toggle').forEach(function(toggle) {
            toggle.querySelectorAll('.bg-btn').forEach(function(btn) {
                btn.addEventListener('click', function() {
                    var mapId = toggle.dataset.mapId;
                    var activeBg = this.dataset.bg;
                    var layers = allBgLayers[mapId];
                    if (!layers) return;
                    toggle.querySelectorAll('.bg-btn').forEach(function(b) { b.classList.remove('active'); });
                    this.classList.add('active');
                    Object.keys(layers).forEach(function(key) {
                        if (layers[key] && layers[key].setOpacity) {
                            layers[key].setOpacity(key === activeBg ? 1 : 0);
                        }
                    });
                });
            });
        });
    }

    function bindPanelToggles() {
        document.querySelectorAll('.panel-chip').forEach(function(chip) {
            chip.addEventListener('click', function() {
                var pid = this.dataset.panelId;
                var isActive = this.classList.contains('active');
                if (!isActive) togglePanel(pid, true);
                var cell = document.querySelector('.map-cell[data-panel-id="' + pid + '"]');
                if (cell) cell.scrollIntoView({behavior:'smooth', block:'center'});
            });
        });
        document.querySelectorAll('.hide-panel-btn').forEach(function(btn) {
            btn.addEventListener('click', function() {
                togglePanel(this.dataset.panelId, false);
            });
        });
    }

    function togglePanel(panelId, show) {
        var cell = document.querySelector('.map-cell[data-panel-id="' + panelId + '"]');
        var chip = document.querySelector('.panel-chip[data-panel-id="' + panelId + '"]');
        if (!cell) return;
        if (show) {
            cell.classList.remove('hidden-panel');
            if (chip) chip.classList.add('active');
            if (allMaps[panelId]) {
                setTimeout(function() { allMaps[panelId].invalidateSize(); }, 50);
            }
        } else {
            cell.classList.add('hidden-panel');
            if (chip) chip.classList.remove('active');
        }
    }

    function bindTabSwitching() {
        document.querySelectorAll('.theme-tab').forEach(function(tab) {
            tab.addEventListener('click', function(e) {
                e.preventDefault();
                document.querySelectorAll('.theme-tab').forEach(function(t) { t.classList.remove('active'); });
                this.classList.add('active');
                var target = this.dataset.tab;
                document.querySelectorAll('.tab-content').forEach(function(tc) {
                    tc.classList.toggle('active', tc.id === 'tab-' + target);
                });
                // Two-stage refit: 50ms catches the simple case, 350ms
                // catches tabs whose maps were initialized while hidden
                // (display:none → 0×0 container → fitBounds clamped). The
                // late refit recomputes from the now-laid-out container.
                // Native-zoom maps preserve their fixed zoom 0 instead of
                // refitting bounds.
                [50, 350].forEach(function(delay) {
                    setTimeout(function() {
                        Object.values(allMaps).forEach(function(m) {
                            m.invalidateSize();
                            if (m._nativeZoom) {
                                // Re-center if the cell size changed but
                                // keep zoom 0 so 1 bound-unit = 1 screen px.
                                if (m._imgBounds) {
                                    var b = m._imgBounds;
                                    m.setView([b[1][0] / 2, b[1][1] / 2], 0);
                                }
                            } else if (m._imgBounds) {
                                m.fitBounds(m._imgBounds);
                            }
                        });
                        // Now that the new tab is laid out, build any chart
                        // whose canvas finally has non-zero width.
                        if (typeof buildChartsFromState === 'function') {
                            buildChartsFromState();
                        }
                    }, delay);
                });
            });
        });
    }

    function bindSubTabSwitching() {
        document.querySelectorAll('.sub-tab').forEach(function(tab) {
            tab.addEventListener('click', function(e) {
                e.preventDefault();
                var nav = this.closest('.sub-tab-nav');
                nav.querySelectorAll('.sub-tab').forEach(function(t) { t.classList.remove('active'); });
                this.classList.add('active');
                var target = this.dataset.subtab;
                var parent = nav.parentElement;
                parent.querySelectorAll('.sub-tab-content').forEach(function(sc) {
                    sc.classList.toggle('active', sc.id === 'tab-' + target);
                });
                setTimeout(function() {
                    Object.values(allMaps).forEach(function(m) {
                        m.invalidateSize();
                        if (m._imgBounds) m.fitBounds(m._imgBounds);
                    });
                    // Charts inside sub-tabs (e.g. tab-fire, tab-grazing) only
                    // get a non-zero offsetWidth after the sub-tab is shown.
                    if (typeof buildChartsFromState === 'function') {
                        buildChartsFromState();
                    }
                }, 50);
            });
        });
    }

    // ── Embed mode ───────────────────────────────────────────────────

    function handleEmbedMode() {
        var params = new URLSearchParams(window.location.search);
        var isEmbed = params.get('embed') === '1' || window.self !== window.top;
        if (isEmbed) {
            var logo = document.querySelector('.des-logo');
            if (logo) logo.style.display = 'none';
            var divider = document.querySelector('.header-divider');
            if (divider) divider.style.display = 'none';
            var footer = document.querySelector('.footer');
            if (footer) footer.style.display = 'none';
        }
    }

    // ── LULC Chart initialization ────────────────────────────────────

    function initLulcCharts() {
        fetch('data/lulc-data.json')
            .then(function(r) { return r.json(); })
            .then(function(LULC_DATA) {
                var chartOpts = {
                    indexAxis: 'y',
                    responsive: true,
                    plugins: {legend: {display: false}},
                    scales: {
                        x: {beginAtZero:true, title:{display:true, text:'%'},
                            grid:{color:'rgba(255,255,255,0.04)'}},
                        y: {grid:{display:false}}
                    }
                };

                // Per-class IoU chart
                if (LULC_DATA.per_class_iou && LULC_DATA.per_class_iou.labels.length > 0) {
                    var iouCanvas = document.getElementById('chart-lulc-iou');
                    if (iouCanvas) {
                        new Chart(iouCanvas, {
                            type: 'bar',
                            data: {
                                labels: LULC_DATA.per_class_iou.labels,
                                datasets: [{
                                    label: 'IoU (%)',
                                    data: LULC_DATA.per_class_iou.values,
                                    backgroundColor: LULC_DATA.per_class_iou.colors,
                                    borderColor: LULC_DATA.per_class_iou.colors.map(function(c) {
                                        return c.replace('0.85','1');
                                    }),
                                    borderWidth: 1
                                }]
                            },
                            options: chartOpts
                        });
                    }
                }

                // Per-class accuracy chart
                if (LULC_DATA.per_class_accuracy && LULC_DATA.per_class_accuracy.labels.length > 0) {
                    var accCanvas = document.getElementById('chart-lulc-accuracy');
                    if (accCanvas) {
                        new Chart(accCanvas, {
                            type: 'bar',
                            data: {
                                labels: LULC_DATA.per_class_accuracy.labels,
                                datasets: [{
                                    label: 'Träffsäkerhet (%)',
                                    data: LULC_DATA.per_class_accuracy.values,
                                    backgroundColor: LULC_DATA.per_class_accuracy.colors,
                                    borderColor: LULC_DATA.per_class_accuracy.colors.map(function(c) {
                                        return c.replace('0.85','1');
                                    }),
                                    borderWidth: 1
                                }]
                            },
                            options: chartOpts
                        });
                    }
                }

                // Update summary cards with real data if available
                if (LULC_DATA.summary && LULC_DATA.summary.tiles > 0) {
                    var lulcTab = document.getElementById('tab-lulc');
                    if (lulcTab) {
                        var cards = lulcTab.querySelectorAll('.summary-card');
                        var s = LULC_DATA.summary;
                        if (cards.length >= 4) {
                            // Update high-conf wrong card
                            cards[1].querySelector('.value').textContent =
                                s.high_confidence_wrong.toLocaleString() + ' px';
                            // Update low-conf correct card
                            cards[2].querySelector('.value').textContent =
                                s.low_confidence_correct.toLocaleString() + ' px';
                            // Update tiles card
                            cards[3].querySelector('.value').textContent =
                                s.tiles.toString();
                        }
                    }
                }
            })
            .catch(function(e) { console.warn('Could not load LULC chart data:', e); });
    }

    // ── Chart initialization ─────────────────────────────────────────

    // Chart.js needs a canvas with non-zero width to acquire its 2D context.
    // initCharts() runs at boot, but the lulc/fire/grazing tabs may be hidden
    // (display:none) — calling new Chart on a 0-width canvas logs "Failed to
    // create chart: can't acquire context from the given item" and silently
    // does nothing. Same pattern as lilla-karlso-chart.js. Returns the new
    // chart instance or null when the canvas is missing/hidden.
    function safeChart(canvasId, configFactory) {
        var canvas = document.getElementById(canvasId);
        if (!canvas || canvas.offsetWidth === 0) return null;
        if (typeof Chart !== 'undefined' && Chart.getChart) {
            var existing = Chart.getChart(canvas);
            if (existing) existing.destroy();
        }
        return new Chart(canvas, configFactory());
    }

    var chartsState = { built: {} };  // canvasId → true once successfully built

    function initCharts() {
        fetch('data/chart-data.json')
            .then(function(r) { return r.json(); })
            .then(function(CHART_DATA) {
                Chart.defaults.color = 'rgba(207,248,228,0.6)';
                Chart.defaults.borderColor = 'rgba(207,248,228,0.1)';
                Chart.defaults.font.family = "'Space Grotesk', sans-serif";

                var chartOpts = {
                    indexAxis: 'y',
                    responsive: true,
                    plugins: {legend: {display: false}},
                    scales: {
                        x: {beginAtZero:true, title:{display:true, text:'Andel (%)'},
                            grid:{color:'rgba(255,255,255,0.04)'}},
                        y: {grid:{display:false}}
                    }
                };

                // Stash the loaded data so re-tries on tab-switch can build
                // charts whose canvases were 0-width at boot time.
                chartsState.data = CHART_DATA;
                chartsState.chartOpts = chartOpts;
                buildChartsFromState();
            })
            .catch(function(e) { console.warn('Could not load chart data:', e); });
    }

    // Build any chart whose canvas is now visible. safeChart() returns null
    // for missing or 0-width canvases, so we mark `built[id]=true` only on
    // success — re-running this on tab-switch finishes the unbuilt ones.
    function buildChartsFromState() {
        var CHART_DATA = chartsState.data;
        if (!CHART_DATA) return;
        var chartOpts = chartsState.chartOpts;
        var built = chartsState.built;

        // Change detection
        if (!built['chart-change'] && CHART_DATA.change && CHART_DATA.change.labels.length > 0) {
            var c = safeChart('chart-change', function() { return {
                type: 'bar',
                data: {
                    labels: CHART_DATA.change.labels,
                    datasets: [{
                        label: 'Förändringsandel (%)',
                        data: CHART_DATA.change.fractions,
                        backgroundColor: CHART_DATA.change.colors,
                        borderColor: CHART_DATA.change.borders,
                        borderWidth: 1
                    }]
                },
                options: chartOpts
            }; });
            if (c) built['chart-change'] = true;
        }

        // dNBR
        if (!built['chart-dnbr'] && CHART_DATA.change && CHART_DATA.change.dnbr) {
            var dnbrColors = CHART_DATA.change.dnbr.map(function(v) {
                if (v < -0.1) return 'rgba(26,152,80,0.85)';
                if (v < 0.1) return 'rgba(217,239,139,0.85)';
                if (v < 0.27) return 'rgba(254,224,139,0.85)';
                if (v < 0.44) return 'rgba(253,174,97,0.85)';
                if (v < 0.66) return 'rgba(244,109,67,0.85)';
                return 'rgba(215,48,39,0.85)';
            });
            var d = safeChart('chart-dnbr', function() { return {
                type: 'bar',
                data: {
                    labels: CHART_DATA.change.labels,
                    datasets: [{
                        label: 'Medel-dNBR',
                        data: CHART_DATA.change.dnbr,
                        backgroundColor: dnbrColors,
                        borderColor: dnbrColors.map(function(c) { return c.replace('0.85','1'); }),
                        borderWidth: 1
                    }]
                },
                options: {
                    indexAxis: 'y', responsive: true,
                    plugins: {legend:{display:false}},
                    scales: {
                        x: {title:{display:true, text:'Medel-dNBR'},
                            grid:{color:'rgba(255,255,255,0.04)'}},
                        y: {grid:{display:false}}
                    }
                }
            }; });
            if (d) built['chart-dnbr'] = true;
        }

        // Prithvi (canvas may not exist in current HTML — safeChart returns
        // null and we mark built so we don't keep retrying on every tab switch)
        if (!built['chart-prithvi'] && CHART_DATA.prithvi && CHART_DATA.prithvi.labels.length > 0) {
            var canvas = document.getElementById('chart-prithvi');
            if (!canvas) {
                built['chart-prithvi'] = true;  // canvas removed from HTML; stop trying
            } else {
                var p = safeChart('chart-prithvi', function() { return {
                    type: 'bar',
                    data: {
                        labels: CHART_DATA.prithvi.labels,
                        datasets: CHART_DATA.prithvi.classes.map(function(cls) {
                            return {
                                label: cls.label, data: cls.data,
                                backgroundColor: cls.color, borderColor: cls.border, borderWidth: 1
                            };
                        })
                    },
                    options: {
                        indexAxis: 'y', responsive: true,
                        plugins: {legend:{position:'top'}},
                        scales: {
                            x: {stacked:true, beginAtZero:true, max:100,
                                title:{display:true, text:'Andel (%)'},
                                grid:{color:'rgba(255,255,255,0.04)'}},
                            y: {stacked:true, grid:{display:false}}
                        }
                    }
                }; });
                if (p) built['chart-prithvi'] = true;
            }
        }

        // Grazing
        if (CHART_DATA.grazing) {
            var g = CHART_DATA.grazing;
            var gBarOpts = {
                indexAxis: 'y', responsive: true,
                plugins: {legend:{display:false}},
                scales: {
                    x: {beginAtZero:true, grid:{color:'rgba(255,255,255,0.04)'}},
                    y: {grid:{display:false}}
                }
            };

            if (!built['chart-grazing-class']) {
                var gc = safeChart('chart-grazing-class', function() { return {
                    type: 'bar',
                    data: {
                        labels: g.classification.labels,
                        datasets: [{
                            label: 'Antal block',
                            data: g.classification.counts,
                            backgroundColor: g.classification.colors,
                            borderColor: g.classification.borders,
                            borderWidth: 1
                        }]
                    },
                    options: Object.assign({}, gBarOpts, {
                        scales: Object.assign({}, gBarOpts.scales, {
                            x: {beginAtZero:true, title:{display:true, text:'Antal block'},
                                grid:{color:'rgba(255,255,255,0.04)'}}
                        })
                    })
                }; });
                if (gc) built['chart-grazing-class'] = true;
            }

            if (!built['chart-grazing-area']) {
                var ga = safeChart('chart-grazing-area', function() { return {
                    type: 'bar',
                    data: {
                        labels: g.classification.labels,
                        datasets: [{
                            label: 'Areal (ha)',
                            data: g.classification.areas,
                            backgroundColor: g.classification.colors,
                            borderColor: g.classification.borders,
                            borderWidth: 1
                        }]
                    },
                    options: Object.assign({}, gBarOpts, {
                        scales: Object.assign({}, gBarOpts.scales, {
                            x: {beginAtZero:true, title:{display:true, text:'Areal (ha)'},
                                grid:{color:'rgba(255,255,255,0.04)'}}
                        })
                    })
                }; });
                if (ga) built['chart-grazing-area'] = true;
            }

            if (!built['chart-grazing-conf']) {
                var gconf = safeChart('chart-grazing-conf', function() { return {
                    type: 'bar',
                    data: {
                        labels: g.confidence.labels,
                        datasets: [{
                            label: 'Antal block',
                            data: g.confidence.counts,
                            backgroundColor: g.confidence.colors,
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {legend:{display:false}},
                        scales: {
                            x: {title:{display:true, text:'Konfidens'},
                                grid:{color:'rgba(255,255,255,0.04)'}},
                            y: {beginAtZero:true, title:{display:true, text:'Antal block'},
                                grid:{color:'rgba(255,255,255,0.04)'}}
                        }
                    }
                }; });
                if (gconf) built['chart-grazing-conf'] = true;
            }

            if (!built['chart-grazing-nmd']) {
                var gnmd = safeChart('chart-grazing-nmd', function() { return {
                    type: 'bar',
                    data: {
                        labels: g.nmd_within_lpis.labels,
                        datasets: [{
                            label: 'Andel (%)',
                            data: g.nmd_within_lpis.fractions,
                            backgroundColor: g.nmd_within_lpis.colors,
                            borderColor: g.nmd_within_lpis.borders,
                            borderWidth: 1
                        }]
                    },
                    options: Object.assign({}, gBarOpts, {
                        scales: Object.assign({}, gBarOpts.scales, {
                            x: {beginAtZero:true, max:100, title:{display:true, text:'Andel (%)'},
                                grid:{color:'rgba(255,255,255,0.04)'}}
                        })
                    })
                }; });
                if (gnmd) built['chart-grazing-nmd'] = true;
            }
        }

        // L2
        if (!built['chart-l2'] && CHART_DATA.l2) {
            var l2 = safeChart('chart-l2', function() { return {
                type: 'bar',
                data: {
                    labels: CHART_DATA.l2.labels,
                    datasets: [{
                        label: 'Andel (%)', data: CHART_DATA.l2.fractions,
                        backgroundColor: CHART_DATA.l2.colors, borderWidth: 1
                    }]
                },
                options: chartOpts
            }; });
            if (l2) built['chart-l2'] = true;
        }
    }

    // ── Initialize a single tab ──────────────────────────────────────

    function initTab(tabId, config) {
        var container = document.querySelector('#tab-' + tabId + ' .tab-dynamic');
        if (!container) return Promise.resolve();

        // Render dynamic HTML
        container.innerHTML = renderTabDynamic(config);

        // Decorate first occurrence of every glossary term in this tab.
        // Walks both the dynamic block AND the sibling .tab-description
        // (handwritten prose in index.html) so acronyms first introduced
        // there also get the tooltip. Per-tab seen-set keeps it to one
        // decoration per term per tab.
        var subTabRoot = container.parentElement;
        if (subTabRoot) decorateGlossary(subTabRoot);

        // Collect GeoJSON files to load
        var geojsonFiles = {};
        var promises = [];
        config.panels.forEach(function(p) {
            if (p.vector && p.geojsonFile && !geojsonFiles[p.geojsonFile]) {
                geojsonFiles[p.geojsonFile] = true;
                promises.push(
                    loadGeoJSON(p.geojsonFile).then(function(data) {
                        return {key: p.geojsonFile, data: data};
                    })
                );
            }
        });

        return Promise.all(promises).then(function(results) {
            var geojsonMap = {_default: null};
            results.forEach(function(r) {
                if (r.data) geojsonMap[r.key] = r.data;
            });
            // Set default GeoJSON (first loaded one, for backward compat)
            if (results.length > 0 && results[0].data) {
                geojsonMap._default = results[0].data;
            }
            initMaps(config.panels, config.images, config.imgH, config.imgW, config.hasBgToggle, geojsonMap, config.nativeZoom);
        });
    }

    // ── Boot ─────────────────────────────────────────────────────────

    handleEmbedMode();
    ensureGlossCard();
    bindGlossHover();
    bindTabSwitching();
    bindSubTabSwitching();

    // Render and initialize all tabs
    var tabIds = Object.keys(TAB_CONFIG);
    var initPromises = tabIds.map(function(tabId) {
        return initTab(tabId, TAB_CONFIG[tabId]);
    });

    Promise.all(initPromises).then(function() {
        // Bind event handlers after all tabs are rendered
        bindOpacitySliders();
        bindBgToggles();
        bindPanelToggles();

        // Initialize charts (fire tab)
        if (TAB_CONFIG.fire && TAB_CONFIG.fire.hasCharts) {
            initCharts();
        }

        // LULC charts disabled — tab moved to fetch/training dashboard
        // if (TAB_CONFIG.lulc && TAB_CONFIG.lulc.hasCharts) {
        //     initLulcCharts();
        // }

        // License toggle
        var licBtn = document.querySelector('.license-toggle');
        if (licBtn) {
            licBtn.addEventListener('click', function() {
                var section = document.getElementById('license-info');
                if (section) {
                    section.classList.toggle('open');
                    this.textContent = section.classList.contains('open')
                        ? 'Dölj licenser' : 'Visa licenser och upphovsrätt';
                }
            });
        }
    });

})();
