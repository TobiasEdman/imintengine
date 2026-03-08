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

    function renderTabDynamic(config) {
        var html = '<div class="section-header"><h2>' + config.title + '</h2></div>';
        html += renderSummaryCards(config.summary);
        html += '<div class="tab-intro"><p>' + config.intro + '</p></div>';
        html += renderPanelToolbar(config.panels);
        html += renderMapGrid(config.panels);
        return html;
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
        if (p.class_label) {
            layer.bindPopup(
                '<b>Block ' + (p.blockid || '') + '</b><br>' +
                p.class_label + ' (' + Math.round((p.confidence||0)*100) + '%)'
            );
        }
    }

    function makeGeoJSON(gjData, map) {
        return L.geoJSON(gjData, {
            style: gjStyle,
            onEachFeature: gjPopup,
            coordsToLatLng: function(coords) { return L.latLng(coords[1], coords[0]); }
        }).addTo(map);
    }

    // ── Map initialization ───────────────────────────────────────────

    function initMaps(panels, images, imgH, imgW, hasBgToggle, geojsonMap) {
        var bounds = [[0, 0], [imgH, imgW]];
        var maps = [];

        panels.forEach(function(panel) {
            var container = document.getElementById(panel.id);
            var isVector = panel.vector || false;
            if (!container || (!images[panel.id] && !isVector)) return;

            var map = L.map(panel.id, {
                crs: L.CRS.Simple,
                minZoom: -2,
                maxZoom: 5,
                attributionControl: false,
                zoomSnap: 0.25
            });

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
                    allOverlays[panel.id] = makeGeoJSON(gjData, map);
                }
            } else {
                allOverlays[panel.id] = L.imageOverlay(images[panel.id], bounds, {zIndex:1}).addTo(map);
            }

            map.fitBounds(bounds);
            map._imgBounds = bounds;
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
                setTimeout(function() {
                    Object.values(allMaps).forEach(function(m) {
                        m.invalidateSize();
                        if (m._imgBounds) m.fitBounds(m._imgBounds);
                    });
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

    // ── Chart initialization ─────────────────────────────────────────

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

                // Change detection chart
                if (CHART_DATA.change && CHART_DATA.change.labels.length > 0) {
                    new Chart(document.getElementById('chart-change'), {
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
                    });

                    // dNBR chart
                    if (CHART_DATA.change.dnbr) {
                        var dnbrColors = CHART_DATA.change.dnbr.map(function(v) {
                            if (v < -0.1) return 'rgba(26,152,80,0.85)';
                            if (v < 0.1) return 'rgba(217,239,139,0.85)';
                            if (v < 0.27) return 'rgba(254,224,139,0.85)';
                            if (v < 0.44) return 'rgba(253,174,97,0.85)';
                            if (v < 0.66) return 'rgba(244,109,67,0.85)';
                            return 'rgba(215,48,39,0.85)';
                        });
                        new Chart(document.getElementById('chart-dnbr'), {
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
                        });
                    }
                }

                // Prithvi chart
                if (CHART_DATA.prithvi && CHART_DATA.prithvi.labels.length > 0) {
                    new Chart(document.getElementById('chart-prithvi'), {
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
                    });
                }

                // L2 chart
                if (CHART_DATA.l2) {
                    new Chart(document.getElementById('chart-l2'), {
                        type: 'bar',
                        data: {
                            labels: CHART_DATA.l2.labels,
                            datasets: [{
                                label: 'Andel (%)', data: CHART_DATA.l2.fractions,
                                backgroundColor: CHART_DATA.l2.colors, borderWidth: 1
                            }]
                        },
                        options: chartOpts
                    });
                }
            })
            .catch(function(e) { console.warn('Could not load chart data:', e); });
    }

    // ── Initialize a single tab ──────────────────────────────────────

    function initTab(tabId, config) {
        var container = document.querySelector('#tab-' + tabId + ' .tab-dynamic');
        if (!container) return Promise.resolve();

        // Render dynamic HTML
        container.innerHTML = renderTabDynamic(config);

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
            initMaps(config.panels, config.images, config.imgH, config.imgW, config.hasBgToggle, geojsonMap);
        });
    }

    // ── Boot ─────────────────────────────────────────────────────────

    handleEmbedMode();
    bindTabSwitching();

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
