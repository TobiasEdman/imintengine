'use strict';

// ── Wetland (Pirinen 2023) input-stack gallery loader ──────────────
// Reads showcase/wetland_pirinen/manifest.json and renders one card
// per layer (#3, #4, #6–#10) with frame thumbnail + status badge +
// stats. Status badges: "ok" (green), "saknas" (orange), "fel" (red).

(function () {
    var MANIFEST_URL = 'showcase/wetland_pirinen/manifest.json';

    function statusBadge(status) {
        switch (status) {
            case 'ok':
                return { label: 'OK', color: '#cff8e4' };
            case 'missing-data':
                return { label: 'Saknas', color: '#e67e22' };
            default:
                return { label: 'Fel', color: '#c0392b' };
        }
    }

    function renderStats(layer) {
        if (layer.status !== 'ok') {
            return '<span style="color:rgba(207,248,228,0.55);font-size:12px;">' +
                   'väntar på k8s/prefetch-nvv-aux-job</span>';
        }
        var s = layer.stats || {};
        return '<span style="font-variant-numeric:tabular-nums; color:rgba(207,248,228,0.7);">' +
               'mean ' + (s.mean !== undefined ? s.mean : '?') +
               ' ' + (layer.units || '') +
               ' &middot; land ' + (s.land_pct !== undefined ? s.land_pct : '?') + '%' +
               '</span>';
    }

    function renderGallery(manifest) {
        var gallery = document.getElementById('wetland-frame-gallery');
        if (!gallery) return;
        gallery.innerHTML = '';

        manifest.layers.forEach(function (layer) {
            var card = document.createElement('div');
            card.className = 'wetland-frame-card';
            // Match .summary-card palette (styles.css:135) so kort matchar
            // övriga tabbar — mörkgrön bg + ljus mint ribba till vänster.
            card.style.cssText =
                'background:#1a4338; border:1px solid #245045; border-radius:10px; ' +
                'overflow:hidden; display:flex; flex-direction:column; position:relative;';

            // 3-px mint accent ribbon på vänsterkanten (matchar .summary-card::before)
            var ribbon = document.createElement('span');
            ribbon.style.cssText =
                'position:absolute; left:0; top:0; bottom:0; width:3px; ' +
                'background:#cff8e4; z-index:1;';
            card.appendChild(ribbon);

            var img = document.createElement('img');
            img.src = layer.frame_path;
            img.alt = layer.title;
            img.style.cssText = 'width:100%; height:auto; display:block;';
            img.loading = 'lazy';

            var meta = document.createElement('div');
            meta.style.cssText =
                'padding:12px 16px; display:flex; flex-direction:column; gap:6px; ' +
                'font-size:13px; color:#cff8e4;';

            var head = document.createElement('div');
            head.style.cssText =
                'display:flex; justify-content:space-between; align-items:center; gap:10px;';
            var b = statusBadge(layer.status);
            head.innerHTML =
                '<span><strong>#' + layer.pirinen_idx + '</strong> ' +
                layer.title + '</span>' +
                '<span style="background:' + b.color + '20; color:' + b.color +
                '; padding:2px 9px; border-radius:999px; font-size:10px; ' +
                'text-transform:uppercase; letter-spacing:1.2px; ' +
                'font-weight:600;">' + b.label + '</span>';

            var statsLine = document.createElement('div');
            statsLine.style.cssText = 'font-size:12px;';
            statsLine.innerHTML = renderStats(layer);

            var src = document.createElement('div');
            src.style.cssText = 'font-size:11px; color:rgba(207,248,228,0.4);';
            src.textContent = layer.source;

            meta.appendChild(head);
            meta.appendChild(statsLine);
            meta.appendChild(src);
            card.appendChild(img);
            card.appendChild(meta);
            gallery.appendChild(card);
        });

        var summary = document.getElementById('wetland-summary');
        if (summary) {
            summary.innerHTML =
                '<strong>' + manifest.n_layers_ok + '</strong> ok &middot; ' +
                '<strong>' + manifest.n_layers_missing + '</strong> väntar på prefetch &middot; ' +
                '<strong>' + manifest.n_layers_error + '</strong> fel ' +
                '<span style="color:#888;">(' + manifest.elapsed_s + ' s totalt)</span>';
        }
    }

    function init() {
        var tab = document.getElementById('tab-wetland_pirinen');
        if (!tab) return;
        fetch(MANIFEST_URL)
            .then(function (r) {
                if (!r.ok) throw new Error('manifest ' + r.status);
                return r.json();
            })
            .then(renderGallery)
            .catch(function (e) {
                var gallery = document.getElementById('wetland-frame-gallery');
                if (gallery) {
                    gallery.innerHTML =
                        '<p style="color:#888;">Kunde inte ladda manifest: ' +
                        e.message + '</p>';
                }
            });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
