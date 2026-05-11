'use strict';

// ── ERA5 prefilter showcase — frame gallery loader ─────────────────
// Reads showcase/era5_metafilter/frames/manifest.json and renders one
// card per scene with mean COT, sorted by ascending mean COT (clearest
// scenes first). Switching set-tabs re-renders the gallery in place.

(function () {
    var MANIFEST_URL = 'showcase/era5_metafilter/frames/manifest.json';
    var manifestCache = null;

    function fetchManifest() {
        if (manifestCache) {
            return Promise.resolve(manifestCache);
        }
        return fetch(MANIFEST_URL)
            .then(function (r) {
                if (!r.ok) throw new Error('manifest ' + r.status);
                return r.json();
            })
            .then(function (m) { manifestCache = m; return m; });
    }

    function thresholdLabel(meanCot, manifest) {
        // DES brand palette: green = clear, peach = thin cloud, red = thick.
        // Avoids the bright #27ae60/#f1c40f scheme that looked off-brand.
        if (meanCot < manifest.thin_cloud_threshold) {
            return { label: 'Klart', color: '#1a4338' };
        }
        if (meanCot < manifest.thick_cloud_threshold) {
            return { label: 'Tunt moln', color: '#fdd5c2' };
        }
        return { label: 'Tjockt moln', color: '#ff826c' };
    }

    function renderGallery(setName, manifest) {
        var gallery = document.getElementById('era5-frame-gallery');
        if (!gallery) return;

        var records = (manifest.sets && manifest.sets[setName]) || [];
        gallery.innerHTML = '';

        if (!manifest.sets || !(setName in manifest.sets)) {
            gallery.innerHTML =
                '<p style="color:#ff826c; grid-column:1/-1;">Set <code>' +
                setName + '</code> saknas i manifest — kör <code>python ' +
                'demos/era5_metafilter/render_frames.py</code> för att bygga om.</p>';
            return;
        }
        if (records.length === 0) {
            gallery.innerHTML =
                '<p style="color:#fdd5c2; grid-column:1/-1;">Set <code>' +
                setName + '</code> är tomt (0 scener efter filter).</p>';
            return;
        }

        records.forEach(function (rec) {
            var card = document.createElement('div');
            card.className = 'era5-frame-card';
            card.style.cssText =
                'background:#0f0f0f; border:1px solid #1f1f1f; border-radius:8px; ' +
                'overflow:hidden; display:flex; flex-direction:column;';

            var img = document.createElement('img');
            img.src = rec.frame_path;
            img.alt = setName + ' ' + rec.date + ' COT ' + rec.mean_cot;
            img.style.cssText = 'width:100%; height:auto; display:block;';
            img.loading = 'lazy';

            var meta = document.createElement('div');
            meta.style.cssText =
                'padding:10px 12px; display:flex; justify-content:space-between; ' +
                'align-items:center; gap:10px; font-size:13px; color:#e6e6e6;';

            var t = thresholdLabel(rec.mean_cot, manifest);
            meta.innerHTML =
                '<span style="font-variant-numeric:tabular-nums;">' +
                rec.date + ' &middot; mean COT <strong>' +
                rec.mean_cot.toFixed(4) + '</strong></span>' +
                '<span style="background:' + t.color + '20; color:' + t.color +
                '; padding:2px 9px; border-radius:999px; font-size:11px; ' +
                'text-transform:uppercase; letter-spacing:0.05em; ' +
                'font-weight:600;">' + t.label + '</span>';

            card.appendChild(img);
            card.appendChild(meta);
            gallery.appendChild(card);
        });
    }

    function bindTabs(manifest) {
        var tabs = document.querySelectorAll('.era5-set-tab');
        tabs.forEach(function (tab) {
            tab.addEventListener('click', function (ev) {
                ev.preventDefault();
                tabs.forEach(function (t) { t.classList.remove('active'); });
                tab.classList.add('active');
                renderGallery(tab.dataset.set, manifest);
            });
        });
    }

    function init() {
        var era5Tab = document.getElementById('tab-era5');
        if (!era5Tab) return;
        fetchManifest()
            .then(function (manifest) {
                // Default to whichever set has the active class on its tab
                var activeTab = document.querySelector('.era5-set-tab.active');
                var defaultSet = (activeTab && activeTab.dataset.set) ||
                                 'M4_era5_then_scl';
                renderGallery(defaultSet, manifest);
                bindTabs(manifest);
            })
            .catch(function (e) {
                var gallery = document.getElementById('era5-frame-gallery');
                if (gallery) {
                    gallery.innerHTML =
                        '<p style="color:#888;">Kunde inte ladda frame-manifest: ' +
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
