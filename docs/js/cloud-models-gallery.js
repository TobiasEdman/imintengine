'use strict';

// ── Cloud-models comparison sub-tab — frame loader ─────────────────
// Loads showcase/cloud_models/manifest.json and renders one full-width
// 5-panel frame per scene with stats below.

(function () {
    var MANIFEST_URL = 'showcase/cloud_models/manifest.json';

    function fmtPct(v) { return v == null ? '—' : (v * 100).toFixed(0) + '%'; }

    function renderFrames(manifest) {
        var host = document.getElementById('cloud-models-frames');
        if (!host) return;
        var frames = manifest.frames || [];
        if (!frames.length) {
            host.innerHTML = '<p style="color:#888;">Inga frames i manifest.</p>';
            return;
        }
        host.innerHTML = frames.map(function (fr) {
            var s = fr.stats || {};
            return '' +
                '<div style="background:#0f0f0f; border:1px solid #1f1f1f; border-radius:8px; overflow:hidden;">' +
                  '<img src="' + fr.frame_path + '" alt="' + fr.date + '" loading="lazy" style="width:100%; display:block;">' +
                  '<div style="padding:12px 16px; color:#e6e6e6; font-size:13px; display:flex; flex-wrap:wrap; gap:18px; align-items:center;">' +
                    '<span><b style="color:#fff;">' + fr.date + '</b></span>' +
                    '<span>SCL clouds <b style="color:#fff;">' + fmtPct(s.scl_cloud_frac) + '</b></span>' +
                    '<span>s2cloudless <b style="color:#fff;">' + fmtPct(s.s2cloudless_frac) + '</b></span>' +
                    '<span>OmniCloudMask thick+thin <b style="color:#fff;">' + fmtPct(s.omnicloudmask_thick_frac) + '</b></span>' +
                    '<span>MLP5 thick <b style="color:#fff;">' + fmtPct(s.mlp5_thick_frac) +
                      '</b> (mean COT ' + (s.mlp5_mean_cot != null ? s.mlp5_mean_cot.toFixed(4) : '—') + ')</span>' +
                  '</div>' +
                '</div>';
        }).join('');
    }

    function init() {
        var sub = document.getElementById('tab-era5_cloud_models');
        if (!sub) return;
        fetch(MANIFEST_URL)
            .then(function (r) { return r.json(); })
            .then(renderFrames)
            .catch(function (e) {
                var host = document.getElementById('cloud-models-frames');
                if (host) host.innerHTML =
                    '<p style="color:#888;">Kunde inte ladda manifest: ' + e.message + '</p>';
            });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
