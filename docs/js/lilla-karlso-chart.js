'use strict';

// ── Lilla Karlsö chl-a tidsserie ───────────────────────────────────────
// Läser showcase/lilla_karlso_birds/timeseries.json och renderar
// chl_p50 + chl_p90 i en line-chart (canvas#chart-lilla-karlso-chl).
// Aktiveras när användaren klickar in på Vattenkvalitet → Lilla Karlsö-
// sub-tabben (Chart.js skall vara initierad lazy så vi inte kraschar
// om manifestet saknas).

(function () {
    var MANIFEST_URL = 'showcase/lilla_karlso_birds/timeseries.json';
    var initialized = false;

    function init() {
        if (initialized) return;
        var canvas = document.getElementById('chart-lilla-karlso-chl');
        if (!canvas) return;
        if (typeof Chart === 'undefined') return;
        // Chart.js kräver synlig canvas (offsetWidth > 0) för att kunna acquire
        // context. Om sub-tabben inte är synlig än, vänta — annars misslyckas
        // chart-skapandet tyst med "can't acquire context from the given item"
        // och initialized=true blockerar all framtida retry.
        if (canvas.offsetWidth === 0) return;
        // Reuse: om en (failad) chart fortfarande är registrerad, släng den
        // så vi kan skapa en ny mot samma canvas.
        var existing = Chart.getChart(canvas);
        if (existing) existing.destroy();
        initialized = true;

        fetch(MANIFEST_URL)
            .then(function (r) {
                if (!r.ok) throw new Error('manifest ' + r.status);
                return r.json();
            })
            .then(function (data) {
                var records = data.records || [];
                if (records.length === 0) {
                    canvas.parentElement.innerHTML =
                        '<p style="color:#888;">Inga records — render-jobbet har inte körts ännu.</p>';
                    return;
                }

                var labels = records.map(function (r) { return r.date; });
                var p50 = records.map(function (r) { return r.chl_p50; });
                var p90 = records.map(function (r) { return r.chl_p90; });

                new Chart(canvas, {
                    type: 'line',
                    data: {
                        labels: labels,
                        datasets: [
                            {
                                label: 'chl-a p90 (mg/m³)',
                                data: p90,
                                borderColor: '#ff826c',  // DES red
                                backgroundColor: 'rgba(255,130,108,0.15)',
                                borderWidth: 2.5,
                                pointRadius: 5,
                                pointHoverRadius: 7,
                                fill: true,
                                tension: 0.3,
                            },
                            {
                                label: 'chl-a p50 (mg/m³)',
                                data: p50,
                                borderColor: '#1a4338',  // DES dark green
                                backgroundColor: 'rgba(26,67,56,0.15)',
                                borderWidth: 2.5,
                                pointRadius: 5,
                                pointHoverRadius: 7,
                                fill: true,
                                tension: 0.3,
                            },
                        ],
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: { mode: 'index', intersect: false },
                        plugins: {
                            legend: {
                                position: 'top',
                                labels: { color: '#1f2937', font: { size: 13 } },
                            },
                            tooltip: {
                                callbacks: {
                                    label: function (ctx) {
                                        var v = ctx.parsed.y;
                                        return ctx.dataset.label + ': ' +
                                               (v !== null ? v.toFixed(2) : 'n/a');
                                    },
                                },
                            },
                        },
                        scales: {
                            x: {
                                title: { display: true, text: 'Datum', color: '#374151' },
                                ticks: { color: '#374151' },
                                grid: { color: 'rgba(0,0,0,0.05)' },
                            },
                            y: {
                                type: 'logarithmic',
                                title: { display: true, text: 'Klorofyll-a (mg/m³, log)', color: '#374151' },
                                ticks: { color: '#374151' },
                                grid: { color: 'rgba(0,0,0,0.05)' },
                                min: 0.5,
                                max: 30,
                            },
                        },
                    },
                });
            })
            .catch(function (e) {
                canvas.parentElement.innerHTML =
                    '<p style="color:#888;">Kunde inte ladda timeseries: ' +
                    e.message + '</p>';
            });
    }

    // Initialiserar när Lilla Karlsö-sub-tabben blir aktiv. Lyssnar på
    // sub-tab-klick eftersom Chart.js behöver synlig canvas (offsetWidth).
    function bindSubTabHook() {
        document.querySelectorAll('[data-subtab="water_quality_lilla_karlso"]')
            .forEach(function (tab) {
                tab.addEventListener('click', function () {
                    setTimeout(init, 100);  // Vänta på att DOM uppdaterar
                });
            });
        // Om sidan laddas direkt med sub-tabben aktiv (via URL hash el.dyl.)
        if (document.querySelector('#tab-water_quality_lilla_karlso.active')) {
            setTimeout(init, 100);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', bindSubTabHook);
    } else {
        bindSubTabHook();
    }
})();
