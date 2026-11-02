// Showcase films: a collapsible player at the top of each showcase sub-tab
// (#tab-<case_id>) and the gallery on films.html (#film-gallery). One
// language choice applies to every player and is remembered per browser.
// Data and provenance: showcase/films/films.json and MANIFEST.json.
(function () {
    'use strict';

    var DATA_URL = 'showcase/films/films.json';
    var LANG_KEY = 'imint-film-lang';
    var LANGS = ['sv', 'en'];
    var LABEL = { sv: 'Svenska', en: 'English' };
    var TEXT = {
        sv: { watch: 'Se filmen', youtube: 'Se på YouTube', all: 'Alla filmer', fallback: 'Finns ännu bara på engelska.', seconds: 's' },
        en: { watch: 'Watch the film', youtube: 'Watch on YouTube', all: 'All films', fallback: 'Available in English only.', seconds: 's' }
    };

    var data = null;
    var lang = storedLang() || 'sv';
    var listeners = [];

    function storedLang() {
        try {
            var value = localStorage.getItem(LANG_KEY);
            return LANGS.indexOf(value) >= 0 ? value : null;
        } catch (e) {
            return null;
        }
    }

    function setLang(value) {
        lang = value;
        try { localStorage.setItem(LANG_KEY, value); } catch (e) { /* private mode: keep in memory */ }
        listeners.forEach(function (fn) { fn(); });
    }

    // The version actually shown: the chosen language if it exists, else the other one.
    function versionOf(film) {
        if (film.versions[lang]) return { lang: lang, v: film.versions[lang] };
        var other = LANGS.filter(function (l) { return film.versions[l]; })[0];
        return { lang: other, v: film.versions[other] };
    }

    function el(tag, attrs, children) {
        var node = document.createElement(tag);
        Object.keys(attrs || {}).forEach(function (k) {
            if (k === 'text') node.textContent = attrs[k];
            else node.setAttribute(k, attrs[k]);
        });
        (children || []).forEach(function (c) { if (c) node.appendChild(c); });
        return node;
    }

    function langToggle(film) {
        var group = el('div', { 'class': 'film-lang', role: 'group', 'aria-label': 'Språk / Language' });
        LANGS.forEach(function (l) {
            var btn = el('button', { type: 'button', 'data-lang': l, text: LABEL[l] });
            btn.addEventListener('click', function () { setLang(l); });
            group.appendChild(btn);
        });
        function sync() {
            group.querySelectorAll('button').forEach(function (b) {
                var l = b.getAttribute('data-lang');
                b.classList.toggle('active', l === lang);
                b.setAttribute('aria-pressed', String(l === lang));
                b.disabled = film ? !film.versions[l] : false;
            });
        }
        listeners.push(sync);
        sync();
        return group;
    }

    // Load one film version into a <video>; captions follow the spoken language.
    function load(video, film, autoplay) {
        var shown = versionOf(film);
        video.pause();
        video.replaceChildren();
        video.src = shown.v.mp4;
        video.poster = shown.v.poster;
        video.appendChild(el('track', { kind: 'captions', srclang: shown.lang, label: LABEL[shown.lang], src: shown.v.vtt }));
        if (autoplay) video.play().catch(function () { /* user presses play */ });
        return shown;
    }

    function noteFor(film, shown, withGalleryLink) {
        var t = TEXT[lang];
        var note = el('p', { 'class': 'film-note' });
        note.appendChild(document.createTextNode(lang === 'sv' ? data.voice : data.voice_en));
        if (shown.lang !== lang) note.appendChild(document.createTextNode(' · ' + t.fallback));
        if (shown.v.youtube) {
            note.appendChild(document.createTextNode(' · '));
            note.appendChild(el('a', { href: shown.v.youtube, rel: 'noopener', target: '_blank', text: t.youtube }));
        }
        if (withGalleryLink) {
            note.appendChild(document.createTextNode(' · '));
            note.appendChild(el('a', { href: 'films.html', text: t.all }));
        }
        return note;
    }

    function renderTabFilm(film) {
        var container = document.getElementById('tab-' + film.case_id);
        if (!container) return;
        var video = el('video', { controls: '', preload: 'none', playsinline: '' });
        var summaryText = el('span', { 'class': 'film-summary-text' });
        var note = el('div');
        var block = el('details', { 'class': 'tab-film' }, [
            el('summary', null, [summaryText]),
            el('div', { 'class': 'tab-film-body' }, [langToggle(film), video, note])
        ]);
        function sync() {
            var shown = load(video, film, false);
            summaryText.textContent = '▶ ' + TEXT[lang].watch + ' · ' + film.title[lang] + ' · '
                + Math.round(shown.v.seconds) + ' ' + TEXT[lang].seconds;
            note.replaceChildren(noteFor(film, shown, true));
        }
        listeners.push(sync);
        sync();
        block.addEventListener('toggle', function () { if (!block.open) video.pause(); });
        container.insertBefore(block, container.firstChild);
    }

    function renderGallery(root) {
        var title = el('h2', { 'class': 'film-player-title' });
        var video = el('video', { controls: '', preload: 'metadata', playsinline: '' });
        var note = el('div');
        var grid = el('div', { 'class': 'film-grid' });
        var current = 0;
        root.appendChild(el('section', { 'class': 'film-player', 'aria-label': 'Filmspelare' },
            [langToggle(null), title, video, note]));
        root.appendChild(grid);

        function select(index, autoplay) {
            current = index;
            var film = data.films[index];
            var shown = load(video, film, autoplay);
            title.textContent = film.title[lang];
            note.replaceChildren(noteFor(film, shown, false));
            grid.querySelectorAll('button').forEach(function (b, i) { b.setAttribute('aria-pressed', String(i === index)); });
        }
        function drawGrid() {
            grid.replaceChildren();
            data.films.forEach(function (film, i) {
                var shown = versionOf(film);
                var card = el('button', { type: 'button', 'class': 'film-card' }, [
                    el('img', { src: shown.v.poster, alt: '', loading: 'lazy' }),
                    el('span', { 'class': 'film-card-meta', text: String(i + 1).padStart(2, '0') + ' · ' + Math.round(shown.v.seconds) + ' s' }),
                    el('strong', { text: film.title[lang] })
                ]);
                card.addEventListener('click', function () { select(i, true); window.scrollTo({ top: 0, behavior: 'smooth' }); });
                grid.appendChild(card);
            });
        }
        listeners.push(function () { drawGrid(); select(current, false); });
        drawGrid();
        select(0, false);
    }

    function init() {
        fetch(DATA_URL)
            .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
            .then(function (json) {
                data = json;
                var gallery = document.getElementById('film-gallery');
                if (gallery) renderGallery(gallery);
                else data.films.forEach(renderTabFilm);
            })
            .catch(function (err) { console.warn('Showcase films unavailable:', err); });
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
