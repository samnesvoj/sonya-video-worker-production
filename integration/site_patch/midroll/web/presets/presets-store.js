/* ============================================================
   SONYA Пресеты — store (loaded eagerly by index.html).
   One document: { presets: [{id, name, settings}], default_id }.
   Signed in → GET/PUT /api/presets (server copy wins on load);
   guest → localStorage only. Both always mirror into localStorage,
   so the Cut constructor can apply the default look synchronously.
   settings = only fields the worker already understands:
     subtitleStyle viral|gradient|minimal|basic, quality standard|high,
     edu_template 1–4, streamer_layout 1–5, moment_types [], max_clips
     0(auto)|5|8|10|12, hook, hookTitle, jumpCuts, clipType (default mode),
     midroll (2026-11-22) {on, key, name, dur, w, h, audio, place mid|pre|post,
       at auto|after, after_s, every 1|2, fit auto|blur|fit|fill, max_total 0|60|90}
       — the user's own ad, uploaded once (/api/midroll/upload), signed-in only.
   ============================================================ */
(function () {
	'use strict';
	if (window.SonyaPresetStore) return;      // loaded by index.html and again as the app module: one store

	var KEY = 'sonya_presets_v1';
	var API = '/api/presets';
	var DEFAULTS = {
		clipType: '', subtitleStyle: 'viral', quality: 'standard', edu_template: 1, streamer_layout: 1,
		moment_types: [], max_clips: 0, hook: false, hookTitle: true, jumpCuts: true, midroll: null
	};
	var MIDROLL_JOB = ['key', 'place', 'at', 'after_s', 'every', 'fit', 'max_total', 'dur'];
	var HIGH_UNITS = 3;
	var HIGH_MODES = ['viral', 'storytelling', 'hooks'];

	var doc = read();
	var signedIn = null;           // null = unknown yet
	var listeners = [];
	var saveTimer = null;
	var loading = null;

	function read() {
		try {
			var d = JSON.parse(localStorage.getItem(KEY) || 'null');
			if (d && Array.isArray(d.presets)) return d;
		} catch (_) {}
		return { presets: [], default_id: null };
	}
	function mirror() { try { localStorage.setItem(KEY, JSON.stringify(doc)); } catch (_) {} }
	function emit() { listeners.forEach(function (fn) { try { fn(doc); } catch (_) {} }); }

	function uid() { return 'p' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6); }
	function clone(o) { return JSON.parse(JSON.stringify(o)); }
	function full(settings) {
		var s = clone(DEFAULTS);
		Object.keys(settings || {}).forEach(function (k) { if (k in DEFAULTS) s[k] = settings[k]; });
		return s;
	}

	function serverLoad() {
		if (loading) return loading;
		if (typeof fetch !== 'function') return Promise.resolve(doc);
		loading = fetch(API, { credentials: 'same-origin' }).then(function (r) {
			if (r.status === 401 || r.status === 403) { signedIn = false; return doc; }
			if (!r.ok) throw new Error('presets ' + r.status);
			return r.json().then(function (srv) {
				signedIn = true;
				if (srv && Array.isArray(srv.presets) && srv.presets.length) {
					doc = { presets: srv.presets, default_id: srv.default_id || null };
					mirror(); emit();
				} else if (doc.presets.length) {
					push();                       // first sign-in: the guest's presets go to the account
				}
				return doc;
			});
		}).catch(function () { return doc; });
		return loading;
	}

	function push() {
		if (signedIn !== true || typeof fetch !== 'function') return Promise.resolve();
		var body = JSON.stringify({ presets: doc.presets.map(function (p) { return { id: p.id, name: p.name, settings: p.settings }; }),
			default_id: doc.default_id });
		return fetch(API, { method: 'PUT', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, body: body })
			.then(function (r) { if (r.status === 401) signedIn = false; })
			.catch(function () {});
	}

	function commit() {
		mirror(); emit();
		clearTimeout(saveTimer);
		saveTimer = setTimeout(push, 500);
	}

	var S = {
		DEFAULTS: DEFAULTS, HIGH_UNITS: HIGH_UNITS, HIGH_MODES: HIGH_MODES,
		get doc() { return doc; },
		get signedIn() { return signedIn; },
		sync: serverLoad,
		onChange: function (fn) { listeners.push(fn); },
		full: full,
		find: function (id) { for (var i = 0; i < doc.presets.length; i++) if (doc.presets[i].id === id) return doc.presets[i]; return null; },
		getDefault: function () { return S.find(doc.default_id); },
		create: function (name, settings) {
			var p = { id: uid(), name: String(name || 'Мой стиль').trim().slice(0, 40) || 'Мой стиль', settings: full(settings) };
			doc.presets.push(p);
			if (!doc.default_id) doc.default_id = p.id;
			commit();
			return p;
		},
		update: function (id, patch) {
			var p = S.find(id);
			if (!p) return null;
			if (patch.name != null) p.name = String(patch.name).trim().slice(0, 40) || p.name;
			if (patch.settings) Object.keys(patch.settings).forEach(function (k) { if (k in DEFAULTS) p.settings[k] = patch.settings[k]; });
			commit();
			return p;
		},
		remove: function (id) {
			doc.presets = doc.presets.filter(function (p) { return p.id !== id; });
			if (doc.default_id === id) doc.default_id = doc.presets.length ? doc.presets[0].id : null;
			commit();
		},
		setDefault: function (id) { doc.default_id = S.find(id) ? id : null; commit(); },
		uniqueName: function (base) {
			var names = doc.presets.map(function (p) { return p.name; });
			if (names.indexOf(base) === -1) return base;
			for (var i = 2; i < 99; i++) if (names.indexOf(base + ' ' + i) === -1) return base + ' ' + i;
			return base;
		},
		/* Job params from a look, for the given mode (clipType). Only existing worker fields. */
		jobParams: function (settings, clipType) {
			var s = full(settings), out = { settings: {}, top: {} };
			var high = s.quality === 'high' && HIGH_MODES.indexOf(clipType) !== -1;
			out.settings.quality = high ? 'high' : 'standard';
			if (clipType === 'streamer') { out.settings.hookTitle = !!s.hookTitle; out.settings.jumpCuts = !!s.jumpCuts; }
			if (s.max_clips) out.top.max_clips = s.max_clips;
			if (s.hook && (clipType === 'viral' || clipType === 'storytelling')) out.top.hook = 1;
			out.top.midroll = S.midrollJob(s);
			return out;
		},
		/* params.midroll for a job: the preset's ad when it is on, else null (= no ad, nothing looked up later) */
		midrollJob: function (settings) {
			var m = settings && settings.midroll;
			if (!m || !m.key || m.on === false) return null;
			var o = {};
			MIDROLL_JOB.forEach(function (k) { if (m[k] != null) o[k] = m[k]; });
			return o;
		},
		midrollOn: function (settings) { var m = settings && settings.midroll; return !!(m && m.key && m.on !== false); },
		units: function (settings, clipType) {
			return full(settings).quality === 'high' && HIGH_MODES.indexOf(clipType) !== -1 ? HIGH_UNITS : 1;
		},
		LABELS: {
			subtitleStyle: { viral: 'Жёлтые', gradient: 'Градиент', minimal: 'Жемчуг', basic: 'Простые' },
			quality: { standard: 'Стандарт', high: 'Высокое — 2K' },
			clipType: { '': 'Не выбирать', viral: 'Вирал', educational: 'Образовательный', storytelling: 'Сторителлинг', hooks: 'Хуки', streamer: 'Стример' },
			max_clips: { 0: 'Авто', 5: 'До 5', 8: 'До 8', 10: 'До 10', 12: 'До 12' }
		}
	};
	window.SonyaPresetStore = S;
	window.addEventListener('storage', function (e) { if (e.key === KEY) { doc = read(); emit(); } });
})();
