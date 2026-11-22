/* ============================================================
   SONYA Пресеты — the look configurator (App Library → «Пресеты»).
   Mounted by the App Shell into #/presets: window.SonyaPresets.mount(el).
   Organised like Apple Settings: a list of sections on the left, glass
   cards with live previews on the right; every change applies at once
   (no Save button) to the preset being edited. Deep links:
   #/presets/<section>. Built from the site's own parts only: Справочник
   layout (.gd-*), Trailer materials (.tr-*: glass, segments, chips,
   switches, active graphite plate). Content rule: only what the worker
   and the site already do; anything else is labelled «Скоро».
   ============================================================ */
(function () {
	'use strict';

	var P = (window.SonyaPresets = window.SonyaPresets || {});
	var store = window.SonyaPresetStore;
	var V = '20261120-presets';     // preview images (unchanged by the mid-roll release)
	var IMG = '/assets/presets/';

	var SECTIONS = [
		{ id: 'style', label: 'Мой стиль', icon: 'fa-sliders' },
		{ id: 'subtitles', label: 'Субтитры', icon: 'fa-closed-captioning' },
		{ id: 'effects', label: 'Эффекты', icon: 'fa-wand-magic-sparkles' },
		{ id: 'modes', label: 'Режимы и шаблоны', icon: 'fa-layer-group' },
		{ id: 'quality', label: 'Качество', icon: 'fa-gem' },
		{ id: 'voices', label: 'Озвучка', icon: 'fa-microphone-lines' },
		{ id: 'library', label: 'Мои пресеты', icon: 'fa-bookmark' }
	];

	var SUBS = [
		{ v: 'viral', name: 'Жёлтые', meta: 'Dela Gothic · слово подсвечивается', img: 'sub_viral' },
		{ v: 'gradient', name: 'Градиент', meta: 'Фиолетовый → розовый → оранжевый', img: 'sub_gradient' },
		{ v: 'minimal', name: 'Жемчуг', meta: 'Жемчуг и шампань, спокойно', img: 'sub_minimal' },
		{ v: 'basic', name: 'Простые', meta: 'Белые, тонкие, внизу', img: 'sub_basic' }
	];
	var MODES = [
		{ v: 'viral', name: 'Вирал', desc: 'Эмоциональные пики: реакция, спор, смех, резкий поворот.' },
		{ v: 'educational', name: 'Образовательный', desc: 'Законченная мысль и объяснение. Своё оформление: 4 шаблона.' },
		{ v: 'storytelling', name: 'Сторителлинг', desc: 'Мини-истории с началом и развязкой, заголовок сверху.' },
		{ v: 'hooks', name: 'Хуки', desc: 'Те же клипы, что в Вирале, а в начале — 1,5–3 с самого сильного момента.' },
		{ v: 'streamer', name: 'Стример', desc: 'Лучшие моменты стрима: вебка + игра, 5 раскладок, типы моментов.' }
	];
	var EDU = [[1, 'Чистый'], [2, 'Ночь'], [3, 'Яркий'], [4, 'Журнал']];
	var LAYOUTS = [[1, 'Вебка сверху'], [2, 'Игра + кружок'], [3, 'Вебка крупно'], [4, 'Игра + окно', '?v=20261119-tpl4'], [5, 'Игра + рамка']];
	var MOMENTS = [['philosophy', 'Философия / мотивация'], ['zavoz', 'Завоз'], ['comment', 'Завозный коммент'], ['chaos', 'Хаос'],
		['guest', 'Гость с подводкой'], ['entrance', 'Выход в образе'], ['joke', 'Пошлая шутка'], ['conflict', 'Конфликт']];
	var VOICES = [
		{ name: 'Fish Audio Drama 3', desc: 'Живая актёрская подача: эмоции, паузы, шёпот и смех.' },
		{ name: 'VoxCPM2', desc: 'Естественный голос рассказчика, клон по короткому образцу.' },
		{ name: 'Qwen3-TTS', desc: 'Быстрая чистая озвучка на русском и других языках.' }
	];

	var root, navEl, docEl, current = 'style', editing = null;

	function esc(s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }
	function img(name) { return IMG + name + '.webp?v=' + V; }
	function find(id) { for (var i = 0; i < SECTIONS.length; i++) if (SECTIONS[i].id === id) return SECTIONS[i]; return null; }
	P.sectionFromHash = function (h) { var m = /^#\/presets\/([a-z-]+)/.exec(h || ''); return m && find(m[1]) ? m[1] : 'style'; };
	P.hashFor = function (id) { return id === 'style' ? '#/presets' : '#/presets/' + id; };

	/* The preset being edited: the default one, else the first, else a fresh «Мой стиль». */
	function cur() {
		var p = store.find(editing) || store.getDefault() || store.doc.presets[0];
		if (!p) p = store.create('Мой стиль', {});
		editing = p.id;
		return p;
	}
	function s() { return store.full(cur().settings); }
	function set(k, v) { var o = {}; o[k] = v; store.update(cur().id, { settings: o }); }

	/* ── Small parts ────────────────────────────────────────── */
	function seg(name, value, opts) {
		return '<div class="tr-seg ps-seg" role="radiogroup">' + opts.map(function (o) {
			return '<label class="tr-seg-opt"><input type="radio" name="' + name + '" value="' + esc(o[0]) + '"' + (String(o[0]) === String(value) ? ' checked' : '') +
				'><span>' + esc(o[1]) + '</span></label>';
		}).join('') + '</div>';
	}
	function row(label, sub, control, cls) {
		return '<div class="ps-row' + (cls ? ' ' + cls : '') + '"><div class="ps-row-text"><span class="ps-row-label">' + esc(label) + '</span>' +
			(sub ? '<span class="ps-row-sub">' + sub + '</span>' : '') + '</div>' + control + '</div>';
	}
	function sw(k, on) { return '<input type="checkbox" class="tr-switch" data-k="' + k + '"' + (on ? ' checked' : '') + ' aria-label="Включить">'; }
	function block(title, inner, extra) {
		return '<section class="tr-surface gd-block ps-block' + (extra || '') + '"><h3 class="gd-block-title">' + esc(title) + '</h3>' + inner + '</section>';
	}
	function hero(title, lead, eyebrow) {
		return '<header class="tr-surface gd-hero ps-hero"><p class="gd-eyebrow">' + esc(eyebrow || 'Пресет · ' + cur().name) + '</p>' +
			'<h2 class="gd-h" id="gd-h" tabindex="-1">' + esc(title) + '</h2>' + (lead ? '<p class="gd-lead">' + esc(lead) + '</p>' : '') + '</header>';
	}
	function card(name, value, checked, picture, title, meta, tag) {
		return '<label class="ps-card"><input type="radio" name="' + name + '" value="' + esc(value) + '"' + (checked ? ' checked' : '') + '>' +
			'<span class="ps-card-pic">' + picture + (tag ? '<span class="ps-card-tag">' + esc(tag) + '</span>' : '') +
			'<span class="ps-card-check" aria-hidden="true"><i class="fa-solid fa-check"></i></span></span>' +
			'<span class="ps-card-name">' + esc(title) + '</span>' + (meta ? '<span class="ps-card-meta">' + esc(meta) + '</span>' : '') + '</label>';
	}
	function pic(src, alt) { return '<img src="' + src + '" alt="' + esc(alt || '') + '" width="360" height="640" loading="lazy" decoding="async">'; }
	function soon(title, text) {
		return '<div class="ps-soon"><div class="ps-row-text"><span class="ps-row-label">' + esc(title) + ' <span class="gd-tag">Скоро</span></span>' +
			'<span class="ps-row-sub">' + esc(text) + '</span></div></div>';
	}
	function subName(v) { return store.LABELS.subtitleStyle[v] || v; }
	function summary(st) {
		var bits = [subName(st.subtitleStyle), store.LABELS.quality[st.quality]];
		if (st.hook) bits.push('крючок');
		if (store.midrollOn(st)) bits.push('мидролл');
		if (st.clipType) bits.push(store.LABELS.clipType[st.clipType]);
		return bits.join(' · ');
	}

	/* ── «Мидролл»: the user's own ad inside every clip (2026-11-22) ── */
	var MR_MAX_MB = 100;
	var mrUrls = {};            // key -> {video_url, poster_url} (fresh presigned links, never stored)
	var mrBusy = null;          // {pct, name} while uploading; {error} after a failed upload
	function mr() { var m = s().midroll; return m && m.key ? m : null; }
	function mrSet(patch) { var m = mr(); if (m) set('midroll', Object.assign({}, m, patch)); }
	function secs(v) { return (Math.round(v * 10) / 10).toString().replace('.', ',') + ' с'; }
	function shape(m) {
		if (!m.w || !m.h) return '';
		var r = m.w / m.h;
		return r < 0.8 ? 'вертикальный' : r > 1.25 ? 'горизонтальный' : 'квадратный';
	}
	function mrPlace(m) { return m.place === 'pre' || m.place === 'post' ? m.place : (m.at === 'after' ? 'after' : 'auto'); }
	function mrLoadUrls(m) {
		if (!m || mrUrls[m.key] || typeof fetch !== 'function') return;
		mrUrls[m.key] = { pending: true };
		fetch('/api/midroll/url?key=' + encodeURIComponent(m.key), { credentials: 'same-origin' })
			.then(function (r) { return r.ok ? r.json() : null; })
			.then(function (j) { mrUrls[m.key] = j || { missing: true }; if (current === 'effects') render(false); })
			.catch(function () { delete mrUrls[m.key]; });
	}
	function mrPreview(m) {
		var u = mrUrls[m.key] || {};
		var bg = u.poster_url ? '<img class="ps-mr-bg" src="' + esc(u.poster_url) + '" alt="" aria-hidden="true">' : '';
		var media = u.video_url
			? '<video class="ps-mr-video" src="' + esc(u.video_url) + '"' + (u.poster_url ? ' poster="' + esc(u.poster_url) + '"' : '') +
				' preload="none" playsinline loop aria-label="Ваш рекламный ролик"></video>' +
				'<button type="button" class="ps-mr-play" data-act="mr-play" aria-label="Смотреть ролик"><i class="fa-solid fa-play" aria-hidden="true"></i></button>'
			: '<span class="ps-mr-ph" aria-hidden="true"><i class="fa-solid fa-film"></i></span>';
		var fit = m.fit === 'fill' ? ' is-fill' : m.fit === 'fit' ? ' is-fit' : '';
		return '<div class="ps-mr-phone' + fit + '">' + bg + media + '<span class="ps-mr-badge">Реклама · ' + esc(secs(m.dur || 0)) + '</span></div>';
	}
	function mrWarn(m) {
		if (!m.dur) return '';
		var t = Math.round(55 + m.dur - 0.2);
		var lim = m.max_total === 60 ? 'Сейчас: «До 60 с» — в клипы, которые станут длиннее, ролик не вставится.'
			: m.max_total === 90 ? 'Сейчас: «До 90 с» — в клипы, которые станут длиннее, ролик не вставится.'
			: 'Нужно строго до 60 или 90 с — выберите лимит ниже: в слишком длинные клипы ролик не вставится.';
		return '<p class="gd-callout ps-mr-warn"><i class="fa-solid fa-triangle-exclamation" aria-hidden="true"></i> ' +
			'Клип станет длиннее на ' + esc(secs(m.dur)) + ': например, 55 с → ' + t + ' с. Shorts, Reels и TikTok принимают и такие, ' +
			'но в некоторых местах (реклама, старые лимиты) ждут до 60 или 90 с. ' + esc(lim) + '</p>';
	}
	function midrollBlock() {
		var m = mr(), on = m && m.on !== false;
		if (m) mrLoadUrls(m);
		var head = row('Мидролл', 'Ваш рекламный ролик внутри каждого клипа — в паузе между фразами, ближе к середине, с мягким переходом. ' +
			'Субтитры на время ролика скрываются, громкость выравнивается к клипу (−14 LUFS), кадр подгоняется под 9:16.', m ? sw('mr-on', on) : '');
		var up;
		if (mrBusy && mrBusy.pct != null) {
			up = '<div class="ps-mr-drop is-busy" role="status"><span class="ps-row-label">Загружаем «' + esc(mrBusy.name) + '»…</span>' +
				'<span class="ps-mr-bar"><span style="transform: scaleX(' + (mrBusy.pct / 100).toFixed(3) + ')"></span></span></div>';
		} else if (!m) {
			up = store.signedIn === false
				? '<div class="ps-mr-drop"><span class="ps-row-sub">Войдите в аккаунт — ролик сохранится в пресете, и загружать его каждый раз не придётся.</span></div>'
				: '<label class="ps-mr-drop" data-mr-drop tabindex="0" role="button" aria-label="Загрузить рекламный ролик"><input type="file" accept="video/*,.mp4,.mov,.webm,.mkv,.m4v" data-mr-file hidden>' +
					'<span class="ps-mr-icon" aria-hidden="true"><i class="fa-solid fa-arrow-up-from-bracket"></i></span>' +
					'<span class="ps-row-text"><span class="ps-row-label">Загрузить рекламный ролик</span>' +
					'<span class="ps-row-sub">Обычно 3–15 с, максимум 30 с и ' + MR_MAX_MB + ' МБ. Любой формат: вертикальный или горизонтальный.</span></span></label>';
		} else {
			up = '<div class="ps-mr-card">' + mrPreview(m) +
				'<div class="ps-mr-meta"><span class="ps-row-label ps-mr-name">' + esc(m.name || 'Рекламный ролик') + '</span>' +
				'<span class="ps-row-sub">' + esc([secs(m.dur || 0), shape(m), m.audio === false ? 'без звука' : 'со звуком'].filter(Boolean).join(' · ')) + '</span>' +
				'<div class="ps-mr-actions"><label class="tr-pillbtn ps-btn ps-mr-btn" tabindex="0" role="button"><input type="file" accept="video/*,.mp4,.mov,.webm,.mkv,.m4v" data-mr-file hidden>' +
					'<i class="fa-solid fa-arrow-rotate-right" aria-hidden="true"></i> Заменить</label>' +
				'<button type="button" class="tr-pillbtn ps-btn ps-mr-btn" data-act="mr-remove"><i class="fa-regular fa-trash-can" aria-hidden="true"></i> Убрать</button></div></div></div>';
		}
		var err = mrBusy && mrBusy.error ? '<p class="ps-mr-error" role="alert">' + esc(mrBusy.error) + '</p>' : '';
		var opts = '';
		if (m) {
			var pl = mrPlace(m);
			opts = '<div class="ps-mr-opts' + (on ? '' : ' is-off') + '">' +
				row('Где вставлять', pl === 'auto' ? 'В естественной паузе ближе к середине клипа, не обрывая фразу.'
					: pl === 'after' ? 'В первой паузе после выбранной секунды (конец фразы — в приоритете).'
					: pl === 'pre' ? 'Пре-ролл: ролик перед клипом, переход растворением.' : 'Пост-ролл: ролик сразу после клипа.',
					seg('ps-mr-place', pl, [['auto', 'В середине'], ['after', 'После N с'], ['pre', 'В начале'], ['post', 'В конце']]), 'ps-row--stack') +
				(pl === 'after' ? row('Через сколько секунд', '', seg('ps-mr-after', m.after_s || 20, [[10, '10 с'], [15, '15 с'], [20, '20 с'], [30, '30 с'], [45, '45 с']]), 'ps-row--stack') : '') +
				row('Как часто', 'Через один — в 1-й, 3-й, 5-й… клип.', seg('ps-mr-every', m.every === 2 ? 2 : 1, [[1, 'Во все клипы'], [2, 'Через один']]), 'ps-row--stack') +
				row('Кадр', 'Авто: свой формат — во весь кадр, другой — по центру на размытом фоне из самого ролика.',
					seg('ps-mr-fit', m.fit || 'auto', [['auto', 'Авто'], ['blur', 'Размытый фон'], ['fit', 'Вписать'], ['fill', 'Обрезать']]), 'ps-row--stack') +
				row('Длина клипа с роликом', '', seg('ps-mr-max', m.max_total || 0, [[0, 'Не ограничивать'], [60, 'До 60 с'], [90, 'До 90 с']]), 'ps-row--stack') +
				mrWarn(m) + '</div>';
		}
		return '<div class="ps-mr">' + head + up + err + opts + '</div>';
	}
	function mrUpload(file) {
		if (!file) return;
		if (file.size > MR_MAX_MB * 1024 * 1024) { mrBusy = { error: 'Файл больше ' + MR_MAX_MB + ' МБ. Сожмите ролик или укоротите его до 3–15 с.' }; render(false); return; }
		var fd = new FormData();
		fd.append('file', file, file.name);
		var x = new XMLHttpRequest();
		mrBusy = { pct: 0, name: file.name };
		render(false);
		x.open('POST', '/api/midroll/upload');
		x.withCredentials = true;
		x.upload.onprogress = function (e) {
			if (!e.lengthComputable) return;
			mrBusy.pct = Math.min(99, Math.round(e.loaded / e.total * 100));
			var bar = docEl.querySelector('.ps-mr-bar span');
			if (bar) bar.style.transform = 'scaleX(' + (mrBusy.pct / 100).toFixed(3) + ')';
		};
		x.onload = function () {
			var j = null;
			try { j = JSON.parse(x.responseText); } catch (_) {}
			if (x.status >= 200 && x.status < 300 && j && j.key) {
				var prev = mr() || {};
				mrUrls[j.key] = { video_url: j.video_url, poster_url: j.poster_url };
				set('midroll', { on: true, key: j.key, name: j.name, dur: j.dur, w: j.w, h: j.h, audio: j.audio,
					place: prev.place || 'mid', at: prev.at || 'auto', after_s: prev.after_s || 20, every: prev.every || 1,
					fit: prev.fit || 'auto', max_total: prev.max_total || 0 });
				mrBusy = null;
				render(false);
				toast('Ролик сохранён в пресете «' + cur().name + '»');
				return;
			}
			var d = j && (j.detail || j);
			mrBusy = { error: (d && d.message) || (x.status === 401 ? 'Войдите в аккаунт, чтобы загрузить ролик.'
				: x.status === 413 ? 'Файл слишком большой: максимум ' + MR_MAX_MB + ' МБ.'
				: d && d.error === 'unsupported_format' ? 'Нужен видеофайл: MP4, MOV, WEBM или MKV.'
				: 'Не получилось загрузить ролик. Попробуйте ещё раз.') };
			render(false);
		};
		x.onerror = function () { mrBusy = { error: 'Нет связи с сервером. Попробуйте ещё раз.' }; render(false); };
		x.send(fd);
	}

	/* ── Sections ───────────────────────────────────────────── */
	var VIEWS = {
		style: function () {
			var st = s(), sub = SUBS.filter(function (x) { return x.v === st.subtitleStyle; })[0] || SUBS[0];
			var isDef = store.doc.default_id === cur().id;
			function link(sec, label, value) {
				return '<button type="button" class="ps-link" data-sec="' + sec + '"><span class="ps-row-label">' + esc(label) + '</span>' +
					'<span class="ps-link-value">' + esc(value) + '</span><i class="fa-solid fa-chevron-right" aria-hidden="true"></i></button>';
			}
			return '<header class="tr-surface gd-hero ps-hero ps-hero--style">' +
					'<div class="ps-phone" aria-hidden="true"><img src="' + img(sub.img) + '" alt="" width="360" height="640">' +
						(st.quality === 'high' ? '<span class="ps-phone-badge">2K</span>' : '') + '</div>' +
					'<div class="ps-hero-text"><p class="gd-wordmark">SONYA</p>' +
						'<h2 class="gd-h gd-h--intro" id="gd-h" tabindex="-1">' + esc(cur().name) + '</h2>' +
						'<p class="gd-lead">Ваш стиль нарезки: субтитры, эффекты, шаблоны и качество. ' +
						(isDef ? 'Он подставляется в каждую новую нарезку.' : 'Сделайте его основным — и он будет подставляться в новые нарезки.') + '</p>' +
						'<div class="ps-hero-actions">' +
							(isDef ? '<span class="ps-default"><i class="fa-solid fa-check" aria-hidden="true"></i> Основной стиль</span>'
								: '<button type="button" class="tr-pillbtn ps-btn" data-act="default">Сделать основным</button>') +
							'<button type="button" class="tr-pillbtn ps-btn" data-act="new"><i class="fa-solid fa-plus" aria-hidden="true"></i> Новый пресет</button>' +
						'</div></div></header>' +
				'<section class="tr-surface gd-block ps-block"><h3 class="gd-block-title">Что внутри</h3><div class="ps-links">' +
					link('subtitles', 'Субтитры', subName(st.subtitleStyle)) +
					link('effects', 'Крючок в начале', st.hook ? 'Вкл.' : 'Выкл.') +
					link('effects', 'Мидролл', store.midrollOn(st) ? 'Вкл.' : 'Выкл.') +
					link('modes', 'Режим по умолчанию', store.LABELS.clipType[st.clipType || '']) +
					link('modes', 'Сколько клипов', store.LABELS.max_clips[st.max_clips] || 'Авто') +
					link('quality', 'Качество', store.LABELS.quality[st.quality]) +
					link('voices', 'Озвучка', 'Скоро') +
				'</div></section>' +
				'<p class="gd-foot-note">' + (store.signedIn ? 'Пресеты хранятся в вашем аккаунте и доступны на всех устройствах.'
					: 'Пресеты хранятся в этом браузере. Войдите — и они будут на всех устройствах.') + '</p>';
		},
		subtitles: function () {
			var st = s();
			return hero('Субтитры', 'Как будут выглядеть слова в клипах. Субтитры вшиваются в видео, активное слово выделяется.') +
				block('Стиль', '<div class="ps-gallery ps-gallery--subs" data-k="subtitleStyle">' + SUBS.map(function (x) {
					return card('ps-sub', x.v, st.subtitleStyle === x.v, '<img src="' + img(x.img + '_45') + '" alt="' + esc(x.name) + '" width="432" height="540" loading="lazy" decoding="async">', x.name, x.meta);
				}).join('') + '</div>' +
				'<p class="gd-callout">Стример: «Жёлтые» превращаются в «Градиент» — так задумано для стримов. У Образовательного свои шаблоны оформления — в разделе «Режимы и шаблоны».</p>') +
				block('Шрифт и место', row('Шрифт', 'Подобран под стиль: Dela Gothic для «Жёлтых» и «Градиента», Montserrat для остальных.', '') +
					row('Положение', 'Внизу, в безопасной зоне. SONYA сама поднимает строки, если внизу уже есть текст из самого видео.', ''));
		},
		effects: function () {
			var st = s();
			return hero('Эффекты', 'То, что добавляется поверх нарезки. Всё работает уже сейчас — кроме отмеченного «Скоро».') +
				block('Начало клипа',
					'<div class="ps-feature"><img class="ps-feature-pic" src="' + img('fx_hook') + '" alt="" width="360" height="640" loading="lazy">' +
						'<div class="ps-feature-body">' + row('Крючок в начале', '1,5–3 с самого сильного момента, белая вспышка и переход в клип. Для Вирала и Сторителлинга (в Хуках — всегда).', sw('hook', st.hook)) + '</div></div>' +
					'<div class="ps-feature"><img class="ps-feature-pic" src="' + img('fx_title') + '" alt="" width="360" height="640" loading="lazy">' +
						'<div class="ps-feature-body">' + row('Заголовок-крючок', 'Крупная фраза сверху в первые секунды. В Сторителлинге — всегда, в Стримере — по переключателю.', sw('hookTitle', st.hookTitle)) + '</div></div>') +
				block('Монтаж',
					row('Убирать паузы', 'Джамп-каты: паузы длиннее 0,7 с вырезаются (Стример).', sw('jumpCuts', st.jumpCuts)) +
					row('Переходы', 'Глитч-переход в «Завозном комменте», стоп-кадр вместо обрывка соседней сцены — автоматически.', '<span class="ps-auto">Авто</span>') +
					row('Громкость', 'Ровная громкость −14 LUFS в каждом клипе, без перегруза.', '<span class="ps-auto">Всегда</span>')) +
				block('Реклама', midrollBlock() +
					soon('Свой баннер', 'Ваш логотип или ссылка поверх клипа.'), ' ps-block--ads') +
				block('Бренд', soon('Концовка SONYA', 'Короткая фирменная концовка в конце клипа.'));
		},
		modes: function () {
			var st = s();
			return hero('Режимы и шаблоны', 'Режим задаёт, что искать в видео. Шаблоны — как собрать кадр.') +
				block('Режим по умолчанию', '<div class="ps-modes">' + [{ v: '', name: 'Не выбирать', desc: 'Каждый раз выбираю сам.' }].concat(MODES).map(function (m) {
					return '<label class="ps-mode"><input type="radio" name="ps-mode" value="' + m.v + '"' + ((st.clipType || '') === m.v ? ' checked' : '') + '>' +
						'<span class="ps-mode-body"><span class="ps-row-label">' + esc(m.name) + '</span><span class="ps-row-sub">' + esc(m.desc) + '</span></span>' +
						'<span class="tr-radio" aria-hidden="true"></span></label>';
				}).join('') + '</div>' +
				'<p class="gd-callout">BEEP (запикивание мата) — отдельное приложение в библиотеке. Трейлер пока доступен только администраторам.</p>') +
				block('Сколько клипов', seg('ps-max', st.max_clips, [[0, 'Авто'], [5, 'До 5'], [8, 'До 8'], [12, 'До 12']]) +
					'<p class="ps-note">«Авто» — до 10 клипов: столько, сколько в видео законченных историй. Длину каждого клипа SONYA подбирает по смыслу.</p>') +
				block('Образовательный · оформление', '<div class="ps-gallery ps-gallery--4">' + EDU.map(function (t) {
					return card('ps-edu', t[0], st.edu_template === t[0], pic('/assets/edu-templates/edu_template_' + t[0] + '.jpg', t[1]), t[0] + ' · ' + t[1]);
				}).join('') + '</div>') +
				block('Стример · раскладка «вебка + игра»', '<div class="ps-gallery ps-gallery--5">' + LAYOUTS.map(function (t) {
					return card('ps-layout', t[0], st.streamer_layout === t[0], pic('/assets/streamer-templates/streamer_template_' + t[0] + '.jpg' + (t[2] || ''), t[1]), t[0] + ' · ' + t[1]);
				}).join('') + '</div>') +
				block('Стример · какие моменты искать', '<div class="tr-chips ps-chips">' +
					'<label class="tr-chip"><input type="checkbox" data-moment="auto"' + (st.moment_types.length ? '' : ' checked') + '><span>Авто · все</span></label>' +
					MOMENTS.map(function (m) {
						return '<label class="tr-chip"><input type="checkbox" data-moment="' + m[0] + '"' + (st.moment_types.indexOf(m[0]) !== -1 ? ' checked' : '') + '><span>' + esc(m[1]) + '</span></label>';
					}).join('') + '</div>');
		},
		quality: function () {
			var st = s();
			function q(v, title, lines, tag) {
				return '<label class="ps-q"><input type="radio" name="ps-quality" value="' + v + '"' + (st.quality === v ? ' checked' : '') + '>' +
					'<span class="ps-q-body"><span class="ps-q-head"><span class="ps-q-title">' + esc(title) + '</span>' + (tag ? '<span class="gd-tag">' + esc(tag) + '</span>' : '') + '</span>' +
					'<ul class="ps-q-list">' + lines.map(function (l) { return '<li>' + esc(l) + '</li>'; }).join('') + '</ul></span>' +
					'<span class="tr-radio" aria-hidden="true"></span></label>';
			}
			return hero('Качество нарезки', 'Стандарт подходит почти всегда. Высокое — когда клип будут смотреть на большом экране или исходник снят в 4K.') +
				block('Качество', '<div class="ps-qs">' +
					q('standard', 'Стандарт', ['1080×1920, как сейчас', 'Быстрая обработка', 'Спишется 1 обработка'], 'По умолчанию') +
					q('high', 'Высокое — 2K', ['1440×2560, высокий битрейт (до 24 Мбит/с)', 'Исходник до 4K, если он есть на YouTube, VK или Rutube', 'Примерно в 3 раза дольше обработка и ожидание', 'Спишется 3 обработки']) +
				'</div>' +
				'<p class="gd-callout">2K пока работает в Вирале, Сторителлинге и Хуках. Стример и Образовательный собираются в 1080×1920 — для них спишется 1 обработка. Превью и доставка в Telegram — как обычно; если файл больше 49 МБ, придёт ссылка.</p>');
		},
		voices: function () {
			return hero('Озвучка', 'Голос рассказчика поверх клипа. Готовим — выберете здесь, когда появится.') +
				block('Голоса', '<div class="ps-voices">' + VOICES.map(function (v) {
					return '<div class="ps-voice"><span class="ps-voice-icon" aria-hidden="true"><i class="fa-solid fa-wave-square"></i></span>' +
						'<span class="ps-row-text"><span class="ps-row-label">' + esc(v.name) + ' <span class="gd-tag">Скоро</span></span>' +
						'<span class="ps-row-sub">' + esc(v.desc) + '</span></span></div>';
				}).join('') + '</div>');
		},
		library: function () {
			var d = store.doc;
			return hero('Мои пресеты', 'Сохраните несколько стилей — например, для разных каналов — и выберите основной.', 'Пресеты') +
				block('Пресеты', '<div class="ps-list">' + d.presets.map(function (p) {
					var on = d.default_id === p.id, ed = p.id === cur().id;
					return '<div class="ps-item' + (ed ? ' is-editing' : '') + '" data-id="' + esc(p.id) + '">' +
						'<button type="button" class="ps-item-main" data-act="edit" aria-label="Настроить «' + esc(p.name) + '»">' +
							'<span class="ps-item-name">' + esc(p.name) + (on ? ' <span class="gd-tag">Основной</span>' : '') + '</span>' +
							'<span class="ps-row-sub">' + esc(summary(store.full(p.settings))) + '</span></button>' +
						'<div class="ps-item-actions">' +
							(on ? '' : '<button type="button" class="tr-iconbtn" data-act="default" title="Сделать основным" aria-label="Сделать основным"><i class="fa-regular fa-star"></i></button>') +
							'<button type="button" class="tr-iconbtn" data-act="rename" title="Переименовать" aria-label="Переименовать"><i class="fa-solid fa-pen"></i></button>' +
							'<button type="button" class="tr-iconbtn" data-act="copy" title="Дублировать" aria-label="Дублировать"><i class="fa-regular fa-copy"></i></button>' +
							(d.presets.length > 1 ? '<button type="button" class="tr-iconbtn" data-act="delete" title="Удалить" aria-label="Удалить"><i class="fa-regular fa-trash-can"></i></button>' : '') +
						'</div></div>';
				}).join('') + '</div>' +
				'<button type="button" class="tr-pillbtn tr-pillbtn--block ps-add" data-act="new"><i class="fa-solid fa-plus" aria-hidden="true"></i> Новый пресет</button>');
		}
	};

	/* ── Name dialog: the site's own modal (auth.js classes) ── */
	function askName(title, value) {
		return new Promise(function (resolve) {
			// the trim dialog's own modal (auth.js askTrimDialog): same id = the same puffy glass in both themes
			var prev = document.getElementById('sonya-trim-modal');
			if (prev) prev.remove();
			var ov = document.createElement('div');
			ov.className = 'sonya-modal-overlay is-open';
			ov.id = 'sonya-trim-modal';
			ov.setAttribute('role', 'dialog'); ov.setAttribute('aria-modal', 'true'); ov.setAttribute('aria-labelledby', 'ps-name-t');
			ov.innerHTML = '<div class="sonya-modal paywall-modal">' +
				'<button class="sonya-modal-close" type="button" data-x aria-label="Закрыть"><i class="fa-solid fa-xmark"></i></button>' +
				'<div class="paywall-view is-active"><h2 class="paywall-title plans-title" id="ps-name-t">' + esc(title) + '</h2>' +
				'<p class="trim-lead">Например, «Мой канал» или «Подкаст».</p>' +
				'<div class="trim-card"><input class="trim-field ps-name-input" type="text" maxlength="40" value="' + esc(value || '') + '" aria-label="Название пресета"></div>' +
				'<button type="button" class="trim-go" data-ok>Сохранить</button>' +
				'<button type="button" class="paywall-later-btn" data-x>Отмена</button></div></div>';
			document.body.appendChild(ov);
			var input = ov.querySelector('input');
			function close(v) { document.removeEventListener('keydown', key); ov.remove(); resolve(v); }
			function key(e) { if (e.key === 'Escape') close(null); if (e.key === 'Enter') ok(); }
			function ok() { var v = input.value.trim(); if (v) close(v); else input.focus(); }
			ov.addEventListener('click', function (e) { if (e.target === ov || e.target.closest('[data-x]')) close(null); else if (e.target.closest('[data-ok]')) ok(); });
			document.addEventListener('keydown', key);
			setTimeout(function () { input.focus(); input.select(); }, 30);
		});
	}

	/* ── Render / events ─────────────────────────────────────── */
	function navHTML() {
		return '<ul class="gd-nav-list" role="list">' + SECTIONS.map(function (x) {
			return '<li><button type="button" class="gd-nav-item" data-sec="' + x.id + '" aria-controls="gd-doc">' +
				'<span class="gd-icon gd-icon--nav gd-icon--glyph" aria-hidden="true"><i class="fa-solid ' + x.icon + '"></i></span>' +
				'<span class="gd-nav-name">' + esc(x.label) + '</span></button></li>';
		}).join('') + '</ul>';
	}

	function show(id, opts) {
		opts = opts || {};
		current = find(id) ? id : 'style';
		Array.prototype.forEach.call(navEl.querySelectorAll('.gd-nav-item'), function (b) {
			var on = b.getAttribute('data-sec') === current;
			b.classList.toggle('is-active', on);
			if (on) b.setAttribute('aria-current', 'true'); else b.removeAttribute('aria-current');
			if (on && opts.reveal !== false && navEl.scrollWidth > navEl.clientWidth) {
				var l = b.offsetLeft, r = l + b.offsetWidth;
				if (l < navEl.scrollLeft || r > navEl.scrollLeft + navEl.clientWidth) navEl.scrollLeft = l - (navEl.clientWidth - b.offsetWidth) / 2;
			}
		});
		render(opts.animate !== false);
		if (opts.updateHash !== false) {
			var want = P.hashFor(current);
			if (location.hash !== want) history[opts.replace ? 'replaceState' : 'pushState'](null, '', location.pathname + location.search + want);
		}
		if (opts.focus) {
			var scroller = root.closest('.sx-ws-canvas');
			if (scroller) scroller.scrollTop = 0;
			var h = docEl.querySelector('#gd-h');
			if (h) h.focus({ preventScroll: true });
		}
	}

	function render(animate) {
		var y = docEl.scrollTop;
		docEl.innerHTML = VIEWS[current]();
		if (animate) { docEl.classList.remove('is-entering'); void docEl.offsetWidth; docEl.classList.add('is-entering'); }
		else { docEl.classList.remove('is-entering'); docEl.scrollTop = y; }
	}

	function onChange(e) {
		var t = e.target;
		if (t.matches('.tr-switch[data-k="mr-on"]')) { mrSet({ on: t.checked }); render(false); return; }
		if (t.matches('.tr-switch[data-k]')) { set(t.getAttribute('data-k'), t.checked); return; }
		if (t.hasAttribute('data-mr-file')) { mrBusy = null; mrUpload(t.files && t.files[0]); return; }
		if (t.name === 'ps-mr-place') {
			var v = t.value;
			mrSet(v === 'pre' || v === 'post' ? { place: v } : { place: 'mid', at: v === 'after' ? 'after' : 'auto' });
			render(false); return;
		}
		if (t.name === 'ps-mr-after') { mrSet({ after_s: Number(t.value) }); return; }
		if (t.name === 'ps-mr-every') { mrSet({ every: Number(t.value) }); return; }
		if (t.name === 'ps-mr-fit') { mrSet({ fit: t.value }); render(false); return; }
		if (t.name === 'ps-mr-max') { mrSet({ max_total: Number(t.value) }); render(false); return; }
		if (t.name === 'ps-sub') { set('subtitleStyle', t.value); return; }
		if (t.name === 'ps-quality') { set('quality', t.value); return; }
		if (t.name === 'ps-mode') { set('clipType', t.value); return; }
		if (t.name === 'ps-max') { set('max_clips', Number(t.value)); return; }
		if (t.name === 'ps-edu') { set('edu_template', Number(t.value)); return; }
		if (t.name === 'ps-layout') { set('streamer_layout', Number(t.value)); return; }
		if (t.hasAttribute('data-moment')) {
			var m = t.getAttribute('data-moment'), boxes = docEl.querySelectorAll('[data-moment]');
			if (m === 'auto' && t.checked) Array.prototype.forEach.call(boxes, function (b) { if (b.getAttribute('data-moment') !== 'auto') b.checked = false; });
			var picked = Array.prototype.filter.call(boxes, function (b) { return b.checked && b.getAttribute('data-moment') !== 'auto'; })
				.map(function (b) { return b.getAttribute('data-moment'); });
			docEl.querySelector('[data-moment="auto"]').checked = !picked.length;
			set('moment_types', picked);
		}
	}

	function onClick(e) {
		var b = e.target.closest('[data-sec]');
		if (b) { show(b.getAttribute('data-sec'), { focus: true }); return; }
		var a = e.target.closest('[data-act]');
		if (!a) return;
		var act = a.getAttribute('data-act'), item = a.closest('[data-id]'), id = item ? item.getAttribute('data-id') : cur().id;
		if (act === 'mr-play') {
			var v = a.parentNode.querySelector('video');
			if (!v) return;
			if (v.paused) { v.play().catch(function () {}); a.classList.add('is-playing'); a.innerHTML = '<i class="fa-solid fa-pause" aria-hidden="true"></i>'; a.setAttribute('aria-label', 'Пауза'); }
			else { v.pause(); a.classList.remove('is-playing'); a.innerHTML = '<i class="fa-solid fa-play" aria-hidden="true"></i>'; a.setAttribute('aria-label', 'Смотреть ролик'); }
			return;
		}
		if (act === 'mr-remove') {
			if (window.confirm('Убрать ролик из пресета «' + cur().name + '»?')) { set('midroll', null); mrBusy = null; render(false); }
			return;
		}
		if (act === 'default') { store.setDefault(id); render(false); toast('Основной стиль: «' + store.find(id).name + '»'); }
		else if (act === 'edit') { editing = id; show('style', { focus: true }); }
		else if (act === 'new') askName('Новый пресет', store.uniqueName('Мой стиль')).then(function (n) {
			if (!n) return; var p = store.create(n, {}); editing = p.id; show('style', { focus: true });
		});
		else if (act === 'copy') { var src = store.find(id); var c = store.create(store.uniqueName(src.name + ' — копия'), src.settings); editing = c.id; render(false); }
		else if (act === 'rename') askName('Название пресета', store.find(id).name).then(function (n) { if (n) { store.update(id, { name: n }); render(false); } });
		else if (act === 'delete') {
			var p = store.find(id);
			if (p && window.confirm('Удалить пресет «' + p.name + '»?')) { if (editing === id) editing = null; store.remove(id); render(false); }
		}
	}

	function toast(msg) { if (typeof window.showToast === 'function') window.showToast(msg, 'success'); }

	P.mount = function (container) {
		if (root) return;
		root = document.createElement('div');
		root.className = 'tr-app gd-app ps-app';
		root.innerHTML = '<div class="gd-layout"><nav class="gd-nav" aria-label="Разделы пресетов">' + navHTML() + '</nav>' +
			'<article class="gd-doc" id="gd-doc"></article></div>';
		container.appendChild(root);
		navEl = root.querySelector('.gd-nav');
		docEl = root.querySelector('.gd-doc');
		root.addEventListener('click', onClick);
		root.addEventListener('change', onChange);
		root.addEventListener('keydown', function (e) {
			var l = e.target.closest && e.target.closest('label[role="button"]');
			if (!l || (e.key !== 'Enter' && e.key !== ' ')) return;
			var inp = l.querySelector('input[type="file"]');
			if (inp) { e.preventDefault(); inp.click(); }
		});
		root.addEventListener('dragover', function (e) { var z = e.target.closest('[data-mr-drop]'); if (z) { e.preventDefault(); z.classList.add('is-over'); } });
		root.addEventListener('dragleave', function (e) { var z = e.target.closest('[data-mr-drop]'); if (z) z.classList.remove('is-over'); });
		root.addEventListener('drop', function (e) {
			var z = e.target.closest('[data-mr-drop]');
			if (!z) return;
			e.preventDefault(); z.classList.remove('is-over');
			mrBusy = null; mrUpload(e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]);
		});
		navEl.addEventListener('keydown', function (e) {
			var keys = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 };
			if (!keys[e.key]) return;
			var items = Array.prototype.slice.call(navEl.querySelectorAll('.gd-nav-item')), i = items.indexOf(document.activeElement);
			if (i < 0) return;
			e.preventDefault();
			items[(i + keys[e.key] + items.length) % items.length].focus();
		});
		window.addEventListener('hashchange', function () {
			if (!/^#\/presets(\/|$)/.test(location.hash)) return;
			var id = P.sectionFromHash(location.hash);
			if (id !== current) show(id, { replace: true, focus: true });
		});
		show(P.sectionFromHash(location.hash), { replace: true, reveal: true });
		// the account's copy arrives → redraw in place (no animation, keeps scroll)
		store.sync().then(function () { if (!store.find(editing)) editing = null; render(false); });
	};
})();
