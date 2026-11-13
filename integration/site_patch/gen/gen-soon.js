/* SONYA Gen window (#/gen): two tabs in the site's own pill (.tr-tabs, as Trailer/BEEP).
   «SONYA Gen» = images: LIVE for admins only (GET /api/gen/access), everyone else sees «Скоро».
   «SONYA Gen Video» = «Скоро» until the video backend is connected. Mounted: window.SonyaGen.mount(el). */
(function () {
	'use strict';
	var TABS = [
		{ id: 'image', label: 'SONYA Gen', icon: 'fa-regular fa-image' },
		{ id: 'video', label: 'SONYA Gen Video', icon: 'fa-solid fa-film' }
	];
	var API = window.SONYA_API_BASE || '';
	var ASPECTS = ['1:1', '4:5', '3:4', '2:3', '9:16', '16:9', '3:2', '5:4'];
	function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
	function api(path, opts) {
		opts = opts || {};
		return fetch(API + path, { credentials: 'include', method: opts.method || 'GET',
			headers: { 'Content-Type': 'application/json' }, body: opts.body ? JSON.stringify(opts.body) : undefined });
	}
	function soonPanel(icon, title, text, chips) {
		return '<section class="tr-surface gn-card"><div class="gn-head"><h2 class="gn-title"><i class="' + icon + '" aria-hidden="true"></i><span>' + title + '</span></h2>' +
			'<span class="tr-badge">Скоро</span></div>' + text +
			'<div class="gn-chips">' + chips.map(function (c) { return '<span class="gn-chip">' + c + '</span>'; }).join('') + '</div>' +
			'<button type="button" class="tr-cta tr-cta--block" disabled aria-disabled="true"><span>Скоро будет доступно</span><i class="fa-solid fa-arrow-right" aria-hidden="true"></i></button></section>';
	}
	function videoPanel() {
		return soonPanel('fa-solid fa-film', 'Gen Video',
			'<p class="gn-text">Короткий ролик 2–5 секунд по описанию или по вашей картинке. Подойдёт для вставки, обложки или фона.</p>' +
			'<p class="gn-text">Ждать придётся несколько минут — мы покажем, на каком этапе идёт работа.</p>',
			['2–5 секунд', '480p и 720p', 'вертикальное и горизонтальное']);
	}
	function imageSoonPanel() {
		return soonPanel('fa-regular fa-image', 'Gen',
			'<p class="gn-text">Картинка по описанию: обложки, превью, кадры для рекламы. Скоро откроем для всех.</p>',
			['по описанию', 'любые форматы', 'PNG']);
	}
	function imagePanel(panel) {
		panel.innerHTML = '<section class="tr-surface gn-card"><div class="gn-head"><h2 class="gn-title"><i class="fa-regular fa-image" aria-hidden="true"></i><span>Gen</span></h2>' +
			'<span class="tr-badge">Админ</span></div>' +
			'<label class="gn-label" for="gn-prompt">Что нарисовать</label>' +
			'<textarea id="gn-prompt" class="gn-input" rows="4" maxlength="1000" placeholder="Например: белый шёлк на светлом фоне, мягкий свет, без логотипов"></textarea>' +
			'<div class="gn-row"><label class="gn-label">Формат <select id="gn-aspect" class="gn-input">' + ASPECTS.map(function (a) { return '<option>' + a + '</option>'; }).join('') + '</select></label>' +
			'<label class="gn-label">Качество <select id="gn-size" class="gn-input"><option value="standard">стандарт</option><option value="draft">черновик</option></select></label>' +
			'<label class="gn-label">Вариантов <select id="gn-count" class="gn-input"><option>1</option><option>2</option><option>4</option></select></label></div>' +
			'<p class="gn-text gn-status" id="gn-status" role="status" aria-live="polite"></p>' +
			'<div class="gn-results" id="gn-results"></div>' +
			'<button type="button" class="tr-cta tr-cta--block" id="gn-go"><span>Создать картинку</span><i class="fa-solid fa-arrow-right" aria-hidden="true"></i></button></section>';
		var go = panel.querySelector('#gn-go'), st = panel.querySelector('#gn-status'), res = panel.querySelector('#gn-results'), busy = false;
		function setSt(t) { st.textContent = t; }
		function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
		var LABELS = { queued: 'В очереди…', downloading: 'Готовим видеокарту…', model_downloading: 'Загружаем модель (первый раз дольше)…', mode_running: 'Рисуем…', uploading_result: 'Сохраняем…' };
		go.addEventListener('click', async function () {
			if (busy) return;
			var prompt = panel.querySelector('#gn-prompt').value.trim();
			if (!prompt) { setSt('Опишите, что нарисовать.'); return; }
			busy = true; go.disabled = true; res.innerHTML = ''; setSt('Отправляем…');
			try {
				var r = await api('/api/gen/image', { method: 'POST', body: { prompt: prompt, aspect: panel.querySelector('#gn-aspect').value,
					size: panel.querySelector('#gn-size').value, count: parseInt(panel.querySelector('#gn-count').value, 10) } });
				if (r.status === 403) { setSt('Скоро.'); return; }
				if (!r.ok) { setSt('Не получилось создать задание. Попробуйте ещё раз.'); return; }
				var id = (await r.json()).job_id, t0 = Date.now();
				for (;;) {
					if (!document.body.contains(panel)) return;
					await sleep(3000);
					var jr = await api('/api/generation/jobs/' + encodeURIComponent(id));
					if (!jr.ok) { setSt('Не удалось получить статус.'); return; }
					var j = await jr.json();
					if (j.status === 'completed') break;
					if (j.status === 'failed' || j.status === 'cancelled') { setSt('Не получилось: ' + esc(j.error_message || 'ошибка генерации')); return; }
					setSt((LABELS[j.status] || 'Работаем…') + ' ' + Math.round((Date.now() - t0) / 1000) + ' с');
					if (Date.now() - t0 > 25 * 60 * 1000) { setSt('Слишком долго. Проверьте позже в истории.'); return; }
				}
				var ur = await api('/api/generation/jobs/' + encodeURIComponent(id) + '/result-url');
				var u = await ur.json();
				res.innerHTML = (u.clips || []).map(function (c) {
					return '<figure class="gn-fig"><img class="gn-img" alt="Сгенерировано ИИ" src="' + esc(c.url) + '"><figcaption><a class="gn-link" href="' + esc(c.download_url) + '">Скачать</a> · сгенерировано ИИ</figcaption></figure>';
				}).join('');
				setSt('Готово.');
			} catch (e) {
				setSt('Ошибка сети. Попробуйте ещё раз.');
			} finally { busy = false; go.disabled = false; }
		});
	}
	window.SonyaGen = { mount: function (container) {
		var imageLive = false, current = 'image';
		container.innerHTML = '<div class="tr-app gn-app"><header class="tr-head"><h1 class="tr-title">Gen</h1></header>' +
			'<div class="tr-main"><div class="tr-center"><div class="gn-wrap">' +
			'<div class="tr-tabs" role="tablist" aria-label="SONYA Gen">' + TABS.map(function (t, i) {
				return '<button type="button" role="tab" id="gn-tab-' + t.id + '" class="tr-tab' + (i ? '' : ' is-active') + '" aria-selected="' + (i ? 'false' : 'true') +
					'" aria-controls="gn-panel" tabindex="' + (i ? '-1' : '0') + '" data-gn-tab="' + t.id + '"><i class="' + t.icon + '" aria-hidden="true"></i><span>' + t.label + '</span></button>';
			}).join('') + '</div>' +
			'<div id="gn-panel" class="gn-panel" role="tabpanel" aria-labelledby="gn-tab-image"></div>' +
			'</div></div></div></div>';
		var tabs = container.querySelectorAll('[data-gn-tab]'), panel = container.querySelector('#gn-panel');
		function show(id, focus) {
			current = id;
			Array.prototype.forEach.call(tabs, function (b) {
				var on = b.getAttribute('data-gn-tab') === id;
				b.classList.toggle('is-active', on); b.setAttribute('aria-selected', on ? 'true' : 'false'); b.tabIndex = on ? 0 : -1;
				if (on && focus) b.focus();
			});
			panel.setAttribute('aria-labelledby', 'gn-tab-' + id);
			if (id === 'video') panel.innerHTML = videoPanel();
			else if (imageLive) imagePanel(panel);
			else panel.innerHTML = imageSoonPanel();
		}
		show('image');   // «скоро» until the access check says otherwise (default = closed)
		api('/api/gen/access').then(function (r) { return r.ok ? r.json() : { image: false }; }).catch(function () { return { image: false }; })
			.then(function (a) { imageLive = a && a.image === true; if (current === 'image') show('image'); });
		Array.prototype.forEach.call(tabs, function (b, i) {
			b.addEventListener('click', function () { show(b.getAttribute('data-gn-tab')); });
			b.addEventListener('keydown', function (e) {
				if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
				e.preventDefault(); show(TABS[(i + 1) % 2].id, true);
			});
		});
	} };
})();
