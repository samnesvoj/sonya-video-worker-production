/* SONYA Работы (#/gallery): history of the user's jobs with clips (9:16 preview + download).
   Data: GET /api/generation/jobs (own jobs), GET /api/generation/jobs/{id}/result-url (presigned clips). */
(function () {
	"use strict";
	var MODES = { virality: "Вирусные клипы", viral: "Вирусные клипы", hooks: "Хуки", storytelling: "Истории", educational: "Обучающие", educational_ru: "Обучающие", streamer: "Стример", sonya_gen: "SONYA Gen", beep: "BEEP", trailer: "Трейлер" };
	var ST = { completed: ["Готово", "ok"], failed: ["Ошибка", "err"], cancelled: ["Отменено", "err"] };
	function esc(s) { return String(s == null ? "" : s).replace(/[&<>"\x27]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "\x27": "&#39;" }[c]; }); }
	function fmt(d) { try { return new Date(d).toLocaleString("ru-RU", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" }); } catch (e) { return ""; } }
	function api(p) { return fetch((window.SONYA_API_BASE || "/api") + p, { credentials: "include" }).then(function (r) { if (!r.ok) throw r.status; return r.json(); }); }
	function clipsHtml(cl) {
		return "<div class=\"gl-clips\">" + cl.map(function (c, i) {
			var img = /\.(png|jpe?g)(\?|$)/i.test(c.url);
			return "<div class=\"gl-clip\">" + (img ? "<img loading=\"lazy\" src=\"" + esc(c.url) + "\" alt=\"\">" :
				"<video controls playsinline preload=\"metadata\" src=\"" + esc(c.url) + "#t=0.1\"></video>") +
				"<a class=\"gl-dl\" href=\"" + esc(c.download_url || c.url) + "\" download>Скачать " + (i + 1) + "</a></div>";
		}).join("") + "</div>";
	}
	window.SonyaGallery = { mount: function (el) {
		el.innerHTML = "<div class=\"tr-app gl-app\"><header class=\"tr-head\"><h1 class=\"tr-title\">Работы</h1></header><div class=\"tr-main\"><div class=\"tr-center\"><div class=\"gl-list\" id=\"gl-list\"><div class=\"gl-empty\">Загружаем…</div></div></div></div></div>";
		var list = el.querySelector("#gl-list");
		function load() {
			api("/generation/jobs?limit=50").then(function (d) {
				var jobs = d.jobs || [];
				if (!jobs.length) { list.innerHTML = "<div class=\"gl-empty\">Пока нет работ. Здесь появятся ваши нарезки и генерации.</div>"; return; }
				list.innerHTML = jobs.map(function (j) {
					var s = ST[j.status] || ["В работе", "run"];
					return "<div class=\"gl-card\" data-id=\"" + esc(j.id) + "\" data-st=\"" + esc(j.status) + "\"><div class=\"gl-top\"><div><div class=\"gl-name\">" +
						esc(MODES[j.mode] || j.mode || "Задание") + "</div><div class=\"gl-meta\">" + esc(fmt(j.created_at)) + (j.clip_count ? " · клипов: " + esc(j.clip_count) : "") +
						"</div></div><span class=\"gl-badge " + s[1] + "\">" + s[0] + "</span></div><div class=\"gl-body\">" +
						(j.status === "completed" ? "<button class=\"gl-btn\" type=\"button\">Показать клипы</button>" : "") + "</div></div>";
				}).join("");
				var first = list.querySelector("[data-st=completed] .gl-btn"); if (first) first.click();
			}).catch(function (e) {
				list.innerHTML = "<div class=\"gl-empty\">" + (e === 401 ? "Войдите в аккаунт, чтобы увидеть свои работы." : "Не удалось загрузить историю. Обновите страницу.") + "</div>";
			});
		}
		list.addEventListener("click", function (ev) {
			var b = ev.target.closest(".gl-btn"); if (!b) return;
			var card = b.closest(".gl-card"), body = card.querySelector(".gl-body");
			b.disabled = true; b.textContent = "Загружаем…";
			api("/generation/jobs/" + encodeURIComponent(card.getAttribute("data-id")) + "/result-url?expires=86400").then(function (r) {
				body.innerHTML = clipsHtml(r.clips || []);
			}).catch(function () { b.disabled = false; b.textContent = "Повторить"; });
		});
		load();
	} };
})();
