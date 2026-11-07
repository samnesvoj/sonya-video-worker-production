#!/usr/bin/env python3
"""
Cost model for the SONYA GPU pool -> docs/COSTS.md. Decimal only. Prices are REAL offers captured read-only from Vast
(data/vast_offers_*.json) and the official CBR rate (data/cbr_usd_*.json). Everything about WORKLOAD (GPU minutes per
video, model size, boot times) is an ASSUMPTION: no GPU run has been done yet. These are projections, not charges.
"""
from __future__ import annotations

import json
import pathlib
from decimal import ROUND_HALF_UP, Decimal as Dec

ROOT = pathlib.Path(__file__).resolve().parents[1]
OFFERS = json.loads(next(ROOT.glob("data/vast_offers_*.json")).read_text())
FX = Dec(json.loads(next(ROOT.glob("data/cbr_usd_*.json")).read_text())["usd_rub"])

DISK_GB = Dec(80)
HOURS_MONTH = Dec(720)
DAYS = Dec(30)
IDLE_TAIL_H = Dec(60) / Dec(3600)                  # IDLE_TIMEOUT_SECONDS=60 is billed running before every stop
CREATE_BOOT_H = Dec(6) / Dec(60)                   # ASSUMPTION: bootstrap + readiness + ~5 GB models at 200+ Mbps, billed as running
RESUME_BOOT_H = Dec("1.5") / Dec(60)               # ASSUMPTION: container start + release cached + models cached + readiness
MODEL_GB, VIDEO_GB = Dec(5), Dec("0.6")            # ASSUMPTIONS (cold-start download; per video in+out)
RETENTION_H = Dec("0.5")                           # hybrid: keep the stopped disk 30 min (SONYA_STOPPED_RETENTION_SECONDS=1800), then destroy
GPUS = [("RTX 3090", ""), ("RTX 4090", ""), ("RTX 3090 Ti", ""), ("RTX A6000", ""),
        ("A10", " *"), ("L4", " *")]               # * = 22-23 GB VRAM: below the 24 GB default, only after a GPU test


def q(x, p="0.01"):
    return Dec(x).quantize(Dec(p), ROUND_HALF_UP)


def eff(o):
    """Effective hourly cost of an offer incl. an 80 GB disk (what the fleet's pick_offer ranks by)."""
    return Dec(str(o["dph_base"])) + Dec(str(o["storage_cost"])) * DISK_GB / HOURS_MONTH


def best(name):
    rows = [o for o in OFFERS["offers"] if o["gpu_name"] == name]
    o = min(rows, key=eff)
    return {k: (Dec(str(v)) if isinstance(v, (int, float)) else v) for k, v in o.items()}, len(rows)


def day_cost(o, strategy, n, p_min, spaced):
    """USD per day. n videos/day, p_min GPU-minutes per video. spaced=True: every video wakes the GPU separately."""
    gpu_h, stor_gb_h, tb_down = o["dph_base"], o["storage_cost"] / HOURS_MONTH * DISK_GB, o["internet_down_cost_per_tb"] / 1000
    tb_up = o["internet_up_cost_per_tb"] / 1000
    work_h = Dec(n) * p_min / 60
    wake = Dec(n) if spaced else Dec(1)
    traffic = lambda cold: cold * MODEL_GB * tb_down + Dec(n) * VIDEO_GB * (tb_down + tb_up) / 2
    if strategy == "24/7":
        return Dec(24) * (gpu_h + stor_gb_h) + traffic(Dec(0)), Dec(24)
    if strategy == "stop60":                   # always STOP (disk kept), RESUME on demand
        run = wake * (RESUME_BOOT_H + IDLE_TAIL_H) + work_h
        return run * gpu_h + Dec(24) * stor_gb_h + traffic(Dec(0)), run
    if strategy == "destroy60":                # always DESTROY after the queue, CREATE on demand
        run = wake * (CREATE_BOOT_H + IDLE_TAIL_H) + work_h
        return run * (gpu_h + stor_gb_h) + traffic(wake), run
    if strategy == "hybrid":                   # STOP at 60 s, DESTROY after RETENTION_H without work
        gap = Dec(24) / wake
        if gap <= RETENTION_H:                 # next wake-up inside retention: RESUME each time, disk lives all day
            run = wake * (RESUME_BOOT_H + IDLE_TAIL_H) + work_h
            return run * gpu_h + Dec(24) * stor_gb_h + traffic(Dec(0)), run
        run = wake * (CREATE_BOOT_H + IDLE_TAIL_H) + work_h
        kept = wake * RETENTION_H              # disk is kept stopped for RETENTION_H after each wake-up, then destroyed
        return run * (gpu_h + stor_gb_h) + kept * stor_gb_h + traffic(wake), run
    raise ValueError(strategy)


def fmt(v):
    return f"${q(v)} / {q(v * FX)} ₽"


def main():
    out = []
    w = out.append
    w("# Расчёт расходов Vast.ai (сгенерировано `tools/cost_model.py`)\n")
    w("> **Это прогноз, а не фактические списания.** Цены — реальные предложения Vast, снятые read-only 2026-11-07 (`data/vast_offers_2026-11-07.json`); "
      f"курс — официальный ЦБ РФ на 07.11.2026: **{FX} ₽/$** (`data/cbr_usd_2026-11-07.json`). Нагрузка (минуты GPU на видео, размер моделей, время старта) — **допущения**: GPU-прогонов ещё не было.\n")
    w("## Источники\n- Цены/поля предложений: Vast API `POST /bundles/` через `vastai search offers` (поля `dph_base`, `storage_cost` $/ГБ/мес, `internet_*_cost_per_tb`), выборка: 1 GPU, CUDA≥12.4, reliability>0.98, verified, диск≥80 ГБ, inet_down>200 Мбит/с.\n"
      "- Правила биллинга: https://docs.vast.ai/documentation/reference/billing , https://docs.vast.ai/documentation/instances/storage , https://docs.vast.ai/documentation/instances/managing-instances\n- Курс: https://www.cbr.ru/scripts/XML_daily.asp\n")
    w("## Что говорит документация Vast (проверено 2026-11-07)\n"
      "- **GPU тарифицируется только пока инстанс активен** (per second). Остановленный инстанс «отсоединён от GPU» — GPU-часы не идут. «Loading» (скачивание образа) **не тарифицируется**.\n"
      "- **Диск тарифицируется каждую секунду существования инстанса, в том числе остановленного** («Stopping an instance does not avoid storage costs»). Ставка диска у остановленных «может быть выше», точный коэффициент в документации не указан — в модели ×1 (риск занижения; в `usage.EstimateParams.stopped_storage_multiplier` настраивается).\n"
      "- **Трафик тарифицируется по использованию, в любом состоянии**, отдельно от $/час.\n"
      "- **Возобновление не гарантировано**: статус `Scheduling`, GPU может быть занят другим арендатором («stuck >30 s — likely rented»), ожидание может быть бесконечным; рекомендация Vast — создать новый инстанс. Истёкший контракт нельзя перезапустить.\n"
      "- Нулевой баланс: инстансы останавливаются, диск продолжает списываться; без карты данные уничтожаются.\n")
    w("## Цены (реальные; из офферов каждой карты выбран самый дешёвый с учётом диска 80 ГБ)\n")
    w("| GPU | офферов | GPU $/ч | GPU ₽/мин | GPU ₽/ч | диск 80 ГБ $/сут | диск ₽/сут | диск ₽/мес | трафик вх/исх $/ТБ |\n|---|---|---|---|---|---|---|---|---|")
    offers = {}
    for name, mark in GPUS:
        o, n = best(name)
        offers[name] = o
        d_day = o["storage_cost"] / HOURS_MONTH * DISK_GB * 24
        w(f"| {name}{mark} | {n} | ${q(o['dph_base'],'0.0001')} | {q(o['dph_base']*FX/60)} | {q(o['dph_base']*FX)} | ${q(d_day)} | {q(d_day*FX)} | {q(d_day*FX*DAYS)} | {q(o['internet_down_cost_per_tb'])} / {q(o['internet_up_cost_per_tb'])} |")
    w("\n\\* 22–23 ГБ VRAM — ниже порога 24 ГБ по умолчанию; допускать только после GPU-теста. Цена оффера меняется (рынок), здесь — снимок.\n")
    w("## Холодный старт и модели (допущения, не измерено)\n"
      f"- CREATE: образ (не тарифицируется) → bootstrap+readiness+модели ≈ {q(CREATE_BOOT_H*60,'0.1')} мин **тарифицируемого** времени (модели {MODEL_GB} ГБ) — стоимость старта RTX 3090: **{fmt(CREATE_BOOT_H*offers['RTX 3090']['dph_base'] + MODEL_GB*offers['RTX 3090']['internet_down_cost_per_tb']/1000 + CREATE_BOOT_H*offers['RTX 3090']['storage_cost']/HOURS_MONTH*DISK_GB)}**.\n"
      f"- RESUME: ≈ {q(RESUME_BOOT_H*60,'0.1')} мин (релиз и модели уже на диске) — **{fmt(RESUME_BOOT_H*offers['RTX 3090']['dph_base'])}**; но GPU может оказаться занят.\n"
      f"- Хвост простоя `IDLE_TIMEOUT_SECONDS=60`: 1 мин GPU после каждого пробуждения ({fmt(IDLE_TAIL_H*offers['RTX 3090']['dph_base'])} для RTX 3090).\n")
    strategies = [("24/7", "Постоянно включён 24/7"), ("stop60", "STOP через 60 с + RESUME (диск остаётся)"), ("destroy60", "DESTROY через 60 с + CREATE"), ("hybrid", "Гибрид: STOP 60 с, DESTROY через 30 мин простоя")]
    for gname in ("RTX 3090", "RTX 4090"):
        o = offers[gname]
        for p_min in (Dec(3), Dec(10), Dec(30)):
            w(f"## {gname}: {p_min} мин GPU на видео — расходы в сутки / в месяц (30 дн.)\n")
            w("Паттерн «разрозненно»: каждое видео будит GPU отдельно (худший случай для старта). «Пачкой»: все видео дня — одним пробуждением.\n")
            w("| видео/сут | паттерн | " + " | ".join(s for _, s in strategies) + " | экономия лучшего vs 24/7 |\n|---|---|" + "---|" * (len(strategies) + 1))
            base_month = None
            for n in (1, 5, 10, 30, 100):
                for spaced in (True, False):
                    if n == 1 and not spaced:
                        continue
                    cells, vals = [], {}
                    for key, _ in strategies:
                        d, run = day_cost(o, key, n, p_min, spaced)
                        vals[key] = d
                        cells.append(f"{fmt(d)} /сут<br>{fmt(d*DAYS)} /мес")
                    best_key = min((k for k in vals if k != "24/7"), key=lambda k: vals[k])
                    sav = (vals["24/7"] - vals[best_key]) * DAYS
                    w(f"| {n} | {'разрозненно' if spaced else 'пачкой'} | " + " | ".join(cells) + f" | **{fmt(sav)} /мес** ({best_key}) |")
            w("")
    w("## Итог для небольшого числа пользователей\n")
    o = offers["RTX 3090"]
    for n, p in ((5, Dec(10)), (10, Dec(10)), (30, Dec(10))):
        vals = {k: day_cost(o, k, n, p, True)[0] * DAYS for k, _ in strategies}
        w(f"- RTX 3090, {n} видео/сут × {p} мин: " + "; ".join(f"{lab} — {fmt(vals[k])}/мес" for k, lab in strategies))
    w("\n## Точка безубыточности STOP/RESUME против DESTROY/CREATE\n")
    w("| GPU | диск за сутки | старт CREATE | старт RESUME | пробуждений в сутки, после которых STOP+RESUME дешевле DESTROY+CREATE |\n|---|---|---|---|---|")
    for gname in ("RTX 3090", "RTX 4090", "RTX A6000"):
        o = offers[gname]
        stor_h = o["storage_cost"] / HOURS_MONTH * DISK_GB
        cold = CREATE_BOOT_H * (o["dph_base"] + stor_h) + MODEL_GB * o["internet_down_cost_per_tb"] / 1000
        res = RESUME_BOOT_H * o["dph_base"]
        w(f"| {gname} | {fmt(stor_h*24)} | {fmt(cold)} | {fmt(res)} | ≈ {q(stor_h*24/(cold-res),'1')} |")
    w("\nДиск 80 ГБ — главная статья расходов при редком использовании: сутки хранения стоят примерно как 2–3 часа работы GPU. Размер диска влияет линейно (диск 40 ГБ вдвое дешевле); минимальный безопасный размер определится после GPU-теста (образ torch+CUDA, модели, временные видео).\n")
    o = offers["RTX 3090"]
    stor_h = o["storage_cost"] / HOURS_MONTH * DISK_GB
    cold = CREATE_BOOT_H * (o["dph_base"] + stor_h) + MODEL_GB * o["internet_down_cost_per_tb"] / 1000
    w("**Рекомендация для SONYA с небольшим числом пользователей (до ~10–20 видео в сутки).** Останавливать инстанс через 60 с простоя (требование), "
      "но **не держать остановленный диск долго**: `SONYA_STOPPED_RETENTION_SECONDS=1800` — 30 минут нужны, чтобы повторные загрузки сразу после первой шли через быстрый RESUME; "
      f"затем DESTROY. Стоимость хранения 30 минут ≈ {fmt(stor_h/2)}, холодный CREATE ≈ {fmt(cold)}. "
      "STOP+RESUME на целые сутки оправдан только при очень частых пробуждениях (таблица выше). RESUME не гарантирован (GPU может быть занят), поэтому в коде есть fallback на CREATE и уничтожение зависшего диска. "
      "Режим 24/7 для такой нагрузки дороже в разы (≈3–15× в зависимости от числа видео, см. таблицы).\n")
    w("## Чего здесь нет (и что нельзя выдавать за факт)\n- Фактические списания: нужен ключ Vast и `/api/v0/charges/`; формат строк ответа не проверен.\n- Реальное время обработки видео на GPU, размер моделей, скорость скачивания — после GPU-теста.\n"
      "- Ставка диска у остановленного инстанса (может быть выше), динамика цен, комиссии платёжных систем и НДС/налоги Vast.\n")
    (ROOT / "docs" / "COSTS.md").write_text("\n".join(out) + "\n")
    print("docs/COSTS.md written,", len(out), "blocks")


if __name__ == "__main__":
    main()
