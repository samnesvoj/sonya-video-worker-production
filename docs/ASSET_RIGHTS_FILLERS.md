# ASSET_RIGHTS — нижние заполнители (bottom_filler)

Правило: в продакшен попадают только ролики с лицензией из списка (`own-generated`, `CC0`, `Pexels-commercial`, `Pixabay-commercial`),
без водяных знаков/логотипов, с совпадающим sha256 (`release/fillers.manifest.json`). Чужие летсплеи/YouTube — не берём.

| id | тип | происхождение | лицензия | проверено |
|---|---|---|---|---|
| gameplay_runner_side, gameplay_runner_lanes, gameplay_hopper | gameplay | сгенерированы нами `tools/gen_fillers.py` (PIL + ffmpeg), ни одного стороннего пикселя | own-generated (права полностью наши) | 2026-11-10 |
| asmr_slime_pink, asmr_slime_blue, asmr_soap_cut, asmr_bubbles | asmr | то же | own-generated | 2026-11-10 |

Сторонние ролики (Pexels/Pixabay/CC0) в стартовую библиотеку НЕ включены: скачивание с их сайтов требует API-ключа/ручной проверки лицензии
каждого файла; добавлять строкой в манифест + в эту таблицу (URL страницы, дата, текст лицензии) только после проверки.
Звука нет (`-an`), в сборку фильтром не попадает даже если появится.
