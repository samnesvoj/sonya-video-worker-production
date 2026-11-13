"""stream_layout: the webcam rectangle by its borders, chat columns, windows that leave them out
(owner 2026-11-10: «в верхней половине — ТОЛЬКО вебка», a Twitch chat under the webcam ruined a clip)."""
import cv2
import numpy as np
import pytest

from scripts.shared.crop import stream_layout as sl

W, H = 960, 540
CAM = (20, 20, 300, 170)            # webcam overlay, top-left, with a 2 px border
CHAT = (20, 200, 200, 300)          # chat column right under it


def _frames(n=16, seed=0, chat=True, border=True, face=True):
    rng = np.random.default_rng(seed)
    out = []
    room = rng.integers(40, 90, (CAM[3], CAM[2])).astype(np.uint8)       # a still room behind the streamer
    room = cv2.GaussianBlur(room, (9, 9), 3)
    for i in range(n):
        g = rng.integers(0, 255, (H // 8, W // 8)).astype(np.uint8)      # the game: changes everywhere
        g = cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        x, y, w, h = CAM
        g[y:y + h, x:x + w] = room
        if face:                                                       # the streamer moves a little
            cx = x + w // 2 + int(6 * np.sin(i))
            cv2.ellipse(g, (cx, y + 80), (28, 36), 0, 0, 360, 200, -1)
            cv2.rectangle(g, (cx - 60, y + 120), (cx + 60, y + h - 1), 150, -1)
        if border:
            cv2.rectangle(g, (x, y), (x + w, y + h), 255, 2)
        if chat:
            cx0, cy0, cw, ch = CHAT
            g[cy0:cy0 + ch, cx0:cx0 + cw] = 25                         # dark chat box
            for line in range(14):                                     # scrolling messages
                yy = cy0 + 8 + line * 20
                txt = "".join(rng.choice(list("abcdefghij klmnopqrs")) for _ in range(rng.integers(8, 18)))
                cv2.putText(g, txt, (cx0 + 4, yy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 230, 1)
        out.append(g)
    return out


def test_webcam_rect_found_by_its_border_and_excludes_chat():
    fr = _frames()
    face = [CAM[0] + CAM[2] // 2 - 28, CAM[1] + 44, 56, 72]
    cz = sl.chat_zones(fr)
    cam = sl.webcam_rect(fr, face, chat=[z["box"] for z in cz["zones"]])
    assert cam is not None and cam["by"] == "face"
    x, y, w, h = cam["rect"]
    assert abs(x - CAM[0]) <= 4 and abs(y - CAM[1]) <= 4
    assert abs(w - CAM[2]) <= 6 and abs(h - CAM[3]) <= 6
    assert y + h < CHAT[1]                                    # stops at the border, not at the chat's bottom


def test_webcam_rect_without_a_face_by_lines():
    """Face paint / a mask: no YuNet face, the bordered still-room-with-motion rectangle is still found."""
    fr = _frames(face=True)
    cam = sl.webcam_rect(fr, None)
    assert cam is not None and cam["by"] == "lines"
    assert abs(cam["rect"][0] - CAM[0]) <= 4 and abs(cam["rect"][2] - CAM[2]) <= 6


def test_chat_column_detected_and_static_hud_ignored():
    fr = _frames()
    for g in fr:                                              # static HUD text: same every frame
        cv2.putText(g, "$4300  100  HP", (600, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
    zones = sl.chat_zones(fr)["zones"]
    assert zones, "chat column not found"
    z = max(zones, key=lambda z: z["box"][3])["box"]
    assert sl._overlap(z, CHAT) > 0.4 * CHAT[2] * CHAT[3]
    assert all(sl._overlap(zz["box"], (560, 490, 300, 50)) == 0 for zz in zones)


def test_no_chat_no_zones():
    assert sl.chat_zones(_frames(chat=False))["zones"] == []


def test_cam_window_stays_inside_the_rect():
    rect = [100, 50, 480, 270]                                # 16:9 webcam
    cw = sl.cam_window(rect, 340, 160)
    x, y, w, h = cw
    assert x >= rect[0] and y >= rect[1] and x + w <= rect[0] + rect[2] and y + h <= rect[1] + rect[3]
    assert abs(w / h - 1080 / 960) < 0.02
    assert h == 270                                           # as tall as the webcam (only the sides go)


def test_cam_window_narrow_rect_keeps_full_width():
    x, y, w, h = sl.cam_window([0, 0, 300, 400], 150, 120)
    assert w == 300 and h == int(300 / (1080 / 960)) // 2 * 2 and y >= 0


def test_game_window_avoids_chat_column():
    Wsrc, Hsrc = 1920, 1080
    chat = [1500, 100, 420, 900]                              # chat panel on the right
    box, cov = sl.game_window(Wsrc, Hsrc, [chat])
    assert cov == 0.0
    assert box[0] + box[2] <= 1500
    box2, _ = sl.game_window(Wsrc, Hsrc, [])
    assert abs(box2[0] + box2[2] / 2 - Wsrc / 2) <= 8         # nothing to avoid: the crosshair centre


def test_fallback_cam_narrows_away_from_chat():
    face = [200, 100, 60, 80]
    chat = [[330, 0, 300, 1080]]
    x, y, w, h = sl.fallback_cam(face, 1920, 1080, chat)
    assert x + w <= 330 + 2 or w <= 2.2 * 60 + 2
    assert x <= 200 and x + w >= 260                          # the face stays whole


def test_stream_game_track_uses_survey_rect_when_border_not_seen():
    an = {"samples": [{"t": i * 0.1, "faces": [[220, 120, 60, 80, 0.9, 0.0]]} for i in range(20)], "cuts": []}
    lay = {"webcam": None, "chat": [[0, 400, 300, 600]]}
    tr = sl.stream_game_track(an, lay, 30.0, width=1920, height=1080,
                              survey_rects=[{"rect": [0, 0, 560, 320], "seen": 5}])
    assert tr["webcam_source"] == "rect:survey"
    x, y, w, h = tr["cam"]
    assert x >= 0 and y >= 0 and x + w <= 560 and y + h <= 320
    # game-v3: the whole game stays; the chat box on the game (not an edge column ≤17 %) and the webcam are masked
    assert tr["game"] == [0, 0, 1920, 1080]
    assert any(sl._overlap(m, [0, 400, 300, 600]) > 0 for m in tr["game_masks"])
    assert any(sl._overlap(m, [0, 0, 560, 320]) > 0 for m in tr["game_masks"])


def test_stream_game_track_none_without_webcam():
    an = {"samples": [{"t": i * 0.1, "faces": []} for i in range(20)], "cuts": []}
    assert sl.stream_game_track(an, {"webcam": None, "chat": []}, 30.0, width=1920, height=1080) is None


def test_moving_face_is_not_a_webcam():
    an = {"samples": [{"t": i * 0.1, "faces": [[100 + i * 40, 300, 60, 80, 0.9, 0.0]]} for i in range(20)]}
    assert sl.webcam_face(an, 1920, 1080) is None


def test_render_game_layout_one_pass_absolute_paths(monkeypatch, tmp_path):
    """One ffmpeg pass straight from the source: lanczos, cam + game stacked, 6 Mb/s args, and absolute paths
    (ffmpeg runs from the subtitle folder; a relative output path broke every clip of the first v2 run)."""
    from scripts.shared.crop import smart_crop_adapter as sca
    from scripts.shared import cut_profile
    seen = {}

    def fake_run(make, args, **kw):
        seen["cmd"] = make(args)
        seen["cwd"] = kw.get("cwd")
        import subprocess
        return subprocess.CompletedProcess(seen["cmd"], 0, "", ""), "cpu", 0.1

    monkeypatch.setattr(cut_profile, "run_encode", fake_run)
    monkeypatch.chdir(tmp_path)
    sub = tmp_path / "subs"
    sub.mkdir()
    (sub / "stream_clip_01.ass").write_text("[Script Info]\n", encoding="utf-8")
    track = {"layout": "game", "cam": [10, 10, 540, 480], "game": [400, 54, 1092, 972]}
    sca.render_tracked("src.mp4", "out/clip.mp4", 100.0, 40.0, track,
                       encoder_args=cut_profile.encoder_args("cpu"), subtitle_path=str(sub / "stream_clip_01.ass"))
    cmd = seen["cmd"]
    assert cmd[cmd.index("-i") + 1] == str(tmp_path / "src.mp4")
    assert cmd[-1] == str(tmp_path / "out" / "clip.mp4")
    vf = cmd[cmd.index("-vf") + 1]
    assert vf.count("flags=lanczos") == 2 and "vstack" in vf and "crop=540:480:10:10" in vf
    assert cmd.count("-i") == 1 and cmd[cmd.index("-maxrate") + 1] == "8M"
    assert seen["cwd"] == str(sub)


def test_game_window_shows_less_of_a_big_top_left_webcam_than_the_centred_window():
    """Buster stream: webcam 0..1092 x 0..604 top-left reaches the crosshair, so it cannot be left out entirely;
    the window must still show less of it than the centred 90 % window did, crosshair kept in the picture."""
    cam = [0, 0, 1092, 604]
    box, cov = sl.game_window(1920, 1080, [cam])
    gh = int(1080 * 0.9) // 2 * 2
    gw = int(gh * 1080 / 960) // 2 * 2
    centred = [(1920 - gw) // 2, (1080 - gh) // 2, gw, gh]
    assert cov < sl._overlap(centred, cam) / float(gw * gh)
    gx, gy, w, h = box
    assert gx + 0.2 * w <= 960 <= gx + 0.8 * w and gy + 0.2 * h <= 540 <= gy + 0.8 * h


def test_full_game_layout_whole_game_fit_to_width():
    """Owner 2026-11-10: «ВСЮ игру было видно» — 16:9 game → 1080x608 at the bottom, no zoom."""
    geo = sl.full_game_geometry(1920, 1080, [1500, 700, 400, 360], 1700, [[1500, 700, 400, 360]])
    assert geo["game"] == [0, 0, 1920, 1080] and geo["game_h"] == 608 and geo["top_h"] == 1920 - 608
    assert geo["game_masks"]                                   # the webcam on the game is blurred, not cropped


def test_full_game_layout_trims_edge_chat_column():
    geo = sl.full_game_geometry(1920, 1080, [0, 0, 400, 300], 200, [[1620, 0, 300, 1080], [0, 0, 400, 300]])
    gx, gy, gw, gh = geo["game"]
    assert gx + gw <= 1620 and gw >= 1600                     # only the chat column went
    assert geo["game_h"] == round(1080 * gh / gw / 2) * 2


def test_small_webcam_is_never_stretched_over_2x():
    """«качество вебки плохое»: a 320x180 camera is shown ≤ 2x (640 px wide), not blown up to 1080."""
    geo = sl.full_game_geometry(1920, 1080, [0, 0, 320, 180], 160, [[0, 0, 320, 180]])
    assert geo["cam_upscale"] <= 2.0 + 1e-6
    assert geo["cam_size"][0] <= 2 * 320 and geo["cam_size"][0] < 1080
    px, py = geo["cam_pos"]
    assert abs(px + geo["cam_size"][0] / 2 - 540) <= 2         # centred


def test_big_webcam_fills_the_top():
    geo = sl.full_game_geometry(1920, 1080, [0, 0, 1092, 604], 546, [[0, 0, 1092, 604]])
    assert geo["cam_size"][0] >= 1070 or geo["cam_size"][1] >= geo["top_h"] - 10
    assert geo["cam_upscale"] <= 2.0
