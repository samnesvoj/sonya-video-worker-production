"""tools/slim_site_packages.py: removes only dead files, and refuses when its assumptions do not hold."""
import pytest

from tools.slim_site_packages import record_hash, slim


def _site(tmp_path, owner="opencv_contrib_python"):
    sp = tmp_path / "sp"
    (sp / "cv2").mkdir(parents=True)
    so = sp / "cv2" / "cv2.abi3.so"
    so.write_bytes(b"ELF-contrib")
    for w in ("opencv_contrib_python", "opencv_python", "opencv_python_headless"):
        (sp / f"{w}.libs").mkdir()
        (sp / f"{w}.libs" / "libopenblas.so").write_bytes(b"x" * 100)
        di = sp / f"{w}-4.14.0.94.dist-info"
        di.mkdir()
        h = record_hash(so) if w == owner else "sha256=other"
        (di / "RECORD").write_text(f"cv2/cv2.abi3.so,{h},11\n")
    (sp / "mediapipe" / "tasks").mkdir(parents=True)
    (sp / "mediapipe" / "tasks" / "a.so").write_bytes(b"m" * 50)
    (sp / "mediapipe-1.2.2.dist-info").mkdir()
    (sp / "onnx" / "backend" / "test").mkdir(parents=True)
    (sp / "onnx" / "backend" / "test" / "d.pb").write_bytes(b"t")
    (sp / "onnx" / "__init__.py").write_text("")
    return sp


def test_removes_dead_opencv_copies_mediapipe_and_onnx_tests_keeps_metadata(tmp_path):
    sp = _site(tmp_path)
    freed = slim(sp)
    assert (sp / "opencv_contrib_python.libs" / "libopenblas.so").exists() and (sp / "cv2" / "cv2.abi3.so").exists()
    assert not (sp / "opencv_python.libs").exists() and not (sp / "opencv_python_headless.libs").exists()
    assert not (sp / "mediapipe").exists() and (sp / "mediapipe-1.2.2.dist-info").exists()
    assert not (sp / "onnx" / "backend" / "test").exists() and (sp / "onnx" / "__init__.py").exists()
    assert all((sp / f"{w}-4.14.0.94.dist-info").exists() for w in ("opencv_contrib_python", "opencv_python", "opencv_python_headless"))
    assert freed == 251


def test_refuses_when_the_active_cv2_is_another_wheel(tmp_path):
    sp = _site(tmp_path, owner="opencv_python_headless")
    with pytest.raises(SystemExit):
        slim(sp)
    assert (sp / "opencv_python.libs").exists() and (sp / "mediapipe").exists()      # nothing removed
