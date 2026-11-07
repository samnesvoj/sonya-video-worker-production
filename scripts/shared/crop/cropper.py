"""
cropper.py — SmartCropper module
Reconstructed from cropper.cp311-win_amd64.pyd (Nuitka-compiled, Python 3.11).

Static analysis extracted:
  - class SmartCropper (methods: __init__, crop_video)
  - function get_video_codec_params
  - all variable names, string literals, imports, and ffmpeg/cv2 API calls
  - docstring: "Crops video to 9:16 aspect ratio, centering on the detected face."

Reconstruction is clean (not byte-identical) but matches all recovered logic.
"""

import os
import subprocess
import json

import cv2
import numpy


# ---------------------------------------------------------------------------
# Helper: get video codec parameters via ffprobe
# ---------------------------------------------------------------------------

def get_video_codec_params(video_path: str) -> dict:
    """
    Probe *video_path* with ffprobe and return a dict with codec info.
    Falls back to safe defaults (libx264 / aac) if ffprobe is unavailable.

    Recovered from .pyd: variable 'params', local '_get_vcodec', key 'libx264'.
    """
    params = {
        "vcodec": "libx264",
        "acodec": "aac",
        "strict": "experimental",
    }

    def _get_vcodec(path: str) -> str:
        """Inner helper — call ffprobe to read the video codec name."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            path,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                return "libx264"
            info = json.loads(result.stdout)
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "video":
                    codec_name = stream.get("codec_name", "libx264")
                    # Map raw codec names to ffmpeg encoder names
                    codec_map = {
                        "h264": "libx264",
                        "hevc": "libx265",
                        "h265": "libx265",
                        "vp9": "libvpx-vp9",
                        "vp8": "libvpx",
                        "av1": "libaom-av1",
                    }
                    return codec_map.get(codec_name, "libx264")
        except Exception:
            pass
        return "libx264"

    if os.path.exists(video_path):
        params["vcodec"] = _get_vcodec(video_path)

    return params


# ---------------------------------------------------------------------------
# SmartCropper
# ---------------------------------------------------------------------------

class SmartCropper:
    """
    Crops video to 9:16 aspect ratio, centering on the detected face.

    Recovered from .pyd:
      - face_cascade via cv2.CascadeClassifier + haarcascade_frontalface_default.xml
      - crop_video uses alpha-smoothed center tracking
    """

    def __init__(self) -> None:
        """
        Initialise the Haar Cascade face detector.
        Recovered from .pyd: 'aface_cascade', 'ahaarcascades',
        'uhaarcascade_frontalface_default.xml'.
        """
        cascade_path = (
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"File not found: {cascade_path}")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    # ------------------------------------------------------------------
    def crop_video(
        self,
        input_video: str,
        output_path: str | None = None,
    ) -> str:
        """
        Crops video to 9:16 aspect ratio, centering on the detected face.
        Returns path to the cropped video.

        Parameters
        ----------
        input_video : str
            Path to the source video file.
        output_path : str | None
            Directory for the output file.  Defaults to the source directory.

        Recovered from .pyd:
          variables — cap, fps, width, height, total_frames, output_filename,
                      temp_video_path, final_output_path, fourcc, ret, frame,
                      gray, faces, largest_face, face_center_x,
                      target_center_x, current_center_x, alpha,
                      target_width, target_height, cropped_frame, frame_count,
                      input_audio, params
          strings  — 'Processing … -> …', 'Original: …, Target: …',
                     'Processed … frames…', 'Merging audio…',
                     'Error merging audio: …', '_cropped.mp4', 'temp_'
        """
        # ── validate input ───────────────────────────────────────────────
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"File not found: {input_video}")

        # ── open video ───────────────────────────────────────────────────
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise RuntimeError("Could not open video")

        width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ── compute 9:16 crop target ─────────────────────────────────────
        # Portrait crop: keep full height, narrow the width to height*(9/16).
        target_width  = int(height * 9 / 16)
        target_height = height

        # If the source is already narrower than 9:16, flip the axis.
        if target_width > width:
            target_width  = width
            target_height = int(width * 16 / 9)

        # ── build output paths ───────────────────────────────────────────
        base, _ext     = os.path.splitext(os.path.basename(input_video))
        output_filename = base + "_cropped.mp4"

        output_dir = output_path if output_path is not None else os.path.dirname(
            os.path.abspath(input_video)
        )
        os.makedirs(output_dir, exist_ok=True)

        # Video-only temp file; audio is merged afterwards via ffmpeg.
        temp_video_path  = os.path.join(output_dir, "temp_" + output_filename)
        final_output_path = os.path.join(output_dir, output_filename)

        print(f"Processing {base} -> {output_filename}")
        print(f"Original: {width}x{height}, Target: {target_width}x{target_height}")

        # ── set up VideoWriter ───────────────────────────────────────────
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(
            temp_video_path, fourcc, fps, (target_width, target_height)
        )

        # ── per-frame state ──────────────────────────────────────────────
        current_center_x = width // 2   # initial crop centre (pixels)
        alpha            = 0.1          # EWMA smoothing; lower = smoother pan
        frame_count      = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                )

                if len(faces) > 0:
                    # Pick the largest detected face by area.
                    largest_face   = max(faces, key=lambda face: face[2] * face[3])
                    x, y, w, h     = largest_face
                    face_center_x  = x + w // 2
                    target_center_x = face_center_x
                else:
                    # No face — hold previous centre.
                    target_center_x = current_center_x

                # Exponential moving average smoothing (avoids jitter).
                current_center_x = int(
                    alpha * target_center_x + (1.0 - alpha) * current_center_x
                )

                # ── crop ─────────────────────────────────────────────────
                x_start = current_center_x - target_width // 2
                x_start = max(0, min(x_start, width - target_width))
                x_end   = x_start + target_width

                cropped_frame = frame[:target_height, x_start:x_end]

                # Guard against rounding edge-cases.
                if (
                    cropped_frame.shape[1] != target_width
                    or cropped_frame.shape[0] != target_height
                ):
                    cropped_frame = cv2.resize(
                        cropped_frame, (target_width, target_height)
                    )

                out.write(cropped_frame)
                frame_count += 1

                if frame_count % 100 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames...")

        finally:
            cap.release()
            out.release()
            try:
                cv2.destroyAllWindows()  # not implemented in opencv-*-headless builds
            except cv2.error:
                pass

        # ── merge original audio via ffmpeg ──────────────────────────────
        try:
            import ffmpeg  # ffmpeg-python

            print("Merging audio...")
            params = get_video_codec_params(input_video)

            input_video_stream = ffmpeg.input(temp_video_path)
            input_audio        = ffmpeg.input(input_video)

            (
                ffmpeg
                .output(
                    input_video_stream.video,
                    input_audio.audio,
                    final_output_path,
                    vcodec=params.get("vcodec", "libx264"),
                    acodec="aac",
                    strict="experimental",
                )
                .overwrite_output()
                .run(quiet=True)
            )
            os.remove(temp_video_path)

        except Exception as e:
            print(f"Error merging audio: {e}")
            # Fall back: rename the silent temp file to the final path.
            import shutil
            if os.path.exists(temp_video_path):
                shutil.move(temp_video_path, final_output_path)

        return final_output_path


# ---------------------------------------------------------------------------
# One-pass face analysis (shared by all clips of a source)
# ---------------------------------------------------------------------------

FACE_ANALYSIS_VERSION = "faces-haar-v1"


def analyze_faces(input_video: str, sample_fps: float = 2.0, max_width: int = 640,
                  check=None) -> dict:
    """Detect faces ONCE over the whole source, sampled at `sample_fps`.

    Uses the same Haar cascade as SmartCropper, on frames downscaled to
    `max_width` (boxes are mapped back to source pixels). Headless: no
    windows, no GUI calls. `check()` is called between samples so a
    cancelled job can stop early.

    Returns {"meta": {...}, "samples": [{"t": sec, "faces": [[x,y,w,h], ...]}]}
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"File not found: {cascade_path}")
    cascade = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / float(sample_fps))))
    scale = min(1.0, float(max_width) / max(width, 1))
    samples = []
    idx = 0
    try:
        while True:
            if idx % step == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                if check is not None:
                    check()
                small = cv2.resize(frame, (int(width * scale), int(height * scale))) if scale < 1 else frame
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                boxes = sorted(([int(v / scale) for v in f] for f in faces),
                               key=lambda b: (-b[2] * b[3], b[0], b[1]))  # largest first, deterministic
                samples.append({"t": round(idx / fps, 3), "faces": boxes})
            else:
                if not cap.grab():
                    break
            idx += 1
    finally:
        cap.release()
    return {"meta": {"version": FACE_ANALYSIS_VERSION, "detector": "opencv-haar-frontalface-default",
                     "opencv": cv2.__version__, "width": width, "height": height, "fps": fps,
                     "frames": total, "sample_fps": sample_fps, "max_width": max_width,
                     "samples": len(samples)},
            "samples": samples}
