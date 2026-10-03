# SPDX-License-Identifier: Apache-2.0
"""NpuEmbeddings -- a MediaPipe-shaped Python facade for the body-pose model.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
MediaPipe's ``PoseLandmarker`` is the API a caller porting to this project is
most likely to be holding in their head, so this module spells its names the
same way -- ``create_from_options``, ``PoseLandmarkerOptions``,
``detect``, ``detect_for_video``, ``PoseResult.pose_landmarks``,
``NormalizedLandmark(x, y, z, visibility, presence)`` -- so that the port is a
change of names and not a rewrite.

It is NOT MediaPipe. Two differences are structural, and both are stated here
rather than discovered later:

* **There are no world landmarks.** MediaPipe's ``pose_world_landmarks`` are
  metric, hip-origin coordinates from a second network head. YOLOv8-pose has no
  such head, and inventing one from the letterboxed pixels would be a number
  with no meaning behind it. The attribute does not exist; ``getattr`` on it
  raises, which is what a caller wants to find out on the first line rather than
  on the thousandth frame.

* **There are boxes and scores, which MediaPipe does not have.** YOLOv8-pose
  detects, so it has both, and they are in ``PoseResult.pose_boxes`` and
  ``PoseResult.pose_scores``. Anything that reads only ``pose_landmarks`` -- the
  drawing code, the counting code -- works unchanged.

The numbers themselves are the C++ runtime's, not Python's: this module runs
``npuembeddings pose`` (or POSTs to ``npuembeddings pose-server``) and parses
the JSON that ``npue::pose::result_json`` writes. There is no second
implementation of the network, the front end or the decoder, so the Python and
the CLI cannot disagree about a keypoint by a rounding step -- there is only one
place the number is computed.

TWO BACKENDS, AND WHY
---------------------
``backend="cli"`` (the default) runs the binary once per ``detect`` and needs
nothing running. ``backend="http"`` keeps a ``pose-server`` child process alive
and POSTs to it, which is what a video loop wants: the model is loaded once
instead of once per frame. Both return the same object, from the same emitter.

``detect_for_video`` exists for call-shape parity and does NOT do anything
temporal. YOLOv8-pose is a per-frame network with no memory, so the answer for
frame 900 is computed from frame 900 alone; the timestamp is accepted, echoed
into ``PoseResult.timestamp_ms`` and used for nothing else. It is not refused
because "video mode is not supported" would send a caller looking for a
per-frame mode that behaves identically, and it is not smoothed or filtered
because a filter this module invented would not be the model's answer.

    from npue_pose import PoseLandmarker, PoseLandmarkerOptions

    with PoseLandmarker.create_from_options(PoseLandmarkerOptions(
            container="/tmp/opencode/p_i8.npue",
            backend="http", port=8139)) as lm:
        for person in lm.detect("bus.jpg").pose_landmarks:
            print(person[0].x, person[0].y, person[0].visibility)   # nose
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "NormalizedLandmark",
    "BoundingBox",
    "PoseResult",
    "PoseLandmarkerOptions",
    "PoseLandmarker",
    "find_binary",
]

# The 17 COCO joints, in the model's own order. Read from the container's JSON
# on every run and cross-checked against this list by _check_skeleton; a
# mismatch is refused rather than indexed around, because a landmark list that
# silently means something else is the failure this whole module exists to make
# impossible.
_COCO17 = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def find_binary(explicit: str | os.PathLike[str] | None = None) -> str:
    """Locate the ``npuembeddings`` binary, or say where it looked.

    Three places, in order: the argument, ``$NPUEMBEDDINGS_BIN``, and the build
    tree next to this file (``runtime/build/npuembeddings``), which is where
    ``bootstrap.sh`` leaves it. ``shutil.which`` is the last resort for an
    installed copy.

    Every path that was tried is named in the error, because "command not found"
    for a binary this project builds is a five-minute confusion and one line of
    text fixes it.
    """
    tried: list[str] = []
    if explicit:
        tried.append(str(explicit))
        if os.access(explicit, os.X_OK):
            return str(explicit)
    env = os.environ.get("NPUEMBEDDINGS_BIN")
    if env:
        tried.append("$NPUEMBEDDINGS_BIN=" + env)
        if os.access(env, os.X_OK):
            return env
    # <repo>/python/npue_pose.py -> <repo>/runtime/build/npuembeddings
    here = Path(__file__).resolve().parent.parent
    built = here / "runtime" / "build" / "npuembeddings"
    tried.append(str(built))
    if built.is_file() and os.access(built, os.X_OK):
        return str(built)
    onpath = shutil.which("npuembeddings")
    if onpath:
        return onpath
    tried.append("$PATH")
    raise FileNotFoundError(
        "cannot find the npuembeddings binary. Looked at:\n  "
        + "\n  ".join(tried)
        + "\nBuild it with `cmake --build runtime/build -j` first, or point "
        "NPUEMBEDDINGS_BIN at it."
    )


@dataclass(frozen=True)
class NormalizedLandmark:
    """One joint.

    ``x``/``y`` are in [0, 1] over the SOURCE image, as MediaPipe's are, so
    drawing code that multiplies by the image width works unchanged.

    ``z`` is MediaPipe's depth-relative-to-hips scale. YOLOv8-pose has no depth
    output, so this is ALWAYS ``0.0`` and ``has_depth`` is False -- which is
    more useful than a plausible-looking number.

    ``visibility`` is the model's own keypoint score (the head's third value per
    joint), thresholded by the runtime at ``keypoint``; ``PoseResult`` keeps the
    raw scores too, because "how close to the threshold" is the question a
    caller actually has. ``presence`` is MediaPipe's separate in-frame score and
    this model has none: it is 1.0 for every landmark, and ``has_presence`` is
    False, so a caller that checks it finds out rather than trusting it.
    """

    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0
    presence: float = 1.0
    index: int = 0
    name: str = ""
    visible: bool = True
    has_depth: bool = False
    has_presence: bool = False

    def __iter__(self):
        """Unpack as ``x, y, z``, the way MediaPipe's dataclass does."""
        return iter((self.x, self.y, self.z))


@dataclass(frozen=True)
class BoundingBox:
    """A person's box, in SOURCE pixels, xyxy, clipped to the image.

    ``origin_x``/``origin_y`` are 0: the runtime already undoes the letterbox,
    so these are source pixels and not letterboxed ones.
    """

    origin_x: float = 0.0
    origin_y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    @property
    def x1(self) -> float:
        return self.origin_x

    @property
    def y1(self) -> float:
        return self.origin_y

    @property
    def x2(self) -> float:
        return self.origin_x + self.width

    @property
    def y2(self) -> float:
        return self.origin_y + self.height


@dataclass
class PoseResult:
    """One image's answer.

    ``pose_landmarks`` is one list of 17 landmarks per person, in detection order
    (highest score first) -- MediaPipe's shape, not MediaPipe's ordering guarantee,
    which this runtime states: it is by descending class score after NMS, and two
    people at the same score are in anchor order, not left-to-right.
    """

    pose_landmarks: list[list[NormalizedLandmark]] = field(default_factory=list)
    pose_boxes: list[BoundingBox] = field(default_factory=list)
    pose_scores: list[float] = field(default_factory=list)
    # Everything else the runtime reported, kept rather than dropped: the
    # thresholds this answer was produced under, the letterbox transform, the
    # timings, and which backend ran the convolutions. An answer that cannot be
    # compared with another answer is how a person count gets read as a property
    # of the model.
    image_label: str = ""
    image_width: int = 0
    image_height: int = 0
    input_size: int = 0
    letterbox: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    backend: str = "host"
    dispatches: int = 0
    timing_s: dict[str, float] = field(default_factory=dict)
    timestamp_ms: int = 0
    # The emitter's object, untouched, for anything this dataclass does not name.
    raw: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.pose_landmarks)

    def __iter__(self):
        """Iterate the people, so ``for person in result`` reads like MediaPipe."""
        return iter(self.pose_landmarks)

    @property
    def num_poses(self) -> int:
        return len(self.pose_landmarks)

    def to_dict(self) -> dict[str, Any]:
        return self.raw


@dataclass
class PoseLandmarkerOptions:
    """How to build the landmarker.

    ``container`` is the .npue path, or a model name the runtime's catalogue
    knows. ``backend`` is ``"cli"`` (one process per detect) or ``"http"`` (a
    ``pose-server`` child, started on first use and stopped by ``close()``).
    ``num_poses`` caps the returned list the way MediaPipe's does, by sending
    ``max_det``; it is not a separate filter applied afterwards, because a filter
    applied afterwards would return the runtime's N-th best person rather than
    the N the runtime chose.

    ``threads``, ``artifacts`` and ``conv_on_array`` are this project's own:
    the array path is honoured and is measured slower for this network, so the
    default is off. ``port``/``bind`` are the http backend's.
    """

    container: str
    backend: str = "cli"
    num_poses: int = 300
    min_pose_detection_confidence: float | None = None
    min_pose_presence_confidence: float | None = None
    min_tracking_confidence: float | None = None
    iou_threshold: float | None = None
    threads: int = 16
    artifacts: str = ""
    conv_on_array: bool = False
    binary: str = ""
    port: int = 0
    bind: str = "127.0.0.1"
    host: str = ""
    timeout_s: float = 300.0
    extra_args: Sequence[str] = ()

    def cli_flags(self) -> list[str]:
        """The threshold flags, or nothing.

        MediaPipe's three names do not all mean something here and are not
        silently aliased:

        * ``min_pose_detection_confidence`` -> ``--conf``. The class score.
        * ``min_pose_presence_confidence`` -> ``--kpt``. THIS MODEL'S per-joint
          visibility score is what MediaPipe calls presence, so this alias is
          exact rather than convenient -- see NormalizedLandmark.
        * ``min_tracking_confidence`` -> nothing. It is a video-mode parameter
          and this runtime has no tracker; passing it is ignored and
          ``ignored_options`` says so, rather than being turned into a threshold
          that does something else.
        """
        out: list[str] = []
        if self.min_pose_detection_confidence is not None:
            out += ["--conf", _num(self.min_pose_detection_confidence)]
        if self.min_pose_presence_confidence is not None:
            out += ["--kpt", _num(self.min_pose_presence_confidence)]
        if self.iou_threshold is not None:
            out += ["--iou", _num(self.iou_threshold)]
        return out

    @property
    def ignored_options(self) -> list[str]:
        out = []
        if self.min_tracking_confidence is not None:
            out.append("min_tracking_confidence")
        return out


def _num(v: float) -> str:
    """A float as the shortest text that reads back as itself."""
    if float(v).is_integer():
        return str(int(v))
    return repr(float(v))


def _check_skeleton(doc: dict[str, Any]) -> None:
    names = tuple(doc.get("skeleton", {}).get("keypoints", ()))
    if names != _COCO17:
        raise ValueError(
            "the container's skeleton is not COCO-17, so a MediaPipe-shaped "
            "landmark index would mean something else here.\n"
            f"  expected {list(_COCO17)}\n"
            f"  got      {list(names)}\n"
            "Use the raw JSON (PoseResult.raw) rather than landmark[5]."
        )


def _to_result(doc: dict[str, Any], timestamp_ms: int = 0) -> PoseResult:
    _check_skeleton(doc)
    w = int(doc.get("width", 0))
    h = int(doc.get("height", 0))
    people: list[list[NormalizedLandmark]] = []
    boxes: list[BoundingBox] = []
    scores: list[float] = []
    for person in doc.get("landmarks", []):
        raw_kpts = person.get("keypoints", [])
        pts = []
        for i, k in enumerate(raw_kpts):
            # Normalised by the SOURCE extent, which is what MediaPipe means.
            # A landmark outside [0,1] is kept as it is rather than clamped:
            # decode.cpp clips boxes to the image and lets a joint fall outside
            # it, and clamping here would hide that.
            pts.append(NormalizedLandmark(
                x=k["x"] / w if w else k["x"],
                y=k["y"] / h if h else k["y"],
                z=0.0,
                visibility=float(k.get("score", 0.0)),
                presence=1.0,
                index=int(k.get("index", i)),
                name=str(k.get("name", "")),
                visible=bool(k.get("visible", False)),
            ))
        people.append(pts)
        scores.append(float(person.get("score", 0.0)))
        b = person.get("box", {})
        boxes.append(BoundingBox(
            origin_x=float(b.get("x1", 0.0)),
            origin_y=float(b.get("y1", 0.0)),
            width=float(b.get("x2", 0.0)) - float(b.get("x1", 0.0)),
            height=float(b.get("y2", 0.0)) - float(b.get("y1", 0.0)),
        ))
    backend = doc.get("backend", {})
    return PoseResult(
        pose_landmarks=people,
        pose_boxes=boxes,
        pose_scores=scores,
        image_label=str(doc.get("image", "")),
        image_width=w,
        image_height=h,
        input_size=int(doc.get("input_size", 0)),
        letterbox=dict(doc.get("letterbox", {})),
        thresholds=dict(doc.get("thresholds", {})),
        backend=str(backend.get("conv", "host")),
        dispatches=int(backend.get("dispatches", 0)),
        timing_s=dict(doc.get("timing_s", {})),
        timestamp_ms=timestamp_ms,
        raw=doc,
    )


def _image_bytes(image: Any) -> tuple[bytes, str]:
    """Bytes and a filename, from whatever a caller is likely to hand over.

    Accepted: a path (str/Path), raw bytes, a file-like with ``.read()``, a
    ``numpy`` array (H, W, 3) uint8, or a PIL Image. Anything else is a refusal
    naming what it got -- the runtime itself reads PNG and JPEG only and refuses
    the rest, so accepting a wider set here would mean a second, quieter decoder
    somewhere.
    """
    if isinstance(image, (str, Path)):
        p = Path(image)
        if not p.is_file():
            raise FileNotFoundError(f"no such image: {p}")
        return p.read_bytes(), p.name
    if isinstance(image, (bytes, bytearray, memoryview)):
        return bytes(image), "upload"
    if hasattr(image, "read"):
        return image.read(), "upload"
    # numpy / PIL, without importing either at module scope: this file must work
    # in a process that has neither, and a caller that has them gets them.
    if hasattr(image, "shape") and hasattr(image, "dtype"):
        import numpy as np  # local: only needed on this path
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(
                f"an image array must be (H, W, 3) or (H, W, 4); got {arr.shape}"
            )
        from PIL import Image as PILImage
        mode = "RGB" if arr.shape[2] == 3 else "RGBA"
        buf = tempfile.SpooledTemporaryFile(max_size=8 << 20)
        PILImage.fromarray(arr.astype("uint8"), mode).save(buf, format="PNG")
        buf.seek(0)
        return buf.read(), "array.png"
    # PIL, without importing it. Checked by CAPABILITY and not by class name:
    # PIL hands back a subclass for most formats (Image.open on a JPEG is a
    # JpegImageFile), and `type(x).__name__ == "Image"` is a test that passes in
    # a REPL with Image.open("a.jpg") replaced by a literal and fails on every
    # real JPEG.
    if hasattr(image, "save") and hasattr(image, "mode") and hasattr(image, "size"):
        import io
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue(), "pil.png"
    raise TypeError(
        "an image is a path, bytes, a file object, an (H, W, 3) uint8 numpy "
        f"array or a PIL Image; got {type(image).__name__}"
    )


class _Server:
    """A ``pose-server`` child process, started lazily and stopped on close().

    A thread is not used and requests are not overlapped: the C++ server serves
    one request at a time by design (one Session, one pool), so a client that
    issued concurrent requests would queue anyway. Saying so is better than
    exposing a knob that appears to work.
    """

    def __init__(self, opts: PoseLandmarkerOptions) -> None:
        self.opts = opts
        self.proc: subprocess.Popen | None = None
        self.port = opts.port
        self.base = ""
        self.log = tempfile.NamedTemporaryFile(
            prefix="npue-pose-", suffix=".log", delete=False)

    def start(self) -> None:
        if self.proc is not None:
            return
        port = self.port
        if not port:
            # A free port, found by binding and releasing. The window between the
            # two is small and the alternative -- a fixed port -- collides with
            # whatever else is on the machine, which is worse.
            import socket
            s = socket.socket()
            s.bind((self.opts.host or self.opts.bind, 0))
            port = int(s.getsockname()[1])
            s.close()
        binary = find_binary(self.opts.binary or None)
        cmd = [binary, "pose-server", self.opts.container,
               "--port", str(port), "--bind", self.opts.bind]
        if self.opts.threads:
            cmd += ["--threads", str(self.opts.threads)]
        if self.opts.artifacts:
            cmd += ["--artifacts", self.opts.artifacts]
        if self.opts.conv_on_array:
            cmd += ["--npu-extra-ops", "conv"]
        cmd += list(self.opts.extra_args)
        self.proc = subprocess.Popen(
            cmd, stdout=self.log, stderr=subprocess.STDOUT)
        self.port = port
        self.base = f"http://{self.opts.host or self.opts.bind}:{port}"
        self._await_health()

    def _await_health(self, seconds: float = 60.0) -> None:
        import time
        deadline = time.monotonic() + seconds
        last = ""
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                self.log.flush()
                with open(self.log.name, "r", errors="replace") as f:
                    tail = f.read()[-4000:]
                raise RuntimeError(
                    f"pose-server exited immediately ({self.proc.returncode}):\n"
                    + tail
                )
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception as exc:            # not up yet; keep waiting
                last = str(exc)
            time.sleep(0.1)
        raise TimeoutError(
            f"pose-server did not answer /health within {seconds:.0f}s "
            f"(last error: {last}). Its output is in {self.log.name}."
        )

    def close(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
        self.proc = None
        try:
            os.unlink(self.log.name)
        except OSError:
            pass


class PoseLandmarker:
    """The landmarker. Build it with ``create_from_options`` or the constructor."""

    def __init__(self, options: PoseLandmarkerOptions) -> None:
        if options.backend not in ("cli", "http"):
            raise ValueError(
                f"backend is 'cli' or 'http', not {options.backend!r}. 'cli' "
                "runs the binary per detect; 'http' keeps a pose-server child "
                "alive, which is what a video loop wants."
            )
        if options.num_poses < 1:
            raise ValueError(f"num_poses is at least 1, not {options.num_poses}")
        self._opts = options
        self._server = _Server(options) if options.backend == "http" else None
        atexit.register(self.close)

    @classmethod
    def create_from_options(cls, options: PoseLandmarkerOptions) -> "PoseLandmarker":
        """MediaPipe's spelling. It is the constructor, under the other name."""
        return cls(options)

    # -- the two entry points -------------------------------------------------

    def detect(self, image: Any) -> PoseResult:
        """Landmarks for one image, in source-pixel order and normalised form.

        Blocking, and about 0.3 s for a 640x640 frame on 16 CPU threads with the
        i8 container -- measured on this machine, not a target. The http backend
        pays the model load once and is roughly the same per frame after that,
        because this network's array path is slower than the host's (see
        runtime/include/pose/net.hpp).
        """
        return self._run(image, 0)

    def detect_for_video(self, image: Any, timestamp_ms: int) -> PoseResult:
        """Same as ``detect``, and per-frame: no temporal state is used.

        ``timestamp_ms`` is echoed into ``PoseResult.timestamp_ms`` and nothing
        else reads it. YOLOv8-pose has no tracker and no memory, so this is not
        MediaPipe's video mode -- see the module docstring.
        """
        return self._run(image, int(timestamp_ms))

    # -- the two backends -----------------------------------------------------

    def _run(self, image: Any, timestamp_ms: int) -> PoseResult:
        if self._server is not None:
            return self._run_http(image, timestamp_ms)
        return self._run_cli(image, timestamp_ms)

    def _cli_argv(self, path: Path, thresholds: list[str]) -> list[str]:
        o = self._opts
        cmd = [find_binary(o.binary or None), "pose", o.container, str(path)]
        if o.num_poses != 300:
            cmd += ["--max-det", str(o.num_poses)]
        cmd += thresholds
        if o.threads:
            cmd += ["--threads", str(o.threads)]
        if o.artifacts:
            cmd += ["--artifacts", o.artifacts]
        if o.conv_on_array:
            cmd += ["--npu-extra-ops", "conv"]
        cmd += list(o.extra_args)
        return cmd

    def _run_cli(self, image: Any, timestamp_ms: int = 0) -> PoseResult:
        data, name = _image_bytes(image)
        thresholds = self._opts.cli_flags()
        # A temporary file, because the CLI's image argument is a PATH and the
        # alternative is a second ingest path through the protocol. Deleted in
        # the finally, including on the error paths, and named with mkstemp so
        # two landmarkers in one process cannot collide.
        fd, path = tempfile.mkstemp(prefix="npue-pose-", suffix=".img")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            proc = subprocess.run(
                self._cli_argv(Path(path), thresholds),
                capture_output=True, timeout=self._opts.timeout_s)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        if proc.returncode != 0:
            # The runtime writes diagnostics to stderr and the result to stdout,
            # so stderr IS the error message here and it is passed through
            # rather than replaced by "exit status 1".
            raise RuntimeError(
                f"npuembeddings pose exited {proc.returncode}:\n"
                + proc.stderr.decode("utf-8", "replace").strip()
            )
        res = _to_result(json.loads(proc.stdout.decode("utf-8")), timestamp_ms)
        # The CLI's `image` field is the path it was given, and what it was given
        # is a temporary file -- so the label the caller reads back would be
        # /tmp/npue-pose-XXXX.img, a name that means nothing and that changes
        # every call. Replaced with the name of the image the CALLER passed, and
        # only on the dataclass: PoseResult.raw keeps the emitter's own object
        # untouched, so a caller comparing against the raw CLI output still sees
        # exactly what the runtime printed.
        if name:
            res.image_label = name
        return res

    def _run_http(self, image: Any, timestamp_ms: int) -> PoseResult:
        assert self._server is not None
        self._server.start()
        data, name = _image_bytes(image)
        boundary = "----npue" + uuid.uuid4().hex
        # Built by hand rather than with urllib's multipart helper: there isn't
        # one, and the body here is four short fields plus one blob, which is
        # smaller than the helper's own machinery.
        parts = [("image", name, "application/octet-stream", data)]
        o = self._opts
        if o.min_pose_detection_confidence is not None:
            parts.append(("conf", None, None,
                          _num(o.min_pose_detection_confidence).encode()))
        if o.min_pose_presence_confidence is not None:
            parts.append(("kpt", None, None,
                          _num(o.min_pose_presence_confidence).encode()))
        if o.iou_threshold is not None:
            parts.append(("iou", None, None,
                          _num(o.iou_threshold).encode()))
        if o.num_poses != 300:
            parts.append(("max_det", None, None, str(o.num_poses).encode()))
        chunks: list[bytes] = []
        for field_name, filename, ctype, payload in parts:
            head = f"--{boundary}\r\nContent-Disposition: form-data; " \
                   f"name=\"{field_name}\""
            if filename:
                head += f"; filename=\"{filename}\""
            head += "\r\n"
            if ctype:
                head += f"Content-Type: {ctype}\r\n"
            chunks.append((head + "\r\n").encode())
            chunks.append(payload)
            chunks.append(b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode())
        req = urllib.request.Request(
            self._server.base + "/v1/pose", data=b"".join(chunks),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                     "Content-Length": str(sum(len(c) for c in chunks))},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._opts.timeout_s) as r:
                doc = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            try:
                msg = json.loads(body)["error"]["message"]
            except Exception:
                msg = body.strip()
            # 4xx is the caller's own request and is raised as ValueError; 5xx is
            # ours and is raised as RuntimeError. A caller that catches
            # RuntimeError to retry a server fault must not also retry "conf was
            # not a number", which is what one exception type would tell it.
            if 400 <= exc.code < 500:
                raise ValueError(f"pose-server rejected the request: {msg}")
            raise RuntimeError(f"pose-server failed ({exc.code}): {msg}")
        return _to_result(doc, timestamp_ms)

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Stop the child server, if there is one. Idempotent."""
        if self._server is not None:
            self._server.close()

    def __enter__(self) -> "PoseLandmarker":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:                 # best effort, never raises
        try:
            self.close()
        except Exception:
            pass


def _main(argv: Sequence[str]) -> int:
    """`python3 npue_pose.py MODEL IMAGE...` -- the smoke test, in eleven lines."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("container")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--backend", default="cli", choices=("cli", "http"))
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--artifacts", default="")
    ap.add_argument("--conv-on-array", action="store_true")
    a = ap.parse_args(argv)
    with PoseLandmarker(PoseLandmarkerOptions(
            container=a.container, backend=a.backend,
            min_pose_detection_confidence=a.conf, artifacts=a.artifacts,
            conv_on_array=a.conv_on_array)) as lm:
        for path in a.images:
            r = lm.detect(path)
            print(f"{path}: {r.num_poses} person/people "
                  f"({r.image_width}x{r.image_height}, {r.backend} conv, "
                  f"{r.timing_s.get('total', 0.0):.2f}s)")
            for i, person in enumerate(r.pose_landmarks):
                nose = person[0]
                print(f"  person {i}: score {r.pose_scores[i]:.3f} "
                      f"box {r.pose_boxes[i].x1:.0f},{r.pose_boxes[i].y1:.0f} "
                      f"{r.pose_boxes[i].width:.0f}x{r.pose_boxes[i].height:.0f}  "
                      f"nose ({nose.x:.3f},{nose.y:.3f}) vis {nose.visibility:.2f}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
