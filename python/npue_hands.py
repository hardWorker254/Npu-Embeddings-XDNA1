# SPDX-License-Identifier: Apache-2.0
"""NpuEmbeddings -- a MediaPipe-shaped Python facade for the hand-landmark model.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
MediaPipe's ``HandLandmarker`` is the API a caller porting hands over to this
project is most likely holding in their head, so this module spells its names the
same way -- ``create_from_options``, ``HandLandmarkerOptions``, ``detect``,
``detect_for_video``, ``HandLandmarkerResult.landmarks``,
``.world_landmarks``, ``.handedness`` -- so that the port is a change of names
and not a rewrite.

It is NOT MediaPipe, and four differences are structural. All four are stated
here rather than discovered on the thousandth frame:

* **The three confidence options are REFUSED, not ignored.** MediaPipe's
  ``min_detection_confidence``, ``min_presence_confidence`` and
  ``min_tracking_confidence`` all mean something there because MediaPipe owns
  those thresholds. Here they live in the CONTAINER -- ``score_threshold``,
  ``nms_threshold`` -- because they are part of how this checkpoint was trained,
  and the OpenCV zoo demo that recorded the golden used the container's values,
  not a command line's. Setting one raises ``ValueError`` naming the flag and
  the container key. Silently dropping them would hand back a landmarker whose
  answers do not match the threshold the caller asked for, with nothing in the
  output saying so.

* **There is no tracker.** ``detect_for_video`` exists for call-shape parity
  and does NOT do anything temporal: both networks are per-frame, so the answer
  for frame 900 is computed from frame 900 alone. ``min_tracking_confidence`` is
  refused for the same reason the pose facade ignores it, except that here it is
  refused rather than ignored because here it is also one of the three
  thresholds that have no flag.

* **``landmarks[].z`` is in PIXELS, relative to the wrist.** MediaPipe normalizes
  depth by the image width, so its ``z`` is comparable across resolutions. This
  runtime does not: the network regresses depth in the 224px crop's own units
  and ``decode.cpp`` scales it by the crop's scale and subtracts the wrist, so a
  ``z`` of 30 means thirty pixels nearer than the wrist -- thirty pixels of a
  *crop*, not thirty pixels of real distance, and not comparable across two
  hands cropped at different scales. ``Landmark.has_depth`` is True and
  ``z_is_pixels`` is True, so a caller finds out here rather than assuming
  MediaPipe's units.

* **Confidence is PER HAND, not per landmark.** ``presence`` is one sigmoid for
  the whole hand; there is no per-joint score, because this architecture has no
  per-joint head. So ``Landmark`` has no ``visibility`` and no ``presence``
  attribute at all -- not a constant, absent -- and a caller that reaches for one
  finds out on the first frame instead of drawing every point at full confidence
  because a placeholder said 1.0. The per-hand values are on ``HandResult``.

WHAT IS THE SAME AS THE POSE FACADE
-----------------------------------
The numbers are the C++ runtime's, not Python's: this module runs
``npuembeddings hands`` and parses the JSON that ``hands_mode.hpp`` writes. There
is no second implementation of the network, the front end or the decoder, so the
Python and the CLI cannot disagree about a landmark by a rounding step.
``find_binary`` and the image-bytes helper are IMPORTED from ``npue_pose`` rather
than copied, because "bytes and a filename from whatever a caller hands you" has
one right answer and two copies of it is how they stop matching.

ONE BACKEND ONLY, AND WHY IT IS NOT A CHOICE
--------------------------------------------
``backend="cli"`` and nothing else. The ``http`` backend is REFUSED with the
reason rather than implemented or silently downgraded, because arch 7 has no
HTTP endpoint: there is no ``/v1/hands``, ``npuembeddings serve`` on a hands
container is refused by name, and a Python layer that quietly fell back to a
process per frame while claiming to be a server would make a video loop slower
with no error raised.

What it costs, measured on this machine at 520x512 with one hand and 16
threads, six runs:

    this backend, end to end   111 - 120 ms   (median 116)
    the model alone            82 -  92 ms   (median 89)
    the backend's own share                ~27 ms   process, load, parse

So a ``serve`` child would be about 1.36x the throughput of this one -- 8.6 fps
against about 11 -- which is a real gain and not the order of magnitude it first
looks like, because the model is the cost and the model is paid either way. That
is measured rather than asserted: adding ``server/hands_backend.cpp`` beside
``server/pose_backend.cpp`` would take the ~27 ms out of the loop, and the
session, the JSON emitter and the multipart reader are already written, so it is
a binding and not a network.

    from npue_hands import HandLandmarker, HandLandmarkerOptions

    with HandLandmarker.create_from_options(HandLandmarkerOptions(
            container="models/mediapipe-hands/hands.npue")) as hl:
        for hand in hl.detect("hand.jpg").landmarks:
            print(hand[0].x, hand[0].y, hand[0].z)   # wrist, z==0.0 by definition
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from npue_pose import _image_bytes, find_binary

__all__ = [
    "NormalizedLandmark",
    "WorldLandmark",
    "HandBox",
    "Category",
    "HandResult",
    "PalmDetection",
    "HandLandmarkerOptions",
    "HandLandmarker",
    "HAND_NAMES",
    "HAND_CONNECTIONS",
    "find_binary",
]

# MediaPipe's canonical 21-landmark hand layout, in the network's own order.
# This is the ONE place the topology is written down, and it is checked against
# what the runtime actually emitted -- see _check_hands. A landmark list that
# silently means something else is the failure this whole module exists to make
# impossible, so the check raises rather than indexing around a mismatch.
HAND_NAMES = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip",
    "ring_finger_tip",
    "pinky_finger_mcp", "pinky_finger_pip", "pinky_finger_dip",
    "pinky_finger_tip",
)

# Bones, as (from, to) index pairs. Edges only -- drawn as lines between joints,
# the same choice the pose skeleton's render makes, because a filled polygon
# would imply the hand is a solid and occludes the fingers it is supposed to show.
#
# MediaPipe's HAND_CONNECTIONS, joint for joint. Five chains of four bones, with
# each finger hanging off the PREVIOUS finger's knuckle rather than the wrist --
# that is MediaPipe's choice and it is kept rather than "corrected", because a
# caller porting drawing code from MediaPipe would otherwise get a different
# picture. 21 joints, 21 edges, so four cycles over 20: the four loops between
# adjacent knuckles.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),                 # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),                 # index
    (5, 9), (9, 10), (10, 11), (11, 12),            # middle, off index mcp
    (9, 13), (13, 14), (14, 15), (15, 16),          # ring, off middle mcp
    (13, 17), (17, 18), (18, 19), (19, 20),         # pinky, off ring mcp
    (0, 17),                                          # wrist to pinky mcp
)


def _check_hands(doc: dict[str, Any]) -> None:
    """Refuse a landmark count this module's topology does not describe."""
    n = len(HAND_NAMES)
    for i, hand in enumerate(doc.get("hands", ())):
        got = len(hand.get("landmarks", ()))
        if got != n:
            raise ValueError(
                f"hand {i} has {got} landmarks, but this module's topology has "
                f"{n} -- MediaPipe's hand layout, which is what the names in "
                f"HAND_NAMES and the bones in HAND_CONNECTIONS index.\n"
                "  The container's num_landmarks says something else, so a "
                "landmark index would mean something different here.\n"
                "Use HandResult.raw, which is the runtime's own object, rather "
                "than landmark[9]."
            )


@dataclass(frozen=True)
class NormalizedLandmark:
    """One hand joint, in the SOURCE image.

    ``x``/``y`` are in [0, 1] over the source extent, as MediaPipe's are, so
    drawing code that multiplies by the image width works unchanged. A joint
    outside [0, 1] is kept as it is rather than clamped: the landmark network's
    output is not clipped to the frame, and clamping here would hide a hand that
    is half out of the picture.

    ``z`` is NOT MediaPipe's z. See the module docstring: it is pixels relative
    to the wrist, so it is exactly 0.0 on ``landmarks[0]`` and not comparable
    between two hands cropped at different scales. ``z_is_pixels`` says so on
    the object, for the same reason ``has_depth`` does.

    There is no ``visibility`` and no ``presence``. This architecture has no
    per-joint score -- confidence is one number per HAND, on ``HandResult`` --
    so the attributes are absent rather than 0.0, which a caller reading a
    constant could mistake for a measurement.
    """

    x: float
    y: float
    z: float = 0.0
    index: int = 0
    name: str = ""
    has_depth: bool = True
    z_is_pixels: bool = True

    def __iter__(self):
        """Unpack as ``x, y, z``, the way MediaPipe's dataclass does."""
        return iter((self.x, self.y, self.z))


@dataclass(frozen=True)
class WorldLandmark:
    """One hand joint in METRES.

    NOT in the frame's pixels and not in [0, 1]. This is the metric frame the
    landmark checkpoint regresses, with the crop's rotation applied so the axes
    match the frame's orientation -- but WITHOUT a translation, and that is why
    ``world_landmarks[0]`` (the wrist) is NOT ``(0, 0, 0)``: the crop's own metric
    origin stays where the network put it, near the wrist but not on it. Measured
    on the zoo's ``hand_plain.png``, the wrist lands at about
    ``(-0.004, 0.082, 0.009)`` and the 21 points span 0.12 x 0.17 x 0.05 m, which
    is a hand. So do NOT test these for validity by expecting a zero wrist --
    test the extent.

    This is the list for measuring a hand across two frames or two cameras, not
    for drawing: a caller multiplying by the image width gets metres per pixel.

    A separate class from ``NormalizedLandmark`` even though MediaPipe uses one
    for both, because the same attribute name meaning "pixels from the wrist" in
    one list and "metres from the wrist" in the other is a unit confusion with no
    compile-time symptom.
    """

    x: float
    y: float
    z: float = 0.0
    index: int = 0
    name: str = ""
    has_depth: bool = True
    z_is_pixels: bool = False
    units: str = "m"

    def __iter__(self):
        return iter((self.x, self.y, self.z))


@dataclass(frozen=True)
class HandBox:
    """The box to draw around a hand: xyxy, SOURCE pixels.

    NOT clipped to the image, and that is deliberate: it is the shifted, 1.65x
    enlarged rectangle around the 21 joints, computed in ``decode.cpp`` from
    landmarks that may be outside the frame, so on a hand that runs off the edge
    the box legitimately has a negative coordinate. Clipping it here would make
    a box and the landmarks inside it disagree about where the frame ends.
    """

    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass(frozen=True)
class Category:
    """One label with its score -- the handedness of one hand.

    MediaPipe's ``Category`` also carries an ``index``, the class id from the
    classifier's own output list. The runtime reports only the display name and
    the sigmoid, so ``index`` is absent rather than guessed at: "Right" is index
    1 in one checkpoint's output order and 0 in another's, and reading it off
    the name would be a coincidence dressed as a fact.
    """

    display_name: str = ""
    score: float = 0.0


@dataclass(frozen=True)
class PalmDetection:
    """One palm-detector box, before the landmark network ran on it.

    Kept because it is the honest view of stage 1 and it is what tells two
    hands apart: the landmark net's answer has no box of its own, only joints,
    and a caller drawing a box has to choose between the palm box (tight, from
    the detector, 7 keypoints, no fingers) and the hand box (looser, from the
    joints, the one most people want). Both are here so the choice is the
    caller's rather than this module's.

    ``keypoints`` is the detector's 7 palm landmarks as ``(x, y)`` in source
    pixels -- wrist, index base, middle base, ring base, little base, and two
    more -- which is a subset of the palm head, not the 21 the second network
    produces. Index 0 is the wrist and index 2 the middle-finger base: the two
    the rotation is computed from, and both are read from the container rather
    than assumed here.
    """

    score: float = 0.0
    box: HandBox = field(default_factory=HandBox)
    keypoints: tuple[tuple[float, float], ...] = ()


@dataclass
class HandResult:
    """One image's answer.

    ``landmarks`` is one list of 21 joints per hand, in detection order --
    highest score first, which is the order NMS produces. MediaPipe does not
    promise an order at all; this runtime states this one.

    ``world_landmarks`` is the same shape in metres, and ``handedness`` is one
    ``Category`` per hand. Everything else the runtime reported is kept rather
    than dropped -- the letterbox transform, the per-stage timings, the palm
    detections -- because an answer that cannot be compared with another answer
    is how a per-frame cost gets read as a property of the model.
    """

    landmarks: list[list[NormalizedLandmark]] = field(default_factory=list)
    world_landmarks: list[list[WorldLandmark]] = field(default_factory=list)
    handedness: list[Category] = field(default_factory=list)
    presence: list[float] = field(default_factory=list)
    hand_boxes: list[HandBox] = field(default_factory=list)
    hand_scores: list[float] = field(default_factory=list)
    palm_detections: list[PalmDetection] = field(default_factory=list)
    image_label: str = ""
    image_width: int = 0
    image_height: int = 0
    letterbox: dict[str, float] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    timestamp_ms: int = 0
    # The emitter's object, untouched, for anything this dataclass does not name.
    raw: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.landmarks)

    def __iter__(self):
        """Iterate the hands, so ``for hand in result`` reads like MediaPipe."""
        return iter(self.landmarks)

    def __getitem__(self, i):
        """``result[0]`` is the first hand, as in MediaPipe.

        Present for the reason spelled out in the pose facade: ``__len__`` and
        ``__iter__`` without it make an object that looks indexable and raises
        TypeError on the first attempt, and ``result[0]`` is the most idiomatic
        thing anyone writes against a landmarker result. Slices come back as a
        plain list.
        """
        return self.landmarks[i]

    @property
    def num_hands(self) -> int:
        return len(self.landmarks)

    def handedness_label(self, i: int) -> str:
        """``"Right"``, ``"Left"``, or ``""`` if the index is out of range.

        A method rather than an attribute so a caller reaching past the end gets
        an empty string it can skip instead of an IndexError in a drawing loop.
        """
        return self.handedness[i].display_name if 0 <= i < len(self.handedness) else ""

    def to_dict(self) -> dict[str, Any]:
        return self.raw


@dataclass
class HandLandmarkerOptions:
    """How to build the landmarker.

    ``container`` is the .npue path, or a model name the runtime's catalogue
    knows. ``num_hands`` caps the returned list the way MediaPipe's does, by
    sending ``--max-hands``; it is an UPPER BOUND and it is applied by the
    runtime BEFORE the landmark network runs, not as a filter afterwards, so the
    hands you keep are the top-N detections rather than the top-N of everything.

    There is no ``backend="http"``: the field exists only so the value can be
    refused with its reason. See the module docstring.

    ``threads`` and ``binary`` are this project's own. ``extra_args`` is passed
    through verbatim, which is a documented escape hatch and not a second way to
    set a threshold: ``--conf`` and ``--iou`` are refused by the runtime for
    this architecture, so anything sent here that looks like a threshold will be
    turned away by the runtime's own refusal rather than by this dataclass.
    """

    container: str
    backend: str = "cli"
    num_hands: int = 0
    min_detection_confidence: float | None = None
    min_hand_presence_confidence: float | None = None
    min_tracking_confidence: float | None = None
    threads: int = 16
    binary: str = ""
    timeout_s: float = 300.0
    extra_args: Sequence[str] = ()


def _box4(v: Any) -> tuple[float, float, float, float]:
    """Four numbers out of a box field, defaulting rather than indexing.

    ``HandBox(*...)`` unpacks, so a box that arrived short would be a TypeError
    about argument count rather than a message naming the field. The defaults
    keep a malformed box a zero rectangle, which draws as nothing -- visible,
    not a crash.
    """
    vals = [float(x) for x in list(v)[:4]]
    while len(vals) < 4:
        vals.append(0.0)
    return (vals[0], vals[1], vals[2], vals[3])


def _to_result(doc: dict[str, Any], timestamp_ms: int = 0) -> HandResult:
    _check_hands(doc)
    w = int(doc.get("width", 0))
    h = int(doc.get("height", 0))

    def _px(xy: Sequence[float]) -> tuple[float, float]:
        return float(xy[0]), float(xy[1])

    landmarks: list[list[NormalizedLandmark]] = []
    world: list[list[WorldLandmark]] = []
    handedness: list[Category] = []
    presence: list[float] = []
    boxes: list[HandBox] = []
    scores: list[float] = []
    for hand in doc.get("hands", ()):
        pts = [
            NormalizedLandmark(
                x=float(p[0]) / w if w else float(p[0]),
                y=float(p[1]) / h if h else float(p[1]),
                z=float(p[2]) if len(p) > 2 else 0.0,
                index=i,
                name=HAND_NAMES[i] if i < len(HAND_NAMES) else "",
            )
            for i, p in enumerate(hand.get("landmarks", ()))
        ]
        landmarks.append(pts)
        world.append([
            WorldLandmark(
                x=float(p[0]), y=float(p[1]),
                z=float(p[2]) if len(p) > 2 else 0.0,
                index=i,
                name=HAND_NAMES[i] if i < len(HAND_NAMES) else "",
            )
            for i, p in enumerate(hand.get("world_landmarks", ()))
        ])
        handedness.append(Category(
            display_name=str(hand.get("handedness_category", "")),
            score=float(hand.get("handedness", 0.0)),
        ))
        presence.append(float(hand.get("presence", 0.0)))
        b = hand.get("bbox", (0.0, 0.0, 0.0, 0.0))
        boxes.append(HandBox(x1=float(b[0]), y1=float(b[1]),
                             x2=float(b[2]), y2=float(b[3])))
        scores.append(float(hand.get("score", 0.0)))

    dets = [
        PalmDetection(
            score=float(d.get("score", 0.0)),
            box=HandBox(*_box4(d.get("box", ()))),
            keypoints=tuple(_px(k) for k in d.get("keypoints", ())),
        )
        for d in doc.get("detections", ())
    ]
    return HandResult(
        landmarks=landmarks,
        world_landmarks=world,
        handedness=handedness,
        presence=presence,
        hand_boxes=boxes,
        hand_scores=scores,
        palm_detections=dets,
        image_label=str(doc.get("image", "")),
        image_width=w,
        image_height=h,
        letterbox=dict(doc.get("letterbox", {})),
        timings_ms=dict(doc.get("timings_ms", {})),
        timestamp_ms=timestamp_ms,
        raw=doc,
    )


class HandLandmarker:
    """The landmarker. Build it with ``create_from_options`` or the constructor."""

    # The three MediaPipe thresholds, and the container key each one would have
    # to become. Spelled out here because the refusal is the whole point: a
    # caller porting code sets min_detection_confidence out of habit, and the
    # useful message is "that lives in the container as score_threshold", not
    # "unexpected keyword argument".
    _THRESHOLDS = (
        ("min_detection_confidence", "score_threshold"),
        ("min_hand_presence_confidence", "score_threshold"),
        ("min_tracking_confidence", None),
    )

    def __init__(self, options: HandLandmarkerOptions) -> None:
        if options.backend != "cli":
            raise ValueError(
                f"backend is 'cli', not {options.backend!r}. arch 7 has no HTTP "
                "endpoint: there is no /v1/hands, and `npuembeddings serve` on "
                "a hands container is refused by name. This is refused rather "
                "than honoured, and the refusal is about the missing endpoint "
                "rather than about speed -- asking for a server that does not "
                "exist is the error worth reporting. What the difference WOULD "
                "be, measured on this machine (520x512, one hand, 16 threads, "
                "six runs): this backend is 111-120 ms end to end and the model "
                "is 82-92 ms, so a serve child would save the ~27 ms of "
                "process-spawn, model-load and parse -- 1.36x, not a dramatic "
                "figure. A quiet fallback that got that wrong silently would be "
                "worse than either."
            )
        for name, key in self._THRESHOLDS:
            v = getattr(options, name)
            if v is None:
                continue
            raise ValueError(
                f"{name}={v}: this architecture has no flag for it. The score "
                "and NMS thresholds are read from the CONTAINER"
                + (f" (`{key}`)" if key else "")
                + ", because they are part of how this checkpoint was trained "
                "and the OpenCV zoo demo that recorded the golden used the "
                "container's values, not a command line's. The runtime refuses "
                "`--conf` and `--iou` by name for the same reason. Read the "
                "values the answer was produced under from HandResult.raw."
            )
        if options.num_hands < 0:
            raise ValueError(
                f"num_hands is 0 (all of them) or a positive cap, not "
                f"{options.num_hands}"
            )
        self._opts = options

    @classmethod
    def create_from_options(cls, options: HandLandmarkerOptions) -> "HandLandmarker":
        """MediaPipe's spelling. It is the constructor, under the other name."""
        return cls(options)

    # -- the two entry points -------------------------------------------------

    def detect(self, image: Any) -> HandResult:
        """Landmarks for one image, in source-pixel order and normalised form.

        Blocking. MEASURED on this machine, a 520x512 frame with one hand, 16 CPU
        threads, six runs:

            this call end to end   111 - 120 ms   (median 116)
            what the runtime says
              for the model alone   82 -  92 ms   (median 89)
                palm detector       33 -  42 ms
                crop                 11 -  13 ms
                landmark network    26 -  31 ms

        The ~27 ms median between them is the ``cli`` backend itself -- spawning
        a process, reading two graphs, writing the answer, parsing it. The gap
        runs 19-38 ms across those six runs, so it is an estimate of the
        backend's cost rather than a figure to quote; it is why this backend is
        not what a fast loop wants. It is not part of the model's cost, and the
        footer in examples/pose_webcam.py prints the runtime's own number rather
        than this call's, so the two never get quoted as one.

        The landmark network runs ONCE PER HAND, so two hands cost a second
        26-31 ms and no more of the palm detector's time; the demo's cap
        (``num_hands`` / ``--max-hands``) is the lever on that. This module does
        NOT state a total for two hands: there is no two-hand fixture here to
        time, and a total derived by adding two numbers is a projection, not a
        measurement.

        The numbers are the runtime's, byte for byte: this is the JSON
        ``npuembeddings hands`` prints, parsed.
        """
        return self._run(image, 0)

    def detect_for_video(self, image: Any, timestamp_ms: int) -> HandResult:
        """Same as ``detect``, and per-frame: no temporal state is used.

        ``timestamp_ms`` is accepted and echoed into
        ``HandResult.timestamp_ms``, and used for nothing else. It is not
        refused because "video mode is not supported" would send a caller
        looking for a per-frame mode that behaves identically, and it is not
        smoothed or filtered because a filter this module invented would not be
        the model's answer. MediaPipe's video mode DOES track, and ``landed``
        -- the ids it assigns so a hand keeps its identity across frames -- does
        not exist here, because there is nothing in either network that could
        produce one.
        """
        return self._run(image, int(timestamp_ms))

    # -- the one backend -------------------------------------------------------

    def _cli_argv(self, path: Path) -> list[str]:
        o = self._opts
        cmd = [find_binary(o.binary or None), "hands", o.container, str(path)]
        if o.num_hands:
            cmd += ["--max-hands", str(o.num_hands)]
        if o.threads:
            cmd += ["--threads", str(o.threads)]
        cmd += list(o.extra_args)
        return cmd

    def _run(self, image: Any, timestamp_ms: int = 0) -> HandResult:
        data, name = _image_bytes(image)
        # A temporary file, because the CLI's image argument is a PATH and the
        # alternative is a second ingest path. Deleted in the finally,
        # including on the error paths, and named with mkstemp so two landmarkers
        # in one process cannot collide.
        fd, path = tempfile.mkstemp(prefix="npue-hands-", suffix=".img")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            proc = subprocess.run(
                self._cli_argv(Path(path)),
                capture_output=True, timeout=self._opts.timeout_s)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        if proc.returncode != 0:
            # The runtime writes diagnostics to stderr and the result to stdout,
            # so stderr IS the error message here and is passed through rather
            # than replaced by "exit status 1".
            raise RuntimeError(
                f"npuembeddings hands exited {proc.returncode}:\n"
                + proc.stderr.decode("utf-8", "replace").strip()
            )
        res = _to_result(json.loads(proc.stdout.decode("utf-8")), timestamp_ms)
        # The CLI's `image` field is the path it was given, and what it was
        # given is a temporary file -- so the label the caller reads back would
        # be /tmp/npue-hands-XXXX.img, a name that means nothing and that changes
        # every call. Replaced with the name of the image the CALLER passed, and
        # only on the dataclass: HandResult.raw keeps the emitter's own object
        # untouched.
        if name:
            res.image_label = name
        return res

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Nothing to stop -- the only backend spawns per detect and exits.

        Present so a caller written against ``npue_pose``'s landmarker, which
        is ``with``-shaped and calls ``close()``, runs unchanged. It is not a
        no-op stub that exists to look symmetric: there is genuinely no long-
        lived process, which is the same fact as the missing ``http`` backend.
        """

    def __enter__(self) -> "HandLandmarker":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _main(argv: Sequence[str]) -> int:
    """`python3 npue_hands.py MODEL IMAGE...` -- the smoke test."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("container")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--max-hands", type=int, default=0)
    a = ap.parse_args(argv)
    with HandLandmarker(HandLandmarkerOptions(
            container=a.container, num_hands=a.max_hands)) as hl:
        for path in a.images:
            r = hl.detect(path)
            t = r.timings_ms.get("total", 0.0)
            print(f"{path}: {r.num_hands} hand(s) "
                  f"({r.image_width}x{r.image_height}, {t:.0f} ms total)")
            for i, hand in enumerate(r.landmarks):
                wrist = hand[0]
                print(f"  hand {i}: {r.handedness_label(i)} "
                      f"presence {r.presence[i]:.3f} "
                      f"score {r.hand_scores[i]:.3f} "
                      f"box {r.hand_boxes[i].x1:.0f},{r.hand_boxes[i].y1:.0f} "
                      f"{r.hand_boxes[i].width:.0f}x{r.hand_boxes[i].height:.0f}  "
                      f"wrist ({wrist.x:.3f},{wrist.y:.3f}) z {wrist.z:.1f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv[1:]))