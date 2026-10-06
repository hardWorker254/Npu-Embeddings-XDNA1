#!/usr/bin/env python3
#===- pose_webcam.py ----------------------------------------*- python -*-===#
#
# NpuEmbeddings -- draw your skeleton from a webcam, with OpenCV.
#
#   python examples/pose_webcam.py                       the default camera
#   python examples/pose_webcam.py --camera 1            a second one
#   python examples/pose_webcam.py --image bus.jpg       a still, no camera
#   python examples/pose_webcam.py --array               the convolutions on the NPU
#
#   q / ESC   quit          s   save a frame to out/      space  pause
#   c         cycle the palette       -/+  line thickness
#
# WHAT THIS IS AND IS NOT
# ----------------------
# A THIN drawing loop over python/npue_pose.py. Every number it draws comes from
# the runtime: the joints, the confidence per joint, the box and the score come
# out of the same /v1/pose answer the CLI prints, and this file invents none of
# them. It exists because "is the detection any good" is a question about
# MOTION and about a body at an angle and at a distance, and a still of three
# people on a bus cannot answer it.
#
# It is not a tracker. YOLOv8-pose has no temporal state at all -- every frame is
# computed from scratch -- so there is no smoothing, no identity and no
# "occluded" case here to smooth over. A joint that is missing from a frame is
# missing because the model did not find it this frame, and the script draws it
# grey rather than guessing where it went. That is the model's behaviour, not a
# simplification: see PoseLandmarker.detect_for_video's docstring.
#
# A PERSON MOSTLY OUT OF FRAME STILL SCORES HIGH. Measured on this machine with
# a sitter at the left edge, shoulders and head partly cropped: class score 0.85,
# surviving --conf 0.7, and the skeleton drawn is the visible part with the rest
# of the bones absent rather than extrapolated. That is the model reporting what
# it can see, so the default --conf below stays at YOLO's 0.25 and is not raised
# to tidy a demo -- an indoor room has real false positives at 0.5 too, and a
# threshold that removes the visible person is worse than one that occasionally
# admits a bag.
#
# WHY backend="http" AND NOT "cli"
# --------------------------------
# The cli backend runs the binary once per detect, so every frame pays a process
# spawn and a model load. At the ~0.3 s a 640x640 frame measures on 16 threads,
# that is a slideshow. The http backend starts one `serve` child on first use and
# keeps it, which is what a loop needs; the answer is the same object either way.
#
# --task hands, AND WHY IT IS NOT AN http LOOP
# --------------------------------------------
# Two architectures, one loop: --task pose draws COCO-17 over npue_pose, --task
# hands draws MediaPipe's 21 over npue_hands. Both are here because both are
# webcam-shaped -- the only honest way to judge a hand detector is a hand moving,
# and a still of one hand does not answer it.
#
# The hands branch is on the CLI backend, and cannot be otherwise: arch 7 has no
# HTTP endpoint. There is no /v1/hands, `npuembeddings serve` on a hands
# container is refused by name, and npue_hands refuses backend="http" with that
# same reason rather than falling back quietly. So every frame pays a process
# spawn and a model load. MEASURED here, 520x512, one hand, 16 threads, six runs:
#
#     npue_hands detect() end to end   111 - 120 ms   (median 116)
#     what the runtime reports
#       for the model alone             82 -  92 ms   (median 89)
#     the backend's own share                      ~27 ms   spawn, load, parse
#
# and this loop, one still, cold then warm: 6.0 fps then 7.8-8.8 fps -- i.e. the
# ~27 ms plus JPEG encode and drawing, not something the model is doing. The
# footer prints the RUNTIME's time for that frame, not the loop's, so the two are
# never quoted as one number. Fixing the backend properly means a
# server/hands_backend.cpp, which does not exist yet.
#
# SPDX-License-Identifier: Apache-2.0
#===----------------------------------------------------------------------===#

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# The facade lives beside this file's parent, so the demo runs from a checkout
# without being installed. A copy of this file elsewhere should import npue_pose
# from wherever it is on sys.path instead -- it is one module and no packaging.
REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT / "python"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from npue_pose import PoseLandmarker, PoseLandmarkerOptions  # noqa: E402
from npue_hands import (  # noqa: E402
    HAND_CONNECTIONS,
    HandLandmarker,
    HandLandmarkerOptions,
)

# --- THE COCO-17 SKELETON ---------------------------------------------------
#
# Index pairs into the runtime's 17 landmarks, in the order COCO defines and the
# order the runtime emits. These are the 16 bones that make a person recognisable
# at a glance; nose-to-eye and the two ear points are deliberately absent because
# they add three near-coincident dots around one nose and cost more in clutter
# than they add in information.
#
# Read from the runtime rather than hardcoded twice? The endpoint's answer
# carries a `skeleton` object naming all 17, and npue_pose._check_skeleton
# already refuses an answer whose names are not COCO-17. So this list CANNOT
# silently drift into drawing the wrong bones -- a runtime that renumbered its
# joints would fail the facade's check before this file drew anything.
BONES = (
    (5, 7),   # left shoulder  -> left elbow
    (7, 9),   # left elbow     -> left wrist
    (6, 8),   # right shoulder -> right elbow
    (8, 10),  # right elbow    -> right wrist
    (5, 6),   # shoulder       -> shoulder
    (5, 11),  # left shoulder  -> left hip
    (6, 12),  # right shoulder -> right hip
    (11, 12),  # hip           -> hip
    (11, 13),  # left hip       -> left knee
    (13, 15),  # left knee      -> left ankle
    (12, 14),  # right hip      -> right knee
    (14, 16),  # right knee     -> right ankle
    (0, 1),    # nose           -> left eye
    (0, 2),    # nose           -> right eye
    (1, 3),    # left eye       -> left ear
    (2, 4),    # right eye      -> right ear
)
# (0, 6) and (0, 5) -- nose to shoulder -- are the two lines MediaPipe draws and
# this does not: with a 640-wide frame they cross the chest and read as a
# mistake. Left here so the omission is visible rather than looking forgotten.

# One colour per person, cycled. Chosen to stay apart on a webcam's usual
# low-contrast background, and to stay apart for the two most common colour
# vision deficiencies: no red/green pair carries meaning here, so a viewer who
# cannot separate them still sees two distinct skeletons by shape.
PALETTE = (
    (0, 200, 255),    # amber
    (255, 120, 0),    # blue
    (0, 230, 120),    # green
    (200, 80, 220),   # magenta
    (80, 255, 255),   # yellow
    (255, 100, 180),  # pink
)
BONE_COLOR = (60, 60, 60)     # the same grey as a low-confidence joint
BOX_COLOR = (200, 200, 200)


def resolve_container(name: str) -> str:
    """A `--model` value that is a PATH, made findable from examples/ too.

    A bare name ("yolov8n-pose") is the runtime catalogue's business and is
    returned untouched. A path is this file's, and the trap is that the runtime
    resolves one "exactly as given, against the CURRENT directory" -- so the
    spelling that works from the repo root is the spelling that fails from
    examples/, with a "no such container" that reads as though the container were
    missing rather than as a spelling mistake.

    So a relative path that is not there is retried against the repo root, and
    whichever one is used is returned so the caller prints a path that can be
    re-run. If neither exists the ORIGINAL string is returned unchanged: the
    runtime's refusal names the file it wanted, and inventing a path here would
    only move the message somewhere less honest.

    This matters more for the hands task than for pose, because the pose default
    is a catalogue name and the hands default is a real path. It USED to be the
    other way round as well: the hands container was packed into
    models/mediapipe-hands/hands.npue, beside the two ONNX files it was built
    from, while every table in the binary globs models/*.npue -- so `list` could
    not see it and `npuembeddings hands mediapipe-hands` could not name it. It
    lives at models/mediapipe-hands.npue now, and the model directory keeps the
    weights and the panel variant.
    """
    if not (name.endswith(".npue") or os.sep in name):
        return name                      # a catalogue name, not our problem
    if Path(name).is_file():
        return name
    candidate = REPO_ROOT / name
    if candidate.is_file():
        return str(candidate)
    return name


def find_binary() -> str:
    """The runtime executable, resolved beside this repository.

    PATH first, because a built-and-installed runtime is the more useful thing
    to run and preferring a stale in-tree build over it would be the opposite.
    The in-tree build is the fallback and the message says which one it found,
    so a demo running against a binary nobody rebuilt is visible rather than
    silent.
    """
    from shutil import which
    env = os.environ.get("NPUEMBEDDINGS_BINARY")
    if env:
        return env
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("npuembeddings", "npuembeddings.exe"):
        found = which(name)
        if found:
            return found
        for build in ("build", "build-relwithdebinfo", "build-release"):
            cand = os.path.join(here, "runtime", build, name)
            if os.path.exists(cand):
                return cand
    raise SystemExit(
        "error: cannot find the npuembeddings binary.\n"
        "  Build it with `cmake -S runtime -B runtime/build && cmake --build "
        "runtime/build`, or point NPUEMBEDDINGS_BINARY at one.\n"
        "  --task mppose calls it directly -- there is no arch=8 Python facade "
        "yet -- and the other two tasks need it too.")


def frame_ms(result) -> float:
    """The model's own time for one frame, in milliseconds.

    NOT sum(timing_s.values()). The runtime's timing object carries `total`
    ALONGSIDE the breakdown that adds up to it -- network, front_end, decode --
    so summing the dict counts the frame twice. That reported 571 ms a frame
    where the runtime had measured 280, which is how a demo that runs at 3.5 fps
    came to be described as 1.7. `total` is the runtime's own arithmetic and is
    preferred; the sum is the fallback for an answer that has a breakdown and no
    total, and it drops `total` from the dict it walks so the fallback cannot
    reintroduce the same double count.

    The two facades disagree about the key AND the unit, and neither is wrong:
    the pose runtime emits `timing_s` in seconds, the hands runtime emits
    `timings_ms` in milliseconds because it has seven stages to report and a
    single-second figure would print as 0.09 for the whole frame. So the key is
    looked up first and the unit is decided by WHICH key was found, rather than
    by which class the object is -- a duck-typed check that keeps working for a
    caller who wrote their own result object with either spelling.
    """
    t = getattr(result, "timings_ms", None)
    if t:
        return float(t["total"]) if "total" in t else float(
            sum(v for k, v in t.items() if k != "total"))
    t = result.timing_s or {}
    if not t:
        return 0.0
    if "total" in t:
        return 1000.0 * float(t["total"])
    return 1000.0 * sum(v for k, v in t.items() if k != "total")


def _put(img, text, org, scale=0.5, color=(255, 255, 255), thick=1, bar=False):
    """One line of text, legibly, on a background that is not cooperating.

    An outline alone is not enough and this was measured on bus.jpg rather than
    reasoned about: the footer sits on pale paving stones, white text with a black
    outline vanishes into them, and the one line that states the timing -- the
    number this demo exists to produce -- became the least readable thing on
    screen. So ``bar=True`` puts the text on its own dark strip. The bar is
    drawn as a rectangle rather than an alpha blend because the loop copies no
    frames and a blend would allocate a second image per frame for one label.
    """
    if bar:
        # Measured at the OUTLINE's thickness, not the fill's. getTextSize
        # describes one weight of glyph, and the outline is drawn two heavier --
        # measured on this string, the bar came out 6 px short of its own last
        # letter, which is the kind of error that reads as a rendering bug and is
        # really a measurement taken of the wrong thing.
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale,
                                      thick + 2)
        x, y = org
        cv2.rectangle(img, (x - 6, y - th - 6), (x + tw + 8, y + 7), (25, 25, 25),
                      -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                cv2.LINE_AA)


def draw(img: np.ndarray, result, thickness: int = 2, show_boxes: bool = True) -> np.ndarray:
    """Draw one PoseResult onto a BGR frame, in place, and return it.

    ``result`` is an npue_pose.PoseResult. The landmarks arrive NORMALISED over
    the source image, so every coordinate is multiplied by the frame's own width
    and height here -- which also means this keeps working when the runtime
    letterboxes a non-square frame, because the facade has already undone that.
    """
    h, w = img.shape[:2]

    for person_i, landmarks in enumerate(result.pose_landmarks):
        color = PALETTE[person_i % len(PALETTE)]

        # A bone is drawn only when BOTH of its joints cleared the runtime's
        # --kpt threshold. Drawing the half that survived would connect a wrist to
        # a shoulder across a gap the model said was not there, and the eye reads
        # that as a straight arm rather than as a low-confidence one.
        for a, b in BONES:
            ja, jb = landmarks[a], landmarks[b]
            if not (ja.visible and jb.visible):
                continue
            pa = (int(ja.x * w), int(ja.y * h))
            pb = (int(jb.x * w), int(jb.y * h))
            # Blend the colour with the mean confidence of its two joints: a bone
            # the model is sure about is drawn at full strength and one it is not
            # fades towards the background. A per-joint dot alone cannot do this,
            # because the informative end of a limb is its middle.
            conf = (ja.visibility + jb.visibility) / 2.0
            col = tuple(int(BONE_COLOR[i] + (color[i] - BONE_COLOR[i]) * conf)
                        for i in range(3))
            cv2.line(img, pa, pb, col, thickness, cv2.LINE_AA)

        for j in landmarks:
            if not j.visible:
                continue
            p = (int(j.x * w), int(j.y * h))
            cv2.circle(img, p, max(2, thickness + 1), color, -1, cv2.LINE_AA)

        if show_boxes and person_i < len(result.pose_boxes):
            box = result.pose_boxes[person_i]
            x1, y1 = int(box.x1), int(box.y1)
            x2, y2 = int(box.x2), int(box.y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 1, cv2.LINE_AA)
            score = (result.pose_scores[person_i]
                     if person_i < len(result.pose_scores) else 0.0)
            _put(img, f"{score:.2f}", (x1, max(12, y1 - 4)), 0.45, BOX_COLOR, 1)

    # The footer states what produced the frame rather than decorating it. Three
    # numbers a reader of a demo otherwise has to guess: how long the model took,
    # where the work ran, and how many people it thinks there are.
    t = result.timing_s or {}
    _put(img, f"{result.num_poses} person(s)  {frame_ms(result):.0f} ms  "
              f"{result.backend} backend  {result.dispatches} dispatches",
         (8, h - 10), 0.5, (255, 255, 255), 1, bar=True)
    return img


def draw_hands(img: np.ndarray, result, thickness: int = 2,
               show_boxes: bool = True) -> np.ndarray:
    """Draw one HandResult onto a BGR frame, in place, and return it.

    ``result`` is an npue_hands.HandResult. The landmarks arrive NORMALISED over
    the source image, so every coordinate is multiplied by the frame's own width
    and height here, exactly as for the pose -- which also means this keeps
    working on a non-square frame, because both facades have already undone the
    runtime's letterbox.

    WHAT IS DIFFERENT FROM THE POSE RENDER, AND WHY IT IS NOT A SIMPLIFICATION
    --------------------------------------------------------------------------
    The pose render fades a bone by its two joints' mean confidence and greys a
    joint that missed -- there is nothing to do here instead. This architecture
    has NO per-joint score: the landmark head's confidences are one `presence`
    for the whole hand and one `handedness`, and every one of the 21 joints is
    drawn at full strength whenever the hand is drawn at all.

    So the honest substitute is per-HAND: `presence` colours the whole hand, from
    the shared BONE_COLOR grey at 0 to the palette colour at 1, and the joint
    dots follow it. That is a weaker statement than the pose render's -- a hand
    can have a confident palm and a doubtful fingertip and there is no number
    here that says so -- and saying that in a comment is better than drawing every
    fingertip as certain.
    """
    h, w = img.shape[:2]

    for hand_i, landmarks in enumerate(result.landmarks):
        color = PALETTE[hand_i % len(PALETTE)]
        presence = (result.presence[hand_i]
                    if hand_i < len(result.presence) else 1.0)
        col = tuple(int(BONE_COLOR[i] + (color[i] - BONE_COLOR[i]) * presence)
                    for i in range(3))
        pts = [(int(j.x * w), int(j.y * h)) for j in landmarks]

        # Every edge is drawn, with no per-joint gate, because there is no
        # per-joint score to gate on. HAND_CONNECTIONS is MediaPipe's own edge
        # list, imported rather than written out here: one copy of a topology is
        # one thing to keep right, and npue_hands already refuses an answer whose
        # landmark count is not the 21 these indices address.
        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(img, pts[a], pts[b], col, thickness, cv2.LINE_AA)

        for p in pts:
            cv2.circle(img, p, max(2, thickness), col, -1, cv2.LINE_AA)

        # The wrist gets a ring rather than a filled dot, so the hand's root is
        # findable at a glance among 20 other joints. It is the only joint this
        # draws differently, and it is the one both the landmark network's z and
        # the world frame are measured FROM, so it is the one whose position has to
        # be exactly where it is drawn.
        if pts:
            cv2.circle(img, pts[0], max(4, thickness * 3), col, thickness,
                       cv2.LINE_AA)

        if show_boxes and hand_i < len(result.hand_boxes):
            box = result.hand_boxes[hand_i]
            # NOT clipped to the frame, and not clipped HERE either: this box is
            # the shifted 1.65x-enlarged rectangle around the 21 joints, and on a
            # hand running off the edge it legitimately has a negative coordinate.
            # cv2.rectangle with a negative vertex draws the visible part and
            # leaves the rest to the image boundary, which is the correct reading
            # of "this hand continues past the edge of the picture".
            cv2.rectangle(img, (int(box.x1), int(box.y1)),
                          (int(box.x2), int(box.y2)), BOX_COLOR, 1, cv2.LINE_AA)
            label = result.handedness_label(hand_i) or "?"
            # The LABEL is clamped where the box is not, and the asymmetry is the
            # point. A rectangle with a vertex off the canvas still draws its
            # visible part; a text origin off the canvas is simply gone, with no
            # partial glyph to read -- measured on hand_plain.png, whose hand box
            # is x1 = -9.1, "Right 1.00" came out as "ght 1.00". So the origin is
            # pulled inside first. pose needs no such clamp because its decoder
            # CLIPS the box to the image (decode.cpp), so its x1 is never
            # negative; hands deliberately does not clip, which is why this branch
            # exists at all.
            lx = max(2, min(int(box.x1), w - 60))
            ly = max(12, int(box.y1) - 4)
            _put(img, f"{label} {presence:.2f}", (lx, ly), 0.45, BOX_COLOR, 1)

    # The footer says "host" rather than naming a backend: arch 7 has no array
    # path at all -- `--npu-ops conv` is refused by name, because no design set in
    # this tree carries the 31 distinct dense (K, N) pairs these two graphs use,
    # so there is no array timing to print a host number against. Printing
    # "host backend  0 dispatches" would be true and would also read as though
    # turning something on could change it, which is the wrong suggestion.
    t = result.timings_ms or {}
    palm = f"palm {t.get('palm', 0.0):.0f} " if "palm" in t else ""
    lm = f"lm {t.get('landmarks', 0.0):.0f}" if "landmarks" in t else ""
    _put(img, f"{result.num_hands} hand(s)  {frame_ms(result):.0f} ms  "
              f"host only  {palm}{lm}".rstrip(),
         (8, h - 10), 0.5, (255, 255, 255), 1, bar=True)
    return img


# ============================================================================
# --task mppose (arch=8, MediaPipe Pose)
#
# WHY THIS TASK IS NOT A THIRD FACADE
# ------------------------------------
# The pose path goes through python/npue_pose.py and the hands path through
# python/npue_hands.py. arch=8 has no facade, and writing one is a separate
# deliverable; this demo therefore calls the binary directly, which is the same
# thing npue_hands' `backend="cli"` does and for the same stated reason: one
# process per frame, so it is the SLOWEST of the three paths and the frame time
# printed below includes the process start.
#
# The consequence worth naming: this path is a demonstration of the runtime's
# output shape and of MediaPipe's topology, not a benchmark. Every number drawn is
# the runtime's.
# ============================================================================

# MediaPipe Pose's 33-landmark topology, taken verbatim from the OpenCV zoo demo
# (opencv_zoo samples, mediapipe, demo.py's `_draw_lines`), which is where the
# golden for this architecture was recorded from as well.
#
# IT IS NOT COCO-17 AND NOT THE 21-POINT HAND SKELETON. COCO-17's 17 joints are a
# different numbering of a different skeleton; drawing this model's landmarks with
# COCO's edges produces a figure with the right joints in the wrong places --
# which is the same class of mistake as reading one container's flag as another's.
MPP_EDGES = (
    # face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    # shoulders
    (9, 10),
    # right arm
    (12, 14), (14, 16), (16, 22), (16, 18), (16, 20), (18, 20),
    # left arm
    (11, 13), (13, 15), (15, 21), (15, 19), (15, 17), (17, 19),
    # torso
    (11, 12), (11, 23), (23, 24), (24, 12),
    # right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    # left leg
    (23, 25), (25, 27), (27, 31), (27, 29), (29, 31),
)


def draw_mppose(img: np.ndarray, result, thickness: int = 2,
                show_boxes: bool = True) -> np.ndarray:
    """Draw one arch=8 result onto a BGR frame, in place, and return it.

    The landmarks arrive in PIXELS here -- this path reads the runtime's own JSON
    rather than going through a facade that normalises them -- so they are used
    as they are, and no width or height is multiplied in. That is a difference
    from the other two draws and it is not an inconsistency: the two facades
    normalise and this one does not, because the two facades exist and this path
    does not.

    The visibility/presence fade is the same one the pose render uses, and it has
    a sharper meaning here: MediaPipe's five columns are
    [x, y, z, visibility, presence] and the visibility of a foot behind the other
    leg is exactly the case where drawing it at full strength is a lie.
    """
    h, w = img.shape[:2]
    for i, pose in enumerate(getattr(result, "poses", ())):
        color = PALETTE[i % len(PALETTE)]
        lm = pose["landmarks"]
        pts = [(int(p[0]), int(p[1])) for p in lm]
        vis = [float(p[3]) for p in lm]
        for a, b in MPP_EDGES:
            if a >= len(pts) or b >= len(pts):
                continue
            # A bone fades by its two joints' mean visibility, and a joint that
            # missed is greyed rather than drawn: an occluded ankle reported at
            # full strength is a claim the network did not make.
            c = tuple(int(BONE_COLOR[k] + (color[k] - BONE_COLOR[k]) *
                          min(1.0, 0.5 * (vis[a] + vis[b]))) for k in range(3))
            cv2.line(img, pts[a], pts[b], c, thickness, cv2.LINE_AA)
        for k, p in enumerate(pts):
            v = min(1.0, max(0.0, vis[k]))
            c = tuple(int(BONE_COLOR[j] + (color[j] - BONE_COLOR[j]) * v)
                      for j in range(3))
            cv2.circle(img, p, thickness, c, -1, cv2.LINE_AA)
        if show_boxes:
            bb = [int(round(v)) for v in pose["bbox"]]
            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color, 1)
            cv2.putText(img, f"{pose['pose_confidence']:.3f}",
                        (bb[0], max(12, bb[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)
    t = result.timings_ms or {}
    det = f"det {t.get('detector', 0.0):.0f} " if "detector" in t else ""
    pose_t = f"pose {t.get('landmarks', 0.0):.0f}" if "landmarks" in t else ""
    _put(img, f"{len(getattr(result, 'poses', ()))} pose(s)  "
              f"{frame_ms(result):.0f} ms  host only  {det}{pose_t}".rstrip(),
         (8, h - 10), 0.5, (255, 255, 255), 1, bar=True)
    return img


class MpposeCli:
    """One arch=8 answer, from one `npuembeddings mppose` process.

    Shaped like the two facades -- a context manager with `detect_for_video` --
    so the loop below does not need a third branch. The frame is written to a
    temporary JPEG because that is what the runtime's image reader takes, and it
    is written AS CV2 GAVE IT (BGR, untouched): `cv2.imencode` files the first
    channel as Blue, and a swap here moves the swap to the wrong side of the
    encoder. That mistake was made once already, in this file, and it moved the
    pose joints by up to 110.8 px while every drawn joint still looked like a
    joint.
    """

    def __init__(self, container: str, binary: str, extra: list[str] | None = None):
        self.container = container
        self.binary = binary
        self.extra = list(extra or [])
        self._tmp = None
        self._tmpdir = None

    def __enter__(self):
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory(prefix="npue-mppose-")
        self._tmp = os.path.join(self._tmpdir.name, "frame.jpg")
        return self

    def __exit__(self, *exc):
        self._tmpdir.cleanup()
        return False

    def detect_for_video(self, frame, timestamp_ms: int = 0):
        """`frame` is the ENCODED image, which is what the loop hands every
        facade: the demo JPEG-encodes once, at --jpeg-quality, and both Python
        facades take the bytes. Encoding again here would be a second lossy pass
        on every frame for no reason, so the bytes are written to disk untouched.

        An ndarray is accepted too, because that is what a caller holding a
        frame rather than a loop would have -- and it is encoded BGR-untouched,
        which is the rule the loop's own comment above spends a paragraph on.
        """
        import subprocess
        del timestamp_ms   # per-frame and stateless: there is no tracker here
        if isinstance(frame, np.ndarray):
            ok, enc = cv2.imencode(".jpg", frame)
            if not ok:
                raise RuntimeError("cv2.imencode failed on the frame")
            payload = enc.tobytes()
        else:
            payload = bytes(frame)
        with open(self._tmp, "wb") as f:
            f.write(payload)
        cmd = [self.binary, "mppose", self.container, self._tmp] + self.extra
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(
                f"`{' '.join(cmd)}` exited {p.returncode}:\n{p.stderr[-400:]}")
        try:
            doc = json.loads(p.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"the runtime printed non-JSON ({e})")
        return _MpposeResult(doc)

    detect = detect_for_video


class _MpposeResult:
    """The runtime's JSON, with the attribute names the loop already uses."""

    def __init__(self, doc: dict):
        self._doc = doc
        self.poses = doc.get("poses", ())
        self.detections = doc.get("detections", ())
        self.timings_ms = doc.get("timings_ms", {})
        self.width = doc.get("width", 0)
        self.height = doc.get("height", 0)
        self.backend = "cli"
        self.dispatches = 0
        # The letterbox as a STRING, because the block above prints it that way
        # and the runtime's JSON carries it as three numbers. Reassembled here so
        # that the two architectures' first-frame lines read the same and a
        # caller can print `result.letterbox` without knowing which it has.
        lb = doc.get("letterbox") or {}
        self.letterbox = (f"scale {lb.get('scale', 1.0):.4f}, "
                          f"pad {lb.get('pad_x', 0)},{lb.get('pad_y', 0)}")
        self.input_size = 224

    # The loop below reads one of these two counts and does not know which
    # architecture it is drawing, so both are here and both are honest: this
    # architecture reports people, and the second-stage run may be fewer of them
    # than the detector found, which is the number that matters.
    @property
    def num_poses(self) -> int:
        return len(self.poses)

    @property
    def num_hands(self) -> int:
        return len(self.poses)


def main(argv: list[str] | None = None) -> int:
    # A literal description rather than a slice of this file's comment header:
    # the header above is all `#` comments, so __doc__ is None here, and reading
    # a slice of it would be a way to make --help break the day someone adds a
    # line at the top.
    ap = argparse.ArgumentParser(
        description="Draw your skeleton or your hands from a webcam, with OpenCV.",
        epilog="q/ESC quit   s save a frame   space pause   c cycle colours   "
               "-/+ line thickness. Requires opencv-python (see "
               "requirements.txt) and a built runtime binary.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=("pose", "hands", "mppose"), default="pose",
                    help="which architecture to draw: %(default)s (COCO-17, over "
                         "python/npue_pose.py), hands (MediaPipe's 21, over "
                         "python/npue_hands.py) or mppose (MediaPipe's 33, "
                         "called straight into the binary). The three differ in "
                         "more than the picture -- the last two have no "
                         "thresholds to set and no HTTP endpoint, so several "
                         "flags below are refused rather than ignored under "
                         "--task hands and --task mppose")
    # The default is a NAME with no ".npue" suffix, and that omission is the whole
    # trick. runtime/include/common/model_catalog.hpp:200 treats any argument
    # ending in ".npue" -- or containing a separator -- as a PATH and uses it
    # exactly as given, resolved against the CURRENT directory. So the spelling
    # "yolov8n-pose.npue" is a path meaning ./yolov8n-pose.npue, which exists in
    # models/ and therefore nowhere you are likely to be standing: it failed from
    # the repo root AND from examples/, with a "no such container" that reads as
    # though the container were missing rather than as a spelling mistake. The
    # bare name goes through the catalogue instead and resolves from any cwd.
    #
    # Empty by default and resolved per --task below, because one static default
    # cannot be right for both: left as "yolov8n-pose", `--task hands` with no
    # --model would look up a body-pose container and get the runtime's arch
    # refusal, which is a confusing way to learn that a default is task-shaped.
    ap.add_argument("--model", default="",
                    help="a model NAME for the runtime's catalogue (no .npue "
                         "suffix, no path separator -- it is looked up under "
                         "models/), or a path to a .npue file, used exactly as "
                         "given, and retried relative to the repo root if that "
                         "is not where you are running from. Default: "
                         "yolov8n-pose for --task pose (a name), "
                         "models/mediapipe-hands.npue for --task hands (a "
                         "path, because that container ships beside the ONNX "
                         "files it was packed from and is untracked)")
    ap.add_argument("--camera", type=int, default=0,
                    help="camera index for cv2.VideoCapture (default: %(default)s)")
    ap.add_argument("--image", metavar="PATH",
                    help="run on one still image instead of a camera -- useful on "
                         "a machine with no webcam, and the way to check the "
                         "drawing without standing up")
    ap.add_argument("--array", action="store_true",
                    help="run the convolutions on the NPU (--npu-ops conv). "
                         "Honoured for --task pose, and measured SLOWER for that "
                         "network -- off by default. This previously said "
                         "'0.43 s against 0.31 s per frame', which does not "
                         "reproduce: re-measured on bus.jpg, 16 threads, best of "
                         "three gives 0.330 s with the array against 0.163 s on "
                         "the host, network alone 0.325 against 0.158, so 2.1x "
                         "SLOWER. Those host runs spread 0.158-0.192 s, so take "
                         "the ratio rather than either figure; it is the same "
                         "~2x as pose_mode.hpp's own 150/290. REFUSED for "
                         "--task hands, which has no array path at all.")
    ap.add_argument("--artifacts", default="",
                    help="the design set's directory. Left empty, --array looks "
                         "for runtime/artifacts/<model>/artifacts_npu1/ and says "
                         "which one it picked; if that is not there either, the "
                         "runtime's refusal names the command that builds it. "
                         "Refused for --task hands")
    ap.add_argument("--arch", type=int, default=1,
                    help="which array the design set under --artifacts is for; "
                         "it is a directory name, artifacts_npu<N> (default: "
                         "%(default)s). Refused for --task hands")
    ap.add_argument("--threads", type=int, default=16,
                    help="host threads (default: %(default)s)")
    # default=None, not 0.25, so that "the user typed this" is distinguishable
    # from "this is the default" -- which is the whole question when --task hands
    # has to refuse it. The default VALUE is applied below for pose, and both are
    # written into the help text so --help still shows what pose gets.
    ap.add_argument("--conf", type=float, default=None, metavar="F",
                    help="minimum class score (pose default: 0.25). Refused for "
                         "--task hands: that threshold is in the container as "
                         "score_threshold, because it is part of how the "
                         "checkpoint was trained")
    ap.add_argument("--kpt", type=float, default=None, metavar="F",
                    help="minimum per-joint score; a joint under it is not drawn "
                         "(pose default: 0.5). Refused for --task hands, which "
                         "has no per-joint score at all -- its confidence is one "
                         "number per hand")
    ap.add_argument("--max-det", type=int, default=None, metavar="N",
                    help="most people to report (pose default: 10). Refused for "
                         "--task hands: use --max-hands")
    ap.add_argument("--max-hands", type=int, default=0, metavar="N",
                    help="most hands to report, 0 for all of them (default: "
                         "%(default)s, meaning no cap). --task hands only. Note "
                         "this caps the DETECTIONS the landmark network runs on, "
                         "so it is an upper bound applied before stage 2 rather "
                         "than a filter on finished hands")
    ap.add_argument("--width", type=int, default=1280,
                    help="requested capture width (default: %(default)s)")
    ap.add_argument("--height", type=int, default=720,
                    help="requested capture height (default: %(default)s)")
    ap.add_argument("--jpeg-quality", type=int, default=95,
                    help="quality of the JPEG the frame is encoded into before it "
                         "is sent (default: %(default)s). PNG would be lossless "
                         "and cost 277 ms a frame at 1280x720 against 2 ms for "
                         "this; the runtime sniffs the format, so the name does "
                         "not matter. Below about 85 the block artefacts start "
                         "reaching the 640px letterbox and the joints move.")
    ap.add_argument("--out", default="out",
                    help="directory for frames saved with `s` (default: %(default)s)")
    ap.add_argument("--no-window", action="store_true",
                    help="do not open a window; still detect, and write every "
                         "frame to --out. For a machine with no display, where "
                         "cv2.imshow raises rather than returning.")
    ap.add_argument("--max-frames", type=int, default=500,
                    help="stop after this many frames, so --no-window terminates "
                         "(default: %(default)s)")
    args = ap.parse_args(argv)

    hands = args.task == "hands"
    mp = args.task == "mppose"
    # The hands default is a PATH, not a catalogue name, and it has to be: the
    # hands container is packed to models/mediapipe-hands.npue because it
    # ships beside the two ONNX files it was built from, while the catalogue's
    # name form only ever means models/<name>.npue. It is also untracked --
    # models/** is gitignored apart from CHECKPOINT.json -- so a checkout without
    # it has to be told to pack it rather than to wait for a download.
    model = args.model or (
        "models/mediapipe-hands.npue" if hands else
        "models/mediapipe-pose.npue" if mp else
        "yolov8n-pose")
    resolved = resolve_container(model)
    if resolved != model:
        print(f"model: {resolved}   (found relative to the repo root)")
        model = resolved

    # The pose-only flags are REFUSED under --task hands, and each refusal names
    # the flag and the reason rather than being ignored. Ignoring them would be
    # worse here than for pose: --conf and --kpt do not merely fail to apply, they
    # imply a control the caller believes they have. There is no threshold flag on
    # this architecture at all -- the score and NMS cuts live in the container,
    # because they are part of how the checkpoint was trained -- and there is no
    # per-joint score to threshold, so --kpt has no meaning rather than a
    # different one. The array flags are refused for the other reason stated in
    # hands_mode.hpp: there is no design set carrying these graphs' 31 distinct
    # dense (K, N) pairs, so there is no array timing to compare against.
    if hands:
        refused = [
            ("--conf", args.conf is not None,
             "the score threshold is read from the container as score_threshold, "
             "because it is part of how this checkpoint was trained; the runtime "
             "refuses --conf by name for this architecture too"),
            ("--kpt", args.kpt is not None,
             "this architecture has no per-joint score: its confidence is one "
             "presence number per hand, so there is nothing for a per-joint "
             "threshold to select"),
            ("--max-det", args.max_det is not None,
             "this model has no person detections to cap; use --max-hands, "
             "which caps the detections the landmark network runs on"),
            ("--array", args.array,
             "arch 7 has no array path, so there is nothing for the flag to "
             "move: both networks run on the host"),
            ("--artifacts", bool(args.artifacts),
             "with no array path a design set would be loaded and never read"),
        ]
        for name, typed, why in refused:
            if not typed:
                continue
            raise SystemExit(
                f"error: {name} is not a flag of --task hands. {why}.\n"
                "  Every number this demo draws would still be the runtime's; "
                "the point of refusing is that the run you asked for and the run "
                "you would get are not the same run.")
        # --arch is checked on its VALUE rather than "was it typed", because its
        # default of 1 is the default and a bare `--arch 1` is not a request to
        # change anything. Naming a design set is the only thing it can mean.
        if args.arch != 1:
            raise SystemExit(
                f"error: --arch {args.arch} is not a flag of --task hands. It "
                "names a design set, and arch 7 has no array path (see --array "
                "above).")
    elif mp:
        # The SAME refusals as hands, and for the same reasons, with two that are
        # this architecture's own rather than inherited: arch 8 has a person
        # DETECTOR with a score threshold in its container (so --conf is not a
        # free parameter here either), and it has a SEPARATE confidence gate for
        # the landmark net -- pose_conf_threshold -- which is a different number
        # from the detector's and has no flag either. Saying "use --max-det" would
        # also be wrong here in a way it is not for hands: this model DOES have
        # person detections to cap.
        refused = [
            ("--conf", args.conf is not None,
             "the detector's score threshold is read from the container as "
             "score_threshold, and the landmark net's confidence gate is a "
             "separate container key, pose_conf_threshold. Both are part of how "
             "this checkpoint was trained, and the OpenCV zoo demo whose answer "
             "is the golden used the container's values, not a command line's"),
            ("--kpt", args.kpt is not None,
             "this architecture has no per-joint gate either: visibility and "
             "presence are reported per landmark and the front end draws them, "
             "but nothing selects on them"),
            ("--max-det", args.max_det is not None,
             "this model does have person detections, so use --max-hands to cap "
             "how many of them reach the landmark network"),
            ("--array", args.array,
             "arch 8 has no array path, so there is nothing for the flag to "
             "move: both networks run on the host"),
            ("--artifacts", bool(args.artifacts),
             "with no array path a design set would be loaded and never read"),
        ]
        for name, typed, why in refused:
            if not typed:
                continue
            raise SystemExit(
                f"error: {name} is not a flag of --task mppose. {why}.\n"
                "  Every number this demo draws would still be the runtime's; "
                "the point of refusing is that the run you asked for and the run "
                "you would get are not the same run.")
        if args.arch != 1:
            raise SystemExit(
                f"error: --arch {args.arch} is not a flag of --task mppose. It "
                "names a design set, and arch 8 has no array path.")

    # --array with no --artifacts could not work at all: the runtime refuses
    # --npu-ops conv by name unless a design set is named, and its message says
    # there is "nothing already on disk that could serve it" -- which is true of
    # the runtime, which cannot infer a set, and false of this checkout, where
    # runtime/artifacts/yolov8n-pose/artifacts_npu1/ holds all 14 designs this
    # container needs. So the demo looks, and prints what it found, because a
    # silently chosen design set is exactly the kind of thing that makes an
    # --artifacts-flagged run comparable to itself for the wrong reason. If the
    # directory is absent, --artifacts stays empty and the runtime's refusal --
    # which names the command that builds one -- is the better message than
    # anything invented here.
    #
    # Resolved against the REPO ROOT, not the current directory: this file is
    # run from examples/ as often as from the root, and a relative guess would
    # work in one and fail in the other, which is the same trap as the .npue
    # default this same function already had to be taught about.
    if args.array and not args.artifacts:
        stem = Path(model).stem          # a name, or a path -> its stem
        guess = (REPO_ROOT / "runtime" / "artifacts" / stem /
                 f"artifacts_npu{args.arch}")
        if guess.is_dir():
            args.artifacts = str(guess)
            print(f"design set: {guess}")
        else:
            print(f"note: no design set at {guess}; the runtime will refuse "
                  f"--array and name the command that builds one", file=sys.stderr)

    # The two facades are constructed in their own branches rather than behind a
    # common variable, because they do not share a signature: pose takes five
    # thresholds and an array path, hands takes none of those and refuses the
    # backend outright. A single options object with the union of the fields would
    # be a way of saying `--conf` applies to both, which is the thing this file
    # refuses three screens above.
    if hands:
        opts = HandLandmarkerOptions(
            container=model,
            backend="cli",     # no /v1/hands exists; see the header
            num_hands=args.max_hands,
            threads=args.threads,
        )
        draw_fn = draw_hands
        make_lm = HandLandmarker.create_from_options
        window = "npue hands"
        stem = "hands"
        noun = "hand"
    elif mp:
        # No facade: this path runs the binary once per frame, which is what
        # npue_hands' `backend="cli"` does and for the same reason. The frame time
        # it prints therefore includes a process start, and that is stated here
        # rather than discovered.
        extra = []
        if args.max_hands:
            extra += ["--max-hands", str(args.max_hands)]
        opts = MpposeCli(resolved, find_binary(), extra)
        draw_fn = draw_mppose
        make_lm = lambda o: o          # MpposeCli is already a context manager
        window = "npue mppose"
        stem = "mppose"
        noun = "pose"
    else:
        opts = PoseLandmarkerOptions(
            container=model,
            backend="http",           # one `serve` child for the whole loop
            num_poses=args.max_det if args.max_det is not None else 10,
            min_pose_detection_confidence=(
                args.conf if args.conf is not None else 0.25),
            min_pose_presence_confidence=(
                args.kpt if args.kpt is not None else 0.5),
            threads=args.threads,
            artifacts=args.artifacts,
            conv_on_array=args.array,
        )
        ignored = opts.ignored_options
        if ignored:
            print(f"note: this runtime ignores {', '.join(ignored)} -- it has no "
                  f"tracker, so every frame is computed from scratch", file=sys.stderr)
        draw_fn = draw
        make_lm = PoseLandmarker.create_from_options
        window = "npue pose"
        stem = "pose"
        noun = "person"

    cap = None
    still = None
    if args.image:
        still = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if still is None:
            print(f"error: cannot read {args.image}", file=sys.stderr)
            return 2
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"error: camera {args.camera} did not open. Try --camera 1, or "
                  f"--image <path> to run on a still.", file=sys.stderr)
            return 2
        # A webcam will happily hand back 640x480 if asked for 1920x1080, so the
        # request is a request and the actual size is printed rather than assumed
        # -- the frame the model saw is the frame the drawing is about.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"camera {args.camera}: {w}x{h} at "
              f"{cap.get(cv2.CAP_PROP_FPS):.0f} fps reported")

    out_dir = Path(args.out)
    paused = False
    thickness = 2
    saved = 0
    frames = 0
    t_start = time.monotonic()

    try:
        # `make_lm` rather than a conditional expression inline: both facades
        # expose MediaPipe's `create_from_options`, so the classmethod -- not the
        # signature -- is what differs, and picking it in the branch above keeps
        # the `with` line readable.
        with make_lm(opts) as lm:
            print("q/ESC quit   s save   space pause   c colours   -/+ thickness")
            while True:
                if cap is not None:
                    if paused:
                        # Keep reading while paused. A camera buffer that is not
                        # drained hands the NEXT unpause several stale frames at
                        # once, which looks like a burst of wrong detections.
                        cap.grab()
                    else:
                        ok, bgr = cap.read()
                        if not ok:
                            print("\ncamera stopped delivering frames", file=sys.stderr)
                            break
                else:
                    bgr = still.copy()

                if bgr is None:
                    break

                # THE FRAME GOES TO imencode AS cv2 GAVE IT -- BGR, UNTOUCHED.
                #
                # This used to `cv2.cvtColor(bgr, COLOR_BGR2RGB)` first, on the
                # reasoning that "the runtime's front end reads RGB". That is true
                # and the swap is still wrong, because it moves the swap to the
                # wrong side of the encoder: `cv2.imencode` takes a BGR array and
                # writes its three channels to the file as Blue, Green, Red. Hand
                # it RGB and it dutifully files your Red under Blue, so the JPEG's
                # stored samples are the original's with R and B exchanged -- and
                # the runtime's decoder, reading the file correctly, then hands the
                # network exactly that.
                #
                # IT LOOKED FINE FOR POSE, which is why it survived: the person count was
                # always right, so nothing ever failed, and a skeleton displaced by
                # tens of pixels still looks attached to the body it came from.
                # The comment here used to say so -- "the pose still mostly lands the
                # wrong way round, which is what makes this worth a comment instead of
                # leaving to chance" -- and read it as a tolerance of the model rather
                # than as the bug it was.
                #
                # Measured three ways -- the file as it is, and the demo's own encoding
                # with and without the swap -- through the same `npuembeddings pose`
                # the loop calls, one frame, no camera in the way:
                #
                #   the file as it is                  3 people, joints = reference to 0 px
                #   this file as it was, imencode(rgb) 3 people, joints off by up to 110.8 px
                #   this file now,      imencode(bgr) 3 people, joints off by up to 2.0 px
                #
                # 110.8 px against 2.0 px is the whole bug: same three people, scores
                # within 0.01, and a nose drawn a fifth of the frame width away from
                # the face. The 2.0 px is genuine JPEG error at q95 and is what the
                # paragraph further down promises.
                #
                # The palm detector does not tolerate it at all, because it was
                # trained on skin. Measured on the zoo's hand_plain.png, 520x512:
                #
                #   this file as it was, imencode(rgb)   0 hands, score 0.097
                #   this file now,      imencode(bgr)   1 hand,  score 0.89387
                #   the PNG, no JPEG at all               1 hand,  score 0.89094
                #
                # 0.894 down to 0.097 is not noise: it is a detector that has been
                # shown a blue hand, against a container threshold of 0.5. So there
                # is no pre-swap here at all -- `bgr` goes straight into the encoder,
                # and anything else that reads this line's logic should do the same.
                #
                # The frame is JPEG-encoded HERE and handed over as bytes, rather
                # than as an array. Handed an array, npue_pose._image_bytes
                # encodes PNG -- correctly, it is a library and lossless is the
                # right default for one -- and PNG costs 277 ms a frame at
                # 1280x720, measured against 2.1 ms for JPEG at q95. That is more
                # than the whole network costs, so a demo that ships the array
                # path spends half its frame budget turning RGB into bytes. The
                # runtime sniffs the format by magic bytes, so the name "upload"
                # does not matter; q95 keeps JPEG's own error far below the
                # letterbox to 640px that follows it.
                ok_jpg, enc = cv2.imencode(".jpg", bgr,
                                          [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                if not ok_jpg:
                    print("\nframe could not be JPEG-encoded", file=sys.stderr)
                    break

                t0 = time.monotonic()
                result = lm.detect_for_video(enc.tobytes(), int(t0 * 1000))
                draw_fn(bgr, result, thickness)
                frames += 1

                if frames == 1:
                    # Printed once, because they are properties of the runtime
                    # and this process rather than of any frame, and re-printing
                    # them every frame would bury the numbers that do change.
                    #
                    # The two tasks report DIFFERENT properties, from the fields
                    # each answer actually has. The pose side names its input size
                    # and its array dispatches; the hands side has no array path
                    # to name and two input sizes rather than one (192 px for the
                    # palm detector, 224 px for the landmark net, neither of which
                    # is in the JSON), so it reports the container's own stage
                    # timings instead. Printing `result.input_size` under --task
                    # hands would be an AttributeError, and printing zeros would
                    # be worse.
                    if hands:
                        t = result.timings_ms or {}
                        print(f"letterbox {result.letterbox}, host only; "
                              f"palm {t.get('palm', 0.0):.0f} ms, landmarks "
                              f"{t.get('landmarks', 0.0):.0f} ms, crop "
                              f"{t.get('crop', 0.0):.0f} ms")
                    elif mp:
                        t = result.timings_ms or {}
                        print(f"detector 224px + landmarks 256px, letterbox "
                              f"{result.letterbox}, host only; detector "
                              f"{t.get('detector', 0.0):.0f} ms, landmarks "
                              f"{t.get('landmarks', 0.0):.0f} ms, crop "
                              f"{t.get('crop', 0.0):.0f} ms")
                        print(f"one `npuembeddings mppose` process per frame, "
                              f"so the frame time above includes its start -- "
                              f"this is the slowest of the three paths and it is "
                              f"a demonstration, not a benchmark")
                    else:
                        print(f"input {result.input_size}px, letterbox "
                              f"{result.letterbox}, backend {result.backend}, "
                              f"{result.dispatches} dispatches")
                    if not args.no_window:
                        count = result.num_hands if hands else result.num_poses
                        print(f"first frame: {count} {noun}(s) in "
                              f"{frame_ms(result):.0f} ms")

                if not args.no_window:
                    cv2.imshow(window, bgr)
                    # A STILL waits for a key; a camera does not. One frame of a
                    # live view with waitKey(1) is a window that flashes for a
                    # millisecond and is gone, which is not a preview -- and for a
                    # still there is no next frame coming, so the loop would exit
                    # having shown nothing and saved nothing.
                    key = cv2.waitKey(0 if still is not None else 1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    if key == ord("s"):
                        out_dir.mkdir(parents=True, exist_ok=True)
                        path = out_dir / f"{stem}_{saved:04d}.jpg"
                        cv2.imwrite(str(path), bgr)
                        saved += 1
                        print(f"saved {path}")
                    elif key == 32:
                        paused = not paused
                        print("paused" if paused else "running")
                    elif key == ord("c"):
                        # Cycle by rotating the module-level palette, so the next
                        # frame is all the new colours and no half-and-half.
                        global PALETTE
                        PALETTE = PALETTE[1:] + PALETTE[:1]
                        if still is not None:
                            continue   # redraw the same still in the new colours
                    elif key in (ord("-"), ord("+")):
                        thickness = max(1, min(6, thickness + (1 if key == ord("+") else -1)))
                        if still is not None:
                            continue
                else:
                    # Headless: there is no window and no key, so writing the frame
                    # IS the output. Every frame, for a camera as well as for a
                    # still -- this is the batch-job mode, and saving only the
                    # still case (which is how this branch first read) produced a
                    # run that detected a person in every frame and reported
                    # nothing at all.
                    out_dir.mkdir(parents=True, exist_ok=True)
                    path = out_dir / f"{stem}_{frames:04d}.jpg"
                    cv2.imwrite(str(path), bgr)

                if still is not None:
                    break   # one still is one frame
                if frames >= args.max_frames:
                    print(f"\n{args.max_frames} frames is the --max-frames cap; "
                          f"stopping", file=sys.stderr)
                    break
    finally:
        if cap is not None:
            cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()

    dt = time.monotonic() - t_start
    if frames:
        print(f"{frames} frame(s) in {dt:.1f} s = {frames / dt:.2f} fps"
              + (f", {saved} saved" if saved else ""))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
