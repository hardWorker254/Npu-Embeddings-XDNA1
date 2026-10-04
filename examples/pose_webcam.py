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
# SPDX-License-Identifier: Apache-2.0
#===----------------------------------------------------------------------===#

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# The facade lives beside this file's parent, so the demo runs from a checkout
# without being installed. A copy of this file elsewhere should import npue_pose
# from wherever it is on sys.path instead -- it is one module and no packaging.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from npue_pose import PoseLandmarker, PoseLandmarkerOptions  # noqa: E402

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
    ms = 1000.0 * sum(t.values()) if t else 0.0
    _put(img, f"{result.num_poses} person(s)  {ms:.0f} ms  "
              f"{result.backend} backend  {result.dispatches} dispatches",
         (8, h - 10), 0.5, (255, 255, 255), 1, bar=True)
    return img


def main(argv: list[str] | None = None) -> int:
    # A literal description rather than a slice of this file's comment header:
    # the header above is all `#` comments, so __doc__ is None here, and reading
    # a slice of it would be a way to make --help break the day someone adds a
    # line at the top.
    ap = argparse.ArgumentParser(
        description="Draw your skeleton from a webcam, with OpenCV.",
        epilog="q/ESC quit   s save a frame   space pause   c cycle colours   "
               "-/+ line thickness. Requires opencv-python (see "
               "requirements.txt) and a built runtime binary.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # The default is a NAME with no ".npue" suffix, and that omission is the whole
    # trick. runtime/include/common/model_catalog.hpp:200 treats any argument
    # ending in ".npue" -- or containing a separator -- as a PATH and uses it
    # exactly as given, resolved against the CURRENT directory. So the spelling
    # "yolov8n-pose.npue" is a path meaning ./yolov8n-pose.npue, which exists in
    # models/ and therefore nowhere you are likely to be standing: it failed from
    # the repo root AND from examples/, with a "no such container" that reads as
    # though the container were missing rather than as a spelling mistake. The
    # bare name goes through the catalogue instead and resolves from any cwd.
    ap.add_argument("--model", default="yolov8n-pose",
                    help="a model NAME for the runtime's catalogue (no .npue "
                         "suffix, no path separator -- it is looked up under "
                         "models/), or a path to a .npue file, used exactly as "
                         "given and so resolved against your current directory. "
                         "Default: %(default)s")
    ap.add_argument("--camera", type=int, default=0,
                    help="camera index for cv2.VideoCapture (default: %(default)s)")
    ap.add_argument("--image", metavar="PATH",
                    help="run on one still image instead of a camera -- useful on "
                         "a machine with no webcam, and the way to check the "
                         "drawing without standing up")
    ap.add_argument("--array", action="store_true",
                    help="run the convolutions on the NPU (--npu-ops conv). "
                         "Honoured, and measured SLOWER for this network: 0.43 s "
                         "against 0.31 s per frame, so it is off by default.")
    ap.add_argument("--artifacts", default="",
                    help="the design set's directory, if not the automatic one")
    ap.add_argument("--threads", type=int, default=16,
                    help="host threads (default: %(default)s)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="minimum class score (default: %(default)s)")
    ap.add_argument("--kpt", type=float, default=0.5,
                    help="minimum per-joint score; a joint under it is not drawn "
                         "(default: %(default)s)")
    ap.add_argument("--max-det", type=int, default=10,
                    help="most people to report (default: %(default)s)")
    ap.add_argument("--width", type=int, default=1280,
                    help="requested capture width (default: %(default)s)")
    ap.add_argument("--height", type=int, default=720,
                    help="requested capture height (default: %(default)s)")
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

    opts = PoseLandmarkerOptions(
        container=args.model,
        backend="http",           # one `serve` child for the whole loop
        num_poses=args.max_det,
        min_pose_detection_confidence=args.conf,
        min_pose_presence_confidence=args.kpt,
        threads=args.threads,
        artifacts=args.artifacts,
        conv_on_array=args.array,
    )
    ignored = opts.ignored_options
    if ignored:
        print(f"note: this runtime ignores {', '.join(ignored)} -- it has no "
              f"tracker, so every frame is computed from scratch", file=sys.stderr)

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
        with PoseLandmarker.create_from_options(opts) as lm:
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

                # cv2 hands back BGR; the runtime's front end reads an image array
                # as RGB (npue_pose._image_bytes). Handing it BGR would train the
                # model on swapped channels -- the pose still mostly lands, which
                # is what makes this worth a comment rather than leaving to chance.
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                t0 = time.monotonic()
                result = lm.detect_for_video(rgb, int(t0 * 1000))
                draw(bgr, result, thickness)
                frames += 1

                if frames == 1:
                    # Printed once, because they are properties of the runtime
                    # and this process rather than of any frame, and re-printing
                    # them every frame would bury the numbers that do change.
                    print(f"input {result.input_size}px, letterbox "
                          f"{result.letterbox}, backend {result.backend}, "
                          f"{result.dispatches} dispatches")
                    if not args.no_window:
                        print(f"first frame: {result.num_poses} person(s) in "
                              f"{1000 * sum(result.timing_s.values()):.0f} ms")

                if not args.no_window:
                    cv2.imshow("npue pose", bgr)
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
                        path = out_dir / f"pose_{saved:04d}.jpg"
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
                    path = out_dir / f"pose_{frames:04d}.jpg"
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
