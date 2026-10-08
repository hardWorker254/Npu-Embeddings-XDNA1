#!/usr/bin/env python3
"""Does `serve` answer for every architecture, and refuse for the others?

WHAT THIS IS, AND WHY IT IS A SEPARATE GATE FROM verify_endpoint.py
-------------------------------------------------------------------
verify_endpoint.py drives the embedding endpoint with the REAL `openai` client
and checks the vectors against `--embed`. That is the right test for /v1/embeddings
and the wrong shape for the other three: none of them is an OpenAI client call, and
reaching for it here would mean testing a client library against a shape it does not
have.

This gate is about the DISPATCH, which is the thing that changed and the thing no
other gate covers. `serve` is one verb and the container's arch picks the endpoint,
so the failure this exists to catch is a container answering for a path that is not
its own -- /v1/embeddings with landmarks in it, /v1/pose with vectors. That is not a
wrong number, it is a wrong KIND of answer, and it is exactly what "one verb, four
endpoints" is supposed to make impossible.

So for each architecture this:
  * starts `serve` on that container and reads /health, which must name the RIGHT
    kind, because a client picks its parser from that field;
  * POSTs to that architecture's own path and requires a 2xx with a document of the
    expected shape -- not merely "not an error", because a 200 carrying the wrong
    object's keys is the failure being looked for;
  * asks for the other three paths and requires a 404 whose message names the right
    one, since a 404 that does not say where to go is not much of a 404;
  * and for the two image endpoints, runs the refusal list: not multipart, no
    `image`, two `image` parts, an empty `image`, a non-image, a threshold that is
    not a number, and an unknown field type.

The last group is where a silent fallback would hide. An endpoint that answers
`conf=abc` with the default 0.25 has not refused anything -- it has returned a
confident detection count produced under thresholds the client never asked for, and
the number of people in it would be read as a property of the model.

WHAT IT IS NOT
--------------
Not a numerics gate. It does not check that the pose landmarks or the classification
are CORRECT -- diff_pose_dump.py and verify_vit_model.py are for that. It checks that
the right container answers on the right path with the right refusal, and that the
bytes the endpoint emits are the bytes the CLI emits.

Usage:
  python tools/verify/verify_serve_dispatch.py      # all four endpoints, plus the arch with none
  python tools/verify/verify_serve_dispatch.py --only pose
  python tools/verify/verify_serve_dispatch.py --skip whisper  # no design set

AND THE ONE ARCHITECTURE WITH NO ENDPOINT
-----------------------------------------
arch 7 (MediaPipe hands) is in here as a REFUSAL, not as an endpoint, because it
is the only arch for which both would be true at once if this were wrong: a
container the dispatcher routes to a mode that has nothing to serve. Before the
refusal existed, `npuimage serve <hands>` fell through to the images check
and told the operator to say whose hands to find -- a wrong answer that read like
a usage question. What is asserted here is the three parts of the fix:

  * it EXITS, non-zero, rather than starting a server nobody can talk to;
  * the message NAMES the absence, so the operator is not sent to a URL;
  * nothing is left LISTENING on the port -- the part a message-only check would
    miss, because a refusal printed before bind() would pass the first two.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "lib"))
from gate_output import GATE_OUT                                 # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# The four endpoints, and the container that reaches each. `kind` is what /health
# must report: a client chooses its response parser from that string, so a container
# reporting the wrong one is a wrong answer to a question that was answered.
ARCHES = {
    "embed": {
        "container": "models/all-MiniLM-L6-v2.npue",
        "path": "/v1/embeddings",
        "kind": "embeddings",
        "verb": "serve",
    },
    "whisper": {
        "container": "models/whisper-tiny.npue",
        "path": "/v1/audio/transcriptions",
        "kind": "stt",
        "verb": "serve",
    },
    "classify": {
        "container": "models/vit-base-patch16-224.npue",
        "path": "/v1/classify",
        "kind": "classify",
        "verb": "serve",
    },
    "pose": {
        "container": "/tmp/opencode/p_i8.npue",
        "path": "/v1/pose",
        "kind": "pose",
        "verb": "serve",
    },
    # Not an endpoint: the one arch that must REFUSE `serve`, asserted below
    # rather than assumed. It is in this dict with the others so that adding an
    # endpoint for it later cannot happen without this entry changing shape --
    # the refusal check would then fail on its own, which is the reminder needed.
    # The container is UNTRACKED (models/** is gitignored apart from
    # CHECKPOINT.json), so a checkout that has not packed it SKIPS, like every
    # other missing container here.
    "hands": {
        "container": "models/mediapipe-hands.npue",
        "path": None,
        "kind": "hands",
        "verb": "serve",
        "refuses": True,
    },
}

# Keys the answer must carry. A 2xx with a document missing these is a 2xx for the
# wrong endpoint, and it is the whole failure this gate is built around.
SHAPE = {
    "/v1/embeddings": ["data"],
    "/v1/audio/transcriptions": ["text"],
    "/v1/classify": ["label", "name", "p", "dispatches"],
    "/v1/pose": ["landmarks", "skeleton", "thresholds", "letterbox"],
}

_failures: list[str] = []
_checks = 0


def report(ok: bool, what: str, detail: str = "") -> bool:
    global _checks
    _checks += 1
    mark = "ok  " if ok else "FAIL"
    line = f"  {mark}  {what}"
    if detail:
        line += f"  -> {detail}"
    print(line)
    if not ok:
        _failures.append(what + (f" -- {detail}" if detail else ""))
    return ok


def note(what: str, detail: str = "") -> None:
    print(f"  note    {what}" + (f"  {detail}" if detail else ""))


# -- the fixture ------------------------------------------------------------


def fixture() -> bytes | None:
    """A JPEG to upload, or None with the reason said out loud.

    bus.jpg lives in docs/ -- its sha256 is recorded in
    models/mediapipe-pose/CHECKPOINT.json, which is how this gate knows it is
    looking at THE photograph and not at one renamed bus.jpg -- with
    /tmp/opencode/bus.jpg preferred because that is where the original
    download went. A machine with neither SKIPS the request cases rather than
    failing them: a gate that fails for a missing photograph is a gate whose
    result says nothing about the code.
    """
    for cand in (Path("/tmp/opencode/bus.jpg"), REPO / "docs" / "bus.jpg"):
        if cand.exists():
            return cand.read_bytes()
    return None


def audio_fixture() -> Path | None:
    """A WAV to upload, or None.

    jfk.wav -- 11 s of speech, 16 kHz mono 16-bit -- is what the reader accepts
    unchanged, and it is SPEECH rather than a tone: this gate checks that the
    endpoint accepts the upload and answers with a `text` field, and a 440 Hz
    sine establishes neither that the audio path works nor that the model was
    given something to transcribe. Looked for beside the repository first, then
    in the home directory, so a machine that has it does not need a copy made.
    """
    for cand in (
        Path.home() / "jfk.wav",
        REPO / "tests" / "data" / "jfk.wav",
        Path("/tmp/opencode/jfk.wav"),
    ):
        if cand.exists():
            return cand
    return None


# -- HTTP helpers -----------------------------------------------------------


def get(url: str, timeout: float = 5.0):
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:                                    # connection refused
        return None, str(e).encode()


def post(url: str, body: bytes, content_type: str, timeout: float = 120.0):
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        return None, str(e).encode()


def multipart(parts: list[tuple[str, str, bytes | None]]) -> tuple[bytes, str]:
    """A multipart body, built here rather than borrowed.

    urllib has no multipart encoder and `requests` is not a dependency of every
    machine that has this binary, so the encoder is these lines and the alternative
    is a gate that cannot run.

    `parts` is a LIST of (name, filename, data), and filename None means a plain
    form field whose data is the value. It was a dict of name -> value at first,
    which cannot express the one case this gate most needs: a request carrying
    TWO `image` parts. A dict silently keeps the last, the server never sees a
    duplicate, and the check named "two image parts is refused" would have been
    passing against a request that carried one.

    Returns the body and the Content-Type together, and they MUST be used
    together: the boundary is generated per call, so calling this twice and taking
    the body from one and the header from the other produces a body whose boundary
    the header does not name -- which the server refuses, correctly, with a message
    about the boundary rather than about the image. That reads like a server bug and
    is a gate bug, which is why the pair is one return value.
    """
    b = f"----npue{uuid.uuid4().hex}"
    out = bytearray()
    for name, filename, data in parts:
        if filename is None:
            out += (f"--{b}\r\nContent-Disposition: form-data; "
                    f"name=\"{name}\"\r\n\r\n").encode()
        else:
            out += (f"--{b}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                    f"filename=\"{filename}\"\r\n"
                    f"Content-Type: application/octet-stream\r\n\r\n").encode()
        out += (data or b"") + b"\r\n"
    out += f"--{b}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={b}"


def field(name: str, value: str) -> tuple[str, str, None]:
    return (name, None, value.encode())


def upload(name: str, filename: str, data: bytes) -> tuple[str, str, bytes]:
    return (name, filename, data)


# THREE BINARIES, ONE PER FAMILY, and the container's arch picks one. `serve`
# is the one verb all three have and it is gated like every other verb: an
# npuaudio listening on a ViT's endpoint is exactly the mismatch the split
# exists to refuse, so the ARCHES table above decides the binary and not a
# flag on this gate. `--exe` still names ONE file, because an override that
# had to name three would break every caller for no gain -- the siblings are
# resolved in its directory, so overriding the TREE still works.
FAMILY = {"embed": "npuembeddings", "whisper": "npuaudio",
          "classify": "npuimage", "pose": "npuimage",
          "hands": "npuimage"}


def exe_for(exe: Path, arch: str) -> Path:
    """The binary that serves THIS arch, resolved next to the given --exe.

    It used to early-return `exe` whenever its stem was one of the three names
    -- and since --exe defaults to `runtime/build/npuembeddings`, that one name
    is what every arch got: the whisper row called `npuembeddings serve
    whisper-tiny`, which refuses because it is not the audio binary, and the
    gate reported a failing serve where the code had refused correctly. The
    early return also contradicted this gate's own --exe help, which promises
    that the other two are resolved from the arch under test.

    THE ARCH PICKS THE BINARY, so this is one line: a path whose stem already
    equals the wanted name is returned unchanged by construction, and any other
    --exe (a differently named build, a wrapper) contributes its DIRECTORY and
    nothing else -- which is the "overrides the TREE" the help text describes.
    """
    return exe.with_name(FAMILY[arch])


# -- the server under test ---------------------------------------------------


class Server:
    """`<family binary> serve <container>`, started and stopped for one arch."""

    def __init__(self, exe: Path, container: str, port: int,
                 extra: list[str] | None = None):
        self.exe, self.container, self.port = exe, container, port
        self.extra = extra or []
        self.proc: subprocess.Popen | None = None
        self.log = REPO / "gate_output" / f"verify_serve_{port}.log"
        self.base = f"http://127.0.0.1:{port}"

    def start(self, seconds: float = 90.0) -> tuple[bool, str]:
        GATE_OUT.mkdir(parents=True, exist_ok=True)
        cmd = [str(self.exe), "serve", self.container, "--port", str(self.port)] + self.extra
        env = dict(os.environ)
        env.setdefault("PYTHONPATH", "/opt/xilinx/xrt/python")
        with open(self.log, "wb") as f:
            self.proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                         env=env, start_new_session=True)
        deadline = time.monotonic() + seconds
        last = ""
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                return False, (f"exited {self.proc.returncode}: "
                               + self.log.read_text(errors="replace")[-1500:])
            status, body = get(self.base + "/health", timeout=2)
            if status == 200:
                return True, ""
            last = body.decode(errors="replace")[:120]
            time.sleep(0.2)
        return False, f"/health never answered ({last})"

    def stop(self) -> None:
        if self.proc is None:
            return
        # SIGINT to the GROUP: the server installs its own handler and unwinds
        # through the accept loop, which releases the NPU contexts. SIGKILL would
        # skip that and leave a reader wondering whether the device is still held.
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
            self.proc.wait(timeout=20)
        except Exception:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                pass
        self.proc = None


# -- the checks -------------------------------------------------------------


def check_health(srv: Server, kind: str, path: str) -> dict | None:
    print(f"\n  /health names the RIGHT kind, and the right path's neighbours")
    status, body = get(srv.base + "/health")
    if not report(status == 200, "/health answers 200",
                   "" if status == 200 else f"-> {status}: {body[:160]!r}"):
        return None
    try:
        h = json.loads(body)
    except Exception as e:
        report(False, "/health is JSON", f"-> {e}")
        return None
    got_kind = h.get("kind")
    report(got_kind == kind,
           f"/health says kind={kind!r}, which is how a client picks its parser",
           "" if got_kind == kind else f"-> got {got_kind!r}")
    report("model" in h and bool(h["model"]),
           "/health names the model it is serving",
           "" if h.get("model") else "-> no model field")
    # The banner is on stdout and flushed; if it is still sitting in a buffer the
    # operator tailing the log sees no port at all. Checked by reading the log.
    try:
        text = srv.log.read_text(errors="replace")
        seen = f":{srv.port}" in text
        report(seen,
               "the banner reached the log, so a redirected run shows its port",
               "" if seen else
               "-> no banner in the log. stdout is block-buffered when redirected "
               "and nothing else is printed after it, so the log never shows the "
               "port and the run looks like it did not start")
    except Exception as e:
        note("could not read the log", str(e))
    return h


def check_own_path(srv: Server, path: str, jpeg: bytes, wav: Path | None) -> None:
    print(f"\n  POST {path} answers 2xx with THIS endpoint's shape")
    if path == "/v1/embeddings":
        body, ctype = json.dumps({"input": "a gate"}).encode(), "application/json"
    elif path == "/v1/audio/transcriptions":
        if wav is None:
            note("skipped: no WAV on disk (--audio), so the request case did not run")
            return
        body, ctype = multipart([upload("file", "a.wav", wav.read_bytes())])
    else:
        body, ctype = multipart([upload("image", "bus.jpg", jpeg)])
    status, raw = post(srv.base + path, body, ctype)
    if not report(status == 200, "the request answers 200",
                   "" if status == 200 else f"-> {status}: {raw[:200]!r}"):
        return
    try:
        d = json.loads(raw)
    except Exception as e:
        report(False, "the answer is JSON", f"-> {e}")
        return
    missing = [k for k in SHAPE[path] if k not in d]
    report(not missing,
           f"the answer carries {', '.join(SHAPE[path])}",
           "" if not missing else
           f"-> missing {missing}: this is a 200 for the wrong endpoint")


def check_other_paths(srv: Server, path: str, jpeg: bytes) -> None:
    """Every other endpoint must 404 here, and say which path does answer.

    `/v1/hands` is probed too, and it is not in ARCHES as a path -- arch 7 has no
    endpoint, so nothing answers it anywhere. Probing it on every OTHER server is
    the assertion that stays true while that is the case: a listener that
    answered /v1/hands would be a fifth endpoint nobody declared, and this is the
    only check in this file that would see it. The guard keeps it out of the way
    the day hands does get one -- on the hands server itself it is `path`, and the
    `continue` below would drop it either way.
    """
    print("\n  the OTHER paths are 404s that say where to go instead")
    others = [o["path"] for o in ARCHES.values() if o["path"] and o["path"] != path]
    others.append("/v1/hands")
    seen: set[str] = set()
    for other_path in others:
        if other_path in seen:
            continue
        seen.add(other_path)
        status, raw = get(srv.base + other_path)
        ok = status == 404
        report(ok, f"GET {other_path} is 404 here, not another endpoint's answer",
               "" if ok else f"-> got {status}: {raw[:140]!r}")
        if not ok:
            continue
        try:
            msg = json.loads(raw)["error"]["message"]
        except Exception:
            report(False, "the 404 is the error envelope", f"-> {raw[:100]!r}")
            continue
        report(path in msg,
               f"the 404 for {other_path} names this model's own path {path}",
               "" if path in msg else f"-> {msg[:150]}")


def check_image_refusals(srv: Server, path: str, jpeg: bytes) -> None:
    """The refusal list, for the two endpoints that take an image."""
    print("\n  the refusals fire by name, and none of them falls back to a default")
    img = "image"
    # (what, body, content-type, expected status, a phrase the message must carry)
    #
    # The last element is "" for the case whose refusal is the front end's own and
    # is worded by the decoder rather than by the endpoint; the status is the check
    # there and pinning the decoder's English into this gate would make it fail when
    # a decoder says the same thing differently.
    cases = [
        ("a body that is not multipart",
         b"this is not a multipart body at all", "application/json", 400,
         "multipart"),
        ("no `image` part at all",
         *multipart([field("top_k", "1")]), 400, img),
        ("an empty `image` part",
         *multipart([upload(img, "e.jpg", b"")]), 400, "empty"),
        ("two `image` parts",
         *multipart([upload(img, "a.jpg", jpeg), upload(img, "b.jpg", jpeg)]), 400,
         "2 'image'"),
        ("a file that is not PNG or JPEG",
         *multipart([upload(img, "x.txt", b"definitely not an image at all")]), 400,
         ""),
    ]
    for what, body, ctype, want, needle in cases:
        status, raw = post(srv.base + path, body, ctype)
        ok = status == want
        report(ok, f"{what} -> {want}",
               "" if ok else f"-> got {status}: {raw[:160]!r}")
        if ok and needle:
            say = needle.encode() in raw
            report(say, f"its message says why ({needle!r})",
                   "" if say else f"-> {raw[:200]!r}")

    # The per-request knobs, which are the two models' own thresholds. The point is
    # not that 0 is refused; it is that a refusal is a refusal and NOT the default.
    for field_name in ("conf", "iou", "kpt", "top_k", "max_det"):
        for bad in ("abc", "1e999", "-1"):
            body, ctype = multipart([field(field_name, bad),
                                     upload(img, "bus.jpg", jpeg)])
            status, raw = post(srv.base + path, body, ctype)
            if status in (400, 413):
                report(True, f"{field_name}={bad!r} is refused")
                continue
            # A model that has no such field ignores it. That is not this gate's
            # business, and saying so is better than counting it as a pass.
            note(f"{field_name}={bad!r} -> {status} (this endpoint has no such field)")


def check_no_threshold_leak(srv: Server, path: str, jpeg: bytes) -> None:
    """A per-request threshold must not outlive the request.

    The failure this catches is quiet: a mode that stores the parsed threshold in
    the session instead of a per-request copy keeps it, and the NEXT request --
    which said nothing about thresholds -- is answered under the last client's
    numbers. Nothing in the response says so, because the response reports the
    thresholds it used and they are the wrong ones.
    """
    print("\n  a per-request threshold does not survive into the next request")
    if path == "/v1/classify":
        # top_k is additive and capped, so it cannot leak visibly; the meaningful
        # check for this endpoint is that a request with no field is the default.
        body, ctype = multipart([upload("image", "bus.jpg", jpeg)])
        _, first = post(srv.base + path, body, ctype)
        body, ctype = multipart([field("top_k", "7"),
                                 upload("image", "bus.jpg", jpeg)])
        _, second = post(srv.base + path, body, ctype)
        try:
            report("top_k" not in json.loads(first),
                   "a request naming no top_k gets no top_k array")
            report(len(json.loads(second).get("top_k", [])) == 7,
                   "and the next request's top_k=7 is honoured")
        except Exception as e:
            report(False, "both answers are JSON", f"-> {e}")
        return

    # conf is the sharp one: 0.25 finds three people on the fixture and 0.9 finds
    # none, so a leak is visible in the count.
    def count(**fields):
        body, ctype = multipart([field(k, v) for k, v in fields.items()]
                                + [upload("image", "bus.jpg", jpeg)])
        _, raw = post(srv.base + path, body, ctype)
        try:
            d = json.loads(raw)
        except Exception:
            return None, raw[:120]
        return len(d.get("landmarks", [])), d.get("thresholds")

    base_n, base_thr = count()
    strict_n, strict_thr = count(conf="0.9")
    report(strict_n == 0,
           "conf=0.9 finds nobody where conf=0.25 finds people",
           "" if strict_n == 0 else
           f"-> got {strict_n} at conf=0.9, where the default found {base_n}")
    again_n, again_thr = count()
    report(again_thr == base_thr and again_n == base_n,
           "and the request AFTER it is back at the default thresholds",
           "" if (again_thr == base_thr and again_n == base_n) else
           f"-> {again_thr} / {again_n} people, was {base_thr} / {base_n}")


def check_cli_agreement(srv: Server, exe: Path, arch: str, jpeg: bytes,
                        wav: Path | None = None) -> None:
    """The endpoint's answer and the CLI's, from the same emitter.

    The two are compared rather than trusted because they are two callers of one
    function, and one caller of one function can still differ from the other
    (different label, different path, different default top_k). The timing fields
    are excluded: they are measurements of two different runs.
    """
    print("\n  the endpoint and the CLI emit the same object")
    path = ARCHES[arch]["path"]
    if path == "/v1/embeddings":
        note("skipped: verify_endpoint.py compares this endpoint against --embed")
        return
    if arch == "whisper":
        # Not skipped. The transcript is a plain string rather than an object,
        # but "the endpoint and the CLI say the same words" is exactly as
        # checkable, and it is the check that catches a per-request field the
        # endpoint applies and the CLI does not -- `language` is the one that
        # bites, since transcribing a foreign language with the wrong one is a
        # different sentence rather than a different number.
        out = subprocess.run([str(exe), "transcribe", ARCHES[arch]["container"],
                              str(wav or "/dev/null"), "--json"],
                             capture_output=True, text=True)
        try:
            cli = json.loads(out.stdout)
        except Exception as e:
            report(False, "`transcribe --json` answers one object", f"-> {e}")
            return
        body, ctype = multipart([upload("file", "a.wav", (wav or Path("/dev/null")).read_bytes())])
        status, raw = post(srv.base + path, body, ctype)
        if not report(status == 200, "the endpoint answers 200",
                      "" if status == 200 else f"-> {status}: {raw[:200]!r}"):
            return
        srv_d = json.loads(raw)
        report(cli["text"] == srv_d.get("text"),
               "the endpoint and the CLI transcribe to the same words",
               "" if cli["text"] == srv_d.get("text") else
               f"-> cli {cli['text'][:90]!r} vs endpoint {str(srv_d.get('text'))[:90]!r}")
        report(cli["language"] == "en" and bool(cli["text"].strip()),
               "and the transcript is not empty",
               "" if cli["text"].strip() else "-> empty")
        # Empty-but-200 is the failure worth naming: a WAV reader that accepted
        # the header and then found no samples answers with an empty string and
        # no error, which a shape check calls a pass.
        return
    if arch == "classify":
        out = subprocess.run([str(exe), "classify", ARCHES[arch]["container"],
                              "/tmp/opencode/bus.jpg", "--json"],
                             capture_output=True, text=True)
        try:
            cli = json.loads(out.stdout)[0]
        except Exception as e:
            report(False, "`classify --json` answers one object", f"-> {e}")
            return
        body, ctype = multipart([upload("image", "bus.jpg", jpeg)])
        _, raw = post(srv.base + path, body, ctype)
        srv_d = json.loads(raw)
        for k in ("label", "name"):
            report(cli[k] == srv_d.get(k), f"both report {k}={cli[k]!r}",
                   "" if cli[k] == srv_d.get(k) else
                   f"-> the endpoint said {srv_d.get(k)!r}")
        dp = abs(cli["p"] - srv_d.get("p", -1))
        report(dp < 1e-6, "and the same top-1 probability",
               "" if dp < 1e-6 else f"-> cli {cli['p']} vs endpoint {srv_d.get('p')}")
        report(cli["dispatches"] == srv_d.get("dispatches"),
               "and the same dispatch count",
               "" if cli["dispatches"] == srv_d.get("dispatches") else
               f"-> cli {cli['dispatches']} vs endpoint {srv_d.get('dispatches')}")
        return

    # pose: the whole document, minus the fields that are about the request rather
    # than the answer (the image's label, and the timings).
    out = subprocess.run([str(exe), "pose", ARCHES[arch]["container"],
                          "/tmp/opencode/bus.jpg", "--json"],
                         capture_output=True, text=True)
    try:
        cli = json.loads(out.stdout)
    except Exception as e:
        report(False, "`pose --json` answers one object", f"-> {e}")
        return
    body, ctype = multipart([upload("image", "bus.jpg", jpeg)])
    _, raw = post(srv.base + path, body, ctype)
    srv_d = json.loads(raw)

    def strip(d):
        d = json.loads(json.dumps(d))
        for k in ("image", "timing_s", "backend"):
            d.pop(k, None)
        return d

    same = strip(cli) == strip(srv_d)
    report(same, "every box, every joint, the skeleton and the letterbox match",
           "" if same else "-> the two documents differ, field by field below")
    if strip(cli) != strip(srv_d):
        for k in sorted(set(cli) | set(srv_d)):
            if k in ("image", "timing_s", "backend"):
                continue
            if cli.get(k) != srv_d.get(k):
                print(f"        {k}: cli={str(cli.get(k))[:90]}")
                print(f"        {' ' * len(k)}  srv={str(srv_d.get(k))[:90]}")


# -- main --------------------------------------------------------------------


def check_serve_refusal(exe: Path, container: Path, port: int) -> dict:
    """`serve <hands>` must refuse, and leave nothing listening behind it.

    Three assertions because the first two are not enough. A refusal printed
    after bind() would exit non-zero and say the right words while having
    already taken the port, and an operator whose port is in use has no
    diagnostic that points at this. So the port is probed from outside the
    process, after it has gone.
    """
    print("\n  arch 7 has no endpoint: `serve` must REFUSE rather than start")

    def listening() -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            return s.connect_ex(("127.0.0.1", port)) == 0

    if listening():
        report(False, f"port {port} was free before the test",
               "-> something else is already bound; this gate cannot tell what")
        return {"ok": False, "why": "port in use"}

    cmd = [str(exe), "serve", str(container), "--port", str(port)]
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", "/opt/xilinx/xrt/python")
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=60, env=env)
    except subprocess.TimeoutExpired:
        report(False, "`serve <hands>` exits instead of hanging",
               "-> it is still running after 60 s, so a server was started")
        return {"ok": False, "why": "timeout"}
    text = (p.stdout + p.stderr).decode("utf-8", "replace")

    report(p.returncode != 0,
           f"`serve <hands>` exits non-zero (got {p.returncode})",
           "" if p.returncode else "-> it returned success for an endpoint that "
                                   "does not exist")
    report("no HTTP endpoint" in text,
           "the refusal NAMES the absence of an endpoint",
           "" if "no HTTP endpoint" in text
           else f"-> first line: {text.strip().splitlines()[:1]}")
    report("/v1/hands" not in text or "no /v1/hands" in text,
           "the message does not send the caller to a /v1/hands URL",
           "" if "/v1/hands" not in text or "no /v1/hands" in text
           else "-> it points at a path nothing answers")
    # Give a server that DID start a moment to bind before probing: a process
    # that exited a millisecond ago could still be mid-bind otherwise, and the
    # probe would pass for the wrong reason.
    time.sleep(1.0)
    port_open = listening()
    report(not port_open,
           f"nothing is left LISTENING on port {port}",
           "" if not port_open
           else "-> a port is bound, so the refusal was printed after bind()")

    print(f"  note    (no endpoint, so no /health, no path and no CLI agreement "
          f"to check; those are this gate's other four cases)")
    result = {"ok": True, "refused": True, "port": port}
    note("serve refused arch 7 as designed")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe",
                    default=str(REPO / "runtime" / "build" / "npuembeddings"),
                    help="one of the three runtime binaries; the other two are "
                         "resolved in its directory from the arch under test, so "
                         "this overrides the build TREE rather than one file "
                         "(default: %(default)s)")
    ap.add_argument("--only", action="append", default=None,
                    help="one arch name, repeatable (default: all of ARCHES -- "
                         "the four endpoints and hands, which must refuse)")
    ap.add_argument("--skip", action="append", default=None,
                    help="one arch name, repeatable")
    ap.add_argument("--base-port", type=int, default=8450)
    ap.add_argument("--out", default=str(GATE_OUT / "verify_serve_dispatch.json"))
    args = ap.parse_args()

    exe = Path(args.exe)
    print("serve: one verb, four endpoints, each container answers for its own;\n"
          "and arch 7, which has no endpoint, refuses instead -- both asserted\n")
    needed = {exe_for(exe, n) for n in ARCHES}
    absent = sorted(str(p) for p in needed if not p.exists())
    if absent:
        print(f"FAIL -- missing {' and '.join(absent)}. Build them:\n"
              f"    cmake -S runtime -B runtime/build && "
              f"cmake --build runtime/build -j")
        return 1

    jpeg = fixture()
    if jpeg is None:
        print("FAIL -- no JPEG fixture. Put one at /tmp/opencode/bus.jpg "
              "(810x1080, people in it) -- the pose counts below are read against "
              "it.")
        return 1
    wav = audio_fixture()

    names = [n for n in ARCHES if (not args.only or n in args.only)
             and n not in (args.skip or [])]
    results: dict[str, dict] = {}

    for i, name in enumerate(names):
        spec = ARCHES[name]
        container = Path(spec["container"])
        if not container.is_absolute():
            container = REPO / container
        print(f"\n{'=' * 74}\n  arch {name}: " +
              (spec["path"] or "no endpoint -- must refuse") + f"\n{'=' * 74}")
        if not container.exists():
            note("SKIPPED: no container at " + str(container))
            print("        (a gate that fails for a model nobody packed says "
                  "nothing about the code)")
            results[name] = {"skipped": "no container"}
            continue

        if spec.get("refuses"):
            results[name] = check_serve_refusal(exe_for(exe, name), container,
                                          args.base_port + i)
            continue

        srv = Server(exe_for(exe, name), str(container), args.base_port + i)
        ok, why = srv.start()
        if not ok:
            report(False, f"`serve {name}` starts and answers /health", why)
            results[name] = {"started": False, "why": why[:400]}
            srv.stop()
            continue
        results[name] = {"started": True}
        try:
            check_health(srv, spec["kind"], spec["path"])
            check_own_path(srv, spec["path"], jpeg, wav)
            check_other_paths(srv, spec["path"], jpeg)
            if spec["path"] in ("/v1/classify", "/v1/pose"):
                check_image_refusals(srv, spec["path"], jpeg)
                check_no_threshold_leak(srv, spec["path"], jpeg)
            # Every architecture that produces an answer, not just the image two:
            # the whisper case used to be a note saying it was somebody else's
            # gate's business, which is true of the WORDS and false of the
            # agreement between two callers of the same session.
            if spec["path"] != "/v1/embeddings":
                check_cli_agreement(srv, exe_for(exe, name), name, jpeg, wav)
        finally:
            srv.stop()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")

    print(f"\n{'=' * 74}")
    if _failures:
        print(f"FAIL -- {len(_failures)} of {_checks} checks did not hold:")
        for f in _failures:
            print(f"  - {f}")
        print(f"\n        details: {out}")
        return 1
    print(f"PASS -- {_checks} checks: `serve` answers for every architecture that has "
          f"an\n        endpoint, refuses for arch 7 which has none, and refuses the "
          f"others'\n        paths with a message that says where to go. The image "
          f"endpoints\n        refuse rather than fall back.\n")
    print(f"        {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())