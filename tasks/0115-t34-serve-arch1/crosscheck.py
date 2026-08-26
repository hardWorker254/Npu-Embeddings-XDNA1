"""tasks/0115: does arch=1's HTTP endpoint agree with its own --embed path?

The BERT endpoint has had this check since tasks/0037 (tools/verify_endpoint.py
"vs --embed"), and it is the check that matters: an endpoint that returns
plausible vectors which are not the ones --embed produces is exactly the
fail-open shape this project keeps finding. Run against a server already
listening on --port.
"""
import argparse, json, sys, urllib.request
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8422)
ap.add_argument("--dir", default="tasks/0115-t34-serve-arch1")
a = ap.parse_args()

texts = [l.rstrip("\n") for l in open(f"{a.dir}/in.txt", encoding="utf-8") if l.strip()]
ref = np.fromfile(f"{a.dir}/out_embed.f32", dtype=np.float32).reshape(len(texts), -1)

def post(payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{a.port}/v1/embeddings",
        data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

st, body = post({"input": texts})
assert st == 200, (st, body)
emb = np.stack([np.asarray(d["embedding"], np.float32) for d in body["data"]])
idx_ok = [d["index"] for d in body["data"]] == list(range(len(texts)))
dmax = float(np.abs(emb - ref).max())
norms = np.linalg.norm(emb, axis=1)

print(f"  model                    -> {body['model']}")
print(f"  batch of {len(texts)}              -> shape {emb.shape}, indices ordered: {idx_ok}")
print(f"  norms                    -> min {norms.min():.6f} max {norms.max():.6f}")
print(f"  usage.prompt_tokens      -> {body['usage']['prompt_tokens']}")
print(f"  vs --embed               -> max abs diff {dmax:.3e}")

st64, b64 = post({"input": texts, "encoding_format": "base64"})
import base64 as b64mod
e64 = np.stack([np.frombuffer(b64mod.b64decode(d["embedding"]), np.float32)
                for d in b64["data"]])
d64 = float(np.abs(e64 - emb).max())
d64ref = float(np.abs(e64 - ref).max())
# The JSON float form is written with "%.7g" -- 7 significant digits, a
# deliberate response-size choice in the existing endpoint code -- so the
# float arm can only ever agree to ~1e-7 relative. base64 carries the raw
# fp32, so THAT is where bit-identity against --embed is checkable, and it is
# the arm the OpenAI client uses by default.
print(f"  base64 (decoded) vs float-> max abs diff {d64:.3e} "
      f"(the %.7g JSON text form, not an encode difference)")
print(f"  base64 vs --embed        -> max abs diff {d64ref:.3e}")

errs = []
for name, payload, want in [
        ("no input", {}, 400),
        ("empty list", {"input": []}, 400),
        ("bad format", {"input": ["x"], "encoding_format": "xml"}, 400),
        ("token ids", {"input": [[1, 2, 3]]}, 400)]:
    code, _ = post(payload)
    errs.append((name, code, code == want))
    print(f"    {name:<12} -> HTTP {code}")

ok = (d64ref == 0.0 and dmax < 1e-6 and idx_ok
      and body["usage"]["prompt_tokens"] > 0
      and all(e[2] for e in errs))
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
