"""VLM backends for the attention front-end study (roadmap M18, H6).

The front-end feeds a vision-language model only the attended fovea crops plus a
low-res global view instead of the full-resolution image. The VLM sits behind a
thin interface so the core stays dependency-free and CI-safe (v2 convention:
modern models live Python-side, behind the interchange boundary):

  VLMBackend.answer(payload)          -> a multiple-choice letter
  VLMBackend.estimate_visual_tokens(size) -> provider-independent token proxy
  VLMBackend.count_tokens(payload)    -> real token count, when the backend has one
  VLMBackend.locate(payload)          -> boxes of what the question asks about
                                         (the --top-down source), when it can

Backends:
  mock    deterministic; answers correctly iff the target is visible in the
          supplied images (payload["oracle"]). Makes the whole pipeline
          testable end to end without a model — the fovea arm scores only when
          attention actually lands on the target, exactly the H6 effect.
  claude  Claude via the anthropic SDK (base64 image blocks, model
          claude-opus-5); real count_tokens(). Gated on the SDK + a key.
  ollama  a local VLM served by Ollama (default qwen3.8:27b) over its HTTP API,
          stdlib only; the real token count comes back with every answer. The
          harness default: free, local, open weights.

The token *fraction* (fovea vs full-res) is what H6 reports, so the absolute
patch size below cancels for a fixed backend.
"""

import base64
import hashlib
import io
import json
import os
import re
import sys
import urllib.request

# One visual token per ~28x28 px patch — a Qwen2-VL-style proxy (14px ViT
# patches merged 2x2). Provider-independent; only ratios are reported, so the
# constant cancels. qwen3.8 under Ollama measures 32 px; the real count is
# recorded with --count-tokens on the claude and ollama backends.
VISUAL_TOKEN_PATCH = 28


def _png_base64(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.standard_b64encode(buffer.getvalue()).decode()


def _prompt_text(payload):
    lettered = "\n".join("%s. %s" % (chr(65 + i), c) for i, c in enumerate(payload["choices"]))
    return "%s\n\n%s\n\nAnswer with the single letter of the correct option only." % (
        payload["question"], lettered)


def _parse_letter(text, n_choices):
    """The first isolated option letter in a reply (not part of a longer word
    like "ANSWER"), or None — an abstention, which the harness scores wrong."""
    valid = {chr(65 + i) for i in range(n_choices)}
    for match in re.finditer(r"(?<![A-Z])([A-Z])(?![A-Z])", text.strip().upper()):
        if match.group(1) in valid:
            return match.group(1)
    return None


def _parse_boxes(text, scale):
    """Every [x1, y1, x2, y2] in a grounding reply, as (x0, y0, x1, y1)
    fractions of the image — the model answers in 0..scale coordinates.
    Degenerate boxes are dropped."""
    boxes = []
    for group in re.findall(r"\[([^\[\]]*)\]", text):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", group)
        if len(numbers) != 4:
            continue
        x0, y0, x1, y1 = (min(1.0, max(0.0, float(v) / scale)) for v in numbers)
        (x0, x1), (y0, y1) = sorted((x0, x1)), sorted((y0, y1))
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1, y1))
    return boxes


class VLMBackend:
    """Answer a multiple-choice visual question over a set of images."""

    name = "base"

    def answer(self, payload):
        """Return the chosen option letter ('A'/'B'/...), or None to abstain
        (an unparseable / non-answer, which the harness scores as wrong rather
        than crediting a fixed guess). `payload` has:
        images   list of PIL.Image (global view first, then fovea crops)
        question str
        choices  list[str], the options in order (choice i is letter chr(65+i))
        oracle   dict {target_visible: bool|None, answer: str} — used only by mock."""
        raise NotImplementedError

    def estimate_visual_tokens(self, size):
        """Provider-independent visual-token proxy for an image of (w, h)."""
        w, h = size
        return -(-w // VISUAL_TOKEN_PATCH) * (-(-h // VISUAL_TOKEN_PATCH))

    def count_tokens(self, payload):
        """Real input-token count for the request, or None if unavailable."""
        return None

    def locate(self, payload):
        """Where is what the question asks about? Returns (boxes, real_tokens):
        boxes as (x0, y0, x1, y1) fractions of payload["images"][0], and the
        call's real input-token count (or None). The --top-down source."""
        raise NotImplementedError("the %s backend cannot localize (needed for --top-down)" % self.name)


class MockVLM(VLMBackend):
    """Deterministic oracle-backed stand-in with three regimes, keyed on the
    harness-supplied `oracle.target_visible`:
      True  -> answer correctly (attention delivered a readable target)
      False -> a deterministic *wrong* choice (target lost, e.g. downsampled) so
               a blind arm scores at ~chance, not accidentally high
      None  -> visibility unknown (no ground-truth box, as on V*Bench): a
               deterministic guess over *all* options, i.e. genuine chance
    All branches are seeded by the question, so runs are reproducible. This ties
    mock accuracy to whether attention actually delivered the target."""

    name = "mock"

    def answer(self, payload):
        choices = payload["choices"]
        if not choices:
            return None  # nothing to choose from — abstain rather than crash
        oracle = payload.get("oracle") or {}
        correct = oracle.get("answer")
        visible = oracle.get("target_visible")
        seed = int(hashlib.sha256(payload["question"].encode()).hexdigest(), 16)
        if visible and correct in choices:
            return chr(65 + choices.index(correct))
        if visible is None:
            # Unknown visibility: guess over every option (chance), the mock
            # can't judge without a target box.
            return chr(65 + seed % len(choices))
        # Target known-not-delivered: a deterministic wrong choice.
        wrong = [i for i, c in enumerate(choices) if c != correct] or list(range(len(choices)))
        return chr(65 + wrong[seed % len(wrong)])

    def locate(self, payload):
        # Perfect grounding from the harness-supplied boxes: exercises the
        # --top-down plumbing (map -> fixations -> crops) without a model.
        return list((payload.get("oracle") or {}).get("target_boxes") or []), None


class ClaudeVLM(VLMBackend):
    """Claude via the anthropic SDK — base64 image blocks, model
    claude-opus-5. Real count_tokens(). Constructed only when the SDK is
    importable; the SDK resolves credentials from the environment or an
    `ant auth login` profile, and a missing credential surfaces as a 401 on the
    first request (not at construction)."""

    name = "claude"

    def __init__(self, model="claude-opus-5"):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "claude backend needs the anthropic SDK: eval/.venv/bin/pip install anthropic") from e
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self.model = model

    def _blocks(self, payload):
        """Build (and memoize on the payload) the messages array. answer() and
        count_tokens() are called with the same payload dict per item, so the
        images are base64-encoded once, not twice."""
        cached = payload.get("_claude_messages")
        if cached is not None:
            return cached
        blocks = [{
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": _png_base64(image)},
        } for image in payload["images"]]
        blocks.append({"type": "text", "text": _prompt_text(payload)})
        messages = [{"role": "user", "content": blocks}]
        payload["_claude_messages"] = messages
        return messages

    def answer(self, payload):
        # Thinking is on by default on claude-opus-5 and shares the max_tokens
        # budget with the answer, so a 16-token cap is spent entirely on an
        # empty thinking block and no letter comes back. Disabled here (legal
        # at the default `high` effort): the arms differ only in what the model
        # can *see*, so extended reasoning would confound the H6 comparison —
        # and a one-letter answer has nothing to reason about.
        response = self._client.messages.create(
            model=self.model, max_tokens=16,
            thinking={"type": "disabled"},
            messages=self._blocks(payload),
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        return _parse_letter(text, len(payload["choices"]))

    def count_tokens(self, payload):
        result = self._client.messages.count_tokens(model=self.model, messages=self._blocks(payload))
        return result.input_tokens


class OllamaVLM(VLMBackend):
    """A local VLM served by Ollama over its HTTP API (stdlib only, no SDK),
    default qwen3.8:27b. Greedy decoding with thinking off — as for claude, the
    arms may differ only in what the model can see. count_tokens() is the answer
    call's own prompt_eval_count (text + image tokens, the model's real
    tokenizer), so it costs no extra request.

    Two silent Ollama behaviours to know about: it downscales any image beyond
    a 64x64-token grid (~2048 px/side for qwen3.8 — keep --full-max-side below
    that or the full-res arm isn't full-res), and it truncates a prompt that
    overflows num_ctx (warned on stderr below)."""

    name = "ollama"

    GROUND_PROMPT = ("Question about this image: %s\n\nLocate the object(s) this question is about. "
                     "Reply only with JSON: [{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"<name>\"}]")

    def __init__(self, model="qwen3.8:27b", host=None, num_ctx=8192, timeout=900, grounding_scale=1000):
        self.model = model
        # Qwen3-family models ground in 0..1000 normalized coordinates
        # (measured on V*Bench for qwen3.8); Qwen2.5-VL used absolute pixels.
        self.grounding_scale = grounding_scale
        host = (host or os.environ.get("OLLAMA_HOST") or "localhost:11434").rstrip("/")
        self.host = host if "://" in host else "http://" + host
        self.num_ctx = num_ctx
        self.timeout = timeout
        # Fail here, with the fix spelled out, rather than on the first item.
        try:
            models = self._request("/api/tags").get("models", [])
        except OSError as e:
            raise RuntimeError("ollama backend: no server at %s — start it with `ollama serve` "
                               "(or set OLLAMA_HOST)" % self.host) from e
        entry = next((m for m in models if m.get("name") in (model, model + ":latest")), None)
        if entry is None:
            have = ", ".join(sorted(m.get("name", "?") for m in models)) or "none"
            raise RuntimeError("ollama backend: model '%s' not available (have: %s) — `ollama pull %s`"
                               % (model, have, model))
        # Older servers don't list capabilities; assume vision + thinking then.
        capabilities = entry.get("capabilities", ["vision", "thinking"])
        if "vision" not in capabilities:
            raise RuntimeError("ollama backend: model '%s' has no vision capability" % model)
        self._can_think = "thinking" in capabilities

    def _request(self, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.host + path, data=data,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def _chat(self, payload):
        """One /api/chat round trip per payload, memoized on it: answer() and
        count_tokens() share the same response."""
        cached = payload.get("_ollama_response")
        if cached is not None:
            return cached
        response = self._generate(_prompt_text(payload), payload["images"], num_predict=16)
        payload["_ollama_response"] = response
        return response

    def _generate(self, text, images, num_predict):
        body = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": self.num_ctx},
            "messages": [{"role": "user", "content": text,
                          "images": [_png_base64(image) for image in images]}],
        }
        if self._can_think:
            body["think"] = False  # else a short token cap is spent on reasoning
        response = self._request("/api/chat", body)
        if response.get("prompt_eval_count", 0) >= self.num_ctx - num_predict:
            print("WARNING: ollama prompt reached num_ctx=%d and was likely truncated "
                  "(images dropped) — raise num_ctx" % self.num_ctx, file=sys.stderr)
        return response

    def answer(self, payload):
        return _parse_letter(self._chat(payload)["message"]["content"], len(payload["choices"]))

    def count_tokens(self, payload):
        return self._chat(payload).get("prompt_eval_count")

    def locate(self, payload):
        response = self._generate(self.GROUND_PROMPT % payload["question"], payload["images"][:1],
                                  num_predict=200)
        return (_parse_boxes(response["message"]["content"], self.grounding_scale),
                response.get("prompt_eval_count"))


_BACKENDS = {"mock": MockVLM, "claude": ClaudeVLM, "ollama": OllamaVLM}


def create_backend(name, **kwargs):
    if name not in _BACKENDS:
        raise ValueError("unknown VLM backend '%s' (have: %s)" % (name, ", ".join(sorted(_BACKENDS))))
    return _BACKENDS[name](**kwargs)


def available_backends():
    return sorted(_BACKENDS)
