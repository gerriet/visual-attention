"""VLM backends for the attention front-end study (roadmap M18, H6).

The front-end feeds a vision-language model only the attended fovea crops plus a
low-res global view instead of the full-resolution image. The VLM sits behind a
thin interface so the core stays dependency-free and CI-safe (v2 convention:
modern models live Python-side, behind the interchange boundary):

  VLMBackend.answer(payload)          -> a multiple-choice letter
  VLMBackend.estimate_visual_tokens(size) -> provider-independent token proxy
  VLMBackend.count_tokens(payload)    -> real token count, when the backend has one

Backends:
  mock    deterministic; answers correctly iff the target is visible in the
          supplied images (payload["oracle"]). Makes the whole pipeline
          testable end to end without a model — the fovea arm scores only when
          attention actually lands on the target, exactly the H6 effect.
  claude  Claude via the anthropic SDK (base64 image blocks, model
          claude-opus-4-8); real count_tokens(). Gated on the SDK + a key.

The token *fraction* (fovea vs full-res) is what H6 reports, so the absolute
patch size below cancels for a fixed backend.
"""

import hashlib
import io
import os
import re

# One visual token per ~28x28 px patch — a Qwen2-VL-style proxy (14px ViT
# patches merged 2x2). Provider-independent; only ratios are reported, so the
# constant cancels. Claude's real tokenizer is used when the claude backend runs.
VISUAL_TOKEN_PATCH = 28


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


class ClaudeVLM(VLMBackend):
    """Claude via the anthropic SDK — base64 image blocks, model
    claude-opus-4-8. Real count_tokens(). Constructed only when the SDK is
    importable; the SDK resolves credentials from the environment or an
    `ant auth login` profile, and a missing credential surfaces as a 401 on the
    first request (not at construction)."""

    name = "claude"

    def __init__(self, model="claude-opus-4-8"):
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
        import base64

        blocks = []
        for image in payload["images"]:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(buffer.getvalue()).decode(),
                },
            })
        lettered = "\n".join("%s. %s" % (chr(65 + i), c) for i, c in enumerate(payload["choices"]))
        blocks.append({
            "type": "text",
            "text": "%s\n\n%s\n\nAnswer with the single letter of the correct option only."
                    % (payload["question"], lettered),
        })
        messages = [{"role": "user", "content": blocks}]
        payload["_claude_messages"] = messages
        return messages

    def answer(self, payload):
        response = self._client.messages.create(
            model=self.model, max_tokens=16,
            messages=self._blocks(payload),
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip().upper()
        valid = {chr(65 + i) for i in range(len(payload["choices"]))}
        # An isolated option letter (not part of a longer word like "ANSWER").
        for match in re.finditer(r"(?<![A-Z])([A-Z])(?![A-Z])", text):
            if match.group(1) in valid:
                return match.group(1)
        return None  # no parseable option letter — abstain (scored wrong)

    def count_tokens(self, payload):
        result = self._client.messages.count_tokens(model=self.model, messages=self._blocks(payload))
        return result.input_tokens


_BACKENDS = {"mock": MockVLM, "claude": ClaudeVLM}


def create_backend(name, **kwargs):
    if name not in _BACKENDS:
        raise ValueError("unknown VLM backend '%s' (have: %s)" % (name, ", ".join(sorted(_BACKENDS))))
    return _BACKENDS[name](**kwargs)


def available_backends():
    return sorted(_BACKENDS)
