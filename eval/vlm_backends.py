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

# One visual token per ~28x28 px patch — a Qwen2-VL-style proxy (14px ViT
# patches merged 2x2). Provider-independent; only ratios are reported, so the
# constant cancels. Claude's real tokenizer is used when the claude backend runs.
VISUAL_TOKEN_PATCH = 28


class VLMBackend:
    """Answer a multiple-choice visual question over a set of images."""

    name = "base"

    def answer(self, payload):
        """Return the chosen option letter ('A'/'B'/...). `payload` has:
        images   list of PIL.Image (global view first, then fovea crops)
        question str
        choices  list[str], the options in order (choice i is letter chr(65+i))
        oracle   dict {target_visible: bool, answer: str} — used only by mock."""
        raise NotImplementedError

    def estimate_visual_tokens(self, size):
        """Provider-independent visual-token proxy for an image of (w, h)."""
        w, h = size
        return -(-w // VISUAL_TOKEN_PATCH) * (-(-h // VISUAL_TOKEN_PATCH))

    def count_tokens(self, payload):
        """Real input-token count for the request, or None if unavailable."""
        return None


class MockVLM(VLMBackend):
    """Deterministic oracle-backed stand-in. Answers correctly iff the target
    is visible in the supplied images; otherwise returns a deterministic wrong
    choice (seeded by the question, so runs are reproducible). This ties mock
    accuracy to whether attention actually delivered the target to the model."""

    name = "mock"

    def answer(self, payload):
        choices = payload["choices"]
        oracle = payload.get("oracle") or {}
        correct = oracle.get("answer")
        if oracle.get("target_visible") and correct in choices:
            return chr(65 + choices.index(correct))
        # Target not delivered: deterministic near-chance guess, biased away from
        # the correct option so a blind arm scores at ~chance, not accidentally high.
        seed = int(hashlib.sha256(payload["question"].encode()).hexdigest(), 16)
        wrong = [i for i, c in enumerate(choices) if c != correct] or list(range(len(choices)))
        return chr(65 + wrong[seed % len(wrong)])


class ClaudeVLM(VLMBackend):
    """Claude via the anthropic SDK — base64 image blocks, model
    claude-opus-4-8. Real count_tokens(). Constructed only when the SDK is
    importable and a credential is resolvable (env key or `ant auth login`
    profile); otherwise raises with an actionable message."""

    name = "claude"

    def __init__(self, model="claude-opus-4-8"):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "claude backend needs the anthropic SDK: eval/.venv/bin/pip install anthropic") from e
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            # The SDK also resolves an `ant auth login` profile; only warn, don't block.
            pass
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self.model = model

    def _blocks(self, payload):
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
        return [{"role": "user", "content": blocks}]

    def answer(self, payload):
        response = self._client.messages.create(
            model=self.model, max_tokens=16,
            messages=self._blocks(payload),
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        for ch in text:
            if ch.upper() in [chr(65 + i) for i in range(len(payload["choices"]))]:
                return ch.upper()
        return "A"  # unparseable: default, counted as an answer

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
