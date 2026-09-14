"""The ollama VLM backend against a faked Ollama HTTP API (no server, no model),
plus the answer-letter parsing it shares with the claude backend."""

import io
import json
import os
import unittest
import urllib.error
from unittest import mock

import vlm_backends
from vlm_backends import OllamaVLM, _parse_letter

try:
    from PIL import Image
except ImportError:
    Image = None

TAGS = {"models": [
    {"name": "qwen3.8:27b", "capabilities": ["completion", "vision", "thinking"]},
    {"name": "text-only:latest", "capabilities": ["completion"]},
]}


class FakeOllama:
    """Stands in for urllib.request.urlopen: answers /api/tags and /api/chat,
    recording every chat request body."""

    def __init__(self, reply="B", prompt_eval_count=321):
        self.reply = reply
        self.prompt_eval_count = prompt_eval_count
        self.chats = []

    def __call__(self, request, timeout=None):
        if request.full_url.endswith("/api/tags"):
            body = TAGS
        else:
            self.chats.append(json.loads(request.data))
            body = {"message": {"role": "assistant", "content": self.reply},
                    "prompt_eval_count": self.prompt_eval_count}
        return io.BytesIO(json.dumps(body).encode())


def payload():
    return {"images": [Image.new("RGB", (64, 48)), Image.new("RGB", (32, 32))],
            "question": "What colour is the cup?", "choices": ["red", "blue", "green"]}


@unittest.skipIf(Image is None, "needs PIL (eval venv)")
class TestOllamaVLM(unittest.TestCase):
    def backend(self, fake, **kwargs):
        with mock.patch.object(vlm_backends.urllib.request, "urlopen", fake):
            return OllamaVLM(**kwargs)

    def test_answer_sends_images_greedy_without_thinking(self):
        fake = FakeOllama(reply="B")
        backend = self.backend(fake)
        with mock.patch.object(vlm_backends.urllib.request, "urlopen", fake):
            self.assertEqual(backend.answer(payload()), "B")
        chat = fake.chats[0]
        self.assertEqual(chat["model"], "qwen3.8:27b")
        self.assertIs(chat["think"], False)
        self.assertEqual(chat["options"]["temperature"], 0)
        self.assertEqual(len(chat["messages"][0]["images"]), 2)
        self.assertIn("B. blue", chat["messages"][0]["content"])

    def test_count_tokens_reuses_the_answer_call(self):
        fake = FakeOllama(prompt_eval_count=321)
        backend = self.backend(fake)
        item = payload()
        with mock.patch.object(vlm_backends.urllib.request, "urlopen", fake):
            backend.answer(item)
            self.assertEqual(backend.count_tokens(item), 321)
        self.assertEqual(len(fake.chats), 1)

    def test_truncated_prompt_warns(self):
        fake = FakeOllama(prompt_eval_count=8190)
        backend = self.backend(fake)
        with mock.patch.object(vlm_backends.urllib.request, "urlopen", fake), \
                mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            backend.answer(payload())
        self.assertIn("truncated", err.getvalue())

    def test_missing_model_names_the_pull(self):
        with self.assertRaisesRegex(RuntimeError, "ollama pull nope"):
            self.backend(FakeOllama(), model="nope")

    def test_model_without_vision_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "no vision"):
            self.backend(FakeOllama(), model="text-only")

    def test_no_server_names_the_fix(self):
        down = mock.Mock(side_effect=urllib.error.URLError("refused"))
        with self.assertRaisesRegex(RuntimeError, "ollama serve"):
            self.backend(down)

    def test_host_from_environment(self):
        with mock.patch.dict(os.environ, {"OLLAMA_HOST": "10.0.0.5:11434"}):
            self.assertEqual(self.backend(FakeOllama()).host, "http://10.0.0.5:11434")


class TestParseLetter(unittest.TestCase):
    def test_bare_and_decorated_letters(self):
        self.assertEqual(_parse_letter("B", 3), "B")
        self.assertEqual(_parse_letter(" answer: c.", 3), "C")

    def test_no_valid_letter_abstains(self):
        self.assertIsNone(_parse_letter("ANSWER", 3))
        self.assertIsNone(_parse_letter("D", 3))
        self.assertIsNone(_parse_letter("", 3))


if __name__ == "__main__":
    unittest.main()
