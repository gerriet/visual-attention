"""The ollama VLM backend against a faked Ollama HTTP API (no server, no model),
plus the answer-letter parsing it shares with the claude backend."""

import io
import json
import os
import unittest
import urllib.error
from unittest import mock

import vlm_backends
from vlm_backends import MockVLM, OllamaVLM, _parse_boxes, _parse_letter

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


@unittest.skipIf(Image is None, "needs PIL (eval venv)")
class TestLocate(unittest.TestCase):
    def test_ollama_grounding_reply_becomes_fractions(self):
        fake = FakeOllama(reply='```json\n[{"bbox_2d": [277, 81, 366, 198], "label": "glove"}]\n```',
                          prompt_eval_count=263)
        with mock.patch.object(vlm_backends.urllib.request, "urlopen", fake):
            boxes, tokens = OllamaVLM().locate({"images": [Image.new("RGB", (512, 341))],
                                                "question": "What colour is the glove?"})
        self.assertEqual(tokens, 263)
        self.assertEqual(len(boxes), 1)
        for got, want in zip(boxes[0], (0.277, 0.081, 0.366, 0.198)):
            self.assertAlmostEqual(got, want)
        self.assertIn("glove", fake.chats[0]["messages"][0]["content"])
        self.assertEqual(fake.chats[0]["options"]["num_predict"], 200)

    def test_mock_grounds_perfectly_from_the_oracle(self):
        boxes, tokens = MockVLM().locate({"oracle": {"target_boxes": [(0.1, 0.2, 0.3, 0.4)]}})
        self.assertEqual(boxes, [(0.1, 0.2, 0.3, 0.4)])
        self.assertIsNone(tokens)


class TestParseBoxes(unittest.TestCase):
    def test_every_four_number_group(self):
        text = '[{"bbox_2d": [100, 200, 300, 400]}, {"bbox_2d": [900, 900, 800, 1100]}]'
        self.assertEqual(_parse_boxes(text, 1000), [(0.1, 0.2, 0.3, 0.4), (0.8, 0.9, 0.9, 1.0)])

    def test_degenerate_and_malformed_dropped(self):
        self.assertEqual(_parse_boxes("[1, 2, 3] [5, 5, 5, 9] no boxes", 1000), [])


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
