import os
import unittest
from unittest.mock import Mock, patch

from ai_providers import provider_manager as pm


class ProviderManagerTests(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(os.environ, {
            "AI_PROVIDER": "auto",
            "AI_PROVIDER_ORDER": "gemini,groq",
            "GEMINI_API_KEY": "test-gemini-key",
            "GEMINI_MODEL": "gemini-test",
            "GROQ_API_KEY": "test-groq-key",
            "GROQ_MODEL": "openai/gpt-oss-20b",
            "AI_REQUEST_TIMEOUT_SECONDS": "15",
        }, clear=True)
        self.env_patch.start()
        pm._LAST_PROVIDER = None

    def tearDown(self):
        self.env_patch.stop()

    @patch("ai_providers.provider_manager.requests.post")
    def test_gemini_response_parsing(self, post):
        post.return_value = Mock(ok=True, json=lambda: {
            "candidates": [{"content": {"parts": [{"text": "ENTITY=Client"}]}}]
        })
        text = pm._generate_gemini("question")
        self.assertEqual(text, "ENTITY=Client")
        self.assertIn("x-goog-api-key", post.call_args.kwargs["headers"])

    @patch("ai_providers.provider_manager.requests.post")
    def test_groq_response_parsing(self, post):
        post.return_value = Mock(ok=True, json=lambda: {
            "choices": [{"message": {"content": "ENTITY=Client"}}]
        })
        text = pm._generate_groq("question")
        self.assertEqual(text, "ENTITY=Client")
        self.assertEqual(post.call_args.kwargs["json"]["model"], "openai/gpt-oss-20b")

    @patch("ai_providers.provider_manager.requests.post")
    def test_ollama_response_parsing(self, post):
        os.environ["AI_PROVIDER"] = "ollama"
        post.return_value = Mock(ok=True, json=lambda: {
            "message": {"content": "ENTITY=Client"}
        })
        text = pm._generate_ollama("question")
        self.assertEqual(text, "ENTITY=Client")
        self.assertTrue(post.call_args.args[0].endswith("/api/chat"))

    def test_fallback_from_gemini_to_groq(self):
        gemini = Mock(side_effect=pm.ProviderError("Gemini returned HTTP 503"))
        groq = Mock(return_value="ENTITY=Client")
        with patch.dict(pm._PROVIDER_CALLS, {"gemini": gemini, "groq": groq}):
            text = pm.generate_text("question")
        self.assertEqual(text, "ENTITY=Client")
        self.assertEqual(pm.get_last_provider(), "groq")
        gemini.assert_called_once()
        groq.assert_called_once()

    def test_missing_keys_report_clear_error(self):
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("GROQ_API_KEY", None)
        with self.assertRaisesRegex(pm.ProviderError, "No configured AI provider"):
            pm.generate_text("question")

    def test_explicit_provider_is_first(self):
        os.environ["AI_PROVIDER"] = "groq"
        self.assertEqual(pm._provider_order()[0], "groq")


if __name__ == "__main__":
    unittest.main()
