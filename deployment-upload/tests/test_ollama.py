import os
import unittest
from unittest.mock import patch

from hr_rag.workflow import chat_model


class OllamaConfigTests(unittest.TestCase):
    @patch.dict(os.environ, {"LLM_BACKEND": "nebius", "CHAT_MODEL": "test/model",
                            "NEBIUS_API_KEY": "fake-test-key"}, clear=True)
    def test_nebius_uses_its_own_credentials_and_endpoint(self):
        model = chat_model()
        self.assertEqual(model.model_name, "test/model")
        self.assertEqual(model.openai_api_base, "https://api.tokenfactory.nebius.com/v1/")
        self.assertEqual(model.openai_api_key.get_secret_value(), "fake-test-key")

    @patch.dict(os.environ, {"LLM_BACKEND": "ollama", "CHAT_MODEL": "local-test-model",
                            "OLLAMA_BASE_URL": "http://localhost:11434/v1/"}, clear=True)
    def test_local_client_needs_no_openai_credentials(self):
        model = chat_model()
        self.assertEqual(model.model_name, "local-test-model")
        self.assertEqual(model.openai_api_base, "http://localhost:11434/v1/")
        self.assertEqual(model.openai_api_key.get_secret_value(), "ollama")
        # Schema bindings must be constructible without tool-calling support.
        from hr_rag.workflow import Context
        self.assertIsNotNone(model.with_structured_output(Context, method="json_schema"))


if __name__ == "__main__":
    unittest.main()
