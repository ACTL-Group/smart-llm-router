import os
import unittest
from unittest.mock import patch
from smart_router.config.settings import Settings


class TestSettings(unittest.TestCase):

    def test_default_settings(self):
        settings = Settings()
        self.assertEqual(settings.server_url, "http://localhost:8080")
        self.assertEqual(settings.base_api_url, "http://localhost:8080")
        self.assertEqual(settings.openai_base_url, "http://localhost:8080/v1")
        self.assertEqual(settings.model_embed, "text-embedding")
        self.assertEqual(settings.model_cheap, "bonsai-27b")
        self.assertEqual(settings.model_expensive, "qwen-27b")
        self.assertEqual(settings.router_margin_threshold, 0.12)
        self.assertEqual(settings.router_logprob_threshold, -0.80)

    def test_from_env_with_v1_suffix(self):
        env_vars = {
            "LLAMA_SERVER_URL": "http://my-server:9090/v1",
            "API_KEY": "secret-key",
            "MODEL_EMBED": "custom-embed",
            "MODEL_CHEAP": "small-model",
            "MODEL_EXPENSIVE": "large-model",
            "ROUTER_MARGIN_THRESHOLD": "0.25",
            "ROUTER_LOGPROB_THRESHOLD": "-0.50",
        }
        with patch.dict(os.environ, env_vars):
            settings = Settings.from_env()
            self.assertEqual(settings.base_api_url, "http://my-server:9090")
            self.assertEqual(settings.openai_base_url, "http://my-server:9090/v1")
            self.assertEqual(settings.api_key, "secret-key")
            self.assertEqual(settings.model_embed, "custom-embed")
            self.assertEqual(settings.router_margin_threshold, 0.25)
            self.assertEqual(settings.router_logprob_threshold, -0.50)


if __name__ == "__main__":
    unittest.main()
