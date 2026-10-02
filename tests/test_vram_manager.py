import unittest
from unittest.mock import MagicMock, patch
from smart_router.infrastructure.vram.manager import DynamicVRAMManager


class TestVRAMManager(unittest.TestCase):

    def setUp(self):
        self.vram = DynamicVRAMManager(
            base_url="http://mock-server:8080",
            container_name="test-container",
            purge_settle_time=0.0,
            switch_settle_time=0.0,
        )

    @patch("requests.post")
    @patch("subprocess.run")
    def test_purge_all(self, mock_run, mock_post):
        self.vram.purge_all()
        self.assertEqual(mock_post.call_count, 3)
        mock_run.assert_called_once()
        self.assertIsNone(self.vram.current_model)

    @patch("requests.post")
    @patch("subprocess.run")
    def test_switch_to_new_model(self, mock_run, mock_post):
        self.vram.switch_to("model-a")
        self.assertEqual(self.vram.current_model, "model-a")
        self.assertEqual(mock_post.call_count, 0)

        # Switching to a different model triggers purge of previous
        self.vram.switch_to("model-b")
        self.assertEqual(self.vram.current_model, "model-b")
        self.assertEqual(mock_post.call_count, 3)

        # Switching to same model is a no-op
        mock_post.reset_mock()
        self.vram.switch_to("model-b")
        self.assertEqual(mock_post.call_count, 0)

    @patch("requests.post")
    @patch("subprocess.run")
    def test_session_context(self, mock_run, mock_post):
        with self.vram.session("test-model"):
            self.assertEqual(self.vram.current_model, "test-model")


if __name__ == "__main__":
    unittest.main()
