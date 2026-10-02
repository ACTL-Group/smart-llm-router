import importlib.util
from pathlib import Path
import unittest


class TestLegacyCompatibility(unittest.TestCase):

    def test_legacy_script_exports(self):
        script_path = Path(__file__).resolve().parent.parent / "smart_router.py"
        spec = importlib.util.spec_from_file_location("smart_router_script", script_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Verify required globals exist
        self.assertTrue(hasattr(module, "DynamicVRAMManager"))
        self.assertTrue(hasattr(module, "vram"))
        self.assertTrue(hasattr(module, "client"))
        self.assertTrue(hasattr(module, "MODEL_EMBED"))
        self.assertTrue(hasattr(module, "MODEL_CHEAP"))
        self.assertTrue(hasattr(module, "MODEL_EXPENSIVE"))
        self.assertTrue(hasattr(module, "MARGIN_THRESHOLD"))
        self.assertTrue(hasattr(module, "LOGPROB_THRESHOLD"))
        self.assertTrue(hasattr(module, "CALIBRATION_SET"))
        self.assertTrue(hasattr(module, "log_step"))
        self.assertTrue(hasattr(module, "log_sub"))
        self.assertTrue(hasattr(module, "log_end"))
        self.assertTrue(hasattr(module, "get_embedding"))
        self.assertTrue(hasattr(module, "init_hnsw_index"))
        self.assertTrue(hasattr(module, "evaluate_slm_uncertainty"))
        self.assertTrue(hasattr(module, "route_query"))
        self.assertTrue(hasattr(module, "render_chat_turn"))
        self.assertTrue(hasattr(module, "dispatch"))


if __name__ == "__main__":
    unittest.main()
