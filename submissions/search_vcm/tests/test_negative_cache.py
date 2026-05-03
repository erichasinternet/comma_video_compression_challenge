import tempfile
import unittest
from pathlib import Path

from submissions.search_vcm.negative_cache import NegativeCache


class NegativeCacheTest(unittest.TestCase):
    def test_dead_family_and_config(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "negative_cache.yaml"
            path.write_text(
                "dead_families:\n"
                "  - dead_family\n"
                "dead_configs:\n"
                "  q55_pareto_controls:\n"
                "    - class_lut_existing\n"
            )
            cache = NegativeCache(path)
            self.assertFalse(cache.check(family="dead_family", config_id="x").allowed)
            self.assertFalse(cache.check(family="q55_pareto_controls", config_id="class_lut_existing").allowed)
            self.assertTrue(
                cache.check(
                    family="q55_pareto_controls",
                    config_id="class_lut_existing",
                    novelty_reason="materially different",
                ).allowed
            )
            self.assertTrue(cache.check(family="dead_family", config_id="x", allow_negative_cache=True).allowed)


if __name__ == "__main__":
    unittest.main()

