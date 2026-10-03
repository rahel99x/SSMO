"""Selection safety: incomplete runs cannot win and both seeds are required."""
import unittest

from singular_sensitivity.tuning import pareto_front, select_confirmed


def row(candidate, seed, score, seconds, status="complete"):
    return {"candidate_id": candidate, "seed": seed, "score": score,
            "complete_seconds": seconds, "status": status, "host_peak_rss_bytes": 100}


class TuningSelectionTests(unittest.TestCase):
    def test_front_excludes_failed_and_nonfinite_runs(self):
        rows = [row("cheap", 17, 0.2, 1), row("accurate", 17, 0.1, 2),
                row("dominated", 17, 0.3, 3), row("failed", 17, 0.001, 0.1, "failed"),
                row("nan", 17, float("nan"), 0.1)]
        self.assertEqual({item["candidate_id"] for item in pareto_front(rows)}, {"cheap", "accurate"})

    def test_seed_failure_and_missing_replication_cannot_win(self):
        rows = [row("valid", 17, 0.2, 2), row("valid", 29, 0.3, 2),
                row("unconfirmed", 17, 0.001, 1), row("failed", 17, 0.001, 1),
                row("failed", 29, 0.001, 1, "failed")]
        self.assertEqual(select_confirmed(rows)["candidate_id"], "valid")

    def test_worst_seed_quality_then_near_tie_runtime(self):
        rows = [row("accurate", 17, 0.05, 4), row("accurate", 29, 0.1, 4),
                row("near-tie", 17, 0.08, 2), row("near-tie", 29, 0.104, 2),
                row("cheap-bad", 17, 0.02, 1), row("cheap-bad", 29, 0.2, 1)]
        self.assertEqual(select_confirmed(rows)["candidate_id"], "near-tie")

    def test_no_confirmation_is_an_error(self):
        with self.assertRaises(ValueError):
            select_confirmed([row("incomplete", 17, 0.1, 1)])
