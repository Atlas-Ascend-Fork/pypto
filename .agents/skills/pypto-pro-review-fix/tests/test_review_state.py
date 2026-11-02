# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "review_state.py"
SPEC = importlib.util.spec_from_file_location("review_state", SCRIPT)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class ReviewStateTests(unittest.TestCase):
    def setUp(self):
        self.state = review.new_state({"base": "fixed-base", "plan": "approved-plan"})

    def event(self, name, **payload):
        self.state = review.transition(self.state, name, payload)

    def begin(self, **extra):
        self.event("begin", snapshot=f"source-{self.state['total_rounds'] + 1}", input="input.md", **extra)

    def result(self, count, **extra):
        ledger = copy.deepcopy(self.state["findings"])
        for index in range(len(ledger), count):
            ledger.append({"id": f"F{index + 1}", "root": f"root-{index + 1}", "severity": "Minor"})
        for index, item in enumerate(ledger):
            item.update(status="open" if index < count else "resolved", evidence="independent source evidence")
        snapshot = self.state["rounds"][-1]["snapshot"]
        return {
            "snapshot": snapshot,
            "roles": {"functional": "complete", "rules": "complete", "verifier": "complete"},
            "coverage_complete": True,
            "coverage_evidence": "coverage.md",
            "validation": {"status": "passed", "snapshot": snapshot, "evidence": "test command and output"},
            "report": "report.md",
            "findings": ledger,
            **extra,
        }

    def round(self, count):
        self.begin()
        self.event("finish", **self.result(count))

    def confirm_first_plan_problem(self):
        self.round(10)
        self.round(8)
        self.assertEqual(self.state["status"], "assess")
        self.event("assess", outcome="flawed", evidence="consumer contract cannot be met", proposal="move ownership")

    def apply_revision(self):
        self.event("approve", evidence="user approved the described revision")
        self.event(
            "revise", plan="approved revised ownership", snapshot="implemented-revision", evidence="revision diff"
        )

    def test_first_clean_round_finishes_without_padding(self):
        self.round(0)
        self.assertEqual(self.state["status"], "passed")
        self.assertEqual(self.state["total_rounds"], 1)
        self.assertEqual(self.state["full_reviews_completed"], 1)

    def test_two_full_then_incremental_and_fifth_round_success(self):
        for count in (8, 4, 2, 1, 0):
            self.round(count)
        self.assertEqual(
            [item["mode"] for item in self.state["rounds"]],
            ["full", "full", "incremental", "incremental", "incremental"],
        )
        self.assertEqual(self.state["status"], "passed")
        self.assertEqual(self.state["total_rounds"], 5)
        self.assertEqual(len(self.state["findings"]), 8)

    def test_two_reviewers_and_verifier_consume_one_attempt(self):
        self.begin()
        self.assertEqual(review.summary(self.state)["mode"], "full")
        self.event("finish", **self.result(3))
        self.assertEqual(self.state["total_rounds"], 1)

    def test_exact_thirty_percent_drop_does_not_trigger_slow_progress(self):
        self.round(10)
        self.round(7)
        self.assertEqual(self.state["status"], "fix")
        self.assertEqual(review.next_mode(self.state), "incremental")

    def test_slow_progress_requires_assessment_before_next_review(self):
        self.round(10)
        self.round(8)
        before = copy.deepcopy(self.state)
        with self.assertRaises(ValueError):
            self.begin()
        self.assertEqual(self.state, before)
        self.event(
            "assess", outcome="sound", evidence="late-discovered independent issues", next_action="fix all locations"
        )
        self.assertEqual(self.state["total_rounds"], 2)
        self.assertEqual(self.state["full_reviews_completed"], 2)

    def test_approval_and_implementation_are_both_required_for_reset(self):
        self.confirm_first_plan_problem()
        with self.assertRaises(ValueError):
            self.event("revise", plan="new", snapshot="new-code", evidence="diff")
        self.event("approve", evidence="user approval")
        self.assertEqual(self.state["full_reviews_completed"], 2)
        with self.assertRaises(ValueError):
            self.event("revise", plan="new", snapshot="source-2", evidence="no source change")
        self.event("revise", plan="new", snapshot="new-code", evidence="implemented revision")
        self.assertEqual(self.state["total_rounds"], 2)
        self.assertEqual(self.state["full_reviews_completed"], 0)
        self.assertEqual(self.state["consecutive_plan_problems"], 1)
        self.assertEqual(self.state["base"], "fixed-base")
        self.assertEqual(len(self.state["findings"]), 10)
        self.assertEqual(review.next_mode(self.state), "full")

    def test_second_consecutive_plan_problem_stops_before_limit(self):
        self.confirm_first_plan_problem()
        self.apply_revision()
        self.round(6)
        self.round(5)
        self.event("assess", outcome="flawed", evidence="revised design still violates the consumer contract")
        self.assertEqual(self.state["status"], "needs_human")
        self.assertEqual(self.state["total_rounds"], 4)
        self.assertEqual(self.state["consecutive_plan_problems"], 2)
        with self.assertRaises(ValueError):
            self.event("approve", evidence="automatic approval attempt")

    def test_only_evidence_backed_sound_assessment_breaks_streak(self):
        self.confirm_first_plan_problem()
        self.apply_revision()
        self.round(6)
        self.assertEqual(self.state["consecutive_plan_problems"], 1)
        self.event(
            "assess",
            outcome="sound",
            evidence="new design satisfies the traced consumers",
            next_action="fix local errors",
        )
        self.assertEqual(self.state["consecutive_plan_problems"], 0)
        with self.assertRaises(ValueError):
            self.event("assess", outcome="flawed", evidence="same assessment again", proposal="another plan")

    def test_five_attempts_stop_even_with_only_minor_findings(self):
        for _ in range(5):
            self.round(1)
            if self.state["status"] == "assess":
                self.event(
                    "assess", outcome="sound", evidence="local repair error", next_action="correct the verified path"
                )
        self.assertEqual(self.state["status"], "needs_human")
        self.assertEqual(self.state["total_rounds"], 5)
        with self.assertRaises(ValueError):
            self.begin()

    def test_revision_does_not_create_a_new_cumulative_budget(self):
        self.confirm_first_plan_problem()
        self.apply_revision()
        for count in (4, 2, 1):
            self.round(count)
        self.assertEqual(self.state["status"], "needs_human")
        self.assertEqual(self.state["total_rounds"], 5)

    def test_failed_role_is_counted_but_not_a_completed_full_review(self):
        self.begin()
        self.event(
            "finish", **self.result(0, roles={"functional": "failed", "rules": "complete", "verifier": "incomplete"})
        )
        self.assertEqual(self.state["status"], "needs_human")
        self.assertEqual(self.state["total_rounds"], 1)
        self.assertEqual(self.state["full_reviews_completed"], 0)

    def test_rules_only_mode_does_not_require_functional_role(self):
        self.state = review.new_state({"base": "base", "plan": "plan", "review_kind": "rules"})
        self.begin()
        self.event("finish", **self.result(0, roles={"rules": "complete", "verifier": "complete"}))
        self.assertEqual(self.state["status"], "passed")

    def test_coverage_gap_or_unresolved_candidate_prevents_success(self):
        for gap in ("coverage", "candidate"):
            with self.subTest(gap=gap):
                self.setUp()
                self.begin()
                result = self.result(1 if gap == "candidate" else 0)
                if gap == "coverage":
                    result["coverage_complete"] = False
                else:
                    result["findings"][0]["status"] = "unresolved"
                self.event("finish", **result)
                self.assertEqual(self.state["status"], "needs_human")

    def test_failed_validation_needs_repair_and_blocked_validation_needs_human(self):
        for validation, expected in (("failed", "fix"), ("blocked", "needs_human")):
            with self.subTest(validation=validation):
                self.setUp()
                self.begin()
                result = self.result(0)
                result["validation"]["status"] = validation
                self.event("finish", **result)
                self.assertEqual(self.state["status"], expected)

    def test_snapshot_mismatch_cannot_use_old_validation(self):
        self.begin()
        for field in ("snapshot", "validation"):
            with self.subTest(field=field):
                result = self.result(0)
                if field == "snapshot":
                    result[field] = "wrong-source"
                else:
                    result[field]["snapshot"] = "old-build"
                with self.assertRaises(ValueError):
                    self.event("finish", **result)
                self.assertEqual(self.state["status"], "reviewing")

    def test_omitted_old_findings_cannot_manufacture_zero_issues(self):
        self.round(1)
        self.begin()
        with self.assertRaises(ValueError):
            self.event("finish", **self.result(0, findings=[]))

    def test_duplicate_id_or_root_is_rejected_without_changing_state(self):
        self.begin()
        for same_id in (True, False):
            result = self.result(1)
            duplicate = copy.deepcopy(result["findings"][0])
            if not same_id:
                duplicate["id"] = "OTHER-REVIEWER-F1"
            result["findings"].append(duplicate)
            before = copy.deepcopy(self.state)
            with self.assertRaises(ValueError):
                self.event("finish", **result)
            self.assertEqual(self.state, before)

    def test_severity_increase_prevents_incremental_mode(self):
        self.round(3)
        self.begin()
        result = self.result(1)
        result["findings"][0]["severity"] = "Major"
        self.event("finish", **result)
        self.assertEqual(review.next_mode(self.state), "full")

    def test_missing_incremental_baseline_can_force_full_without_reset(self):
        self.round(4)
        self.round(2)
        self.assertEqual(review.next_mode(self.state), "incremental")
        self.begin(full_review_reason="Previous artifact unavailable; original baseline retained")
        self.assertEqual(self.state["rounds"][-1]["mode"], "full")
        self.assertEqual(self.state["total_rounds"], 3)
        self.assertEqual(self.state["full_reviews_completed"], 2)

    def test_claimed_fixed_severe_issue_still_open_triggers_assessment(self):
        self.begin()
        result = self.result(1)
        result["findings"][0]["severity"] = "Major"
        self.event("finish", **result)
        self.begin(claimed_fixed=["F1"])
        self.event("finish", **self.result(1))
        self.assertEqual(self.state["status"], "assess")

    def test_repair_introduced_severe_issue_triggers_assessment(self):
        self.round(1)
        self.begin()
        result = self.result(2)
        result["findings"][1].update(severity="Blocker", introduced_by_fix=True)
        self.event("finish", **result)
        self.assertEqual(self.state["status"], "assess")

    def test_repair_escalating_an_existing_issue_also_triggers_assessment(self):
        self.round(1)
        self.begin()
        result = self.result(1)
        result["findings"][0].update(severity="Major", introduced_by_fix=True)
        self.event("finish", **result)
        self.assertEqual(self.state["status"], "assess")

    def test_reopened_minor_issue_restores_full_mode(self):
        self.round(4)
        self.round(2)
        self.begin()
        result = self.result(1)
        result["findings"][0]["status"] = "resolved"
        result["findings"][3]["status"] = "open"
        self.event("finish", **result)
        self.assertEqual(self.state["rounds"][-1]["reopened"], ["F4"])
        self.assertEqual(review.next_mode(self.state), "full")

    def test_no_executable_direction_or_unknown_plan_requires_human(self):
        for outcome in ("sound", "unknown"):
            with self.subTest(outcome=outcome):
                self.setUp()
                self.round(1)
                self.event("assess", outcome=outcome, evidence="no supported next repair")
                self.assertEqual(self.state["status"], "needs_human")

    def test_user_stop_retains_inflight_attempt(self):
        self.begin()
        self.event("stop", kind="user", reason="User requested stop")
        self.assertEqual(self.state["status"], "stopped")
        self.assertEqual(self.state["total_rounds"], 1)

    def test_same_source_cannot_be_relabelled_as_another_review(self):
        self.round(1)
        with self.assertRaises(ValueError):
            self.event("begin", snapshot="source-1", input="same-input.md")

    def test_cli_persistence_resume_and_rejected_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path, payload_path = Path(directory) / "state.json", Path(directory) / "payload.json"

            def invoke(event, payload=None):
                command = [sys.executable, str(SCRIPT), "--state", str(state_path), event]
                if payload is not None:
                    payload_path.write_text(json.dumps(payload), encoding="utf-8")
                    command.extend(["--input", str(payload_path)])
                return subprocess.run(command, capture_output=True, text=True, check=False)

            initial = {"base": "base", "plan": "plan"}
            self.assertEqual(invoke("init", initial).returncode, 0)
            self.assertEqual(invoke("begin", {"snapshot": "source", "input": "input.md"}).returncode, 0)
            saved = state_path.read_bytes()
            restored = json.loads(invoke("show").stdout)
            self.assertEqual(restored["total_rounds"], 1)
            self.assertEqual(restored["status"], "reviewing")
            self.assertNotEqual(invoke("init", initial).returncode, 0)
            self.assertEqual(state_path.read_bytes(), saved)
            self.assertNotEqual(invoke("finish", {"snapshot": "wrong"}).returncode, 0)
            self.assertEqual(state_path.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
