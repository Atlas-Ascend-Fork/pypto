#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Persist one review/fix run. Evidence and source identities are supplied by the main agent."""

import argparse
import copy
import json
from pathlib import Path
import sys

MAX_ROUNDS = 5
FULL_ROUNDS = 2
SEVERITIES = ("Blocker", "Major", "Minor")
TERMINAL = {"passed", "needs_human", "stopped"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty(data, key):
    value = data.get(key)
    require(isinstance(value, str) and bool(value.strip()), f"{key} must be a nonempty string")
    return value


def new_state(data):
    mode = data.get("review_kind", "all")
    require(mode in {"all", "rules"}, "review_kind must be all or rules")
    return {
        "base": nonempty(data, "base"),
        "plan": nonempty(data, "plan"),
        "review_kind": mode,
        "plan_revision": 1,
        "total_rounds": 0,
        "full_reviews_completed": 0,
        "consecutive_plan_problems": 0,
        "status": "ready",
        "reason": "Initialized; preserve this file when resuming the task",
        "rounds": [],
        "findings": [],
        "plan_history": [],
    }


def completed_rounds(state):
    return [
        item
        for item in state["rounds"]
        if item["plan_revision"] == state["plan_revision"] and item.get("review_complete")
    ]


def converging(previous, current):
    return (
        current["counts"]["total"] < previous["counts"]["total"]
        and all(current["counts"][level] <= previous["counts"][level] for level in SEVERITIES[:2])
        and not current["reopened"]
    )


def next_mode(state):
    rounds = completed_rounds(state)
    if state["full_reviews_completed"] >= FULL_ROUNDS and len(rounds) >= 2:
        if converging(rounds[-2], rounds[-1]):
            return "incremental"
    return "full"


def begin(state, data):
    require(state["status"] in {"ready", "fix"}, "Finish the pending action before starting another round")
    require(state["total_rounds"] < MAX_ROUNDS, "The five-round limit cannot be reset")
    snapshot = nonempty(data, "snapshot")
    require(
        not state["rounds"] or snapshot != state["rounds"][-1]["snapshot"],
        "A new round needs a changed source snapshot; do not repeat the same review",
    )
    claimed = data.get("claimed_fixed", [])
    require(
        isinstance(claimed, list) and all(isinstance(item, str) for item in claimed),
        "claimed_fixed must be a list of finding IDs",
    )
    known = {item["id"] for item in state["findings"]}
    require(set(claimed) <= known, "claimed_fixed contains an unknown finding ID")
    full_reason = data.get("full_review_reason")
    if full_reason is not None:
        nonempty(data, "full_review_reason")
    mode = "full" if full_reason is not None else next_mode(state)
    state["total_rounds"] += 1
    state["rounds"].append(
        {
            "number": state["total_rounds"],
            "plan_revision": state["plan_revision"],
            "mode": mode,
            "snapshot": snapshot,
            "input": nonempty(data, "input"),
            "claimed_fixed": claimed,
            "repair_evidence": data.get("repair_evidence"),
            "full_review_reason": full_reason,
        }
    )
    state.update(status="reviewing", reason=f"Run one {mode} review group on the fixed snapshot")


def checked_findings(state, data):
    findings = data.get("findings")
    require(isinstance(findings, list), "findings must contain the complete finding ledger")
    previous = {item["id"]: item for item in state["findings"]}
    ids, roots = set(), set()
    for item in findings:
        require(isinstance(item, dict), "Each finding must be an object")
        identity, root = nonempty(item, "id"), nonempty(item, "root")
        require(
            identity not in ids and root not in roots, "Merge duplicate finding IDs or root causes before recording"
        )
        ids.add(identity)
        roots.add(root)
        require(item.get("severity") in SEVERITIES, "Invalid finding severity")
        require(item.get("status") in {"open", "resolved", "rejected", "unresolved"}, "Invalid finding status")
        nonempty(item, "evidence")
        require(type(item.get("introduced_by_fix", False)) is bool, "introduced_by_fix must be boolean")
        if identity in previous:
            require(root == previous[identity]["root"], "A finding ID must retain its root cause")
    require(set(previous) <= ids, "Do not omit earlier findings; record resolution or rejection evidence")
    return findings


def finish(state, data):
    require(state["status"] == "reviewing", "There is no active review to finish")
    current = state["rounds"][-1]
    require(data.get("snapshot") == current["snapshot"], "The result must match the reviewed snapshot")
    roles = data.get("roles", {})
    required_roles = ("rules", "verifier") if state["review_kind"] == "rules" else ("functional", "rules", "verifier")
    require(
        isinstance(roles, dict)
        and all(roles.get(role) in {"complete", "incomplete", "failed"} for role in required_roles),
        "Record every required role's status",
    )
    require(type(data.get("coverage_complete")) is bool, "coverage_complete must be boolean")
    nonempty(data, "coverage_evidence")
    validation = data.get("validation", {})
    require(
        isinstance(validation, dict) and validation.get("status") in {"passed", "failed", "blocked"},
        "validation must record passed, failed, or blocked",
    )
    require(validation.get("snapshot") == current["snapshot"], "Validation must correspond to the reviewed snapshot")
    nonempty(validation, "evidence")
    nonempty(data, "report")
    findings = checked_findings(state, data)
    previous = {item["id"]: item for item in state["findings"]}
    opened = [item for item in findings if item["status"] == "open"]
    unresolved = [item["id"] for item in findings if item["status"] == "unresolved"]
    reopened = [item["id"] for item in opened if previous.get(item["id"], {}).get("status") == "resolved"]
    severe = [item for item in opened if item["severity"] in SEVERITIES[:2]]
    recurring = [item["id"] for item in severe if item["id"] in current["claimed_fixed"] or item["id"] in reopened]
    regressions = [
        item["id"]
        for item in severe
        if item.get("introduced_by_fix")
        and (
            previous.get(item["id"], {}).get("status") != "open"
            or SEVERITIES.index(item["severity"]) < SEVERITIES.index(previous[item["id"]]["severity"])
        )
    ]
    complete = (
        all(roles[role] == "complete" for role in required_roles) and data["coverage_complete"] and not unresolved
    )
    current.update(
        {
            "result": copy.deepcopy(data),
            "review_complete": complete,
            "counts": {
                **{level: sum(item["severity"] == level for item in opened) for level in SEVERITIES},
                "total": len(opened),
            },
            "reopened": reopened,
        }
    )
    state["findings"] = findings
    if complete and current["mode"] == "full":
        state["full_reviews_completed"] += 1
    if complete and not opened and validation["status"] == "passed":
        state.update(status="passed", reason="The reviewed snapshot has no open findings or verification gaps")
        return
    if state["total_rounds"] >= MAX_ROUNDS:
        state.update(
            status="needs_human", reason="Five review attempts exhausted; do not apply further automatic fixes"
        )
        return
    if not complete or validation["status"] == "blocked":
        state.update(status="needs_human", reason="Incomplete roles, coverage, candidates, or required validation")
        return
    rounds = completed_rounds(state)
    full = [item for item in rounds if item["mode"] == "full"]
    slow = (
        len(full) == FULL_ROUNDS
        and current["mode"] == "full"
        and current["counts"]["total"] >= 5
        and 10 * current["counts"]["total"] > 7 * full[0]["counts"]["total"]
    )
    stalled = (
        len(rounds) >= 3
        and rounds[-1]["counts"]["total"] >= rounds[-2]["counts"]["total"] >= rounds[-3]["counts"]["total"]
    )
    current["assessment_triggers"] = {
        "slow": slow,
        "stalled": stalled,
        "recurring_severe": recurring,
        "repair_regressions": regressions,
    }
    if slow or stalled or recurring or regressions:
        state.update(status="assess", reason="Reconsider the plan using the recorded counts and root-cause evidence")
    else:
        state.update(
            status="fix", reason="Repair confirmed findings or validation failures, then review the changed snapshot"
        )


def assess(state, data):
    require(state["status"] in {"assess", "fix"}, "A completed review is required for plan assessment")
    current = state["rounds"][-1]
    require("assessment" not in current, "This round's assessment has already been recorded")
    outcome = data.get("outcome")
    require(outcome in {"sound", "flawed", "unknown"}, "Invalid plan assessment outcome")
    nonempty(data, "evidence")
    current["assessment"] = copy.deepcopy(data)
    if outcome == "sound":
        state["consecutive_plan_problems"] = 0
        if isinstance(data.get("next_action"), str) and data["next_action"].strip():
            state.update(status="fix", reason="Plan supported by evidence; perform the recorded repair action")
        else:
            state.update(status="needs_human", reason="No executable repair direction after assessment")
    elif outcome == "flawed":
        state["consecutive_plan_problems"] += 1
        if state["consecutive_plan_problems"] >= 2:
            state.update(status="needs_human", reason="Two consecutive assessments confirmed plan problems")
        else:
            nonempty(data, "proposal")
            state.update(status="awaiting_approval", reason="Obtain authorization for the proposed plan revision")
    else:
        state.update(status="needs_human", reason="Plan assessment needs missing evidence or a human decision")


def approve(state, data):
    require(state["status"] == "awaiting_approval", "There is no plan revision awaiting approval")
    state["rounds"][-1]["approval"] = nonempty(data, "evidence")
    state.update(
        status="awaiting_revision", reason="Implement the authorized revision before resetting the full-review stage"
    )


def revise(state, data):
    require(state["status"] == "awaiting_revision", "A plan reset requires a recorded authorization")
    plan, snapshot = nonempty(data, "plan"), nonempty(data, "snapshot")
    require(plan != state["plan"], "Record the revised plan, not just a counter reset")
    require(snapshot != state["rounds"][-1]["snapshot"], "Implement the revised plan before resetting")
    nonempty(data, "evidence")
    state["plan_history"].append(
        {
            "previous_revision": state["plan_revision"],
            "previous_plan": state["plan"],
            "after_round": state["total_rounds"],
            **copy.deepcopy(data),
        }
    )
    state["plan"] = plan
    state["plan_revision"] += 1
    state["full_reviews_completed"] = 0
    state.update(status="ready", reason="Revised plan implemented; cumulative limits and finding history are retained")


def stop(state, data):
    require(data.get("kind") in {"user", "human"}, "stop kind must be user or human")
    state.update(status="stopped" if data["kind"] == "user" else "needs_human", reason=nonempty(data, "reason"))


def transition(state, event, data):
    """Apply one event without modifying the caller's state if the event is rejected."""
    require(state["status"] not in TERMINAL, "This run has ended; do not restart it automatically")
    updated = copy.deepcopy(state)
    handlers = {"begin": begin, "finish": finish, "assess": assess, "approve": approve, "revise": revise, "stop": stop}
    require(event in handlers, "Unknown state transition")
    handlers[event](updated, data)
    return updated


def summary(state):
    return {
        key: state[key]
        for key in (
            "status",
            "reason",
            "total_rounds",
            "full_reviews_completed",
            "consecutive_plan_problems",
            "plan_revision",
        )
    } | {
        "next_mode": next_mode(state) if state["status"] in {"ready", "fix"} else None,
        "mode": state["rounds"][-1]["mode"] if state["status"] == "reviewing" else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("event", choices=("init", "show", "begin", "finish", "assess", "approve", "revise", "stop"))
    parser.add_argument("--input", type=Path, help="JSON event payload file; required except for show")
    args = parser.parse_args()
    try:
        if args.event == "show":
            print(json.dumps(json.loads(args.state.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
            return 0
        require(args.input is not None, "--input is required for a state transition")
        data = json.loads(args.input.read_text(encoding="utf-8"))
        require(isinstance(data, dict), "The event payload must be a JSON object")
        if args.event == "init":
            require(not args.state.exists(), "State already exists; resume it instead of resetting counters")
            state = new_state(data)
        else:
            state = transition(json.loads(args.state.read_text(encoding="utf-8")), args.event, data)
        args.state.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.state.with_name(args.state.name + ".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.state)
        print(json.dumps(summary(state), ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Review state error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
