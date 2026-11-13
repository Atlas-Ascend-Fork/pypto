#!/usr/bin/env python3
"""Check review-record structure and render both coverage views, without judging code."""

import argparse
import json
from pathlib import Path


def keyed(items, label):
    result = {}
    for item in items:
        key = item.get("id")
        if not isinstance(key, str) or not key.strip() or key in result:
            raise ValueError(f"{label}: missing or duplicate id: {key!r}")
        result[key] = item
    return result


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def inspect(manifest, record):
    """Return structural errors and unfinished coverage; neither establishes semantic validity."""
    errors, gaps = [], []
    snapshot = manifest.get("snapshot", {})
    if not all(nonempty(snapshot.get(key)) for key in ("source", "base", "target", "rules")):
        errors.append("manifest: source/base/target/rules snapshot identifiers are required")
    if record.get("snapshot") != snapshot:
        errors.append("record snapshot differs from manifest")
    rules, files = manifest.get("rules", []), manifest.get("files", [])
    for name, values in (("rules", rules), ("files", files)):
        if not values or not all(nonempty(value) for value in values) or len(set(values)) != len(values):
            errors.append(f"manifest: {name} must be a nonempty list of unique strings")
    subjects = keyed(record.get("subjects", []), "subjects")
    evidence = keyed(record.get("evidence", []), "evidence")
    checks = keyed(record.get("checks", []), "checks")
    excluded = record.get("excluded_files", {})
    for path, reason in excluded.items():
        if path not in files or not nonempty(reason):
            errors.append(f"invalid file exclusion: {path}")
    for sid, subject in subjects.items():
        if subject.get("path") not in files or not nonempty(subject.get("symbol")):
            errors.append(f"{sid}: requires a manifest path and a symbol/check object")
        if subject.get("path") in excluded:
            errors.append(f"{sid}: path is also excluded")
    for eid, fact in evidence.items():
        if not nonempty(fact.get("fact")) or not fact.get("locations"):
            errors.append(f"{eid}: fact and source locations required")
        elif not all(nonempty(location) for location in fact["locations"]):
            errors.append(f"{eid}: invalid source location")
    for cid, check in checks.items():
        status = check.get("status")
        if check.get("rule") not in rules:
            errors.append(f"{cid}: unknown rule")
        if status not in ("checked", "not_applicable", "pending") or not nonempty(check.get("basis")):
            errors.append(f"{cid}: valid status and a judgment/reason required")
        for field, known in (("subjects", subjects), ("evidence", evidence)):
            refs = check.get(field, [])
            if len(set(refs)) != len(refs) or any(ref not in known for ref in refs):
                errors.append(f"{cid}: invalid {field} references")
        if status == "checked" and (not check.get("subjects") or not check.get("evidence")):
            errors.append(f"{cid}: checked requires subjects and evidence")
        if status == "pending":
            gaps.append(f"{cid}: pending")
    covered_files = {subject.get("path") for subject in subjects.values()} | set(excluded)
    gaps.extend(f"file has no check objects or exclusion: {path}" for path in files if path not in covered_files)
    for rule in rules:
        if not any(check.get("rule") == rule for check in checks.values()):
            gaps.append(f"rule has no record: {rule}")
    for sid in subjects:
        if not any(sid in check.get("subjects", []) for check in checks.values()):
            gaps.append(f"check object has no rule: {sid}")
    return errors, gaps


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def render(manifest, record, errors, gaps):
    lines = ["# Coverage index", "", "Structural validation only; independent source verification is required.", "",
             "| Rule | Check IDs |", "|---|---|"]
    checks = record.get("checks", [])
    for rule in manifest.get("rules", []):
        ids = [check["id"] for check in checks if check.get("rule") == rule]
        lines.append(f"| {cell(rule)} | {cell(', '.join(ids) or 'MISSING')} |")
    lines.extend(["", "| Changed object | Check IDs |", "|---|---|"])
    for subject in record.get("subjects", []):
        ids = [check["id"] for check in checks if subject["id"] in check.get("subjects", [])]
        name = f"{subject['id']}: {subject['path']}:{subject['symbol']}"
        lines.append(f"| {cell(name)} | {cell(', '.join(ids) or 'MISSING')} |")
    lines.extend(["", "## Errors and unfinished coverage", ""])
    lines.extend(f"- {cell(item)}" for item in errors + gaps)
    if not errors and not gaps:
        lines.append("No structural omissions detected; this is not a review approval.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("record", type=Path)
    parser.add_argument("--partial", action="store_true", help="allow unfinished coverage, never invalid records")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text())
        record = json.loads(args.record.read_text())
        errors, gaps = inspect(manifest, record)
        output = render(manifest, record, errors, gaps)
        if args.output:
            args.output.write_text(output)
        else:
            print(output, end="")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        parser.exit(1, f"Invalid coverage input: {exc}\n")
    print(f"Structural errors: {len(errors)}; unfinished items: {len(gaps)}")
    return int(bool(errors or (gaps and not args.partial)))


if __name__ == "__main__":
    raise SystemExit(main())
