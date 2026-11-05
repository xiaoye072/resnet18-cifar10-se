#!/usr/bin/env python3
"""Fail if 大作业报告.md does not cite the logged summary numbers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resnet_cifar.models import VARIANTS
from resnet_cifar.utils import read_json


def must_contain(report: str, token: str, label: str, missing: list[str]) -> None:
    if token not in report:
        missing.append(f"{label}: {token}")


def main() -> None:
    report_path = Path("大作业报告.md")
    report = report_path.read_text(encoding="utf-8")
    missing: list[str] = []
    present = []
    for name in VARIANTS:
        summary_path = Path("runs") / name / "summary.json"
        if not summary_path.exists():
            continue
        present.append(name)
        summary = read_json(summary_path)
        complexity = summary["complexity"]
        must_contain(report, f"{summary['best_test_acc'] * 100:.2f}%", f"{name} best acc", missing)
        must_contain(report, f"{summary['final_test_acc'] * 100:.2f}%", f"{name} final acc", missing)
        must_contain(report, f"{complexity['params']:,}", f"{name} params", missing)
        must_contain(report, f"{complexity['macs']:,}", f"{name} macs", missing)
        must_contain(report, str(summary["best_epoch"]), f"{name} best epoch", missing)
        history = read_json(Path("runs") / name / "history.json")
        first = history[0]
        must_contain(
            report,
            f"{first['test_acc'] * 100:.2f}%",
            f"{name} epoch-1 test acc",
            missing,
        )
    if "resnet18" not in present or len(present) < 3:
        print(f"need baseline plus at least two variant logs, found {present}")
        sys.exit(1)
    if missing:
        print("report is missing logged values:")
        for item in missing:
            print(" -", item)
        sys.exit(1)
    print("report cites every logged summary value")


if __name__ == "__main__":
    main()
