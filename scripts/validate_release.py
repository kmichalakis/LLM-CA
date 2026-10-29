"""Run the offline suite and retain test logs, JUnit and measured coverage."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import sys
import time
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import coverage


class RecordedResult(unittest.TextTestResult):
    def startTest(self, test):
        self.started = time.perf_counter()
        super().startTest(test)

    def stopTest(self, test):
        elapsed = time.perf_counter() - self.started
        self.records.append({"name": test.id(), "seconds": elapsed})
        super().stopTest(test)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []


def validate_release(root: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root))
    cov = coverage.Coverage(source=[str(root / "src" / "llmca")], branch=True, data_file=str(out / ".coverage"))
    cov.start()
    log = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(log), patch("socket.create_connection", side_effect=AssertionError("network disabled in release tests")), patch("urllib.request.urlopen", side_effect=AssertionError("provider requests disabled in release tests")):
        suite = unittest.TestLoader().discover(str(root / "tests"))
        result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=RecordedResult).run(suite)
    cov.stop()
    cov.save()
    elapsed = time.perf_counter() - started
    (out / "tests.log").write_text(log.getvalue(), encoding="utf-8")
    cov.json_report(outfile=str(out / "coverage.json"))
    cov.xml_report(outfile=str(out / "coverage.xml"))
    with (out / "coverage.txt").open("w", encoding="utf-8") as fh:
        cov.report(file=fh, show_missing=True)
    errors = {t.id(): message for t, message in result.errors}
    failures = {t.id(): message for t, message in result.failures}
    skipped = {t.id(): reason for t, reason in result.skipped}
    testsuite = ET.Element("testsuite", name="llmca-offline", tests=str(result.testsRun),
                           failures=str(len(failures)), errors=str(len(errors)), skipped=str(len(skipped)), time=f"{elapsed:.6f}")
    for record in result.records:
        testcase = ET.SubElement(testsuite, "testcase", name=record["name"], time=f"{record['seconds']:.6f}")
        for tag, outcomes in (("error", errors), ("failure", failures), ("skipped", skipped)):
            if record["name"] in outcomes:
                ET.SubElement(testcase, tag).text = outcomes[record["name"]]
    ET.ElementTree(testsuite).write(out / "junit.xml", encoding="utf-8", xml_declaration=True)
    summary = {"tests": result.testsRun, "failures": len(failures), "errors": len(errors),
               "skipped": len(skipped), "successful": result.wasSuccessful(), "seconds": elapsed,
               "python": sys.version, "platform": platform.platform(),
               "network": "socket.create_connection and urllib.request.urlopen blocked",
               "scope": "automated unittest suite; GUI smoke is separate"}
    (out / "test_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result.wasSuccessful():
        raise RuntimeError(f"release tests failed; see {out / 'tests.log'}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(validate_release(Path(__file__).resolve().parents[1], Path(args.out)), indent=2))
