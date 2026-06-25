# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run all unit tests."""

import argparse
import os
import subprocess
import sys
import time
import unittest
import warnings
from unittest.runner import TextTestResult, TextTestRunner
from unittest.signals import registerResult

# --- TEMP CI-TIMING INSTRUMENTATION (branch ci-timing-probe, do not merge) ---
_TIMING_OUT = os.environ.get("TZREC_TIMING_OUT", "test_durations.tsv")
_DURATIONS = []  # list of (duration_seconds, test_id, kind)
_TIMING_FH = open(_TIMING_OUT, "w", buffering=1)
_TIMING_FH.write("duration_s\tkind\ttest_id\n")


def _record_timing(duration, test_id, kind):
    _DURATIONS.append((duration, test_id, kind))
    _TIMING_FH.write(f"{duration:.3f}\t{kind}\t{test_id}\n")
    _TIMING_FH.flush()


# --- END INSTRUMENTATION ---

SUBPROC_TEST_PATTERN = [
    ".dataset_test.",
    ".odps_dataset_test.",
    ".parquet_dataset_test.",
    ".sampler_test.",
    ".tdm.gen_tree.",
    ".convert_easyrec_config_to_tzrec_config_test.",
]


def _is_in_scope(test_case: unittest.TestCase, scope: str) -> bool:
    method = getattr(test_case, test_case._testMethodName, None)
    method_scopes = set(getattr(method, "_ci_scopes", ()) or ())
    if scope in method_scopes:
        return True
    cls = type(test_case)
    cls_scopes = set(getattr(cls, "_ci_scopes", ()) or ())
    return scope in cls_scopes


def _filter_in_scope(suite, scope: str) -> unittest.TestSuite:
    """Prune tests not tagged with ``scope``; preserves nested suite layout.

    Robust against ``_FailedTest`` placeholders (modules whose import
    failed at discovery): they aren't iterable, so handle the leaf-case
    first and just drop them from a scoped run.
    """
    new_suite = unittest.TestSuite()
    if isinstance(suite, unittest.TestCase):
        if _is_in_scope(suite, scope):
            new_suite.addTest(suite)
        return new_suite
    for child in suite:
        if isinstance(child, unittest.TestCase):
            if _is_in_scope(child, scope):
                new_suite.addTest(child)
        else:
            filtered = _filter_in_scope(child, scope)
            if filtered.countTestCases() > 0:
                new_suite.addTest(filtered)
    return new_suite


def _gather_test_cases(args):
    test_dir = os.path.abspath(args.test_dir)
    test_suite_main = unittest.TestSuite()
    test_suite_subproc = unittest.TestSuite()
    discover = unittest.defaultTestLoader.discover(
        test_dir, pattern=args.pattern, top_level_dir=None
    )
    for suite_discovered in discover:
        for test_case in suite_discovered:
            if args.scope is not None:
                test_case = _filter_in_scope(test_case, args.scope)
                if test_case.countTestCases() == 0:
                    continue
            test_case_str = str(test_case)
            is_subproc = False
            for subp_pattern in SUBPROC_TEST_PATTERN:
                if subp_pattern in test_case_str:
                    is_subproc = True
                    break
            if not is_subproc:
                test_suite_main.addTest(test_case)
            else:
                test_suite_subproc.addTest(test_case)

            if hasattr(test_case, "__iter__"):
                for subcase in test_case:
                    if args.list_tests:
                        print(subcase)
            else:
                if args.list_tests:
                    print(test_case)
    return test_suite_main, test_suite_subproc


def _error_info_of_subproc_result(result):
    start = False
    endline = 0
    error_info = ""
    for line in result.stderr.splitlines():
        if "====" in line:
            start = True
            endline = 0
        if start:
            error_info += line + "\n"
            if "----" in line:
                endline += 1
            if endline == 2:
                break
    return error_info


class _TimingTextResult(TextTestResult):
    """TextTestResult that records per-test wall time (setUp+method+tearDown)."""

    def startTest(self, test):
        self._t0 = time.perf_counter()
        super().startTest(test)

    def stopTest(self, test):
        super().stopTest(test)
        dur = time.perf_counter() - getattr(self, "_t0", time.perf_counter())
        _record_timing(dur, str(test.id()), "main")


class TZRecTestRunner(TextTestRunner):
    """Test Runner for TorchEasyRec."""

    resultclass = _TimingTextResult

    def _makeResult(self):
        return self.resultclass(self.stream, self.descriptions, self.verbosity)

    def run(self, main_test, subproc_test):
        """Run the given test case or test suite."""
        result = self._makeResult()
        registerResult(result)
        result.failfast = self.failfast
        result.buffer = self.buffer
        result.tb_locals = self.tb_locals
        with warnings.catch_warnings():
            if self.warnings:
                # if self.warnings is set, use it to filter all the warnings
                warnings.simplefilter(self.warnings)
                # if the filter is 'default' or 'always', special-case the
                # warnings from the deprecated unittest methods to show them
                # no more than once per module, because they can be fairly
                # noisy.  The -Wd and -Wa flags can be used to bypass this
                # only when self.warnings is None.
                if self.warnings in ["default", "always"]:
                    warnings.filterwarnings(
                        "module",
                        category=DeprecationWarning,
                        message=r"Please use assert\w+ instead.",
                    )
            startTime = time.perf_counter()
            startTestRun = getattr(result, "startTestRun", None)
            if startTestRun is not None:
                startTestRun()
            try:
                main_test(result)
                # run subproc test in subprocess to prevent some global var fork problem
                for subp_test_group in subproc_test:
                    for one_subp_test in subp_test_group:
                        subp_test_module = f"tzrec.{one_subp_test.__module__}"
                        subp_test_name = f"{one_subp_test.__class__.__name__}.{one_subp_test._testMethodName}"  # NOQA
                        _sub_t0 = time.perf_counter()
                        try:
                            run_env = {}
                            if hasattr(one_subp_test, "_run_env"):
                                run_env = one_subp_test._run_env()
                            subp_result = subprocess.run(
                                ["python", "-m", subp_test_module, subp_test_name],
                                capture_output=True,
                                text=True,
                                timeout=600,
                                env=dict(os.environ, **run_env),
                            )
                            if subp_result.returncode != 0:
                                result.failures.append(
                                    (
                                        one_subp_test,
                                        _error_info_of_subproc_result(subp_result),
                                    )
                                )
                                self.stream.write("F")
                            else:
                                result.addSuccess(one_subp_test)
                                self.stream.write(".")
                        except subprocess.TimeoutExpired:
                            result.failures.append((one_subp_test, "timeout"))
                            self.stream.write("F")
                        _record_timing(
                            time.perf_counter() - _sub_t0,
                            f"{subp_test_module}.{subp_test_name}",
                            "subproc",
                        )
                        self.stream.flush()
            finally:
                stopTestRun = getattr(result, "stopTestRun", None)
                if stopTestRun is not None:
                    stopTestRun()

            stopTime = time.perf_counter()
        timeTaken = stopTime - startTime
        result.printErrors()
        if hasattr(result, "separator2"):
            self.stream.writeln(result.separator2)
        run = result.testsRun
        self.stream.writeln(
            "Ran %d test%s in %.3fs" % (run, run != 1 and "s" or "", timeTaken)
        )
        self.stream.writeln()

        expectedFails = unexpectedSuccesses = skipped = 0
        try:
            results = map(
                len,
                (result.expectedFailures, result.unexpectedSuccesses, result.skipped),
            )
        except AttributeError:
            pass
        else:
            expectedFails, unexpectedSuccesses, skipped = results

        infos = []
        if not result.wasSuccessful():
            self.stream.write("FAILED")
            failed, errored = len(result.failures), len(result.errors)
            if failed:
                infos.append("failures=%d" % failed)
            if errored:
                infos.append("errors=%d" % errored)
        else:
            self.stream.write("OK")
        if skipped:
            infos.append("skipped=%d" % skipped)
        if expectedFails:
            infos.append("expected failures=%d" % expectedFails)
        if unexpectedSuccesses:
            infos.append("unexpected successes=%d" % unexpectedSuccesses)
        if infos:
            self.stream.writeln(" (%s)" % (", ".join(infos),))
        else:
            self.stream.write("\n")
        self.stream.flush()

        # --- dump per-test timings sorted slowest-first ---
        sorted_out = _TIMING_OUT + ".sorted"
        with open(sorted_out, "w") as f:
            f.write("duration_s\tkind\ttest_id\n")
            for dur, tid, kind in sorted(_DURATIONS, reverse=True):
                f.write(f"{dur:.3f}\t{kind}\t{tid}\n")
        total = sum(d for d, _, _ in _DURATIONS)
        self.stream.writeln(
            f"[timing] {len(_DURATIONS)} tests, sum={total:.1f}s, "
            f"top20 below; full list -> {sorted_out}"
        )
        for dur, tid, kind in sorted(_DURATIONS, reverse=True)[:20]:
            self.stream.writeln(f"[timing] {dur:8.1f}s  {kind:7s}  {tid}")
        self.stream.flush()
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TorchEasyRec")
    parser.add_argument("--list_tests", action="store_true", help="list all tests")
    parser.add_argument(
        "--pattern", type=str, default="*_test.py", help="test file pattern"
    )
    parser.add_argument(
        "--test_dir", type=str, default="tzrec", help="directory to be tested"
    )
    parser.add_argument(
        "--scope",
        type=str,
        default=None,
        help="filter tests by CI scope marker (e.g. 'h20'); "
        "default None runs the full suite",
    )
    args = parser.parse_args()

    runner = TZRecTestRunner()
    test_suite_main, test_suite_subproc = _gather_test_cases(args)

    if not args.list_tests:
        runner_result = runner.run(test_suite_main, test_suite_subproc)
        failed, errored = len(runner_result.failures), len(runner_result.errors)
        if failed > 0 or errored > 0:
            sys.exit(1)
