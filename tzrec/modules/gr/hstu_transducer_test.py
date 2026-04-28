# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Focused unit tests for ``HSTUTransducer`` plan-replay glue.

The full ``forward`` requires a configured input preprocessor + output
postprocessor; tests here exercise the post-stack truncation block in
isolation via the static ``_replay_truncation_state`` helper.
"""

import unittest

import torch

from tzrec.modules.gr.hstu_transducer import HSTUTransducer
from tzrec.ops import Kernel
from tzrec.ops.hstu_attention_utils import compute_stu_truncation_plan
from tzrec.ops.hstu_attention_utils_test import _reference_truncation


class ReplayTruncationStateTest(unittest.TestCase):
    """Cover ``HSTUTransducer._replay_truncation_state`` invariants.

    - When ``plan is None``, pass through untouched.
    - When ``plan`` fires: truncate ``seq_timestamps`` via the same plan,
      use ``plan.new_lengths`` for ``seq_lengths`` (not, e.g.,
      ``plan.new_x_offsets`` which would be a wrong field assignment),
      take ``seq_offsets`` from ``post_stu_seq_offsets`` and ``max_seq_len``
      from ``post_stu_max_seq_len``, and emit ``total_uih_len = None``.
    """

    def _setup(self):
        lengths = [10, 14, 6]
        targets = [2, 3, 1]
        ctx = 2
        tail = 4
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            torch.tensor(lengths, dtype=torch.int64)
        )
        total = int(offsets[-1].item())
        seq_lengths = offsets[1:] - offsets[:-1]
        # Identity-marked timestamps so we can verify positional alignment.
        seq_timestamps = torch.arange(total, dtype=torch.float32) * 7.0
        plan = compute_stu_truncation_plan(
            x_offsets=offsets,
            num_targets=torch.tensor(targets, dtype=torch.int64),
            max_seq_len=int(seq_lengths.max().item()),
            truncate_tail_len=tail,
            contextual_seq_len=ctx,
        )
        return {
            "lengths": lengths,
            "targets": targets,
            "ctx": ctx,
            "tail": tail,
            "offsets": offsets,
            "seq_lengths": seq_lengths,
            "seq_timestamps": seq_timestamps,
            "plan": plan,
            "max_seq_len": int(seq_lengths.max().item()),
            "total_uih_len": int(seq_lengths.sum().item()) - sum(targets),
            "total_targets": sum(targets),
        }

    def test_pass_through_when_plan_is_none(self) -> None:
        s = self._setup()
        out = HSTUTransducer._replay_truncation_state(
            seq_timestamps=s["seq_timestamps"],
            seq_lengths=s["seq_lengths"],
            seq_offsets=s["offsets"],
            max_seq_len=s["max_seq_len"],
            total_uih_len=s["total_uih_len"],
            post_stu_seq_offsets=s["offsets"],
            post_stu_max_seq_len=s["max_seq_len"],
            plan=None,
            kernel=Kernel.PYTORCH,
        )
        out_ts, out_lens, out_offsets, out_max, out_total_uih = out
        self.assertIs(out_ts, s["seq_timestamps"])
        self.assertIs(out_lens, s["seq_lengths"])
        self.assertIs(out_offsets, s["offsets"])
        self.assertEqual(out_max, s["max_seq_len"])
        self.assertEqual(out_total_uih, s["total_uih_len"])

    def test_replays_plan_and_assigns_correct_fields(self) -> None:
        s = self._setup()
        post_stu_seq_offsets = s["plan"].new_x_offsets
        post_stu_max_seq_len = s["plan"].new_max_seq_len
        out = HSTUTransducer._replay_truncation_state(
            seq_timestamps=s["seq_timestamps"],
            seq_lengths=s["seq_lengths"],
            seq_offsets=s["offsets"],
            max_seq_len=s["max_seq_len"],
            total_uih_len=s["total_uih_len"],
            post_stu_seq_offsets=post_stu_seq_offsets,
            post_stu_max_seq_len=post_stu_max_seq_len,
            plan=s["plan"],
            kernel=Kernel.PYTORCH,
        )
        out_ts, out_lens, out_offsets, out_max, out_total_uih = out

        # 1. Timestamps truncated against an independent reference (catches
        #    a bug in the unsqueeze / squeeze round-trip).
        ref_ts, _ = _reference_truncation(
            s["seq_timestamps"].unsqueeze(-1),
            s["offsets"],
            s["targets"],
            truncate_tail_len=s["tail"],
            contextual_seq_len=s["ctx"],
        )
        torch.testing.assert_close(out_ts, ref_ts.squeeze(-1))

        # 2. seq_lengths must come from plan.new_lengths, not, say,
        #    plan.new_x_offsets.  Independent reference comparison.
        torch.testing.assert_close(out_lens, s["plan"].new_lengths)
        # Distinct from new_x_offsets (different shape; confirms the
        # right field flowed through).
        self.assertEqual(out_lens.shape[0], len(s["lengths"]))
        self.assertEqual(s["plan"].new_x_offsets.shape[0], len(s["lengths"]) + 1)

        # 3. seq_offsets / max_seq_len come from the post-STU outputs.
        self.assertIs(out_offsets, post_stu_seq_offsets)
        self.assertEqual(out_max, post_stu_max_seq_len)

        # 4. total_uih_len is reset to None so split_2D_jagged derives it.
        self.assertIsNone(out_total_uih)


if __name__ == "__main__":
    unittest.main()
