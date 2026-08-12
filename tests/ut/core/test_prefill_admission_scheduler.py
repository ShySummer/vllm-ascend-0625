# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from types import SimpleNamespace

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import PrefillAdmissionConfig
from vllm_ascend.core.prefill_admission_scheduler import PrefillAdmissionController


def _decode_request(
    request_id: str,
    token_demand: int = 1,
    next_decode_eligible_step: int = 0,
):
    return SimpleNamespace(
        request_id=request_id,
        is_prefill_chunk=False,
        num_tokens_with_spec=token_demand,
        num_output_placeholders=0,
        num_computed_tokens=0,
        next_decode_eligible_step=next_decode_eligible_step,
    )


def _pending_prefill(request_id: str):
    return SimpleNamespace(request_id=request_id)


def _running_prefill(request_id: str):
    return SimpleNamespace(request_id=request_id, is_prefill_chunk=True)


class TestPrefillAdmissionController(TestBase):
    def _make_controller(self, now, **overrides):
        user_config = {
            "enabled": True,
            "decode_low_watermark": 2,
            "prefill_burst_cooldown_ms": 1000,
            "prefill_tokens_per_pp_bubble": 512,
        }
        user_config.update(overrides)
        return PrefillAdmissionController(
            PrefillAdmissionConfig(user_config),
            pipeline_parallel_size=2,
            clock=lambda: now[0],
        )

    @staticmethod
    def _decide(
        controller,
        running,
        pending,
        *,
        scheduler_step=1,
        max_prefill_slots=4,
        all_pending=None,
        waiting=None,
    ):
        all_pending = pending if all_pending is None else all_pending
        waiting = pending if waiting is None else waiting
        return controller.decide(
            running,
            pending,
            all_pending_prefills=lambda: all_pending,
            waiting_prefills=waiting,
            max_prefill_slots=max_prefill_slots,
            scheduler_step=scheduler_step,
            max_token_budget=4096,
        )

    def test_prefill_stays_open_before_running_capacity(self):
        controller = self._make_controller([0.0])
        pending = [_pending_prefill("p0")]

        decision = self._decide(
            controller,
            [_decode_request("d0"), _decode_request("d1"), _decode_request("d2")],
            pending,
        )

        self.assertFalse(decision.throttle_prefills)
        self.assertEqual(decision.reason, "open")
        self.assertEqual(decision.token_budget, 515)

    def test_running_capacity_latches_hold_below_capacity(self):
        controller = self._make_controller([0.0])
        pending = [_pending_prefill("p0")]

        full = self._decide(
            controller,
            [_decode_request(f"d{i}") for i in range(4)],
            pending,
        )
        drained = self._decide(
            controller,
            [_decode_request(f"d{i}") for i in range(3)],
            pending,
            scheduler_step=2,
        )

        self.assertTrue(full.throttle_prefills)
        self.assertEqual(full.reason, "running_full_hold")
        self.assertEqual(full.token_budget, 4)
        self.assertTrue(drained.throttle_prefills)
        self.assertEqual(drained.reason, "running_full_hold")
        self.assertEqual(drained.token_budget, 3)

    def test_low_decode_releases_hold_without_cooldown(self):
        now = [0.0]
        controller = self._make_controller(
            now, prefill_burst_cooldown_ms=99999999999
        )
        pending = [_pending_prefill("p0")]
        self._decide(
            controller,
            [_decode_request(f"d{i}") for i in range(4)],
            pending,
        )

        now[0] = 0.001
        released = self._decide(
            controller,
            [_decode_request("d0")],
            pending,
            scheduler_step=2,
        )

        self.assertFalse(released.throttle_prefills)
        self.assertEqual(released.reason, "low_decode_release")
        self.assertEqual(released.token_budget, 513)

    def test_decode_equal_to_watermark_does_not_release_hold(self):
        controller = self._make_controller([0.0])
        pending = [_pending_prefill("p0")]
        self._decide(
            controller,
            [_decode_request(f"d{i}") for i in range(4)],
            pending,
        )

        decision = self._decide(
            controller,
            [_decode_request("d0"), _decode_request("d1")],
            pending,
            scheduler_step=2,
        )

        self.assertTrue(decision.throttle_prefills)
        self.assertEqual(decision.reason, "running_full_hold")

    def test_release_stays_open_until_capacity_and_then_rearms(self):
        controller = self._make_controller([0.0])
        first_pending = [_pending_prefill("p0")]
        self._decide(
            controller,
            [_decode_request(f"d{i}") for i in range(4)],
            first_pending,
        )
        self._decide(
            controller,
            [_decode_request("d0")],
            first_pending,
            scheduler_step=2,
        )

        next_pending = [_pending_prefill("p1")]
        open_again = self._decide(
            controller,
            [_decode_request(f"d{i}") for i in range(3)],
            next_pending,
            scheduler_step=3,
        )
        rearmed = self._decide(
            controller,
            [_decode_request(f"d{i}") for i in range(4)],
            next_pending,
            scheduler_step=4,
        )

        self.assertFalse(open_again.throttle_prefills)
        self.assertEqual(open_again.reason, "open")
        self.assertTrue(rearmed.throttle_prefills)
        self.assertEqual(rearmed.reason, "running_full_hold")

    def test_low_decode_overrides_full_running_batch(self):
        controller = self._make_controller([0.0])
        running_prefills = [
            _running_prefill("p0"),
            _running_prefill("p1"),
            _running_prefill("p2"),
        ]

        decision = self._decide(
            controller,
            [_decode_request("d0"), *running_prefills],
            running_prefills,
            waiting=[],
        )

        self.assertFalse(decision.throttle_prefills)
        self.assertEqual(decision.reason, "open")

    def test_no_pending_prefill_still_updates_hold_state(self):
        controller = self._make_controller([0.0])

        no_prefill = self._decide(
            controller,
            [_decode_request(f"d{i}") for i in range(4)],
            [],
        )
        blocked_arrival = self._decide(
            controller,
            [_decode_request(f"d{i}") for i in range(3)],
            [_pending_prefill("p0")],
            scheduler_step=2,
        )
        cleared_without_prefill = self._decide(
            controller,
            [_decode_request("d0")],
            [],
            scheduler_step=3,
        )
        admitted_arrival = self._decide(
            controller,
            [_decode_request(f"d{i}") for i in range(3)],
            [_pending_prefill("p1")],
            scheduler_step=4,
        )

        self.assertFalse(no_prefill.throttle_prefills)
        self.assertIsNone(no_prefill.token_budget)
        self.assertEqual(no_prefill.reason, "no_prefill")
        self.assertTrue(blocked_arrival.throttle_prefills)
        self.assertFalse(cleared_without_prefill.throttle_prefills)
        self.assertEqual(cleared_without_prefill.reason, "no_prefill")
        self.assertFalse(admitted_arrival.throttle_prefills)

    def test_elapsed_time_does_not_release_hold(self):
        now = [0.0]
        controller = self._make_controller(
            now, prefill_burst_cooldown_ms=1
        )
        pending = [_pending_prefill("p0")]
        self._decide(
            controller,
            [_decode_request(f"d{i}") for i in range(4)],
            pending,
        )

        now[0] = 1000.0
        decision = self._decide(
            controller,
            [_decode_request(f"d{i}") for i in range(3)],
            pending,
            scheduler_step=2,
        )

        self.assertTrue(decision.throttle_prefills)

    def test_pp_ineligible_decodes_do_not_fake_low_decode_release(self):
        controller = self._make_controller([0.0])
        pending = [_pending_prefill("p0")]
        self._decide(
            controller,
            [_decode_request(f"d{i}") for i in range(4)],
            pending,
        )

        decision = self._decide(
            controller,
            [
                _decode_request("d0", next_decode_eligible_step=5),
                _decode_request("d1", next_decode_eligible_step=5),
            ],
            pending,
            scheduler_step=3,
        )

        self.assertTrue(decision.throttle_prefills)
        self.assertEqual(decision.token_budget, 0)

    def test_released_burst_drains_frozen_cohort(self):
        controller = self._make_controller([0.0])
        initial = [_pending_prefill("p0")]
        self._decide(
            controller,
            [_decode_request(f"d{i}") for i in range(4)],
            initial,
        )

        cohort = [_pending_prefill("p1"), _pending_prefill("p2")]
        released = self._decide(
            controller,
            [_decode_request("d0")],
            cohort,
            scheduler_step=2,
        )
        remaining = [_pending_prefill("p2")]
        continued = self._decide(
            controller,
            [_decode_request("d0")],
            remaining,
            scheduler_step=3,
        )

        self.assertEqual(released.reason, "low_decode_release")
        self.assertEqual(
            released.pending_prefill_ids, frozenset(("p1", "p2"))
        )
        self.assertFalse(continued.throttle_prefills)
        self.assertEqual(continued.reason, "burst")
        self.assertEqual(continued.pending_prefill_ids, frozenset(("p2",)))

    def test_no_prefill_leaves_upstream_budget_unchanged(self):
        controller = self._make_controller([0.0])

        decision = self._decide(
            controller,
            [_decode_request("d0")],
            [],
        )

        self.assertFalse(decision.throttle_prefills)
        self.assertIsNone(decision.token_budget)
        self.assertEqual(decision.reason, "no_prefill")
