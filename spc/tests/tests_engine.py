"""The SPC engine (specification section 5): rows built here, no historian, graph or network."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
from django.test import SimpleTestCase

from spc import engine

T0 = datetime(2026, 10, 12, 8, 0, tzinfo=timezone.utc)
IMR = dict(column="x", states=["RUNNING"], ignore_seconds=0, method="imr", subgroup_size=3, group_by="", period=20)


def rows(values, state="RUNNING", start=T0, period=20, **extra):
    return [{"ts": start + timedelta(seconds=period * i), "machine_state": state, "x": v, **extra} for i, v in enumerate(values)]


def lim(cl=10.0, sigma=1.0, k=3.0, r_ucl=None, r_lcl=0.0):
    """Limits for I-MR with a given sigma."""
    return SimpleNamespace(cl=cl, lcl=cl - k * sigma, ucl=cl + k * sigma, sigma_within=sigma, mean=cl,
                           r_lcl=r_lcl, r_ucl=3.687 * sigma if r_ucl is None else r_ucl)


def judged(values, limits=None, rules=None, **more):
    found = engine.samples(rows(values), **IMR)
    return engine.judge(found, {"": limits or lim()}, rules=rules or list(engine.RULES), method="imr", subgroup_size=3, **more)


class SampleTests(SimpleTestCase):
    def test_only_rows_in_the_tools_states_with_a_number_are_kept(self):
        data = rows([1, 2, 3]) + rows([9], "STOPPED", T0 + timedelta(seconds=60)) + rows([None, 5, True, 6], start=T0 + timedelta(seconds=80))
        found = engine.samples(data, **IMR)
        self.assertEqual([s.value for s in found], [1, 2, 3, 5, 6])
        self.assertEqual([s.segment for s in found], [0, 0, 0, 1, 1])              # the stop ends the segment
        self.assertEqual([s.range for s in found], [None, 1, 1, None, 1])           # no moving range across segments

    def test_a_segment_ends_at_a_gap_at_a_state_change_and_at_a_group_change(self):
        gap = rows([1, 2]) + rows([3, 4], start=T0 + timedelta(seconds=100))        # 80 s without a row: more than two periods
        self.assertEqual([s.segment for s in engine.samples(gap, **IMR)], [0, 0, 1, 1])
        one_missing = rows([1, 2]) + rows([3], start=T0 + timedelta(seconds=60))     # one row missing: the same segment
        self.assertEqual([s.segment for s in engine.samples(one_missing, **IMR)], [0, 0, 0])

        two_states = rows([1, 2]) + rows([3, 4], "COOL DOWN", T0 + timedelta(seconds=40))
        found = engine.samples(two_states, **{**IMR, "states": ["RUNNING", "COOL DOWN"]})
        self.assertEqual([(s.segment, s.state) for s in found], [(0, "RUNNING"), (0, "RUNNING"), (1, "COOL DOWN"), (1, "COOL DOWN")])

        recipes = rows([1, 2], recipe_id="RC-A") + rows([3, 4], start=T0 + timedelta(seconds=40), recipe_id="RC-B")
        found = engine.samples(recipes, **{**IMR, "group_by": "recipe_id"})
        self.assertEqual([(s.segment, s.group) for s in found], [(0, "RC-A"), (0, "RC-A"), (1, "RC-B"), (1, "RC-B")])
        self.assertIsNone(found[2].range)

    def test_the_ignore_period_is_dropped_at_the_start_of_every_segment(self):
        data = rows(range(10)) + rows(range(10, 20), start=T0 + timedelta(seconds=600))
        found = engine.samples(data, **{**IMR, "ignore_seconds": 60})
        self.assertEqual([s.value for s in found], [3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19])
        self.assertIsNone(found[0].range)                                           # the dropped rows give no range either

    def test_subgroups_stay_inside_a_segment_and_leftovers_are_dropped(self):
        data = rows([1, 2, 3, 4, 5, 6, 7]) + rows([10, 20, 30, 40], start=T0 + timedelta(seconds=600))
        found = engine.samples(data, **{**IMR, "method": "xbar_r"})
        self.assertEqual([(s.value, s.range, s.segment) for s in found], [(2, 2, 0), (5, 2, 0), (20, 20, 1)])
        self.assertEqual(found[0].rows, [1, 2, 3])
        self.assertEqual(found[0].ts, T0 + timedelta(seconds=40))                    # the last row of the subgroup


class LimitTests(SimpleTestCase):
    def test_limits_of_an_individuals_chart(self):
        found = engine.samples(rows([10, 12, 10, 12] * 20), **IMR)                   # every moving range is 2
        limits, skipped = engine.baseline_limits(found, method="imr", subgroup_size=3, k=3, min_samples=50)
        only = limits[""]
        self.assertEqual((skipped, only.n_samples, only.mean, only.range_bar), ([], 80, 11.0, 2.0))
        self.assertAlmostEqual(only.sigma_within, 2 / 1.128, places=6)
        self.assertAlmostEqual((only.ucl - only.cl) / only.range_bar, 2.660, places=3)       # the textbook constants
        self.assertAlmostEqual(only.r_ucl / only.range_bar, 3.267, places=2)
        self.assertEqual(only.r_lcl, 0.0)
        self.assertAlmostEqual(only.sigma_overall, float(np.std([10, 12] * 40, ddof=1)), places=6)

    def test_limits_of_subgroups_of_three(self):
        found = engine.samples(rows([10, 11, 12] * 60), **{**IMR, "method": "xbar_r"})   # every subgroup: mean 11, range 2
        limits, _ = engine.baseline_limits(found, method="xbar_r", subgroup_size=3, k=3, min_samples=50)
        only = limits[""]
        self.assertEqual((only.n_samples, only.mean, only.range_bar), (60, 11.0, 2.0))
        self.assertAlmostEqual((only.ucl - only.cl) / only.range_bar, 1.023, places=3)
        self.assertAlmostEqual(only.r_ucl / only.range_bar, 2.574, places=2)

    def test_one_set_of_limits_per_group_and_small_or_flat_groups_get_none(self):
        data = (rows([10, 12] * 30, recipe_id="RC-A") + rows([50, 54] * 30, start=T0 + timedelta(hours=1), recipe_id="RC-B")
                + rows([7, 8] * 10, start=T0 + timedelta(hours=2), recipe_id="RC-C")
                + rows([5] * 60, start=T0 + timedelta(hours=3), recipe_id="RC-D"))
        found = engine.samples(data, **{**IMR, "group_by": "recipe_id"})
        limits, skipped = engine.baseline_limits(found, method="imr", subgroup_size=3, k=3, min_samples=50)
        self.assertEqual((limits["RC-A"].mean, limits["RC-B"].mean, limits["RC-B"].range_bar), (11.0, 52.0, 4.0))
        self.assertEqual(skipped, [{"group": "RC-C", "n_samples": 20, "reason": "fewer than 50 samples"},
                                   {"group": "RC-D", "n_samples": 60, "reason": "no variation in the baseline"}])

    def test_fixed_limits_give_the_sigma_they_imply(self):
        only = engine.fixed_limits("", 100, 94, 106, None, method="imr", subgroup_size=3, k=3)
        self.assertEqual((only.cl, only.lcl, only.ucl, only.sigma_within, only.sigma_overall), (100, 94, 106, 2.0, None))
        self.assertAlmostEqual(only.r_ucl, 3.267 * 1.128 * 2.0, places=1)
        sub = engine.fixed_limits("", 100, 94, 106, 9.0, method="xbar_r", subgroup_size=4, k=3)
        self.assertEqual((sub.sigma_within, sub.r_ucl), (4.0, 9.0))                  # sigma of a mean of four is half
        with self.assertRaisesMessage(ValueError, "LCL < CL < UCL"):
            engine.fixed_limits("", 100, 106, 94, None, method="imr", subgroup_size=3, k=3)


class RuleTests(SimpleTestCase):
    """Limits: centre 10, sigma 1, so 3 sigma is 7 to 13."""

    def only(self, values, rule, **more):
        return [rule in fired for fired in judged(values, rules=[rule], **more)]

    def test_a_point_beyond_the_control_limits(self):
        self.assertEqual(self.only([10, 13, 13.1, 6.9, 7], "beyond_limits"), [False, False, True, True, False])

    def test_the_range_chart(self):
        self.assertEqual(self.only([10, 10.5, 14.5, 14.6], "range_beyond_limits"), [False, False, True, False])     # a jump of 4
        tight = judged([10, 10.5, 10.6], lim(r_lcl=0.2), rules=["range_beyond_limits"])
        self.assertEqual(tight, [[], [], ["range_beyond_limits"]])                   # below a lower range limit above zero

    def test_a_value_beyond_the_specification(self):
        self.assertEqual(self.only([10, 12.1, 7.9, 12], "spec", lsl=8, usl=12), [False, True, True, False])
        self.assertEqual(self.only([10, 12.1], "spec", usl=12), [False, True])       # one limit is enough
        self.assertEqual(self.only([10, 99], "spec"), [False, False])                # none: nothing to violate

    def test_two_of_three_beyond_two_sigma_on_one_side(self):
        self.assertEqual(self.only([10, 12.5, 10, 12.5], "two_of_three"), [False, False, False, True])
        self.assertEqual(self.only([12.5, 7.5, 12.5], "two_of_three"), [False, False, True])
        self.assertEqual(self.only([12.5, 7.5, 10], "two_of_three"), [False, False, False])          # opposite sides do not add up

    def test_four_of_five_beyond_one_sigma_on_one_side(self):
        self.assertEqual(self.only([11.5, 11.5, 10, 11.5, 11.5], "four_of_five"), [False, False, False, False, True])
        self.assertEqual(self.only([11.5, 11.5, 11.5, 10, 10], "four_of_five"), [False] * 5)

    def test_eight_in_a_row_on_one_side(self):
        self.assertEqual(self.only([10.1] * 8 + [9.9], "eight_one_side"), [False] * 7 + [True, False])
        self.assertEqual(self.only([10.1] * 7 + [10.0], "eight_one_side"), [False] * 8)              # on the centre is no side

    def test_six_in_a_row_rising_or_falling(self):
        self.assertEqual(self.only([10, 10.1, 10.2, 10.3, 10.4, 10.5, 10.4], "six_trending"), [False] * 5 + [True, False])
        self.assertEqual(self.only([10.5, 10.4, 10.3, 10.2, 10.1, 10.0], "six_trending")[-1], True)
        self.assertEqual(self.only([10, 10.1, 10.2, 10.2, 10.4, 10.5], "six_trending")[-1], False)   # an equal step breaks it

    def test_no_variation(self):
        self.assertEqual(self.only([10, 11, 11, 11], "no_variation", no_variation_count=3), [False, False, False, True])

    def test_a_run_does_not_continue_across_a_segment(self):
        data = rows([10.1] * 5) + rows([9], "STOPPED", T0 + timedelta(seconds=100)) + rows([10.1] * 5, start=T0 + timedelta(seconds=120))
        found = engine.samples(data, **IMR)
        fired = engine.judge(found, {"": lim()}, rules=["eight_one_side"], method="imr", subgroup_size=3)
        self.assertEqual(len(found), 10)
        self.assertEqual([f for f in fired if f], [])                                # ten on one side, but five and five

    def test_a_group_without_limits_is_not_judged_and_switched_off_rules_are_silent(self):
        data = rows([99, 99], recipe_id="RC-A") + rows([99, 99], start=T0 + timedelta(seconds=40), recipe_id="RC-Z")
        found = engine.samples(data, **{**IMR, "group_by": "recipe_id"})
        fired = engine.judge(found, {"RC-A": lim()}, rules=list(engine.RULES), method="imr", subgroup_size=3, usl=12)
        self.assertEqual(fired, [["beyond_limits", "spec"], ["beyond_limits", "spec", "two_of_three"], [], []])
        self.assertEqual(judged([99], rules=["six_trending"]), [[]])

    def test_subgroup_means_are_judged_with_the_sigma_of_a_mean(self):
        found = engine.samples(rows([11, 11, 11, 11.9, 11.9, 11.9]), **{**IMR, "method": "xbar_r"})
        limits = engine.fixed_limits("", 10, 10 - 3 / 3 ** 0.5, 10 + 3 / 3 ** 0.5, None, method="xbar_r", subgroup_size=3, k=3)
        self.assertAlmostEqual(limits.sigma_within, 1.0, places=9)                   # of a row; a mean of three: 0.577
        fired = engine.judge(found, {"": limits}, rules=["beyond_limits", "four_of_five"], method="xbar_r", subgroup_size=3)
        self.assertEqual(fired, [[], ["beyond_limits"]])                             # 11.9 is over 10 + 3 × 0.577


class CapabilityTests(SimpleTestCase):
    def cap(self, values, **spec):
        found = engine.samples(rows(values), **IMR)
        return engine.capability(found, method="imr", subgroup_size=3, min_samples=100, **{"lsl": None, "usl": None, **spec})

    def test_both_limits(self):
        c = self.cap([9, 11] * 60, lsl=4, usl=16)                                    # mean 10, every moving range 2
        sigma = 2 / 1.128
        self.assertEqual((c["sample_count"], c["mean"], c["reason"]), (120, 10.0, ""))
        self.assertAlmostEqual(c["cp"], 12 / (6 * sigma), places=6)
        self.assertAlmostEqual(c["cpk"], 6 / (3 * sigma), places=6)
        off_centre = self.cap([9, 11] * 60, lsl=7, usl=16)
        self.assertAlmostEqual(off_centre["cpk"], 3 / (3 * sigma), places=6)         # the nearer limit decides
        self.assertGreater(off_centre["cp"], off_centre["cpk"])

    def test_one_limit_gives_the_one_sided_figures_only(self):
        upper = self.cap([9, 11] * 60, usl=16)
        self.assertEqual((upper["cp"], upper["pp"]), (None, None))
        self.assertAlmostEqual(upper["cpk"], 6 / (3 * 2 / 1.128), places=6)
        lower = self.cap([9, 11] * 60, lsl=7)
        self.assertAlmostEqual(lower["cpk"], 3 / (3 * 2 / 1.128), places=6)
        self.assertIsNotNone(lower["ppk"])

    def test_not_available_with_the_reason(self):
        self.assertEqual(self.cap([9, 11] * 60)["reason"], "no specification limit")
        few = self.cap([9, 11] * 10, lsl=4, usl=16)
        self.assertEqual((few["reason"], few["cpk"], few["ppk"]), ("fewer than 100 values in the window", None, None))
        self.assertEqual(self.cap([10] * 120, lsl=4, usl=16)["reason"], "no variation in the window")
        self.assertEqual(self.cap([], lsl=4, usl=16)["cpk"], None)

    def test_cpk_uses_the_variation_within_and_ppk_the_overall(self):
        shifted = self.cap([9, 11] * 30 + [19, 21] * 30, lsl=0, usl=30)             # a level shift in the middle
        self.assertGreater(shifted["cpk"], 2 * shifted["ppk"])                       # within stays small, overall grows
        self.assertGreater(shifted["sigma_overall"], 2 * shifted["sigma_within"])


class ShiftStatusAndChartTests(SimpleTestCase):
    def test_sigma_shift_is_the_recent_mean_against_the_baseline(self):
        found = engine.samples(rows([10] * 200 + [12] * 30), **IMR)
        self.assertAlmostEqual(engine.sigma_shift(found, {"": lim()}, recent_seconds=580), 2.0, places=6)    # the last 30 rows
        self.assertIsNone(engine.sigma_shift(found, {}, recent_seconds=580))                                  # no baseline
        self.assertIsNone(engine.sigma_shift(found[:5], {"": lim()}, recent_seconds=580))                     # too few
        self.assertIsNone(engine.sigma_shift([], {"": lim()}, recent_seconds=580))

    def test_status_is_the_first_line_that_applies(self):
        ok = dict(enabled=True, has_column=True, has_limits=True, group_has_limits=True)
        event = lambda kind, rule: {"type": kind, "rule": rule}      # noqa: E731
        everything = [event(engine.WARNING, "two_of_three"), event(engine.WARNING, "six_trending"),
                      event(engine.CONTROL, "beyond_limits"), event(engine.SPEC, "spec")]
        self.assertEqual(engine.status(**{**ok, "enabled": False}, active_events=everything), ("paused", ""))
        self.assertEqual(engine.status(**{**ok, "has_column": False})[0], "configuration_required")
        self.assertEqual(engine.status(**{**ok, "has_limits": False})[0], "configuration_required")
        self.assertEqual(engine.status(**{**ok, "group_has_limits": False}, group="RC-J"), ("configuration_required", "no control limits for RC-J"))
        self.assertEqual(engine.status(**ok, active_events=everything), ("spec_violation", "spec"))
        self.assertEqual(engine.status(**ok, active_events=everything[:3]), ("out_of_control", "beyond_limits"))
        self.assertEqual(engine.status(**ok, active_events=everything[:2]), ("trending", "six_trending"))
        self.assertEqual(engine.status(**ok, active_events=everything[:1]), ("warning", "two_of_three"))
        self.assertEqual(engine.status(**ok), ("stable", ""))

    def test_box_plot_and_quantiles_of_the_values(self):
        box = engine.box_plot([1, 2, 3, 4, 5, 6, 7, 8, 9, 30, -20])                   # 30 and -20 are far from the box
        self.assertEqual((box["q1"], box["median"], box["q3"]), (2.5, 5.0, 7.5))
        self.assertEqual((box["whisker_low"], box["whisker_high"]), (1.0, 9.0))      # the farthest values within 1.5 × IQR
        self.assertEqual((box["outlier_count"], box["outliers"]), (2, [-20.0, 30.0]))
        self.assertEqual([q["percent"] for q in box["quantiles"]], [100, 99.5, 97.5, 90, 75, 50, 25, 10, 2.5, 0.5, 0])
        self.assertEqual((box["quantiles"][0]["value"], box["quantiles"][5]["value"], box["quantiles"][-1]["value"]), (30.0, 5.0, -20.0))
        self.assertIsNone(engine.box_plot([]))
        flat = engine.box_plot([5, 5, 5])                                             # no spread: a box of no height, no outlier
        self.assertEqual((flat["q1"], flat["q3"], flat["whisker_low"], flat["whisker_high"], flat["outlier_count"]), (5.0, 5.0, 5.0, 5.0, 0))
        many = engine.box_plot([0.0] * 1000 + list(range(100, 400)))                  # more outliers than are listed
        self.assertEqual((many["outlier_count"], len(many["outliers"]), many["outliers"][-1]), (300, 200, 399.0))

    def test_trend_and_distribution_for_a_card(self):
        self.assertEqual(engine.downsample([1, 2, 3]), [1, 2, 3])
        self.assertEqual(engine.downsample(list(range(120)), 60)[:2], [0.5, 2.5])
        self.assertLessEqual(len(engine.downsample(list(range(1000)), 60)), 60)
        h = engine.histogram([10, 10.5, 11, 12], bins=4)                             # the bins cover the values, whatever the specification
        self.assertEqual((h["min"], h["max"], h["counts"]), (10, 12, [1, 1, 1, 1]))
        self.assertIsNone(engine.histogram([]))
        self.assertEqual(engine.histogram([5, 5, 5], bins=3)["counts"], [3, 0, 0])
