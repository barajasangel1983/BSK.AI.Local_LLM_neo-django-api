"""hour_stats (P18b) against rows counted by hand, so the chat's numbers and chat_eval's expected numbers are not
only checked against each other."""

from django.test import SimpleTestCase

from monitor import agent

#        state        oee   downtime alarm  die pressure (normal 0–120)   temp (at least 50)
ROWS = [("RUNNING",    80,  0,      "",    100, 60),
        ("RUNNING",    90,  0,      "",    130, 60),
        ("RUNNING",    85,  0,      "A1",  131, 60),
        ("RUNNING",    95,  0,      "A1",  132, 40),      # three running rows above 120 in a row: an alert
        ("STOPPED",    None, 1,     "A1",  140, 60),      # not running: not in the range check
        ("STOPPED",    None, 1,     "",    100, 60),
        ("CHANGEOVER", None, 0,     "",    100, 60),      # not running, not downtime
        ("CHANGEOVER", None, 0,     "A2",  100, 60),
        ("RUNNING",    70,  0,      "A2",  125, 60),      # out, but a stretch of one
        ("RUNNING",    80,  0,      "",    100, 60),
        ("RUNNING",    None, 0,     "A1",  100, 60),      # no OEE in this row
        ("RUNNING",    100, 0,      "",    100, 60)]
SIGNALS = [{"key": "die_pressure_bar", "name": "Die pressure", "unit": "bar", "low": 0, "high": 120},
           {"key": "temp_c", "name": "Zone temp", "unit": "°C", "low": 50, "high": None}]


def rows():
    return [{"machine_state": s, "oee_pct": oee, "availability_pct": 95 if oee else None, "downtime_min": d,
             "alarm_code": a, "die_pressure_bar": p, "temp_c": t} for s, oee, d, a, p, t in ROWS]


class HourStatsTests(SimpleTestCase):
    def test_minute_rows(self):
        s = agent.hour_stats(rows(), SIGNALS, 60, 15)
        self.assertEqual((s["minutes"], s["running_min"], s["not_running_min"], s["downtime_min"]), (12, 8, 4, 2))
        self.assertEqual(s["states"], {"RUNNING": 8, "STOPPED": 2, "CHANGEOVER": 2})
        self.assertEqual(s["oee"], 85.7)                    # (80+90+85+95+70+80+100) / 7 running rows with a value
        self.assertEqual(s["availability"], 95.0)
        self.assertEqual(s["alarms"], {"A1": 2, "A2": 1})   # onsets: A1 at the 3rd and 11th rows, A2 at the 8th
        self.assertEqual(s["no_data_min"], 3)               # 15 min asked, 12 rows
        self.assertEqual([o["name"] for o in s["out_of_range"]], ["Die pressure"])   # temp: a stretch of one
        die = s["out_of_range"][0]
        self.assertEqual((die["minutes"], die["min"], die["max"]), (4, 100, 132))    # running values only

    def test_short_gaps_and_20_second_rows(self):
        self.assertEqual(agent.hour_stats(rows(), SIGNALS, 60, 13)["no_data_min"], 0)   # 1 min short: not reported
        s = agent.hour_stats(rows(), SIGNALS, 20)
        self.assertEqual((s["running_min"], s["downtime_min"], s["states"]["STOPPED"]), (2.7, 0.7, 0.7))
        self.assertEqual(s["out_of_range"], [])             # 3 rows of 20 s is 1 min, under the 3-minute rule

    def test_without_a_downtime_column_every_stop_is_downtime(self):
        plain = [{k: v for k, v in r.items() if k != "downtime_min"} for r in rows()]
        self.assertEqual(agent.hour_stats(plain, SIGNALS, 60)["downtime_min"], 4)
