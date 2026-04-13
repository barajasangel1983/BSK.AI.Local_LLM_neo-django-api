import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from django.core.management.base import BaseCommand
from django.utils import timezone

UTC = timezone.UTC

from historian.models import ExtruderSample


class Command(BaseCommand):
    help = "Generate synthetic EXTR01 extruder data into the historian database."

    def add_arguments(self, parser):
        parser.add_argument(
            "--month",
            type=str,
            help=(
                "Target month in YYYY-MM format (e.g. 2026-03). "
                "If omitted, you must pass --start and --end."
            ),
        )
        parser.add_argument(
            "--start",
            type=str,
            help="Start timestamp (ISO 8601, e.g. 2026-03-01T00:00:00Z).",
        )
        parser.add_argument(
            "--end",
            type=str,
            help="End timestamp (ISO 8601, e.g. 2026-03-31T23:59:00Z).",
        )
        parser.add_argument(
            "--interval-seconds",
            type=int,
            default=60,
            help="Sampling interval in seconds (default: 60).",
        )
        parser.add_argument(
            "--extruder-id",
            type=str,
            default="EXTR01",
            help="Extruder ID to simulate (default: EXTR01).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Generate and print a few samples but do not write to DB.",
        )

    # ---- helpers ---------------------------------------------------------

    def _parse_time_range(
        self, month: str | None, start: str | None, end: str | None
    ) -> Tuple[datetime, datetime]:
        if month:
            year, mon = map(int, month.split("-"))
            start_dt = datetime(year, mon, 1, 0, 0, tzinfo=UTC)
            # naive end: next month minus 1 minute
            if mon == 12:
                end_dt = datetime(year + 1, 1, 1, 0, 0, tzinfo=UTC) - timedelta(
                    minutes=1
                )
            else:
                end_dt = datetime(year, mon + 1, 1, 0, 0, tzinfo=UTC) - timedelta(
                    minutes=1
                )
            return start_dt, end_dt

        if not start or not end:
            raise SystemExit("Either --month or both --start and --end must be provided.")

        def parse_ts(s: str) -> datetime:
            # accept 'Z' or offset; default to UTC if naive
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt

        return parse_ts(start), parse_ts(end)

    def _shift_code_for_ts(self, ts: datetime) -> str:
        hour = ts.hour
        # 7:00–15:00 A, 15:00–23:00 B, 23:00–7:00 C
        if 7 <= hour < 15:
            return "A"
        if 15 <= hour < 23:
            return "B"
        return "C"

    def _pick_recipe_for_batch(self) -> str:
        # RC-A 40%, RC-B 30%, RC-C..RC-J 30% spread
        r = random.random()
        if r < 0.40:
            return "RC-A"
        if r < 0.70:
            return "RC-B"
        others = [
            "RC-C",
            "RC-D",
            "RC-E",
            "RC-F",
            "RC-G",
            "RC-H",
            "RC-I",
            "RC-J",
        ]
        return random.choice(others)

    def _machine_state_blocks(self, total_minutes: int) -> List[str]:
        """Build machine_state list with approximate proportions.

        ~70% RUNNING, 15% STOPPED, remaining ~15% other states.
        Slow running is modeled later via performance, not a separate state.
        """

        states: List[str] = []
        remaining = total_minutes

        while remaining > 0:
            # block size 30–120 minutes
            block_size = min(remaining, random.randint(30, 120))
            r = random.random()
            if r < 0.70:
                label = "RUNNING"
            elif r < 0.85:
                label = "STOPPED"
            else:
                label = random.choice(
                    ["CHANGEOVER", "FLUSH", "WARM UP", "COOL DOWN", "ALARMED"]
                )
            states.extend([label] * block_size)
            remaining -= block_size

        return states[:total_minutes]

    def _alarm_catalog(self) -> List[str]:
        return [
            "HI_MOISTURE",
            "LO_MOISTURE",
            "HI_DIE_PRESSURE",
            "HI_TORQUE",
            "LO_TEMP_ZONE_1",
            "HI_TEMP_ZONE_3",
            "HEAD_TEMP_DRIFT",
            "FEEDER_STARVE",
            "FEEDER_OVERLOAD",
            "MOTOR_OVERLOAD",
            "DIE_PLUG",
            "SCREEN_BLOCK",
            "OUT_OF_SPEC_DIAMETER",
            "OUT_OF_SPEC_DENSITY",
            "LOW_OEE",
            "HIGH_SCRAP",
            "SPEED_UNDER_TARGET",
            "TEMP_SENSOR_FAIL",
            "MOISTURE_SENSOR_FAIL",
            "UNPLANNED_STOP",
        ]

    # ---- main handle -----------------------------------------------------

    def handle(self, *args, **options):
        month = options["month"]
        start_str = options["start"]
        end_str = options["end"]
        interval_sec: int = options["interval_seconds"]
        extruder_id: str = options["extruder_id"]
        dry_run: bool = options["dry_run"]

        start_dt, end_dt = self._parse_time_range(month, start_str, end_str)
        self.stdout.write(
            f"Simulating extruder data from {start_dt} to {end_dt} for {extruder_id}"
        )

        # build timeline
        timestamps: List[datetime] = []
        current = start_dt
        while current <= end_dt:
            timestamps.append(current)
            current += timedelta(seconds=interval_sec)

        total_minutes = len(timestamps)
        states = self._machine_state_blocks(total_minutes)
        alarms = self._alarm_catalog()

        # batch and recipe tracking
        batch_counter = 1
        current_batch_id = f"{extruder_id}-{start_dt.date():%Y%m%d}-{batch_counter:03d}"
        current_recipe = self._pick_recipe_for_batch()
        batch_accum_kg = 0.0
        batch_size_kg = 4000.0

        capacity_kg_hr = 11000.0

        samples: List[ExtruderSample] = []

        # Pre-build recipe targets
        recipe_target_map: Dict[str, float] = {
            "RC-A": 10000.0,
            "RC-B": 9000.0,
        }
        for rcp in ["RC-C", "RC-D", "RC-E", "RC-F", "RC-G", "RC-H", "RC-I", "RC-J"]:
            recipe_target_map[rcp] = random.uniform(8000.0, 10500.0)

        for idx, ts in enumerate(timestamps):
            state = states[idx]
            shift = self._shift_code_for_ts(ts)

            target = recipe_target_map.get(current_recipe, 9000.0)

            # running fraction
            if state == "RUNNING":
                running_frac = random.uniform(0.9, 1.05)  # normal
            elif state in ["CHANGEOVER", "FLUSH", "WARM UP", "COOL DOWN"]:
                running_frac = random.uniform(0.1, 0.5)
            else:  # STOPPED or ALARMED
                running_frac = random.uniform(0.0, 0.2)

            throughput_actual = max(0.0, min(capacity_kg_hr, target * running_frac))
            throughput_target = target

            # screw speed & feeder rate correlate with throughput
            screw_speed = 300.0 + (throughput_actual / capacity_kg_hr) * 150.0 + random.uniform(
                -10, 10
            )
            feeder_rate = throughput_actual + random.uniform(-200, 200)

            # pressures/temps (simple correlations)
            die_pressure = 50.0 + (throughput_actual / capacity_kg_hr) * 50.0 + random.uniform(
                -5, 5
            )
            screw_torque = 40.0 + (die_pressure / 100.0) * 60.0 + random.uniform(-5, 5)
            motor_power = 100.0 + (feeder_rate / capacity_kg_hr) * 300.0 + random.uniform(
                -10, 10
            )

            # barrel temps & head/product temps
            b1 = random.uniform(80, 100)
            b2 = b1 + random.uniform(5, 10)
            b3 = b2 + random.uniform(5, 10)
            head_temp = b3 + random.uniform(-5, 5)
            prod_temp = head_temp + random.uniform(-3, 3)

            steam_flow = (throughput_actual / 8000.0) * 100.0 + random.uniform(-10, 10)
            water_inj = (throughput_actual / 8000.0) * 80.0 + random.uniform(-10, 10)

            # product properties
            moisture = random.uniform(8.0, 11.0)  # %; spec say 9–10.5
            bulk_density = random.uniform(330, 380)  # g/L
            kibble_diam = random.uniform(7.0, 10.0)  # mm

            # downtime and OEE components
            if state in ["STOPPED", "ALARMED"]:
                downtime_min = float(interval_sec / 60.0)
                availability = random.uniform(50.0, 80.0)
            else:
                downtime_min = 0.0
                availability = random.uniform(90.0, 100.0)

            performance = (
                min(100.0, max(0.0, (throughput_actual / throughput_target) * 100.0))
                if throughput_target > 0
                else 0.0
            )
            quality = random.uniform(95.0, 100.0)

            # alarm & quality logic
            alarm_active = False
            alarm_code = ""

            # out-of-spec moisture
            if moisture < 9.0 or moisture > 10.5:
                quality -= random.uniform(5.0, 15.0)
                if random.random() < 0.3:
                    alarm_active = True
                    alarm_code = random.choice(["HI_MOISTURE", "LO_MOISTURE"])

            # diameter out-of-tolerance
            if kibble_diam < 7.5 or kibble_diam > 9.5:
                quality -= random.uniform(5.0, 10.0)
                if not alarm_active and random.random() < 0.3:
                    alarm_active = True
                    alarm_code = "OUT_OF_SPEC_DIAMETER"

            # density drift
            if bulk_density < 340 or bulk_density > 370:
                quality -= random.uniform(2.0, 8.0)
                if not alarm_active and random.random() < 0.3:
                    alarm_active = True
                    alarm_code = "OUT_OF_SPEC_DENSITY"

            # occasional other alarms
            if not alarm_active and random.random() < 0.01:
                alarm_active = True
                alarm_code = random.choice(alarms)

            # ensure quality in [0,100]
            quality = min(100.0, max(0.0, quality))

            oee = (availability * performance * quality) / 10000.0

            # outputs and scrap
            minutes = interval_sec / 60.0
            good_delta = (throughput_actual * minutes) * (
                0.95 if not alarm_active else 0.8
            )
            scrap_delta = (throughput_actual * minutes) - good_delta

            batch_accum_kg += (good_delta + scrap_delta)
            if batch_accum_kg >= batch_size_kg:
                batch_counter += 1
                batch_accum_kg -= batch_size_kg
                current_batch_id = f"{extruder_id}-{ts.date():%Y%m%d}-{batch_counter:03d}"
                current_recipe = self._pick_recipe_for_batch()

            # per-minute good/scrap (not cumulative to avoid overflow)
            total_good = good_delta
            total_scrap = scrap_delta

            sample = ExtruderSample(
                ts=ts,
                extruder_id=extruder_id,
                batch_id=current_batch_id,
                recipe_id=current_recipe,
                shift_code=shift,
                machine_state=state,
                screw_speed_rpm=screw_speed,
                screw_torque_pct=screw_torque,
                motor_power_kw=motor_power,
                feeder_rate_actual_kg_hr=feeder_rate,
                barrel_zone_1_temp_c=b1,
                barrel_zone_2_temp_c=b2,
                barrel_zone_3_temp_c=b3,
                head_temp_c=head_temp,
                steam_flow_kg_hr=steam_flow,
                water_injection_l_hr=water_inj,
                die_pressure_bar=die_pressure,
                product_temp_discharge_c=prod_temp,
                product_moisture_pct=moisture,
                bulk_density_g_l=bulk_density,
                kibble_diameter_mm=kibble_diam,
                throughput_actual_kg_hr=throughput_actual,
                throughput_target_kg_hr=throughput_target,
                good_output_kg=total_good,
                scrap_output_kg=total_scrap,
                downtime_min=downtime_min,
                availability_pct=availability,
                performance_pct=performance,
                quality_pct=quality,
                oee_pct=oee,
                alarm_active=alarm_active,
                alarm_code=alarm_code,
            )
            samples.append(sample)

        if dry_run:
            self.stdout.write(f"Generated {len(samples)} samples (dry run). Showing first 3:")
            for s in samples[:3]:
                self.stdout.write(str(s))
            return

        self.stdout.write(
            f"Writing {len(samples)} samples to extruder_samples in historian DB..."
        )
        ExtruderSample.objects.bulk_create(samples, batch_size=1000)
        self.stdout.write(self.style.SUCCESS("Done."))
