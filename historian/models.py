from django.db import models


class ExtruderSample(models.Model):
    """Snapshot of EXTR01 extruder process conditions at a given timestamp.

    Lives in the Postgres historian DB (via HistorianRouter).
    """

    ts = models.DateTimeField()
    extruder_id = models.CharField(max_length=30)

    batch_id = models.CharField(max_length=50)
    recipe_id = models.CharField(max_length=50)
    shift_code = models.CharField(max_length=10)

    machine_state = models.CharField(max_length=20)  # RUNNING, STOPPED, ...

    screw_speed_rpm = models.DecimalField(max_digits=10, decimal_places=2)
    screw_torque_pct = models.DecimalField(max_digits=10, decimal_places=2)
    motor_power_kw = models.DecimalField(max_digits=10, decimal_places=2)

    feeder_rate_actual_kg_hr = models.DecimalField(max_digits=10, decimal_places=2)

    barrel_zone_1_temp_c = models.DecimalField(max_digits=10, decimal_places=2)
    barrel_zone_2_temp_c = models.DecimalField(max_digits=10, decimal_places=2)
    barrel_zone_3_temp_c = models.DecimalField(max_digits=10, decimal_places=2)
    head_temp_c = models.DecimalField(max_digits=10, decimal_places=2)

    steam_flow_kg_hr = models.DecimalField(max_digits=10, decimal_places=2)
    water_injection_l_hr = models.DecimalField(max_digits=10, decimal_places=2)
    die_pressure_bar = models.DecimalField(max_digits=10, decimal_places=2)
    product_temp_discharge_c = models.DecimalField(max_digits=10, decimal_places=2)
    product_moisture_pct = models.DecimalField(max_digits=10, decimal_places=2)
    bulk_density_g_l = models.DecimalField(max_digits=10, decimal_places=2)
    kibble_diameter_mm = models.DecimalField(max_digits=10, decimal_places=2)

    throughput_actual_kg_hr = models.DecimalField(max_digits=10, decimal_places=2)
    throughput_target_kg_hr = models.DecimalField(max_digits=10, decimal_places=2)
    good_output_kg = models.DecimalField(max_digits=10, decimal_places=2)
    scrap_output_kg = models.DecimalField(max_digits=10, decimal_places=2)
    downtime_min = models.DecimalField(max_digits=10, decimal_places=2)

    availability_pct = models.DecimalField(max_digits=6, decimal_places=2)
    performance_pct = models.DecimalField(max_digits=6, decimal_places=2)
    quality_pct = models.DecimalField(max_digits=6, decimal_places=2)
    oee_pct = models.DecimalField(max_digits=6, decimal_places=2)

    alarm_active = models.BooleanField(default=False)
    alarm_code = models.CharField(max_length=50, blank=True)

    class Meta:
        db_table = "extruder_samples"
        ordering = ["-ts"]

    def __str__(self) -> str:
        return f"{self.ts.isoformat()} {self.extruder_id} {self.recipe_id} {self.machine_state}"


# --- The live historian: the PLC's rows, one every 20 s (TimescaleDB, read-only) -----------------
# Tables of the repository BSKLAB_OPC_collector_node (historian/db/migrations). Not managed here:
# a schema change is a new numbered file in that repository.

class LiveRow(models.Model):
    """One 20 s period of an extruder: the same 32 columns as ExtruderSample; `ts` is the period's start (UTC)."""

    live = True

    pk = models.CompositePrimaryKey("extruder_id", "ts")
    ts = models.DateTimeField()
    extruder_id = models.TextField()

    batch_id = models.TextField()
    recipe_id = models.TextField()
    shift_code = models.TextField()

    machine_state = models.TextField()

    screw_speed_rpm = models.FloatField()
    screw_torque_pct = models.FloatField()
    motor_power_kw = models.FloatField()

    feeder_rate_actual_kg_hr = models.FloatField()

    barrel_zone_1_temp_c = models.FloatField()
    barrel_zone_2_temp_c = models.FloatField()
    barrel_zone_3_temp_c = models.FloatField()
    head_temp_c = models.FloatField()

    steam_flow_kg_hr = models.FloatField()
    water_injection_l_hr = models.FloatField()
    die_pressure_bar = models.FloatField()
    product_temp_discharge_c = models.FloatField()
    product_moisture_pct = models.FloatField()
    bulk_density_g_l = models.FloatField()
    kibble_diameter_mm = models.FloatField()

    throughput_actual_kg_hr = models.FloatField()
    throughput_target_kg_hr = models.FloatField()
    good_output_kg = models.FloatField()
    scrap_output_kg = models.FloatField()
    downtime_min = models.FloatField()

    availability_pct = models.FloatField()
    performance_pct = models.FloatField()
    quality_pct = models.FloatField()
    oee_pct = models.FloatField()

    alarm_active = models.BooleanField()
    alarm_code = models.TextField(blank=True)

    class Meta:
        managed = False
        db_table = "historian_rows"
        ordering = ["-ts"]


class LiveRowMeta(models.Model):
    """How a row was collected: the PLC's counter, the PLC run, the collector's verdict."""

    live = True

    pk = models.CompositePrimaryKey("extruder_id", "ts")
    extruder_id = models.TextField()
    ts = models.DateTimeField()
    period_counter = models.BigIntegerField()        # restarts at 1 when the PLC restarts
    plc_boot_utc = models.DateTimeField(null=True)   # identifies the PLC run
    quality = models.TextField()                     # 'good' or 'suspect'
    quality_notes = models.TextField(blank=True)     # comma-separated findings, may be empty
    collected_at = models.DateTimeField()
    received_at = models.DateTimeField()
    collector_id = models.TextField()
    sim_seed = models.BigIntegerField(null=True)
    sim_version = models.TextField(null=True)

    class Meta:
        managed = False
        db_table = "row_meta"


class LiveSimEvent(models.Model):
    """The injected-anomaly record: one line per row that had an anomaly active."""

    live = True

    pk = models.CompositePrimaryKey("extruder_id", "ts")
    extruder_id = models.TextField()
    ts = models.DateTimeField()
    event_id = models.BigIntegerField()
    event_type = models.IntegerField()   # 1 Step, 2 Drift, 3 Spike, 4 Stuck value, 5 More noise, 6 Broken relation
    event_signal = models.TextField()
    event_size = models.FloatField()

    class Meta:
        managed = False
        db_table = "sim_events"


class LiveCollector(models.Model):
    """Last report of a collector."""

    live = True

    collector_id = models.TextField(primary_key=True)
    version = models.TextField()
    last_seen = models.DateTimeField()
    buffered = models.IntegerField()     # rows still waiting on the collector
    last_row_ts = models.DateTimeField(null=True)

    class Meta:
        managed = False
        db_table = "collectors"
