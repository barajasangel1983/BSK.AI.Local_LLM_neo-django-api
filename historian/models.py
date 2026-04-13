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