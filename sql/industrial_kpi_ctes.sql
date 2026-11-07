-- =============================================================================
-- SQL Joins & CTEs for Industrial Machine Telemetry & KPI Aggregation
-- =============================================================================

WITH variant_baselines AS (
    SELECT
        variant_type,
        COUNT(*) AS variant_units,
        AVG(torque_nm) AS avg_variant_torque,
        AVG(rotational_speed_rpm) AS avg_variant_rpm,
        AVG(tool_wear_min) AS avg_variant_wear
    FROM raw_machine_telemetry
    GROUP BY variant_type
),
physics_features AS (
    SELECT
        m.udi,
        m.product_id,
        m.variant_type,
        m.air_temp_k,
        m.process_temp_k,
        ROUND(m.process_temp_k - m.air_temp_k, 4) AS temp_diff_k,
        m.rotational_speed_rpm,
        m.torque_nm,
        m.tool_wear_min,
        ROUND(m.torque_nm * m.rotational_speed_rpm * 2.0 * 3.141592653589793 / 60.0, 2) AS mechanical_power_w,
        ROUND(m.tool_wear_min * m.torque_nm, 2) AS overstrain_nm_min,
        ROUND(m.torque_nm - v.avg_variant_torque, 4) AS torque_dev_from_variant,
        m.machine_failure,
        m.twf,
        m.hdf,
        m.pwf,
        m.osf,
        m.rnf
    FROM raw_machine_telemetry m
    INNER JOIN variant_baselines v
        ON m.variant_type = v.variant_type
)
SELECT
    variant_type,
    COUNT(*) AS total_units,
    SUM(machine_failure) AS total_failures,
    ROUND(100.0 * SUM(machine_failure) / COUNT(*), 2) AS failure_rate_pct,
    ROUND(AVG(mechanical_power_w), 2) AS avg_power_w,
    ROUND(AVG(overstrain_nm_min), 2) AS avg_overstrain_index
FROM physics_features
GROUP BY variant_type
ORDER BY variant_type;
