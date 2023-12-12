WITH base_data AS (
    SELECT
        t.*,
        COALESCE(l.unique_referral_index, 30) AS unique_referral_index,
        COALESCE(l.merged_referral_index, 5) AS merged_referral_index,
        c1.core10_bracket AS first_core10_bracket,
        c2.core10_bracket AS last_max_core10_bracket,
        p1.pgsi_bracket AS first_pgsi_bracket,
        p2.pgsi_bracket AS last_max_pgsi_bracket,
        EXTRACT(EPOCH FROM (t.first_assessment_date_offered - t.opened_date)) / 86400 AS wait_time_days,
        EXTRACT(EPOCH FROM (t.discharge_date - t.first_assessment_date_offered)) / 86400 AS treatment_time_days
    FROM
        modality_df t
    LEFT JOIN
        referral_source_lookup l ON t.referral_source_general = l.referral_name
    LEFT JOIN
        core10_lookup c1 ON t.first_core10 = c1.core10_score
    LEFT JOIN
        core10_lookup c2 ON t.last_max_core10 = c2.core10_score
    LEFT JOIN
        pgsi_lookup p1 ON t.first_pgsi = p1.pgsi_score
    LEFT JOIN
        pgsi_lookup p2 ON t.last_max_pgsi = p2.pgsi_score
    WHERE
        EXTRACT(YEAR FROM t.opened_date) >= 2013 AND
        EXTRACT(YEAR FROM t.first_assessment_date_offered) >= 2013 AND
        t.first_pgsi BETWEEN 0 AND 27 AND
        t.last_max_pgsi BETWEEN 0 AND 27 AND
        t.first_core10 BETWEEN 0 AND 40 AND
        t.last_max_core10 BETWEEN 0 AND 40
),
stats AS (
    SELECT
        AVG(wait_time_days) AS avg_wait_time,
        STDDEV(wait_time_days) AS stddev_wait_time,
        AVG(treatment_time_days) AS avg_treatment_time,
        STDDEV(treatment_time_days) AS stddev_treatment_time
    FROM base_data
),
filtered_data AS (
    SELECT
        *,
        (wait_time_days - avg_wait_time) / stddev_wait_time AS wait_time_zscore,
        (treatment_time_days - avg_treatment_time) / stddev_treatment_time AS treatment_time_zscore
    FROM
        base_data, stats
    WHERE
        wait_time_days >= 0 AND
        treatment_time_days >= 0 AND
        ABS((wait_time_days - avg_wait_time) / stddev_wait_time) <= 2.5 AND
        ABS((treatment_time_days - avg_treatment_time) / stddev_treatment_time) <= 2.5
)
SELECT
    client_key,
    unique_referral_index,
    merged_referral_index,
    first_core10_bracket,
    last_max_core10_bracket,
    first_pgsi_bracket,
    last_max_pgsi_bracket,
    first_pgsi,
    last_max_pgsi,
    first_core10,
    last_max_core10,
    modality_type
FROM
    filtered_data
ORDER BY
    client_key ASC;
--Should probably use a VIEW in future