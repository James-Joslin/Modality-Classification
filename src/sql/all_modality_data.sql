SELECT
    t.*,
    COALESCE(l.unique_referral_index, 30) AS unique_referral_index,
    COALESCE(l.merged_referral_index, 5) AS merged_referral_index,
    c.core10_bracket AS last_max_core10_bracket,
    p.pgsi_bracket AS last_max_pgsi_bracket
FROM
    modality_df t
LEFT JOIN
    referral_source_lookup l ON t.referral_source_general = l.referral_name
LEFT JOIN
    core10_lookup c ON t.last_max_core10 = c.core10_score
LEFT JOIN
    pgsi_lookup p ON t.last_max_pgsi = p.pgsi_score
WHERE
    EXTRACT(YEAR FROM t.opened_date) >= 2013 AND
    EXTRACT(YEAR FROM t.first_assessment_date_offered) >= 2013 AND
    t.first_pgsi BETWEEN 0 AND 27 AND
    t.last_max_pgsi BETWEEN 0 AND 27 AND
    t.first_core10 BETWEEN 0 AND 40 AND
    t.last_max_core10 BETWEEN 0 AND 40
    
--Data prior to 2013 is laregly incomplete and there are some erronous cells with one example being prior to 1955
--Making sure we're training the model on data that is within the correct ranges of the pgsi and core10 upper and lower limits