SELECT 
    'PGSI' AS score_type,
    MIN(pgsi_score) AS min_score,
    MAX(pgsi_score) AS max_score
FROM 
    pgsi_lookup

UNION ALL

SELECT 
    'CORE10' AS score_type,
    MIN(core10_score) AS min_score,
    MAX(core10_score) AS max_score
FROM 
    core10_lookup

UNION ALL

SELECT 
    'Ref_Index' AS score_type,
    MIN(unique_referral_index) AS min_score,
    MAX(unique_referral_index) AS max_score
FROM 
    referral_source_lookup;