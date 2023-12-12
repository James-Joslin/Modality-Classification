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
    core10_lookup;