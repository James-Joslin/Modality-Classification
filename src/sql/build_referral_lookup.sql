CREATE TABLE referral_source_lookup (
    category_name VARCHAR(255),
    unique_category_index INT,
    merged_category_index INT
);

-- Insert categories with the merged index
INSERT INTO referral_source_lookup (category_name, unique_category_index, merged_category_index)
VALUES
('GP', 1, 1),
('Health Visitor', 2, 1),
('Mental Health NHS Trust', 3, 1),
('Other Primary Health Care', 4, 1),
('Independent Sector Mental Health Services', 5, 1),
('Accident And Emergency Department', 6, 1),
('Drug Action Team / Drug Misuse Agency', 7, 1),
('Courts', 8, 2),
('Police', 9, 2),
('Probation Service', 10, 2),
('Court Liaison and Diversion Service', 11, 2),
('Prison', 12, 2),
('Social Services', 13, 3),
('Carer', 14, 3),
('Education Service', 15, 3),
('GamCare/partner network', 16, 3),
('Citizen’s Advice', 17, 3),
('Jobcentre plus', 18, 3),
('Self-Referral', 19, 4),
('National Gambling Helpline', 20, 4),
('London Problem Gambling Clinic / CNWL', 21, 4),
('Northern Gambling Service / LYPFT', 22, 4),
('Primary Care Gambling Service (PCGS)', 23, 4),
('Gordon Moody Association (GMA)', 24, 4),
('Not known or declined response', 25, 5),
('Not stated', 26, 5),
(NULL, 27, 5),
('Other service or agency', 28, 5);

-- PREVIOUS METHOD LEFT FOR TRANSPARENCY
-- CREATE TABLE referral_source_lookup (
--     category_name VARCHAR(255),
--     category_index INT
-- );

-- -- Insert categories other than the special ones
-- INSERT INTO referral_source_lookup (category_name, category_index)
-- SELECT
--     referral_source_general,
--     ROW_NUMBER() OVER (ORDER BY referral_source_general)
-- FROM
--     (SELECT DISTINCT referral_source_general FROM modality_df  WHERE referral_source_general IS NOT NULL) AS distinct_categories
-- WHERE referral_source_general NOT IN ('Not known or declined response', 'Not stated');

-- -- Insert the special categories with index 28
-- INSERT INTO referral_source_lookup (category_name, category_index)
-- VALUES
-- ('Not known or declined response', 28),
-- ('Not stated', 28);

-- -- Handle actual NULL values
-- INSERT INTO referral_source_lookup (category_name, category_index)
-- SELECT
--     NULL,
--     28
-- WHERE NOT EXISTS (SELECT 1 FROM referral_source_lookup WHERE category_name IS NULL);
-- -- Values given and index of 28 are treated the same, as they all reflect an absense of data which for personal metrics is a useful category
-- -- A reluctance to reveal personal information, or the data not getting collected either tells us about the individual or their initial treatment