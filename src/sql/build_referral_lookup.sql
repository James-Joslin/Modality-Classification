CREATE TABLE referral_source_lookup (
    referral_name VARCHAR(255),
    unique_referral_index INT,
    merged_referral_index INT
);

-- Insert categories with the merged index
INSERT INTO referral_source_lookup (referral_name, unique_referral_index, merged_referral_index)
VALUES
('GP', 0, 0),
('Health Visitor', 1, 0),
('Mental Health NHS Trust', 2, 0),
('Other Primary Health Care', 3, 0),
('Independent Sector Mental Health Services', 4, 0),
('Accident And Emergency Department', 5, 0),
('Drug Action Team / Drug Misuse Agency', 6, 0),
('Courts', 7, 1),
('Police', 8, 1),
('Probation Service', 9, 1),
('Court Liaison and Diversion Service', 10, 1),
('Prison', 11, 1),
('Social Services', 12, 2),
('Carer', 13, 2),
('Voluntary Sector', 14, 2),
('Education Service', 15, 2),
('GamCare/partner network', 16, 2),
('Citizen’s Advice', 17, 2),
('Jobcentre plus', 18, 2),
('Self-Referral', 19, 3),
('National Gambling Helpline', 20, 3),
('London Problem Gambling Clinic / CNWL', 21, 3),
('Northern Gambling Service / LYPFT', 22, 3),
('Primary Care Gambling Service (PCGS)', 23, 3),
('Gordon Moody Association (GMA)', 24, 3),
('Employer', 25, 4),
('Not known or declined response', 26, 4),
('Not stated', 27, 4),
('Other service or agency', 28, 4),
(NULL, 29, 4);

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