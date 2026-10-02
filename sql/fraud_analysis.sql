-- =============================================================================
-- Phase 7: SQL Analytics Layer
-- Run against: data/processed/fraud_analytics.db, table "transactions"
--
-- NOTE ON SIMULATED COLUMNS:
-- merchant_category, state, transaction_date and transaction_month are
-- SIMULATED (see src/load_to_sqlite.py docstring) because the real Kaggle
-- dataset only contains Time (48 real hours), Amount and Class - no
-- merchant/geography/calendar data exists in the anonymized source.
-- Queries 1 and 2 use REAL data. Queries 3, 4 and 5 use simulated dimensions
-- and are labeled as illustrative of the analysis pattern, not real findings.
-- =============================================================================


-- 1. REAL DATA: Fraud rate by hour of day
-- Shows which hours see the highest concentration of fraud relative to volume.
SELECT
    hour_of_day,
    COUNT(*) AS total_transactions,
    SUM(Class) AS fraud_count,
    ROUND(100.0 * SUM(Class) / COUNT(*), 4) AS fraud_rate_pct
FROM transactions
GROUP BY hour_of_day
ORDER BY fraud_rate_pct DESC;


-- 2. REAL DATA: Fraud rate by amount band
-- Shows whether fraud clusters in small "card testing" amounts or large ones.
SELECT
    amount_band,
    COUNT(*) AS total_transactions,
    SUM(Class) AS fraud_count,
    ROUND(100.0 * SUM(Class) / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(AVG(Amount), 2) AS avg_amount
FROM transactions
GROUP BY amount_band
ORDER BY
    CASE amount_band
        WHEN '0-10' THEN 1 WHEN '10-50' THEN 2 WHEN '50-100' THEN 3
        WHEN '100-500' THEN 4 WHEN '500-1000' THEN 5 ELSE 6
    END;


-- 3. SIMULATED DIMENSION: Fraud rate by merchant category
-- Illustrative only - categories are randomly assigned (see load_to_sqlite.py).
-- In production this would use the bank's real merchant category codes (MCC).
SELECT
    merchant_category,
    COUNT(*) AS total_transactions,
    SUM(Class) AS fraud_count,
    ROUND(100.0 * SUM(Class) / COUNT(*), 4) AS fraud_rate_pct
FROM transactions
GROUP BY merchant_category
ORDER BY fraud_rate_pct DESC;


-- 4. SIMULATED DIMENSION: Geographic fraud pattern by state
-- Illustrative only - states are randomly assigned, no real location signal
-- exists in the anonymized dataset.
SELECT
    state,
    COUNT(*) AS total_transactions,
    SUM(Class) AS fraud_count,
    ROUND(100.0 * SUM(Class) / COUNT(*), 4) AS fraud_rate_pct
FROM transactions
GROUP BY state
ORDER BY fraud_rate_pct DESC;


-- 5. SIMULATED DIMENSION: Monthly fraud trend
-- Illustrative only - the real data spans 48 hours; dates are synthetically
-- stretched across Jan-Jun 2025 to demonstrate a monthly trend query shape.
SELECT
    transaction_month,
    COUNT(*) AS total_transactions,
    SUM(Class) AS fraud_count,
    ROUND(100.0 * SUM(Class) / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN Class = 1 THEN Amount ELSE 0 END), 2) AS fraud_amount_at_risk
FROM transactions
GROUP BY transaction_month
ORDER BY transaction_month;


-- 6. REAL DATA: Night vs day fraud comparison
-- is_night = 1 for transactions between 10pm-6am.
SELECT
    CASE WHEN is_night = 1 THEN 'Night (10pm-6am)' ELSE 'Day (6am-10pm)' END AS period,
    COUNT(*) AS total_transactions,
    SUM(Class) AS fraud_count,
    ROUND(100.0 * SUM(Class) / COUNT(*), 4) AS fraud_rate_pct
FROM transactions
GROUP BY is_night;


-- 7. REAL DATA: Overall summary stats (for dashboard "Fraud Overview" page)
SELECT
    COUNT(*) AS total_transactions,
    SUM(Class) AS total_fraud,
    ROUND(100.0 * SUM(Class) / COUNT(*), 4) AS overall_fraud_rate_pct,
    ROUND(SUM(CASE WHEN Class = 1 THEN Amount ELSE 0 END), 2) AS total_fraud_amount_at_risk,
    ROUND(AVG(CASE WHEN Class = 1 THEN Amount END), 2) AS avg_fraud_amount,
    ROUND(AVG(CASE WHEN Class = 0 THEN Amount END), 2) AS avg_legit_amount
FROM transactions;