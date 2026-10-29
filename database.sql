CREATE DATABASE IF NOT EXISTS customer_churn;
USE customer_churn;

-- After creating the database, run database.py to load customers.csv.

-- 1. Churn rate
SELECT ROUND(100 * AVG(churn = 'Yes'), 2) AS churn_rate
FROM customers;

-- 2. Average monthly charges by churn status
SELECT churn, ROUND(AVG(monthly_charges),2) AS avg_monthly_charges
FROM customers
GROUP BY churn;

-- 3. Churn by contract
SELECT contract, COUNT(*) AS customers,
       SUM(churn='Yes') AS churned_customers
FROM customers
GROUP BY contract
ORDER BY churned_customers DESC;

-- 4. High-value churned customers
SELECT customer_id, monthly_charges, total_charges, contract
FROM customers
WHERE churn='Yes'
ORDER BY monthly_charges DESC
LIMIT 10;

-- 5. Churn by internet service
SELECT internet_service, COUNT(*) AS customers,
       SUM(churn='Yes') AS churned
FROM customers
GROUP BY internet_service;
