CREATE DATABASE IF NOT EXISTS mymarketing;
USE mymarketing;

DROP TABLE IF EXISTS campaigns;

CREATE TABLE campaigns (
    id INT AUTO_INCREMENT PRIMARY KEY,
    campaign_name VARCHAR(255),
    channel VARCHAR(50),                  -- e.g., Facebook, Google, LinkedIn
    campaign_type ENUM('awareness', 'conversion'),  -- Simplified types
    start_date DATE,
    end_date DATE,
    status ENUM('active', 'completed'),   -- Simplified status
    budget DECIMAL(10,2),                 -- Total allocated budget
    spend DECIMAL(10,2),                  -- Actual spend
    impressions INT,
    clicks INT,
    conversions INT,
    revenue DECIMAL(10,2),                -- Revenue generated
    notes TEXT,                           -- Additional notes
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);