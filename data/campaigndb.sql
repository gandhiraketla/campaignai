-- create_campaigns_table.sql
CREATE DATABASE IF NOT EXISTS mymarketing;
USE mymarketing;

DROP TABLE IF EXISTS campaigns;

CREATE TABLE campaigns (
    id INT AUTO_INCREMENT PRIMARY KEY,
    campaign_name VARCHAR(255),
    channel VARCHAR(255),      -- e.g., 'Facebook', 'Google Ads', 'LinkedIn', etc.
    start_date DATE,
    end_date DATE,
    budget FLOAT,              -- total allocated budget
    spend FLOAT,               -- actual spend so far
    impressions INT,           -- how many impressions served
    clicks INT,                -- how many clicks
    conversions INT,           -- how many conversions (leads, signups, etc.)
    revenue FLOAT,             -- total revenue attributed to this campaign
    notes TEXT                 -- free-form text for additional details
);
