import os
import random
import datetime
import mysql.connector
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from util.envutils import EnvUtils
envUtils = EnvUtils()
DB_HOST = envUtils.get_required_env("DB_HOST")
DB_USER = envUtils.get_required_env("DB_USER")
DB_PASS = envUtils.get_required_env("DB_PASSWORD")
DB_NAME = envUtils.get_required_env("DB_NAME")

CAMPAIGN_NAMES = [
    "Spring Sale",
    "Summer Launch",
    "Holiday Promo",
    "Black Friday",
    "New Year Kickoff",
    "Brand Awareness Q3",
    "Upsell Campaign",
]
CHANNELS = ["Facebook", "Google Ads", "LinkedIn", "Twitter", "Email"]

def random_date(year_start=2022, year_end=2023):
    start_date = datetime.date(year_start, 1, 1)
    end_date = datetime.date(year_end, 12, 31)
    delta = (end_date - start_date).days
    random_days = random.randrange(delta)
    return start_date + datetime.timedelta(days=random_days)

def main(num_records=50):
    conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME
    )
    cursor = conn.cursor()

    insert_sql = """
        INSERT INTO campaigns (
            campaign_name, channel, start_date, end_date,
            budget, spend, impressions, clicks, conversions, revenue, notes
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    data_to_insert = []
    for _ in range(num_records):
        name = random.choice(CAMPAIGN_NAMES)
        channel = random.choice(CHANNELS)
        start_date = random_date(2022, 2023)
        # Ensure end_date is after start_date
        end_date = start_date + datetime.timedelta(days=random.randint(15, 90))
        budget = round(random.uniform(1000.0, 20000.0), 2)
        spend = round(random.uniform(0.0, budget), 2)
        impressions = random.randint(1000, 200000)
        clicks = random.randint(0, impressions // 5)  # e.g. up to 20% CTR
        conversions = random.randint(0, clicks // 3)  # e.g. up to 30% conversion of clicks
        revenue = round(random.uniform(0.0, conversions * 50.0), 2)  # e.g. each conversion worth up to $50
        notes = f"Campaign {name} on {channel} from {start_date} to {end_date}."

        data_to_insert.append((
            name,
            channel,
            start_date,
            end_date,
            budget,
            spend,
            impressions,
            clicks,
            conversions,
            revenue,
            notes
        ))

    cursor.executemany(insert_sql, data_to_insert)
    conn.commit()
    cursor.close()
    conn.close()

    print(f"Inserted {num_records} campaign records.")

if __name__ == "__main__":
    main(num_records=1000)
