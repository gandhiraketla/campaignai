import mysql.connector
from datetime import datetime, timedelta
import random

def generate_sample_data(conn, num_campaigns=10):
    """Generate sample campaign data"""
    
    channels = ['Facebook', 'Google', 'LinkedIn']
    campaign_types = ['awareness', 'conversion']
    
    cursor = conn.cursor()
    
    for i in range(num_campaigns):
        # Basic campaign info
        channel = random.choice(channels)
        campaign_type = random.choice(campaign_types)
        
        # Generate dates
        start_date = datetime.now() - timedelta(days=random.randint(1, 90))
        end_date = start_date + timedelta(days=random.randint(15, 45))
        
        # Generate metrics with realistic correlations
        budget = round(random.uniform(1000, 10000), 2)
        spend = round(random.uniform(0.7, 1.0) * budget, 2)  # 70-100% of budget
        
        # Generate correlated metrics
        impressions = random.randint(10000, 100000)
        clicks = int(impressions * random.uniform(0.01, 0.05))  # 1-5% CTR
        conversions = int(clicks * random.uniform(0.02, 0.10))  # 2-10% conversion rate
        
        # Revenue (higher for conversion campaigns)
        avg_value = 100 if campaign_type == 'conversion' else 50
        revenue = round(conversions * avg_value, 2)
        
        # Prepare data
        campaign_data = {
            'campaign_name': f"{channel}_{campaign_type}_Campaign_{i+1}",
            'channel': channel,
            'campaign_type': campaign_type,
            'start_date': start_date.date(),
            'end_date': end_date.date(),
            'status': 'completed' if end_date.date() < datetime.now().date() else 'active',
            'budget': budget,
            'spend': spend,
            'impressions': impressions,
            'clicks': clicks,
            'conversions': conversions,
            'revenue': revenue,
            'notes': f"Sample {campaign_type} campaign for {channel}"
        }
        
        # Insert into database
        columns = ', '.join(campaign_data.keys())
        values = ', '.join(['%s'] * len(campaign_data))
        insert_query = f"INSERT INTO campaigns ({columns}) VALUES ({values})"
        
        try:
            cursor.execute(insert_query, list(campaign_data.values()))
            conn.commit()
        except Exception as e:
            print(f"Error inserting campaign: {e}")
            conn.rollback()
    
    cursor.close()

def main():
    # Database configuration
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'root',
        'database': 'mymarketing'
    }
    
    try:
        # Connect to database
        conn = mysql.connector.connect(**db_config)
        
        # Generate sample data
        generate_sample_data(conn, num_campaigns=1000)
        
        print("Successfully generated sample campaign data")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            conn.close()

if __name__ == "__main__":
    main()