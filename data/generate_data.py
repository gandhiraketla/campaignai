import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

# Initialize Faker
fake = Faker()

# Constants
CAMPAIGN_TYPES = {
    'Social Media': 0.30,
    'Email Marketing': 0.25,
    'Content Marketing': 0.20,
    'TV Advertising': 0.15,
    'Direct Mail': 0.10
}

PRODUCT_TYPES = {
    'Smartphones': {
        'price_range': (600, 1200),
        'conversion_rate': (0.01, 0.03),
        'seasonality': {'Q1': 0.9, 'Q2': 1.0, 'Q3': 1.1, 'Q4': 1.2}
    },
    'Smart Watches': {
        'price_range': (200, 500),
        'conversion_rate': (0.02, 0.04),
        'seasonality': {'Q1': 0.8, 'Q2': 1.3, 'Q3': 1.0, 'Q4': 1.1}
    },
    'Wireless Headphones': {
        'price_range': (150, 300),
        'conversion_rate': (0.03, 0.06),
        'seasonality': {'Q1': 0.9, 'Q2': 1.0, 'Q3': 1.0, 'Q4': 1.2}
    },
    'Smart Home Security': {
        'price_range': (200, 800),
        'conversion_rate': (0.015, 0.035),
        'seasonality': {'Q1': 1.0, 'Q2': 1.1, 'Q3': 0.9, 'Q4': 1.1}
    },
    'Gaming Consoles': {
        'price_range': (300, 600),
        'conversion_rate': (0.02, 0.04),
        'seasonality': {'Q1': 0.8, 'Q2': 0.9, 'Q3': 1.0, 'Q4': 1.4}
    },
    'Laptop Computers': {
        'price_range': (800, 2000),
        'conversion_rate': (0.01, 0.025),
        'seasonality': {'Q1': 0.9, 'Q2': 1.0, 'Q3': 1.3, 'Q4': 1.0}
    }
}

CREATIVE_TYPES = ['Image', 'Video', 'Text', 'Poll']
REGIONS = {
    'Northeast': 0.20,
    'Southeast': 0.25,
    'Midwest': 0.20,
    'Southwest': 0.15,
    'West': 0.20
}
TARGET_AUDIENCES = ['Millennials', 'Gen Z', 'Professionals', 'Tech Enthusiasts', 'Parents', 'Seniors']
OBJECTIVES = [
    'Increase brand awareness',
    'Drive conversions',
    'Boost engagement',
    'Generate leads',
    'Launch product',
    'Increase sales'
]

CHANNELS = {
    'Social Media': 'Instagram, Facebook, Twitter',
    'Email Marketing': 'Email, Landing Pages',
    'Content Marketing': 'Blog, Website, Newsletter',
    'TV Advertising': 'Television, YouTube, Streaming Platforms',
    'Direct Mail': 'Physical Mail, Local Print'
}

class MarketingCampaignGenerator:
    def __init__(self, num_records=200000):
        self.num_records = num_records
        self.campaign_counter = 1
        
    def generate_campaign_id(self):
        campaign_id = f"CAMP{str(self.campaign_counter).zfill(6)}"
        self.campaign_counter += 1
        return campaign_id
    
    def generate_dates(self):
        start_year = 2020
        end_year = 2024
        
        # Generate random start date
        start_date = datetime(
            year=random.randint(start_year, end_year),
            month=random.randint(1, 12),
            day=random.randint(1, 28)
        )
        
        # Generate duration between 7 and 90 days
        duration = random.randint(7, 90)
        end_date = start_date + timedelta(days=duration)
        
        quarter = f"Q{(start_date.month - 1) // 3 + 1}"
        
        return start_date, end_date, start_date.year, quarter, duration
    
    def calculate_budget(self, campaign_type, quarter):
        # Base budget range
        if campaign_type == 'TV Advertising':
            base_budget = random.uniform(100000, 200000)
        else:
            base_budget = random.uniform(50000, 100000)
        
        # Apply seasonal multiplier
        seasonal_multiplier = {
            'Q1': 0.8,
            'Q2': 1.2,
            'Q3': 0.9,
            'Q4': 1.1
        }[quarter]
        
        return base_budget * seasonal_multiplier
    
    def generate_metrics(self, budget, product_type, quarter):
        # Generate impressions
        impressions = random.randint(100000, 1000000)
        
        # Calculate clicks (1-5% of impressions)
        click_rate = random.uniform(0.01, 0.05)
        clicks = int(impressions * click_rate)
        
        # Get conversion rate based on product type
        conv_rate_range = PRODUCT_TYPES[product_type]['conversion_rate']
        conversion_rate = random.uniform(conv_rate_range[0], conv_rate_range[1])
        
        # Calculate leads (2-10% of clicks)
        leads = int(clicks * random.uniform(0.02, 0.10))
        
        # Calculate revenue
        price_range = PRODUCT_TYPES[product_type]['price_range']
        avg_price = random.uniform(price_range[0], price_range[1])
        seasonal_multiplier = PRODUCT_TYPES[product_type]['seasonality'][quarter]
        
        conversions = int(clicks * conversion_rate)
        revenue = conversions * avg_price * seasonal_multiplier
        
        # Calculate ROI
        actual_spend = budget * random.uniform(0.9, 1.1)
        roi = ((revenue - actual_spend) / actual_spend) * 100 if actual_spend > 0 else 0
        
        # Generate engagement rate
        engagement_rate = random.uniform(0.02, 0.08)
        
        return {
            'Impressions': impressions,
            'Clicks': clicks,
            'Conversion_Rate': conversion_rate,
            'Leads': leads,
            'Revenue': revenue,
            'Actual_Spend': actual_spend,
            'ROI': roi,
            'Engagement_Rate': engagement_rate
        }
    
    def generate_campaign_name(self, year, product_type, campaign_type):
        seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        season = random.choice(seasons)
        return f"{season} {year} - {product_type} {campaign_type}"
    
    def generate_optimal_timing(self):
        hour = random.randint(8, 20)
        minute = random.choice([0, 15, 30, 45])
        ampm = 'AM' if hour < 12 else 'PM'
        if hour > 12:
            hour -= 12
        return f"Day {hour:02d}:{minute:02d} {ampm}"
    
    def generate_strategy_overview(self, campaign_type, target_audience, objective):
        return f"Targeted {target_audience} through {campaign_type} with focus on {objective.lower()}. " \
               f"Implemented best practices for {campaign_type.lower()} engagement and conversion optimization."
    
    def generate_key_lessons(self, roi, engagement_rate):
        lessons = []
        if roi > 50:
            lessons.append("Strong ROI performance suggests effective targeting and messaging")
        elif roi < 0:
            lessons.append("Campaign underperformed on ROI, requires strategy adjustment")
            
        if engagement_rate > 0.05:
            lessons.append("High engagement rates indicate strong content resonance")
        else:
            lessons.append("Room for improvement in content engagement strategies")
            
        return " | ".join(lessons)
    
    def generate_dataset(self):
        data = []
        
        for _ in range(self.num_records):
            # Generate campaign basics
            campaign_type = random.choices(
                list(CAMPAIGN_TYPES.keys()),
                weights=list(CAMPAIGN_TYPES.values())
            )[0]
            product_type = random.choice(list(PRODUCT_TYPES.keys()))
            start_date, end_date, year, quarter, duration = self.generate_dates()
            
            # Generate budget and metrics
            budget = self.calculate_budget(campaign_type, quarter)
            metrics = self.generate_metrics(budget, product_type, quarter)
            
            # Generate other fields
            region = random.choices(
                list(REGIONS.keys()),
                weights=list(REGIONS.values())
            )[0]
            
            campaign_data = {
                'Campaign_ID': self.generate_campaign_id(),
                'Year': year,
                'Quarter': quarter,
                'Start_Date': start_date.strftime('%Y-%m-%d'),
                'End_Date': end_date.strftime('%Y-%m-%d'),
                'Product_Type': product_type,
                'Campaign_Type': campaign_type,
                'Campaign_Name': self.generate_campaign_name(year, product_type, campaign_type),
                'Objective': random.choice(OBJECTIVES),
                'Target_Audience': random.choice(TARGET_AUDIENCES),
                'Geographic_Region': region,
                'Budget': budget,
                'Channels': CHANNELS[campaign_type],
                'Creative_Type': random.choice(CREATIVE_TYPES),
                'Duration_Days': duration,
                'Optimal_Timing': self.generate_optimal_timing(),
                **metrics
            }
            
            # Add strategy and lessons after metrics are generated
            campaign_data['Strategy_Overview'] = self.generate_strategy_overview(
                campaign_data['Campaign_Type'],
                campaign_data['Target_Audience'],
                campaign_data['Objective']
            )
            
            campaign_data['Key_Lessons_Learned'] = self.generate_key_lessons(
                campaign_data['ROI'],
                campaign_data['Engagement_Rate']
            )
            
            # Add optional notes
            campaign_data['Notes'] = fake.text(max_nb_chars=100) if random.random() < 0.3 else ''
            
            data.append(campaign_data)
        
        return pd.DataFrame(data)

def main():
    # Create generator instance
    generator = MarketingCampaignGenerator(num_records=200000)
    
    # Generate dataset
    print("Generating synthetic marketing campaign data...")
    df = generator.generate_dataset()
    
    # Save to CSV
    output_file = "synthetic_campaign_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Dataset generated and saved to {output_file}")
    
    # Print basic statistics
    print("\nDataset Statistics:")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['Start_Date'].min()} to {df['End_Date'].max()}")
    print(f"Average ROI: {df['ROI'].mean():.2f}%")
    print(f"Average Conversion Rate: {(df['Conversion_Rate'].mean() * 100):.2f}%")
    
    # Verify data distributions
    print("\nCampaign Type Distribution:")
    campaign_dist = df['Campaign_Type'].value_counts(normalize=True)
    for camp_type, freq in campaign_dist.items():
        print(f"{camp_type}: {freq:.1%}")
    
    print("\nRegional Distribution:")
    region_dist = df['Geographic_Region'].value_counts(normalize=True)
    for region, freq in region_dist.items():
        print(f"{region}: {freq:.1%}")

if __name__ == "__main__":
    main()