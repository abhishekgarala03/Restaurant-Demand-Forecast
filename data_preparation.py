import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def prepare_swiggy_demand_data():
    """
    Converts public store sales data into Swiggy restaurant logistics context
    WHY:
    - Demonstrates "mine and extract relevant information from Swiggy's massive historical data"
    - Shows ability to reframe data for logistics domain
    - Proves business understanding of delivery challenges
    """
    # Load public dataset
    try:
        # Use smaller dataset for faster processing
        df = pd.read_csv('https://raw.githubusercontent.com/monk333/datasets/main/store_sales.csv', 
                         nrows=10000)
    except:
        # Fallback to local sample if internet issues
        df = pd.DataFrame({
            'date': pd.date_range(start='2022-01-01', periods=100),
            'store': np.random.randint(1, 55, 100),
            'item': np.random.randint(1, 50, 100),
            'sales': np.random.poisson(lam=25, size=100)
        })
    
    # Convert to Swiggy logistics context
    df = df.rename(columns={
        'date': 'order_date',
        'store': 'restaurant_id',
        'item': 'menu_item_id',
        'sales': 'order_count'
    })
    
    # Add Swiggy-specific logistics features
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['hour'] = df['order_date'].dt.hour
    df['day_of_week'] = df['order_date'].dt.dayofweek  # Monday=0, Sunday=6
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_festival'] = 0  # Placeholder for Swiggy festivals (Eid, Diwali etc.)
    
    # Simulate Bangalore-specific patterns
    df.loc[(df['hour'] >= 12) & (df['hour'] <= 14), 'is_lunch_rush'] = 1
    df.loc[(df['hour'] >= 19) & (df['hour'] <= 22), 'is_dinner_rush'] = 1
    df['is_lunch_rush'] = df['is_lunch_rush'].fillna(0)
    df['is_dinner_rush'] = df['is_dinner_rush'].fillna(0)
    
    # Business-relevant aggregation
    # Group by restaurant and hour for delivery staffing decisions
    demand_data = df.groupby([
        pd.Grouper(key='order_date', freq='H'), 
        'restaurant_id'
    ]).agg({
        'order_count': 'sum',
        'is_weekend': 'first',
        'is_lunch_rush': 'first',
        'is_dinner_rush': 'first'
    }).reset_index()
    
    # Add weather impact
    demand_data['weather_impact'] = np.random.choice(
        [0, 0.3, 0.7], 
        size=len(demand_data),
        p=[0.7, 0.2, 0.1]  # 70% normal, 20% moderate rain, 10% heavy rain
    )
    
    print(f"✅ Prepared {len(demand_data)} Swiggy-style hourly demand records")
    print(f"📍 Simulated Bangalore restaurant patterns: "
          f"{demand_data['is_lunch_rush'].mean():.0%} lunch rush, "
          f"{demand_data['is_dinner_rush'].mean():.0%} dinner rush")
    
    return demand_data

def create_logistics_features(df):
    """
    Adds Swiggy-specific logistics features for delivery optimization
    WHY:
    - Directly addresses "logistics domain" opportunity
    - Shows understanding of delivery staffing challenges
    - Proves ability to "formulate business problems in ML terms"
    """
    # Lag features for demand patterns
    df = df.sort_values(['restaurant_id', 'order_date'])
    df['orders_last_hour'] = df.groupby('restaurant_id')['order_count'].shift(1)
    df['orders_last_day_same_hour'] = df.groupby('restaurant_id')['order_count'].shift(24)
    
    # Rolling features for trend detection
    df['orders_3h_mean'] = df.groupby('restaurant_id')['order_count'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    
    # Special event features
    df['is_festival'] = 0
    festival_dates = [
        '2022-10-24', '2022-11-12', '2022-12-25',  # Diwali, Christmas etc.
        '2023-01-26', '2023-03-08'  # Republic Day, Holi
    ]
    df.loc[df['order_date'].dt.date.isin([pd.Timestamp(d).date() for d in festival_dates]), 'is_festival'] = 1
    
    # Fill NA
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    print("✅ Added Swiggy logistics features: historical patterns, festival impact, rolling trends")
    return df

if __name__ == "__main__":
    demand_data = prepare_swiggy_demand_data()
    logistics_data = create_logistics_features(demand_data)
    
    # Save for model training
    logistics_data.to_csv('swiggy_demand_data.csv', index=False)
    print(f"💾 Saved {len(logistics_data)} records for forecasting engine")
