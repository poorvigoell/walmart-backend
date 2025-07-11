import pandas as pd
import gdown
import numpy as np
import requests
import lightgbm as lgb
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("WEATHER_API_KEY")


def get_weather_trigger(city='Austin'):
    api_key = "30cde49b96cbba46887872e0e9bcf81b"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    try:
        response = requests.get(url).json()
        if 'weather' not in response:
            return 'unknown'
        weather = response['weather'][0]['main'].lower()
        temp = response['main']['temp'] - 273.15
        if weather in ['rain', 'snow', 'storm']:
            return 'bad_weather'
        elif temp > 35:
            return 'heatwave'
        elif temp < 5:
            return 'cold_snap'
        else:
            return 'normal'
    except:
        return 'unknown'

def create_lag_features(df, lags=[7, 28], windows=[7, 28]):
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('id')['sales'].shift(lag)
    for lag in lags:
        for win in windows:
            df[f'rolling_mean_lag{lag}_w{win}'] = (
                df.groupby('id')[f'lag_{lag}'].transform(lambda x: x.rolling(win).mean())
            )

def load_data():
    sales_id = "1W6aJIYVrdUo_n39pcFwqUphPV6gnOMKo"
    calendar_id = "1t0fbfL9ukoA6ZlcF2B6hEoi5mZgDP9Ag"
    prices_id = "1ZXg52jNdQM_zNgM7B9EqeN2T6SHfH4T6"

    os.makedirs("data", exist_ok=True)

    gdown.download(f"https://drive.google.com/uc?id={calendar_id}", "data/calendar.csv", quiet=False)
    gdown.download(f"https://drive.google.com/uc?id={sales_id}", "data/sales.csv", quiet=False)
    gdown.download(f"https://drive.google.com/uc?id={prices_id}", "data/sell_prices.csv", quiet=False)

    calendar = pd.read_csv("data/calendar.csv")
    sales = pd.read_csv("data/sales.csv")
    sell_prices = pd.read_csv("data/sell_prices.csv")

    return calendar, sales, sell_prices

def run_lazy_forecast(city='Austin'):
    # Weather check
    weather_trigger = get_weather_trigger(city)

    # Load data
    calendar, sales, sell_prices = load_data()

    sales = sales.iloc[:, :6].join(sales.iloc[:, -60:])

    # Melt and merge
    sales_melted = sales.melt(
        id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
        var_name='d', value_name='sales'
    )
    calendar_cols = ['d', 'wm_yr_wk', 'event_name_1', 'event_type_1', 'weekday', 'month', 'year']
    sales_melted = sales_melted.merge(calendar[calendar_cols], on='d', how='left')
    sell_prices['id'] = sell_prices['item_id'] + '_' + sell_prices['store_id'] + '_validation'
    sales_melted = sales_melted.merge(sell_prices[['id', 'wm_yr_wk', 'sell_price']], on=['id', 'wm_yr_wk'], how='left')

    # Categorical
    cat_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'weekday']
    for col in cat_features:
        sales_melted[col] = sales_melted[col].astype('category')

    # Lag features
    create_lag_features(sales_melted)

    # Fill NaNs
    num_cols = sales_melted.select_dtypes(include=['number']).columns
    sales_melted[num_cols] = sales_melted[num_cols].fillna(-1)

    # Train model
    sales_melted['d_int'] = sales_melted['d'].str.extract('d_(\\d+)').astype(int)
    train_data = sales_melted[sales_melted['d_int'] < 1886]
    valid_data = sales_melted[sales_melted['d_int'] >= 1886]
    features = ['sell_price', 'lag_7', 'lag_28', 'rolling_mean_lag7_w7', 'rolling_mean_lag28_w28'] + cat_features
    target = 'sales'
    train_set = lgb.Dataset(train_data[features], label=train_data[target])
    valid_set = lgb.Dataset(valid_data[features], label=valid_data[target])
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 64,
        'min_data_in_leaf': 50,
        'verbosity': -1
    }
    model = lgb.train(params, train_set, valid_sets=[train_set, valid_set], callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(100)
    ])

    # Low stock logic
    recent_sales = sales[['id'] + list(sales.columns[-7:])].copy()
    recent_sales['mean_sales'] = recent_sales.iloc[:, 1:].mean(axis=1)
    low_stock_ids = recent_sales[recent_sales['mean_sales'] < 3]['id'].tolist()

    # Forecast future
    forecast_days = [f'd_{i}' for i in range(1914, 1942)]
    future = sales[sales['id'].isin(low_stock_ids)].copy()
    future = future.loc[future.index.repeat(28)].reset_index(drop=True)
    future['d'] = forecast_days * len(low_stock_ids)
    future = future.merge(calendar[calendar_cols], on='d', how='left')
    future['id'] = future['item_id'] + '_' + future['store_id'] + '_validation'
    future = future.merge(sell_prices[['id', 'wm_yr_wk', 'sell_price']], on=['id', 'wm_yr_wk'], how='left')

    # Categorical setup
    future_cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    for col in ['event_name_1', 'event_type_1', 'weekday']:
        if col in future.columns:
            future_cat.append(col)
    for col in future_cat:
        future[col] = future[col].astype('category')

    # Lag again
    all_data = pd.concat([sales_melted, future], sort=False)
    create_lag_features(all_data)
    future = all_data[all_data['d'].isin(forecast_days)].copy()

    # Fill
    num_cols = future.select_dtypes(include=['float64', 'int64']).columns
    future[num_cols] = future[num_cols].fillna(-1)
    for col in future.select_dtypes(include='category').columns:
        future[col] = future[col].cat.add_categories(['missing']).fillna('missing')
    for col in future_cat:
        if col in future.columns and col in sales_melted.columns:
            future[col] = future[col].astype(
                pd.CategoricalDtype(categories=sales_melted[col].cat.categories)
            )

    # Predict
    future['sales'] = model.predict(future[features])
    future['weather_condition'] = weather_trigger

    # Compare future vs past
    predicted_totals = future.groupby('id')['sales'].sum()
    past_cols = list(sales.columns[-28:])
    past_sales = sales[sales['id'].isin(predicted_totals.index)].copy()
    past_sales['past_28_day_total'] = past_sales[past_cols].sum(axis=1)
    past_totals = past_sales.set_index('id')['past_28_day_total']

    comparison = pd.DataFrame({
        'predicted': predicted_totals,
        'past': past_totals
    })
    comparison['change_%'] = ((comparison['predicted'] - comparison['past']) / comparison['past']) * 100
    comparison = comparison.replace([np.inf, -np.inf], np.nan).dropna()

    # Combine top 3 ↑ and top 2 ↓ (total 5), or adjust as you like
    top_spike = comparison.sort_values(by='change_%', ascending=False).head(3)
    top_drop = comparison.sort_values(by='change_%').head(2)
    top_ids = pd.concat([top_spike, top_drop]).index

    # Format: one row per SKU (not 28 per day)
    top_summary = comparison.loc[top_ids].reset_index()
    top_summary[['item_id', 'store_id']] = top_summary['id'].str.extract(r'^(.*)_([^_]+)_validation$')
    top_summary['weather_condition'] = weather_trigger

    return top_summary[['id', 'item_id', 'store_id', 'predicted', 'past', 'change_%', 'weather_condition']]

def get_stockout_alerts(city='Austin'):
    # Repeat shared setup steps
    weather_trigger = get_weather_trigger(city)
    
    calendar, sales, sell_prices = load_data()

    sales = sales.iloc[:, :6].join(sales.iloc[:, -60:])
    forecast_days = [f'd_{i}' for i in range(1914, 1942)]

    # Prepare recent sales
    recent_sales = sales[['id'] + list(sales.columns[-7:])].copy()
    recent_sales['mean_sales'] = recent_sales.iloc[:, 1:].mean(axis=1)
    low_stock_ids = recent_sales[recent_sales['mean_sales'] < 3]['id'].tolist()

    # Melt for lag features
    sales_melted = sales.melt(
        id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
        var_name='d', value_name='sales'
    )
    calendar_cols = ['d', 'wm_yr_wk', 'event_name_1', 'event_type_1', 'weekday', 'month', 'year']
    sales_melted = sales_melted.merge(calendar[calendar_cols], on='d', how='left')
    sell_prices['id'] = sell_prices['item_id'] + '_' + sell_prices['store_id'] + '_validation'
    sales_melted = sales_melted.merge(sell_prices[['id', 'wm_yr_wk', 'sell_price']], on=['id', 'wm_yr_wk'], how='left')

    cat_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'weekday']
    for col in cat_features:
        sales_melted[col] = sales_melted[col].astype('category')

    create_lag_features(sales_melted)
    num_cols = sales_melted.select_dtypes(include=['number']).columns
    sales_melted[num_cols] = sales_melted[num_cols].fillna(-1)

    sales_melted['d_int'] = sales_melted['d'].str.extract('d_(\\d+)').astype(int)
    train_data = sales_melted[sales_melted['d_int'] < 1886]
    valid_data = sales_melted[sales_melted['d_int'] >= 1886]
    features = ['sell_price', 'lag_7', 'lag_28', 'rolling_mean_lag7_w7', 'rolling_mean_lag28_w28'] + cat_features
    target = 'sales'
    train_set = lgb.Dataset(train_data[features], label=train_data[target])
    valid_set = lgb.Dataset(valid_data[features], label=valid_data[target])
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 64,
        'min_data_in_leaf': 50,
        'verbosity': -1
    }
    model = lgb.train(params, train_set, valid_sets=[train_set, valid_set], callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(100)
    ])

    # Forecast future for low-stock
    future = sales[sales['id'].isin(low_stock_ids)].copy()
    future = future.loc[future.index.repeat(28)].reset_index(drop=True)
    future['d'] = forecast_days * len(low_stock_ids)
    future = future.merge(calendar[calendar_cols], on='d', how='left')
    future['id'] = future['item_id'] + '_' + future['store_id'] + '_validation'
    future = future.merge(sell_prices[['id', 'wm_yr_wk', 'sell_price']], on=['id', 'wm_yr_wk'], how='left')

    future_cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    for col in ['event_name_1', 'event_type_1', 'weekday']:
        if col in future.columns:
            future_cat.append(col)
    for col in future_cat:
        future[col] = future[col].astype('category')

    all_data = pd.concat([sales_melted, future], sort=False)
    create_lag_features(all_data)
    future = all_data[all_data['d'].isin(forecast_days)].copy()

    num_cols = future.select_dtypes(include=['float64', 'int64']).columns
    future[num_cols] = future[num_cols].fillna(-1)
    for col in future.select_dtypes(include='category').columns:
        future[col] = future[col].cat.add_categories(['missing']).fillna('missing')
    for col in future_cat:
        if col in future.columns and col in sales_melted.columns:
            future[col] = future[col].astype(
                pd.CategoricalDtype(categories=sales_melted[col].cat.categories)
            )

    future['sales'] = model.predict(future[features])
    next_7_days = forecast_days[:7]
    future_next_7 = future[future['d'].isin(next_7_days)]
    forecast_7d = future_next_7.groupby('id')['sales'].sum()

    # Merge with past 7d avg
    past_7d = recent_sales.set_index('id')['mean_sales']
    risk_df = pd.DataFrame({
        'avg_past_7d_sales': past_7d,
        'forecast_next_7d': forecast_7d
    }).dropna()
    risk_df['risk'] = (risk_df['avg_past_7d_sales'] < 3) & (risk_df['forecast_next_7d'] > 10)

    at_risk = risk_df[risk_df['risk']].reset_index()
    # Sort by forecasted next 7 days sales (descending)
    at_risk = at_risk.sort_values(by='forecast_next_7d', ascending=False).head(3).reset_index(drop=True)

    # Extract item/store + add weather
    at_risk[['item_id', 'store_id']] = at_risk['id'].str.extract(r'^(.*)_([^_]+)_validation$')
    at_risk['weather_condition'] = weather_trigger

    return at_risk[['id', 'item_id', 'store_id', 'avg_past_7d_sales', 'forecast_next_7d', 'weather_condition']].to_dict(orient='records')
