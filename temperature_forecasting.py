import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools
import seaborn as sns

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

print("="*80)
print("WINTER SEASON MIN & MAX TEMPERATURE FORECASTING WITH ARIMA AND SARIMA")
print("="*80)

# Load the data
print("Loading comprehensive ski analysis data...")
df = pd.read_csv('comprehensive_ski_analysis.csv')

print(f"Dataset loaded: {len(df)} records")
print(f"Winter season data: Week {df['Week'].min()}-{df['Week'].max()} across {df['Year'].nunique()} years ({df['Year'].min()}-{df['Year'].max()})")
print(f"Resorts: {df['Resort'].nunique()}")
print(f"Total seasonal periods: {df['Year'].nunique()} winters × 15 weeks = {df['Year'].nunique() * 15} weeks")
print(f"Min temperature range: {df['Avg_Min_Temp'].min():.1f}°C to {df['Avg_Min_Temp'].max():.1f}°C")
print(f"Max temperature range: {df['Avg_Max_Temp'].min():.1f}°C to {df['Avg_Max_Temp'].max():.1f}°C")

# Check for missing values
print(f"\nMissing values:")
print(f"  Avg_Min_Temp: {df['Avg_Min_Temp'].isnull().sum()}")
print(f"  Avg_Max_Temp: {df['Avg_Max_Temp'].isnull().sum()}")

# Display basic statistics
print("\nTemperature Statistics:")
print("\nMin Temperature:")
print(df['Avg_Min_Temp'].describe())
print("\nMax Temperature:")
print(df['Avg_Max_Temp'].describe())

print("\n" + "-"*60)
print("PREPARING TIME SERIES DATA")
print("-"*60)

# For winter season data, create time series for both min and max temperatures
weekly_temp = df.groupby(['Year', 'Week'])[['Avg_Min_Temp', 'Avg_Max_Temp']].mean().reset_index()
weekly_temp = weekly_temp.sort_values(['Year', 'Week']).reset_index(drop=True)

# Create a period index for seasonal data (15-week seasons)
weekly_temp['Season_Week'] = range(1, len(weekly_temp) + 1)
weekly_temp['Season'] = ((weekly_temp['Season_Week'] - 1) // 15) + 1

print(f"Time series prepared: {len(weekly_temp)} weekly observations")
print(f"Covering {weekly_temp['Season'].max()} complete winter seasons")
print(f"Season structure: 15 weeks × {weekly_temp['Season'].max()} years = {len(weekly_temp)} total weeks")

# Set up time series for both min and max temperatures
ts_min = pd.Series(weekly_temp['Avg_Min_Temp'].values, 
                   index=pd.RangeIndex(start=1, stop=len(weekly_temp)+1, name='Week_Number'))
ts_max = pd.Series(weekly_temp['Avg_Max_Temp'].values, 
                   index=pd.RangeIndex(start=1, stop=len(weekly_temp)+1, name='Week_Number'))

print(f"\nMin Temperature Series:")
print(f"  Average: {ts_min.mean():.2f}°C")
print(f"  Standard deviation: {ts_min.std():.2f}°C")

print(f"\nMax Temperature Series:")
print(f"  Average: {ts_max.mean():.2f}°C") 
print(f"  Standard deviation: {ts_max.std():.2f}°C")

# Check stationarity
def check_stationarity(timeseries, title):
    print(f"\n{title}")
    
    # Perform Augmented Dickey-Fuller test
    result = adfuller(timeseries, autolag='AIC')
    
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')
    
    if result[1] <= 0.05:
        print("✓ Series is stationary")
        return True
    else:
        print("✗ Series is non-stationary")
        return False

# Check stationarity for both temperature series
print("\nSTATIONARITY TESTS:")
is_stationary_min = check_stationarity(ts_min, "Min Temperature - Original Series")
is_stationary_max = check_stationarity(ts_max, "Max Temperature - Original Series")

# If not stationary, difference the series
if not is_stationary_min:
    ts_min_diff = ts_min.diff().dropna()
    is_diff_stationary_min = check_stationarity(ts_min_diff, "Min Temperature - Differenced Series")
    
if not is_stationary_max:
    ts_max_diff = ts_max.diff().dropna()
    is_diff_stationary_max = check_stationarity(ts_max_diff, "Max Temperature - Differenced Series")

# For seasonal data with limited observations, use different split strategy
# Use last 2 seasons (30 weeks) for testing, rest for training
test_seasons = 2
test_size = test_seasons * 15
train_size = len(ts_min) - test_size

print(f"\nTime series length: {len(ts_min)} weeks ({len(ts_min)//15} complete seasons)")
print(f"Training data: {train_size} weeks ({train_size//15} seasons + {train_size%15} weeks)")
print(f"Testing data: {test_size} weeks ({test_seasons} complete seasons)")
print(f"Split strategy: Using last {test_seasons} seasons for testing")

print("\n" + "-"*60)
print("ARIMA MODEL IMPLEMENTATION")
print("-"*60)

# Split data using the calculated sizes for both temperature series
train_min, test_min = ts_min[:train_size], ts_min[train_size:]
train_max, test_max = ts_max[:train_size], ts_max[train_size:]

print(f"Training period: Week {train_min.index[0]} to Week {train_min.index[-1]}")
print(f"Testing period: Week {test_min.index[0]} to Week {test_min.index[-1]}")
print(f"Testing represents seasons {(test_min.index[0]-1)//15 + 1}-{(test_min.index[-1]-1)//15 + 1}")

def evaluate_arima_model(train_data, test_data, arima_order):
    """Evaluate ARIMA model with given parameters"""
    try:
        model = ARIMA(train_data, order=arima_order)
        fitted_model = model.fit()
        
        # Make predictions
        forecast = fitted_model.forecast(steps=len(test_data))
        
        # Calculate metrics
        mae = mean_absolute_error(test_data, forecast)
        mse = mean_squared_error(test_data, forecast)
        rmse = np.sqrt(mse)
        
        return fitted_model.aic, mae, rmse, fitted_model, forecast
    except:
        return float('inf'), float('inf'), float('inf'), None, None

print("\nSearching for optimal ARIMA parameters...")

# Grid search for ARIMA parameters
p_values = range(0, 4)
d_values = range(0, 3)
q_values = range(0, 4)

# Initialize results for both temperature series
arima_results_min = []
arima_results_max = []
best_models = {}

# Process Min Temperature ARIMA models
print("\n--- MIN TEMPERATURE ARIMA MODELS ---")
best_aic_min = float('inf')
best_order_min = None
best_model_min = None
best_forecast_min = None

for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p, d, q)
            try:
                aic, mae, rmse, model, forecast = evaluate_arima_model(train_min, test_min, order)
                arima_results_min.append({
                    'order': order,
                    'aic': aic,
                    'mae': mae,
                    'rmse': rmse
                })
                
                if aic < best_aic_min:
                    best_aic_min = aic
                    best_order_min = order
                    best_model_min = model
                    best_forecast_min = forecast
                    
                print(f"Min Temp ARIMA{order} - AIC: {aic:.2f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}")
            except:
                continue

print(f"\n✓ Best Min Temp ARIMA model: ARIMA{best_order_min}")
print(f"  AIC: {best_aic_min:.2f}")
if arima_results_min:
    best_min_result = min(arima_results_min, key=lambda x: x['aic'])
    print(f"  MAE: {best_min_result['mae']:.3f}°C")
    print(f"  RMSE: {best_min_result['rmse']:.3f}°C")

# Process Max Temperature ARIMA models  
print("\n--- MAX TEMPERATURE ARIMA MODELS ---")
best_aic_max = float('inf')
best_order_max = None
best_model_max = None
best_forecast_max = None

for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p, d, q)
            try:
                aic, mae, rmse, model, forecast = evaluate_arima_model(train_max, test_max, order)
                arima_results_max.append({
                    'order': order,
                    'aic': aic,
                    'mae': mae,
                    'rmse': rmse
                })
                
                if aic < best_aic_max:
                    best_aic_max = aic
                    best_order_max = order
                    best_model_max = model
                    best_forecast_max = forecast
                    
                print(f"Max Temp ARIMA{order} - AIC: {aic:.2f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}")
            except:
                continue

print(f"\n✓ Best Max Temp ARIMA model: ARIMA{best_order_max}")
print(f"  AIC: {best_aic_max:.2f}")
if arima_results_max:
    best_max_result = min(arima_results_max, key=lambda x: x['aic'])
    print(f"  MAE: {best_max_result['mae']:.3f}°C")
    print(f"  RMSE: {best_max_result['rmse']:.3f}°C")

# Store best models
best_models['arima_min'] = {
    'order': best_order_min, 
    'model': best_model_min, 
    'forecast': best_forecast_min,
    'aic': best_aic_min
}
best_models['arima_max'] = {
    'order': best_order_max, 
    'model': best_model_max, 
    'forecast': best_forecast_max,
    'aic': best_aic_max
}

print("\n" + "-"*60)
print("SARIMA MODEL IMPLEMENTATION")
print("-"*60)

def evaluate_sarima_model(train_data, test_data, order, seasonal_order):
    """Evaluate SARIMA model with given parameters"""
    try:
        model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)
        
        # Make predictions
        forecast = fitted_model.forecast(steps=len(test_data))
        
        # Calculate metrics
        mae = mean_absolute_error(test_data, forecast)
        mse = mean_squared_error(test_data, forecast)
        rmse = np.sqrt(mse)
        
        return fitted_model.aic, mae, rmse, fitted_model, forecast
    except:
        return float('inf'), float('inf'), float('inf'), None, None

print("Searching for optimal SARIMA parameters...")
print("Note: Using seasonal period of 15 (winter season seasonality)")

# SARIMA grid search adapted for 15-week seasonal patterns
p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]

# Seasonal parameters (P, D, Q, S) - using S=15 for winter season patterns
P_values = [0, 1]
D_values = [0, 1]
Q_values = [0, 1]
S = 15  # 15-week winter season seasonality

best_sarima_aic = float('inf')
best_sarima_order = None
best_sarima_seasonal = None
best_sarima_model = None
best_sarima_forecast = None

sarima_results = []

# Initialize SARIMA results for both series
sarima_results_min = []
sarima_results_max = []

# Process Min Temperature SARIMA models
print("\n--- MIN TEMPERATURE SARIMA MODELS ---")
best_sarima_aic_min = float('inf')
best_sarima_order_min = None
best_sarima_seasonal_min = None
best_sarima_model_min = None
best_sarima_forecast_min = None

print("Testing Min Temp SARIMA configurations...")
for p in p_values:
    for d in d_values:
        for q in q_values:
            for P in P_values:
                for D in D_values:
                    for Q in Q_values:
                        order = (p, d, q)
                        seasonal_order = (P, D, Q, S)
                        
                        try:
                            aic, mae, rmse, model, forecast = evaluate_sarima_model(
                                train_min, test_min, order, seasonal_order
                            )
                            
                            sarima_results_min.append({
                                'order': order,
                                'seasonal_order': seasonal_order,
                                'aic': aic,
                                'mae': mae,
                                'rmse': rmse
                            })
                            
                            if aic < best_sarima_aic_min:
                                best_sarima_aic_min = aic
                                best_sarima_order_min = order
                                best_sarima_seasonal_min = seasonal_order
                                best_sarima_model_min = model
                                best_sarima_forecast_min = forecast
                            
                            print(f"Min Temp SARIMA{order}x{seasonal_order} - AIC: {aic:.2f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}")
                        except Exception as e:
                            continue

print(f"\n✓ Best Min Temp SARIMA model: SARIMA{best_sarima_order_min}x{best_sarima_seasonal_min}")
print(f"  AIC: {best_sarima_aic_min:.2f}")
if sarima_results_min:
    best_sarima_result_min = min(sarima_results_min, key=lambda x: x['aic'])
    print(f"  MAE: {best_sarima_result_min['mae']:.3f}°C")
    print(f"  RMSE: {best_sarima_result_min['rmse']:.3f}°C")

# Process Max Temperature SARIMA models
print("\n--- MAX TEMPERATURE SARIMA MODELS ---")
best_sarima_aic_max = float('inf')
best_sarima_order_max = None
best_sarima_seasonal_max = None
best_sarima_model_max = None
best_sarima_forecast_max = None

print("Testing Max Temp SARIMA configurations...")
for p in p_values:
    for d in d_values:
        for q in q_values:
            for P in P_values:
                for D in D_values:
                    for Q in Q_values:
                        order = (p, d, q)
                        seasonal_order = (P, D, Q, S)
                        
                        try:
                            aic, mae, rmse, model, forecast = evaluate_sarima_model(
                                train_max, test_max, order, seasonal_order
                            )
                            
                            sarima_results_max.append({
                                'order': order,
                                'seasonal_order': seasonal_order,
                                'aic': aic,
                                'mae': mae,
                                'rmse': rmse
                            })
                            
                            if aic < best_sarima_aic_max:
                                best_sarima_aic_max = aic
                                best_sarima_order_max = order
                                best_sarima_seasonal_max = seasonal_order
                                best_sarima_model_max = model
                                best_sarima_forecast_max = forecast
                            
                            print(f"Max Temp SARIMA{order}x{seasonal_order} - AIC: {aic:.2f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}")
                        except Exception as e:
                            continue

print(f"\n✓ Best Max Temp SARIMA model: SARIMA{best_sarima_order_max}x{best_sarima_seasonal_max}")
print(f"  AIC: {best_sarima_aic_max:.2f}")
if sarima_results_max:
    best_sarima_result_max = min(sarima_results_max, key=lambda x: x['aic'])
    print(f"  MAE: {best_sarima_result_max['mae']:.3f}°C")
    print(f"  RMSE: {best_sarima_result_max['rmse']:.3f}°C")

# Store best SARIMA models
best_models['sarima_min'] = {
    'order': best_sarima_order_min,
    'seasonal_order': best_sarima_seasonal_min,
    'model': best_sarima_model_min,
    'forecast': best_sarima_forecast_min,
    'aic': best_sarima_aic_min
}
best_models['sarima_max'] = {
    'order': best_sarima_order_max,
    'seasonal_order': best_sarima_seasonal_max,
    'model': best_sarima_model_max,
    'forecast': best_sarima_forecast_max,
    'aic': best_sarima_aic_max
}

print("\n" + "-"*60)
print("MODEL COMPARISON AND FORECASTING")
print("-"*60)

# Compare models
print("MODEL PERFORMANCE COMPARISON:")
print("="*40)

# Compare models for both temperature series
print("MIN TEMPERATURE MODEL COMPARISON:")
print("="*45)
if arima_results_min and sarima_results_min:
    best_arima_min = min(arima_results_min, key=lambda x: x['aic'])
    best_sarima_min = min(sarima_results_min, key=lambda x: x['aic'])
    
    print(f"Min Temp ARIMA {best_arima_min['order']}:")
    print(f"  AIC: {best_arima_min['aic']:.2f}")
    print(f"  MAE: {best_arima_min['mae']:.3f}°C")
    print(f"  RMSE: {best_arima_min['rmse']:.3f}°C")
    
    print(f"\nMin Temp SARIMA {best_sarima_min['order']}x{best_sarima_min['seasonal_order']}:")
    print(f"  AIC: {best_sarima_min['aic']:.2f}")
    print(f"  MAE: {best_sarima_min['mae']:.3f}°C")
    print(f"  RMSE: {best_sarima_min['rmse']:.3f}°C")
    
    # Determine best model for min temperature
    if best_arima_min['aic'] < best_sarima_min['aic']:
        best_min_overall = "ARIMA"
        best_min_model = best_models['arima_min']['model']
        best_min_forecast = best_models['arima_min']['forecast']
    else:
        best_min_overall = "SARIMA"
        best_min_model = best_models['sarima_min']['model']
        best_min_forecast = best_models['sarima_min']['forecast']
    
    print(f"\n✓ Best Min Temperature Model: {best_min_overall}")

print("\nMAX TEMPERATURE MODEL COMPARISON:")
print("="*45)
if arima_results_max and sarima_results_max:
    best_arima_max = min(arima_results_max, key=lambda x: x['aic'])
    best_sarima_max = min(sarima_results_max, key=lambda x: x['aic'])
    
    print(f"Max Temp ARIMA {best_arima_max['order']}:")
    print(f"  AIC: {best_arima_max['aic']:.2f}")
    print(f"  MAE: {best_arima_max['mae']:.3f}°C")
    print(f"  RMSE: {best_arima_max['rmse']:.3f}°C")
    
    print(f"\nMax Temp SARIMA {best_sarima_max['order']}x{best_sarima_max['seasonal_order']}:")
    print(f"  AIC: {best_sarima_max['aic']:.2f}")
    print(f"  MAE: {best_sarima_max['mae']:.3f}°C")
    print(f"  RMSE: {best_sarima_max['rmse']:.3f}°C")
    
    # Determine best model for max temperature
    if best_arima_max['aic'] < best_sarima_max['aic']:
        best_max_overall = "ARIMA"
        best_max_model = best_models['arima_max']['model']
        best_max_forecast = best_models['arima_max']['forecast']
    else:
        best_max_overall = "SARIMA"
        best_max_model = best_models['sarima_max']['model']
        best_max_forecast = best_models['sarima_max']['forecast']
    
    print(f"\n✓ Best Max Temperature Model: {best_max_overall}")

# Generate future forecasts for next winter season
print(f"\nGENERATING FUTURE FORECASTS:")
print("="*40)

forecast_horizon = 15  # Next complete winter season (15 weeks)

# Generate forecasts for both temperature series
forecast_data = {}

if 'best_min_model' in locals() and best_min_model:
    # Min temperature forecast
    if best_min_overall == "ARIMA":
        full_model_min = ARIMA(ts_min, order=best_models['arima_min']['order']).fit()
    else:
        full_model_min = SARIMAX(ts_min, 
                                order=best_models['sarima_min']['order'], 
                                seasonal_order=best_models['sarima_min']['seasonal_order']).fit(disp=False)
    
    future_forecast_min = full_model_min.forecast(steps=forecast_horizon)
    forecast_ci_min = full_model_min.get_forecast(steps=forecast_horizon).conf_int()
    
    forecast_data['min'] = {
        'forecast': future_forecast_min.values,
        'lower_ci': forecast_ci_min.iloc[:, 0].values,
        'upper_ci': forecast_ci_min.iloc[:, 1].values,
        'model': best_min_overall
    }

if 'best_max_model' in locals() and best_max_model:
    # Max temperature forecast
    if best_max_overall == "ARIMA":
        full_model_max = ARIMA(ts_max, order=best_models['arima_max']['order']).fit()
    else:
        full_model_max = SARIMAX(ts_max, 
                                order=best_models['sarima_max']['order'], 
                                seasonal_order=best_models['sarima_max']['seasonal_order']).fit(disp=False)
    
    future_forecast_max = full_model_max.forecast(steps=forecast_horizon)
    forecast_ci_max = full_model_max.get_forecast(steps=forecast_horizon).conf_int()
    
    forecast_data['max'] = {
        'forecast': future_forecast_max.values,
        'lower_ci': forecast_ci_max.iloc[:, 0].values,
        'upper_ci': forecast_ci_max.iloc[:, 1].values,
        'model': best_max_overall
    }

# Create comprehensive forecast dataframe
if forecast_data:
    last_week = ts_min.index[-1]
    future_weeks = list(range(last_week + 1, last_week + forecast_horizon + 1))
    
    forecast_df = pd.DataFrame({
        'Week_Number': future_weeks,
        'Season_Week': [(w-1) % 15 + 1 for w in future_weeks]
    })
    
    if 'min' in forecast_data:
        forecast_df['Min_Temp_Forecast'] = forecast_data['min']['forecast']
        forecast_df['Min_Temp_Lower_CI'] = forecast_data['min']['lower_ci']
        forecast_df['Min_Temp_Upper_CI'] = forecast_data['min']['upper_ci']
        
    if 'max' in forecast_data:
        forecast_df['Max_Temp_Forecast'] = forecast_data['max']['forecast']
        forecast_df['Max_Temp_Lower_CI'] = forecast_data['max']['lower_ci']
        forecast_df['Max_Temp_Upper_CI'] = forecast_data['max']['upper_ci']
    
    print(f"Forecasting next {forecast_horizon} weeks (complete winter season):")
    if 'min' in forecast_data:
        print(f"  Min Temperature using {forecast_data['min']['model']} model")
    if 'max' in forecast_data:
        print(f"  Max Temperature using {forecast_data['max']['model']} model")
    
    print("\nFORECAST RESULTS:")
    for i, row in forecast_df.iterrows():
        week_info = f"Season Week {int(row['Season_Week']):2d} (Overall Week {int(row['Week_Number']):3d}):"
        
        forecast_line = week_info
        if 'Min_Temp_Forecast' in row:
            forecast_line += f" Min: {row['Min_Temp_Forecast']:6.2f}°C [{row['Min_Temp_Lower_CI']:6.2f}, {row['Min_Temp_Upper_CI']:6.2f}]"
        if 'Max_Temp_Forecast' in row:
            forecast_line += f" Max: {row['Max_Temp_Forecast']:6.2f}°C [{row['Max_Temp_Lower_CI']:6.2f}, {row['Max_Temp_Upper_CI']:6.2f}]"
        
        print(forecast_line)
    
    # Save forecasts
    forecast_df.to_csv('temperature_forecasts.csv', index=False)
    print(f"\n✓ Comprehensive temperature forecasts saved to: temperature_forecasts.csv")

print("\n" + "-"*60)
print("VISUALIZATION AND RESULTS")
print("-"*60)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Original time series (both min and max)
axes[0, 0].plot(ts_min.index, ts_min.values, 'b-', linewidth=1, alpha=0.8, label='Min Temp')
axes[0, 0].plot(ts_max.index, ts_max.values, 'r-', linewidth=1, alpha=0.8, label='Max Temp')
axes[0, 0].axvline(x=train_min.index[-1], color='gray', linestyle='--', alpha=0.7, label='Train/Test Split')
axes[0, 0].set_title('Min & Max Temperature Time Series', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Temperature (°C)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# 2. Test predictions vs actual (both temperatures)
plot_added = False
if 'best_min_forecast' in locals() and best_min_forecast is not None:
    axes[0, 1].plot(test_min.index, test_min.values, 'b-', linewidth=2, label='Actual Min', alpha=0.8)
    axes[0, 1].plot(test_min.index, best_min_forecast, 'b--', linewidth=2, label=f'Min {best_min_overall}')
    plot_added = True

if 'best_max_forecast' in locals() and best_max_forecast is not None:
    axes[0, 1].plot(test_max.index, test_max.values, 'r-', linewidth=2, label='Actual Max', alpha=0.8)
    axes[0, 1].plot(test_max.index, best_max_forecast, 'r--', linewidth=2, label=f'Max {best_max_overall}')
    plot_added = True

if plot_added:
    axes[0, 1].set_title('Test Period Predictions (Min & Max)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
else:
    axes[0, 1].text(0.5, 0.5, 'No test forecasts available', ha='center', va='center', transform=axes[0, 1].transAxes)

# 3. Residuals for both temperature series
residuals_plotted = False
if 'best_min_forecast' in locals() and best_min_forecast is not None:
    residuals_min = test_min.values - best_min_forecast
    axes[1, 0].plot(test_min.index, residuals_min, 'b-', linewidth=1, alpha=0.8, label='Min Temp Residuals')
    residuals_plotted = True

if 'best_max_forecast' in locals() and best_max_forecast is not None:
    residuals_max = test_max.values - best_max_forecast
    axes[1, 0].plot(test_max.index, residuals_max, 'r-', linewidth=1, alpha=0.8, label='Max Temp Residuals')
    residuals_plotted = True

if residuals_plotted:
    axes[1, 0].axhline(y=0, color='gray', linestyle='-', alpha=0.7)
    axes[1, 0].set_title('Prediction Residuals (Min & Max)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Residual (°C)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
else:
    axes[1, 0].text(0.5, 0.5, 'No residuals available', ha='center', va='center', transform=axes[1, 0].transAxes)

# 4. Future forecasts with confidence intervals
if 'forecast_df' in locals():
    # Plot last 30 weeks of historical data (2 seasons)
    historical_tail_min = ts_min.tail(30)
    historical_tail_max = ts_max.tail(30)
    axes[1, 1].plot(historical_tail_min.index, historical_tail_min.values, 'b-', linewidth=2, label='Historical Min', alpha=0.8)
    axes[1, 1].plot(historical_tail_max.index, historical_tail_max.values, 'r-', linewidth=2, label='Historical Max', alpha=0.8)
    
    # Plot forecasts
    if 'Min_Temp_Forecast' in forecast_df.columns:
        axes[1, 1].plot(forecast_df['Week_Number'], forecast_df['Min_Temp_Forecast'], 'b--', linewidth=2, label='Min Forecast')
        axes[1, 1].fill_between(forecast_df['Week_Number'], 
                               forecast_df['Min_Temp_Lower_CI'], 
                               forecast_df['Min_Temp_Upper_CI'], 
                               alpha=0.2, color='blue', label='Min 95% CI')
    
    if 'Max_Temp_Forecast' in forecast_df.columns:
        axes[1, 1].plot(forecast_df['Week_Number'], forecast_df['Max_Temp_Forecast'], 'r--', linewidth=2, label='Max Forecast')
        axes[1, 1].fill_between(forecast_df['Week_Number'], 
                               forecast_df['Max_Temp_Lower_CI'], 
                               forecast_df['Max_Temp_Upper_CI'], 
                               alpha=0.2, color='red', label='Max 95% CI')
    
    axes[1, 1].axvline(x=ts_min.index[-1], color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
    axes[1, 1].set_title('Future Temperature Forecasts (Next Winter Season)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Temperature (°C)')
    axes[1, 1].set_xlabel('Week Number')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('temperature_forecasting_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: temperature_forecasting_results.png")

# Create ARIMA vs SARIMA comparison plots for both temperature series
if 'forecast_df' in locals():
    print("\nCreating ARIMA vs SARIMA comparison visualizations...")
    
    # Min Temperature Comparison
    if all(key in best_models for key in ['arima_min', 'sarima_min']):
        # Generate forecasts from both models for min temperature
        arima_full_model_min = ARIMA(ts_min, order=best_models['arima_min']['order']).fit()
        sarima_full_model_min = SARIMAX(ts_min, 
                                       order=best_models['sarima_min']['order'], 
                                       seasonal_order=best_models['sarima_min']['seasonal_order']).fit(disp=False)
        
        arima_forecast_min = arima_full_model_min.forecast(steps=forecast_horizon)
        sarima_forecast_min = sarima_full_model_min.forecast(steps=forecast_horizon)
        
        arima_ci_min = arima_full_model_min.get_forecast(steps=forecast_horizon).conf_int()
        sarima_ci_min = sarima_full_model_min.get_forecast(steps=forecast_horizon).conf_int()
        
        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot historical data
        historical_tail_min = ts_min.tail(30)
        historical_tail_max = ts_max.tail(30)
        future_weeks = list(range(ts_min.index[-1] + 1, ts_min.index[-1] + forecast_horizon + 1))
        
        # Min Temperature ARIMA
        ax1.plot(historical_tail_min.index, historical_tail_min.values, 'b-', linewidth=2, label='Historical Min', alpha=0.8)
        ax1.plot(future_weeks, arima_forecast_min, 'r-', linewidth=2, label='ARIMA Forecast')
        ax1.fill_between(future_weeks, arima_ci_min.iloc[:, 0], arima_ci_min.iloc[:, 1], 
                         alpha=0.3, color='red', label='95% CI')
        ax1.axvline(x=ts_min.index[-1], color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
        ax1.set_title(f'Min Temp ARIMA{best_models["arima_min"]["order"]}', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Temperature (°C)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Min Temperature SARIMA
        ax2.plot(historical_tail_min.index, historical_tail_min.values, 'b-', linewidth=2, label='Historical Min', alpha=0.8)
        ax2.plot(future_weeks, sarima_forecast_min, 'g-', linewidth=2, label='SARIMA Forecast')
        ax2.fill_between(future_weeks, sarima_ci_min.iloc[:, 0], sarima_ci_min.iloc[:, 1], 
                         alpha=0.3, color='green', label='95% CI')
        ax2.axvline(x=ts_min.index[-1], color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
        ax2.set_title(f'Min Temp SARIMA{best_models["sarima_min"]["order"]}x{best_models["sarima_min"]["seasonal_order"]}', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Temperature (°C)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Max Temperature Comparison
    if all(key in best_models for key in ['arima_max', 'sarima_max']):
        # Generate forecasts from both models for max temperature
        arima_full_model_max = ARIMA(ts_max, order=best_models['arima_max']['order']).fit()
        sarima_full_model_max = SARIMAX(ts_max, 
                                       order=best_models['sarima_max']['order'], 
                                       seasonal_order=best_models['sarima_max']['seasonal_order']).fit(disp=False)
        
        arima_forecast_max = arima_full_model_max.forecast(steps=forecast_horizon)
        sarima_forecast_max = sarima_full_model_max.forecast(steps=forecast_horizon)
        
        arima_ci_max = arima_full_model_max.get_forecast(steps=forecast_horizon).conf_int()
        sarima_ci_max = sarima_full_model_max.get_forecast(steps=forecast_horizon).conf_int()
        
        # Max Temperature ARIMA
        ax3.plot(historical_tail_max.index, historical_tail_max.values, 'r-', linewidth=2, label='Historical Max', alpha=0.8)
        ax3.plot(future_weeks, arima_forecast_max, 'r-', linewidth=2, label='ARIMA Forecast')
        ax3.fill_between(future_weeks, arima_ci_max.iloc[:, 0], arima_ci_max.iloc[:, 1], 
                         alpha=0.3, color='red', label='95% CI')
        ax3.axvline(x=ts_max.index[-1], color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
        ax3.set_title(f'Max Temp ARIMA{best_models["arima_max"]["order"]}', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Temperature (°C)')
        ax3.set_xlabel('Week Number')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Max Temperature SARIMA
        ax4.plot(historical_tail_max.index, historical_tail_max.values, 'r-', linewidth=2, label='Historical Max', alpha=0.8)
        ax4.plot(future_weeks, sarima_forecast_max, 'g-', linewidth=2, label='SARIMA Forecast')
        ax4.fill_between(future_weeks, sarima_ci_max.iloc[:, 0], sarima_ci_max.iloc[:, 1], 
                         alpha=0.3, color='green', label='95% CI')
        ax4.axvline(x=ts_max.index[-1], color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
        ax4.set_title(f'Max Temp SARIMA{best_models["sarima_max"]["order"]}x{best_models["sarima_max"]["seasonal_order"]}', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Temperature (°C)')
        ax4.set_xlabel('Week Number')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('arima_vs_sarima_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ ARIMA vs SARIMA comparison saved: arima_vs_sarima_comparison.png")
        
        # Create overlay comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Min Temperature Overlay
        ax1.plot(historical_tail_min.index, historical_tail_min.values, 'b-', linewidth=3, label='Historical Min', alpha=0.8)
        ax1.plot(future_weeks, arima_forecast_min, 'r-', linewidth=2, label=f'ARIMA Forecast', marker='o', markersize=4)
        ax1.plot(future_weeks, sarima_forecast_min, 'g-', linewidth=2, label=f'SARIMA Forecast', marker='s', markersize=4)
        ax1.fill_between(future_weeks, arima_ci_min.iloc[:, 0], arima_ci_min.iloc[:, 1], 
                        alpha=0.2, color='red', label='ARIMA 95% CI')
        ax1.fill_between(future_weeks, sarima_ci_min.iloc[:, 0], sarima_ci_min.iloc[:, 1], 
                        alpha=0.2, color='green', label='SARIMA 95% CI')
        ax1.axvline(x=ts_min.index[-1], color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
        ax1.set_title('Min Temperature: ARIMA vs SARIMA', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_xlabel('Week Number')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Max Temperature Overlay
        ax2.plot(historical_tail_max.index, historical_tail_max.values, 'r-', linewidth=3, label='Historical Max', alpha=0.8)
        ax2.plot(future_weeks, arima_forecast_max, 'r-', linewidth=2, label=f'ARIMA Forecast', marker='o', markersize=4)
        ax2.plot(future_weeks, sarima_forecast_max, 'g-', linewidth=2, label=f'SARIMA Forecast', marker='s', markersize=4)
        ax2.fill_between(future_weeks, arima_ci_max.iloc[:, 0], arima_ci_max.iloc[:, 1], 
                        alpha=0.2, color='red', label='ARIMA 95% CI')
        ax2.fill_between(future_weeks, sarima_ci_max.iloc[:, 0], sarima_ci_max.iloc[:, 1], 
                        alpha=0.2, color='green', label='SARIMA 95% CI')
        ax2.axvline(x=ts_max.index[-1], color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
        ax2.set_title('Max Temperature: ARIMA vs SARIMA', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_xlabel('Week Number')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('forecast_models_overlay.png', dpi=300, bbox_inches='tight')
        print("✓ Forecast overlay comparison saved: forecast_models_overlay.png")
        
        # Print numerical comparison
        print(f"\nFORECAST COMPARISON SUMMARY:")
        print("="*50)
        if 'arima_forecast_min' in locals():
            print(f"Min Temp ARIMA forecast range: {arima_forecast_min.min():.2f}°C to {arima_forecast_min.max():.2f}°C")
            print(f"Min Temp SARIMA forecast range: {sarima_forecast_min.min():.2f}°C to {sarima_forecast_min.max():.2f}°C")
            print(f"Min Temp avg difference: {abs(arima_forecast_min - sarima_forecast_min).mean():.3f}°C")
        if 'arima_forecast_max' in locals():
            print(f"Max Temp ARIMA forecast range: {arima_forecast_max.min():.2f}°C to {arima_forecast_max.max():.2f}°C")
            print(f"Max Temp SARIMA forecast range: {sarima_forecast_max.min():.2f}°C to {sarima_forecast_max.max():.2f}°C")
            print(f"Max Temp avg difference: {abs(arima_forecast_max - sarima_forecast_max).mean():.3f}°C")

# Create model comparison summary
print(f"\nFINAL SUMMARY:")
print("="*50)

# Summary for both temperature series
total_models = 0
if arima_results_min and sarima_results_min:
    min_models = len(arima_results_min) + len(sarima_results_min)
    total_models += min_models
    print(f"Min Temperature Models: {len(arima_results_min)} ARIMA + {len(sarima_results_min)} SARIMA = {min_models}")
    if 'best_min_overall' in locals():
        print(f"  Best Min Temp Model: {best_min_overall}")
        min_best_aic = min(best_arima_min['aic'], best_sarima_min['aic'])
        min_best_mae = min(best_arima_min['mae'], best_sarima_min['mae'])
        min_best_rmse = min(best_arima_min['rmse'], best_sarima_min['rmse'])
        print(f"  Best AIC: {min_best_aic:.2f}")
        print(f"  Best MAE: {min_best_mae:.3f}°C")
        print(f"  Best RMSE: {min_best_rmse:.3f}°C")

if arima_results_max and sarima_results_max:
    max_models = len(arima_results_max) + len(sarima_results_max)
    total_models += max_models
    print(f"\nMax Temperature Models: {len(arima_results_max)} ARIMA + {len(sarima_results_max)} SARIMA = {max_models}")
    if 'best_max_overall' in locals():
        print(f"  Best Max Temp Model: {best_max_overall}")
        max_best_aic = min(best_arima_max['aic'], best_sarima_max['aic'])
        max_best_mae = min(best_arima_max['mae'], best_sarima_max['mae'])
        max_best_rmse = min(best_arima_max['rmse'], best_sarima_max['rmse'])
        print(f"  Best AIC: {max_best_aic:.2f}")
        print(f"  Best MAE: {max_best_mae:.3f}°C")
        print(f"  Best RMSE: {max_best_rmse:.3f}°C")

print(f"\nTotal Models Evaluated: {total_models}")

print(f"\nForecasts generated: {forecast_horizon} weeks ahead (1 complete winter season)")
if 'forecast_df' in locals():
    if 'Min_Temp_Forecast' in forecast_df.columns:
        print(f"Min Temp forecast range: {forecast_df['Min_Temp_Forecast'].min():.1f}°C to {forecast_df['Min_Temp_Forecast'].max():.1f}°C")
    if 'Max_Temp_Forecast' in forecast_df.columns:
        print(f"Max Temp forecast range: {forecast_df['Max_Temp_Forecast'].min():.1f}°C to {forecast_df['Max_Temp_Forecast'].max():.1f}°C")

print(f"\nOutputs created:")
print(f"  - temperature_forecasts.csv: Dual temperature predictions for next winter season")
print(f"  - temperature_forecasting_results.png: Comprehensive visualization")
print(f"  - arima_vs_sarima_comparison.png: Side-by-side model comparison")
print(f"  - forecast_models_overlay.png: Overlaid forecast comparison")

print(f"\nKey insights:")
print(f"  - Dataset covers {weekly_temp['Season'].max()} winter seasons (15 weeks each)")
print(f"  - Separate optimal models identified for min and max temperatures")
print(f"  - Forecasts provide complete daily temperature range predictions")
print(f"  - Models account for different seasonal patterns in min vs max temperatures")

print("\n" + "="*80)
print("DUAL TEMPERATURE FORECASTING ANALYSIS COMPLETE")
print("="*80)