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
print("WINTER SEASON TEMPERATURE FORECASTING WITH ARIMA AND SARIMA")
print("="*80)

# Load the data
print("Loading comprehensive ski analysis data...")
df = pd.read_csv('comprehensive_ski_analysis.csv')

print(f"Dataset loaded: {len(df)} records")
print(f"Winter season data: Week {df['Week'].min()}-{df['Week'].max()} across {df['Year'].nunique()} years ({df['Year'].min()}-{df['Year'].max()})")
print(f"Resorts: {df['Resort'].nunique()}")
print(f"Total seasonal periods: {df['Year'].nunique()} winters × 15 weeks = {df['Year'].nunique() * 15} weeks")
print(f"Temperature range: {df['Avg_Min_Temp'].min():.1f}°C to {df['Avg_Min_Temp'].max():.1f}°C")

# Check for missing values
print(f"\nMissing values in Avg_Min_Temp: {df['Avg_Min_Temp'].isnull().sum()}")

# Display basic statistics
print("\nAvg_Min_Temp Statistics:")
print(df['Avg_Min_Temp'].describe())

print("\n" + "-"*60)
print("PREPARING TIME SERIES DATA")
print("-"*60)

# For winter season data, create a sequential time series
# Since we have 15 weeks per year across multiple years, we'll create a proper time series
weekly_temp = df.groupby(['Year', 'Week'])['Avg_Min_Temp'].mean().reset_index()
weekly_temp = weekly_temp.sort_values(['Year', 'Week']).reset_index(drop=True)

# Create a period index for seasonal data (15-week seasons)
weekly_temp['Season_Week'] = range(1, len(weekly_temp) + 1)
weekly_temp['Season'] = ((weekly_temp['Season_Week'] - 1) // 15) + 1

print(f"Time series prepared: {len(weekly_temp)} weekly observations")
print(f"Covering {weekly_temp['Season'].max()} complete winter seasons")
print(f"Season structure: 15 weeks × {weekly_temp['Season'].max()} years = {len(weekly_temp)} total weeks")

# Set up time series using sequential indexing
ts = pd.Series(weekly_temp['Avg_Min_Temp'].values, 
               index=pd.RangeIndex(start=1, stop=len(weekly_temp)+1, name='Week_Number'))

print(f"Average temperature across all seasons: {ts.mean():.2f}°C")
print(f"Temperature standard deviation: {ts.std():.2f}°C")

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

is_stationary = check_stationarity(ts, "Stationarity Test - Original Series")

# If not stationary, difference the series
if not is_stationary:
    ts_diff = ts.diff().dropna()
    is_diff_stationary = check_stationarity(ts_diff, "Stationarity Test - Differenced Series")
    
    # Check second difference if needed
    if not is_diff_stationary:
        ts_diff2 = ts_diff.diff().dropna()
        is_diff2_stationary = check_stationarity(ts_diff2, "Stationarity Test - Second Differenced Series")

# For seasonal data with limited observations, use different split strategy
# Use last 2 seasons (30 weeks) for testing, rest for training
test_seasons = 2
test_size = test_seasons * 15
train_size = len(ts) - test_size

print(f"\nTime series length: {len(ts)} weeks ({len(ts)//15} complete seasons)")
print(f"Training data: {train_size} weeks ({train_size//15} seasons + {train_size%15} weeks)")
print(f"Testing data: {test_size} weeks ({test_seasons} complete seasons)")
print(f"Split strategy: Using last {test_seasons} seasons for testing")

print("\n" + "-"*60)
print("ARIMA MODEL IMPLEMENTATION")
print("-"*60)

# Split data using the calculated sizes
train, test = ts[:train_size], ts[train_size:]

print(f"Training period: Week {train.index[0]} to Week {train.index[-1]}")
print(f"Testing period: Week {test.index[0]} to Week {test.index[-1]}")
print(f"Testing represents seasons {(test.index[0]-1)//15 + 1}-{(test.index[-1]-1)//15 + 1}")

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

best_aic = float('inf')
best_order = None
best_model = None
best_forecast = None

arima_results = []

for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p, d, q)
            try:
                aic, mae, rmse, model, forecast = evaluate_arima_model(train, test, order)
                arima_results.append({
                    'order': order,
                    'aic': aic,
                    'mae': mae,
                    'rmse': rmse
                })
                
                if aic < best_aic:
                    best_aic = aic
                    best_order = order
                    best_model = model
                    best_forecast = forecast
                    
                print(f"ARIMA{order} - AIC: {aic:.2f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}")
            except:
                continue

print(f"\n✓ Best ARIMA model: ARIMA{best_order}")
print(f"  AIC: {best_aic:.2f}")
print(f"  MAE: {arima_results[0]['mae']:.3f}°C" if arima_results else "N/A")
print(f"  RMSE: {arima_results[0]['rmse']:.3f}°C" if arima_results else "N/A")

# Show model summary
if best_model:
    print(f"\nARIMA Model Summary:")
    print(best_model.summary().tables[1])

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

print("Testing SARIMA configurations...")
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
                                train, test, order, seasonal_order
                            )
                            
                            sarima_results.append({
                                'order': order,
                                'seasonal_order': seasonal_order,
                                'aic': aic,
                                'mae': mae,
                                'rmse': rmse
                            })
                            
                            if aic < best_sarima_aic:
                                best_sarima_aic = aic
                                best_sarima_order = order
                                best_sarima_seasonal = seasonal_order
                                best_sarima_model = model
                                best_sarima_forecast = forecast
                            
                            print(f"SARIMA{order}x{seasonal_order} - AIC: {aic:.2f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}")
                        except Exception as e:
                            continue

print(f"\n✓ Best SARIMA model: SARIMA{best_sarima_order}x{best_sarima_seasonal}")
print(f"  AIC: {best_sarima_aic:.2f}")

# Find the best result for metrics
if sarima_results:
    best_sarima_result = min(sarima_results, key=lambda x: x['aic'])
    print(f"  MAE: {best_sarima_result['mae']:.3f}°C")
    print(f"  RMSE: {best_sarima_result['rmse']:.3f}°C")

# Show SARIMA model summary
if best_sarima_model:
    print(f"\nSARIMA Model Summary:")
    print(best_sarima_model.summary().tables[1])

print("\n" + "-"*60)
print("MODEL COMPARISON AND FORECASTING")
print("-"*60)

# Compare models
print("MODEL PERFORMANCE COMPARISON:")
print("="*40)

if arima_results and sarima_results:
    best_arima_result = min(arima_results, key=lambda x: x['aic'])
    best_sarima_result = min(sarima_results, key=lambda x: x['aic'])
    
    print(f"ARIMA {best_arima_result['order']}:")
    print(f"  AIC: {best_arima_result['aic']:.2f}")
    print(f"  MAE: {best_arima_result['mae']:.3f}°C")
    print(f"  RMSE: {best_arima_result['rmse']:.3f}°C")
    
    print(f"\nSARIMA {best_sarima_result['order']}x{best_sarima_result['seasonal_order']}:")
    print(f"  AIC: {best_sarima_result['aic']:.2f}")
    print(f"  MAE: {best_sarima_result['mae']:.3f}°C")
    print(f"  RMSE: {best_sarima_result['rmse']:.3f}°C")
    
    # Determine best overall model
    if best_arima_result['aic'] < best_sarima_result['aic']:
        overall_best = "ARIMA"
        best_overall_model = best_model
        best_overall_forecast = best_forecast
    else:
        overall_best = "SARIMA"
        best_overall_model = best_sarima_model
        best_overall_forecast = best_sarima_forecast
    
    print(f"\n✓ Best Overall Model: {overall_best}")

# Generate future forecasts for next winter season
print(f"\nGENERATING FUTURE FORECASTS:")
print("="*40)

forecast_horizon = 15  # Next complete winter season (15 weeks)

if best_overall_model:
    # Refit model on full dataset
    if overall_best == "ARIMA":
        full_model = ARIMA(ts, order=best_order).fit()
    else:
        full_model = SARIMAX(ts, order=best_sarima_order, seasonal_order=best_sarima_seasonal).fit(disp=False)
    
    # Generate forecast
    future_forecast = full_model.forecast(steps=forecast_horizon)
    forecast_ci = full_model.get_forecast(steps=forecast_horizon).conf_int()
    
    # Create future week numbers
    last_week = ts.index[-1]
    future_weeks = list(range(last_week + 1, last_week + forecast_horizon + 1))
    
    print(f"Forecasting next {forecast_horizon} weeks (complete winter season) using {overall_best} model:")
    
    forecast_df = pd.DataFrame({
        'Week_Number': future_weeks,
        'Season_Week': [(w-1) % 15 + 1 for w in future_weeks],
        'Forecast_Temp': future_forecast.values,
        'Lower_CI': forecast_ci.iloc[:, 0].values,
        'Upper_CI': forecast_ci.iloc[:, 1].values
    })
    
    print("\nFORECAST RESULTS:")
    for i, row in forecast_df.iterrows():
        print(f"Season Week {int(row['Season_Week']):2d} (Overall Week {int(row['Week_Number']):3d}): {row['Forecast_Temp']:6.2f}°C [{row['Lower_CI']:6.2f}, {row['Upper_CI']:6.2f}]")
    
    # Save forecasts
    forecast_df.to_csv('temperature_forecasts.csv', index=False)
    print(f"\n✓ Forecasts saved to: temperature_forecasts.csv")

print("\n" + "-"*60)
print("VISUALIZATION AND RESULTS")
print("-"*60)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Original time series
axes[0, 0].plot(ts.index, ts.values, 'b-', linewidth=1, alpha=0.8)
axes[0, 0].axvline(x=train.index[-1], color='red', linestyle='--', alpha=0.7, label='Train/Test Split')
axes[0, 0].set_title('Average Minimum Temperature Time Series', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Temperature (°C)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# 2. Test predictions vs actual
if best_overall_forecast is not None:
    axes[0, 1].plot(test.index, test.values, 'b-', linewidth=2, label='Actual', alpha=0.8)
    axes[0, 1].plot(test.index, best_overall_forecast, 'r--', linewidth=2, label=f'{overall_best} Forecast')
    axes[0, 1].set_title(f'{overall_best} Model: Test Period Predictions', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

# 3. Residuals
if best_overall_forecast is not None:
    residuals = test.values - best_overall_forecast
    axes[1, 0].plot(test.index, residuals, 'g-', linewidth=1, alpha=0.8)
    axes[1, 0].axhline(y=0, color='red', linestyle='-', alpha=0.7)
    axes[1, 0].set_title('Prediction Residuals', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Residual (°C)')
    axes[1, 0].grid(True, alpha=0.3)

# 4. Future forecasts with confidence intervals
if 'forecast_df' in locals():
    # Plot last 30 weeks of historical data (2 seasons)
    historical_tail = ts.tail(30)
    axes[1, 1].plot(historical_tail.index, historical_tail.values, 'b-', linewidth=2, label='Historical', alpha=0.8)
    
    # Plot forecasts
    axes[1, 1].plot(forecast_df['Week_Number'], forecast_df['Forecast_Temp'], 'r-', linewidth=2, label=f'{overall_best} Forecast')
    axes[1, 1].fill_between(forecast_df['Week_Number'], 
                           forecast_df['Lower_CI'], 
                           forecast_df['Upper_CI'], 
                           alpha=0.3, color='red', label='95% Confidence Interval')
    
    axes[1, 1].axvline(x=ts.index[-1], color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
    axes[1, 1].set_title('Future Temperature Forecast (Next Winter Season)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Temperature (°C)')
    axes[1, 1].set_xlabel('Week Number')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('temperature_forecasting_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: temperature_forecasting_results.png")

# Create ARIMA vs SARIMA comparison plot
if best_model and best_sarima_model and 'forecast_df' in locals():
    print("\nCreating ARIMA vs SARIMA comparison visualization...")
    
    # Generate forecasts from both models for comparison
    arima_full_model = ARIMA(ts, order=best_order).fit()
    sarima_full_model = SARIMAX(ts, order=best_sarima_order, seasonal_order=best_sarima_seasonal).fit(disp=False)
    
    arima_forecast = arima_full_model.forecast(steps=forecast_horizon)
    sarima_forecast = sarima_full_model.forecast(steps=forecast_horizon)
    
    arima_ci = arima_full_model.get_forecast(steps=forecast_horizon).conf_int()
    sarima_ci = sarima_full_model.get_forecast(steps=forecast_horizon).conf_int()
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot last 30 weeks of historical data
    historical_tail = ts.tail(30)
    future_weeks = list(range(ts.index[-1] + 1, ts.index[-1] + forecast_horizon + 1))
    
    # ARIMA forecast plot
    ax1.plot(historical_tail.index, historical_tail.values, 'b-', linewidth=2, label='Historical', alpha=0.8)
    ax1.plot(future_weeks, arima_forecast, 'r-', linewidth=2, label='ARIMA Forecast')
    ax1.fill_between(future_weeks, arima_ci.iloc[:, 0], arima_ci.iloc[:, 1], 
                     alpha=0.3, color='red', label='95% Confidence Interval')
    ax1.axvline(x=ts.index[-1], color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
    ax1.set_title(f'ARIMA{best_order} Forecast', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_xlabel('Week Number')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # SARIMA forecast plot
    ax2.plot(historical_tail.index, historical_tail.values, 'b-', linewidth=2, label='Historical', alpha=0.8)
    ax2.plot(future_weeks, sarima_forecast, 'g-', linewidth=2, label='SARIMA Forecast')
    ax2.fill_between(future_weeks, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1], 
                     alpha=0.3, color='green', label='95% Confidence Interval')
    ax2.axvline(x=ts.index[-1], color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
    ax2.set_title(f'SARIMA{best_sarima_order}x{best_sarima_seasonal} Forecast', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_xlabel('Week Number')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('arima_vs_sarima_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ ARIMA vs SARIMA comparison saved: arima_vs_sarima_comparison.png")
    
    # Create overlay comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot historical data
    ax.plot(historical_tail.index, historical_tail.values, 'b-', linewidth=3, label='Historical Data', alpha=0.8)
    
    # Plot both forecasts
    ax.plot(future_weeks, arima_forecast, 'r-', linewidth=2, label=f'ARIMA{best_order} Forecast', marker='o', markersize=4)
    ax.plot(future_weeks, sarima_forecast, 'g-', linewidth=2, label=f'SARIMA{best_sarima_order}x{best_sarima_seasonal} Forecast', marker='s', markersize=4)
    
    # Add confidence intervals with transparency
    ax.fill_between(future_weeks, arima_ci.iloc[:, 0], arima_ci.iloc[:, 1], 
                    alpha=0.2, color='red', label='ARIMA 95% CI')
    ax.fill_between(future_weeks, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1], 
                    alpha=0.2, color='green', label='SARIMA 95% CI')
    
    ax.axvline(x=ts.index[-1], color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
    ax.set_title('ARIMA vs SARIMA Temperature Forecasts Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_xlabel('Week Number', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('forecast_models_overlay.png', dpi=300, bbox_inches='tight')
    print("✓ Forecast overlay comparison saved: forecast_models_overlay.png")
    
    # Print numerical comparison
    print(f"\nFORECAST COMPARISON SUMMARY:")
    print("="*50)
    print(f"ARIMA{best_order} forecast range: {arima_forecast.min():.2f}°C to {arima_forecast.max():.2f}°C")
    print(f"SARIMA{best_sarima_order}x{best_sarima_seasonal} forecast range: {sarima_forecast.min():.2f}°C to {sarima_forecast.max():.2f}°C")
    print(f"Average difference: {abs(arima_forecast - sarima_forecast).mean():.3f}°C")
    print(f"Maximum difference: {abs(arima_forecast - sarima_forecast).max():.3f}°C")

# Create model comparison summary
print(f"\nFINAL SUMMARY:")
print("="*50)

if arima_results and sarima_results:
    print(f"Models evaluated: {len(arima_results)} ARIMA + {len(sarima_results)} SARIMA")
    print(f"Best model: {overall_best}")
    print(f"Best AIC: {min(best_arima_result['aic'], best_sarima_result['aic']):.2f}")
    print(f"Best MAE: {min(best_arima_result['mae'], best_sarima_result['mae']):.3f}°C")
    print(f"Best RMSE: {min(best_arima_result['rmse'], best_sarima_result['rmse']):.3f}°C")

print(f"\nForecasts generated: {forecast_horizon} weeks ahead (1 complete winter season)")
if 'forecast_df' in locals():
    print(f"Forecast range: {forecast_df['Forecast_Temp'].min():.1f}°C to {forecast_df['Forecast_Temp'].max():.1f}°C")

print(f"\nOutputs created:")
print(f"  - temperature_forecasts.csv: Future temperature predictions for next winter season")
print(f"  - temperature_forecasting_results.png: Comprehensive visualization")
print(f"  - arima_vs_sarima_comparison.png: Side-by-side model comparison")
print(f"  - forecast_models_overlay.png: Overlaid forecast comparison")

print(f"\nKey insights:")
print(f"  - Dataset covers {weekly_temp['Season'].max()} winter seasons (15 weeks each)")
print(f"  - Best model identified for 15-week seasonal temperature patterns")
print(f"  - Forecasts account for within-season temperature variations")

print("\n" + "="*80)
print("WINTER SEASON TEMPERATURE FORECASTING ANALYSIS COMPLETE")
print("="*80)