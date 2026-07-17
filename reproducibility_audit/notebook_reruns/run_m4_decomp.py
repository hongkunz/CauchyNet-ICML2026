import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import traceback
_n=[0]
def _show(*a,**k):
    plt.gcf().savefig(f'm4decomp_fig_{_n[0]:02d}.png', dpi=110, bbox_inches='tight'); _n[0]+=1; plt.close('all')
plt.show=_show

print('=== CELL 0 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from statsmodels.tsa.seasonal import MSTL
    import os
    from enum import Enum
    import pandas as pd
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.seasonal import seasonal_decompose, STL , MSTL
    pass
    pass
    pass
    import pandas as pd
    pass
    from statsforecast.models import SeasonalNaive
    from statsforecast.utils import ConformalIntervals
    
    
    INPUT_DIR = "input"
    Datasets = Enum('Datasets', 'Hourly Daily Monthly Quarterly')
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 0 FAILED, continuing', flush=True)

print('=== CELL 1 ===', flush=True)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style and context
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.5)
    
    # Set the color palette
    sns.set_palette("colorblind")
    
    # Configure Matplotlib parameters
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "lines.linewidth": 2,
        "grid.color": "lightblue",
        "grid.linestyle": "--",
        "grid.linewidth": 0.1
    })
    
    # Example plot
    x = range(10)
    y = [xi**2 for xi in x]
    
    plt.figure()
    plt.plot(x, y, label="Quadratic", color=sns.color_palette("colorblind")[0])
    plt.title("Sample Plot for Journal")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("sample_plot.png", dpi=300)
    
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 1 FAILED, continuing', flush=True)

print('=== CELL 2 ===', flush=True)
try:
    #url = "https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data-raw/vic_elec/VIC2015/demand.csv"
    #df = pd.read_csv(url)
    
    import pandas as pd
    import numpy as np
    from statsmodels.tsa.seasonal import seasonal_decompose, STL , MSTL
    pass
    pass
    pass
    
    # Input
    # The raw M4 hourly CSVs are NOT shipped with this repo (download the
    # M4 dataset, e.g. from github.com/Mcompetitions/M4-methods, and point
    # M4_DATA_DIR at a folder containing Train/ and Test/ subdirectories).
    _m4_root = os.environ.get("M4_DATA_DIR", "M4data")
    INPUT_DIR_train = os.path.join(_m4_root, "Train")
    INPUT_DIR_test = os.path.join(_m4_root, "Test")
    
    Datasets = Enum('Datasets', 'Hourly Daily Monthly Quarterly')
    
    def load_dataset(dataset):
        train_filepath = os.path.join(INPUT_DIR_train, dataset.name + "-train.csv")
        test_filepath = os.path.join(INPUT_DIR_test, dataset.name + "-test.csv")
        
        return pd.read_csv(train_filepath, sep=',', header=0, index_col=0, engine='python'), pd.read_csv(test_filepath, sep=',', header=0, index_col=0, engine='python')
    hourly_train, hourly_test = load_dataset(Datasets.Hourly)
    daily_train, daily_test = load_dataset(Datasets.Daily)
    monthly_train, monthly_test = load_dataset(Datasets.Monthly)
    quarterly_train, quarterly_test = load_dataset(Datasets.Quarterly)
    hourly_train.shape, hourly_test.shape
    
    print(hourly_train.head())
    # Lots of seasonality in there
    hourly_train.iloc[0:10, ].transpose().plot()
    hourly_test.iloc[0:10, ].transpose().plot()
    print(hourly_train.min().min(), hourly_train.max().max())
    print(hourly_train.iloc[0, ].first_valid_index(),  hourly_train.iloc[0, ].last_valid_index())
    
    def trim(df, index):
        """Return the time series at index, with the end NaN padding removed (not all M4 TS are the same length)."""
        s = df.iloc[index, ]
        return s.loc[:s.last_valid_index()]
    
    ht0 = trim(hourly_train, 0)
    ht0.head()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 2 FAILED, continuing', flush=True)

print('=== CELL 3 ===', flush=True)
try:
    # Let's try to define a proper period
    plt.plot(ht0)
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 3 FAILED, continuing', flush=True)

print('=== CELL 4 ===', flush=True)
try:
    def plot_components(result):
    
      df = pd.concat([result.observed, result.trend, result.seasonal, result.resid], axis=1)
      df = df.rename(columns={0:'Original Data', 'season':'seasonal','observed':'Original Data'})
      components = df.columns
      rows = len(components)
      fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, subplot_titles = [i for i in components])
      
    # Plot original data
      for i, col in enumerate(components):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col), row=i+1, col=1)
    
    
      # Update layout
      fig.update_layout(
          title='Time Series Decomposition',
          xaxis_title='Time',
          height=1200,
          width=1200
      )
    
      fig.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 4 FAILED, continuing', flush=True)

print('=== CELL 5 ===', flush=True)
try:
    def plot_components(result):
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.5)
        sns.set_palette("colorblind")
        
        plt.rcParams.update({
            "figure.figsize": (10, 12),
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "lines.linewidth": 2,
            "grid.color": "lightblue",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5
        })
        
        df = pd.concat([result.observed, result.trend, result.seasonal, result.resid], axis=1)
        df.columns = ['Original Data', 'Trend', 'Seasonal', 'Residual']
        
        components = df.columns
        rows = len(components)
        
        fig, axes = plt.subplots(rows, 1, sharex=True)
        
        for i, col in enumerate(components):
            axes[i].plot(df.index, df[col], label=col)
            axes[i].set_title(col)
            axes[i].grid(True)
            
        plt.suptitle('Time Series Decomposition', y=0.95)
        plt.xlabel('Time')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('M4_trend.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 5 FAILED, continuing', flush=True)

print('=== CELL 6 ===', flush=True)
try:
    data=ht0
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 6 FAILED, continuing', flush=True)

print('=== CELL 7 ===', flush=True)
try:
    stl_kwargs = {"seasonal_deg": 0} 
    model = MSTL(data, periods=(24, 24 * 7), stl_kwargs=stl_kwargs)
    res = model.fit()
    
    seasonal = res.seasonal # contains both seasonal components
    trend = res.trend
    residual = res.resid
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 7 FAILED, continuing', flush=True)

print('=== CELL 8 ===', flush=True)
try:
    # Already better (some seasonality extracted)
    # result = seasonal.seasonal_decompose(ht0, period=24, model='additive')
    # result.plot()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 8 FAILED, continuing', flush=True)

print('=== CELL 9 ===', flush=True)
try:
    period = 24 #setting the period for decomposition
    # Apply seasonal_decompose
    result_sd = seasonal_decompose(data, model='multiplicative', period=period, extrapolate_trend=1)
    
    # Plot the results
    plot_components(result_sd)
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 9 FAILED, continuing', flush=True)

print('=== CELL 10 ===', flush=True)
try:
    result_sd
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 10 FAILED, continuing', flush=True)

print('=== CELL 11 ===', flush=True)
try:
    stl = STL(data, period=period)
    result_stl = stl.fit()
    
    # Plot the results
    plot_components(result_stl)
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 11 FAILED, continuing', flush=True)

print('=== CELL 12 ===', flush=True)
try:
    mstl = MSTL(data, periods=24*7)
    result_mstl = mstl.fit()
    
    # Plot the results
    plot_components(result_mstl)
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 12 FAILED, continuing', flush=True)

print('=== CELL 13 ===', flush=True)
try:
    mstl = MSTL(data, periods=[24, 24*7])
    result_mstl = mstl.fit()
    
    # Plot the results
    plot_components(result_mstl)
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 13 FAILED, continuing', flush=True)

print('=== CELL 14 ===', flush=True)
try:
    mstl = MSTL(data, periods=[24, 160])
    result_mstl = mstl.fit()
    
    # Plot the results
    plot_components(result_mstl)
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 14 FAILED, continuing', flush=True)

print('=== CELL 15 ===', flush=True)
try:
    from statsmodels.tsa.seasonal import seasonal_decompose, STL , MSTL
    stl = STL(data, period=24)
    result_stl = stl.fit()
    result_stl.plot()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 15 FAILED, continuing', flush=True)

print('=== CELL 16 ===', flush=True)
try:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    # Perform exponential smoothing
    model = SimpleExpSmoothing(ht0)
    fitted_model = model.fit(smoothing_level=0.3, optimized=False) # Adjust smoothing level as needed
    trend_exp_smooth = fitted_model.fittedvalues
    residual_exp_smooth = ht0 - trend_exp_smooth
    # Visualize the components
    plt.figure(figsize=(10, 6))
    plt.subplot(4, 1, 1)
    plt.plot(ht0, label='Original Data')
    plt.title('Original Time Series')
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(trend_exp_smooth, label='Trend (Exp. Smoothing)', color='orange')
    plt.title('Trend Component (Exp. Smoothing)')
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(residual_exp_smooth, label='Residual (Exp. Smoothing)', color='green')
    plt.title('Residual Component (Exp. Smoothing)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 16 FAILED, continuing', flush=True)

print('=== CELL 17 ===', flush=True)
try:
    result.plot()
    plot_components(result)
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 17 FAILED, continuing', flush=True)

print('=== CELL 18 ===', flush=True)
try:
    ts=ht0
    from statsmodels.tsa.seasonal import seasonal_decompose
    # Perform multiplicative decomposition
    result = seasonal_decompose(ts,period =24, model='multiplicative')
    # Extract the components
    trend_mul = result.trend.dropna()
    seasonal_mul = result.seasonal.dropna()
    residual_mul = result.resid.dropna()
    #Let's visualise the results:
    # Visualize the components
    plt.figure(figsize=(10, 8))
    plt.subplot(4, 1, 1)
    plt.plot(ts, label='Original Data')
    plt.title('Original Time Series')
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(trend_mul, label='Trend (Multiplicative)', color='orange')
    plt.title('Trend Component (Multiplicative)')
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(seasonal_mul, label='Seasonal (Multiplicative)', color='green')
    plt.title('Seasonal Component (Multiplicative)')
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(residual_mul, label='Residual (Multiplicative)', color='red')
    plt.title('Residual Component (Multiplicative)')
    plt.legend()
    plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 18 FAILED, continuing', flush=True)

print('=== CELL 19 ===', flush=True)
try:
    from statsmodels.tsa.seasonal import STL
    # Perform STL decomposition
    stl_result = STL(ts, period=24, seasonal=13).fit()
    # Extract the components
    seasonal_stl = stl_result.seasonal
    trend_stl = stl_result.trend
    residual_stl = stl_result.resid
    stl_result.plot()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 19 FAILED, continuing', flush=True)

print('=== CELL 20 ===', flush=True)
try:
    def my_plot(series, title='Time Series Plot', xlabel='Index', ylabel='Trend Value', figsize=(12, 6)):
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.5)
        sns.set_palette("colorblind")
        
        plt.rcParams.update({
            "figure.figsize": figsize,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "lines.linewidth": 2,
            "grid.color": "lightgrey",  # Set grid color to light grey
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "axes.facecolor": "#D6EAF8"  # Set the background color to a slightly darker light blue
        })
        
        plt.plot(series.values)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        
        # Adjust the spines
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 20 FAILED, continuing', flush=True)

print('=== CELL 21 ===', flush=True)
try:
    my_plot(result_sd.trend, title='Trend Component', xlabel='Index', ylabel='Trend Value', figsize=(12, 6))
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 21 FAILED, continuing', flush=True)

print('=== CELL 22 ===', flush=True)
try:
    df= result_sd.trend
    df = df.drop(columns=['ds'])
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 22 FAILED, continuing', flush=True)

print('=== CELL 23 ===', flush=True)
try:
    import pandas as pd
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Define X as numerical indices starting from 0
    X = pd.Series(range(len(result_sd.trend)))
    
    # Define Y as the 'trend' values
    Y = result_sd.trend.values
    
    # Print X and Y to verify
    print("X:", X.shape)
    print("Y:", Y.shape)
    
    df = pd.DataFrame({'X': X, 'Y': Y})
    
    # Save the DataFrame to a CSV file
    df.to_csv('M4_trend_data.csv', index=False)
    
    print("Data saved to 'XY_trend_data.csv'")
    df.head()
    pass
except Exception:
    traceback.print_exc()
    print('!! CELL 23 FAILED, continuing', flush=True)
