#!/usr/bin/env python
# coding: utf-8

# In[103]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# More flexible data loading function
def load_cmapss_data(filename):
    """Load CMAPSS data with flexible column handling"""
    try:
        # Read the file first to see how many columns we have
        data = pd.read_csv(filename, sep='\s+', header=None)
        
        # Define column names based on actual number of columns
        n_cols = data.shape[1]
        print(f"Found {n_cols} columns in the data")
        
        # Standard CMAPSS structure: unit_id, time_cycle, 3 settings, 21 sensors
        if n_cols >= 26:
            columns = ['unit_id', 'time_cycle'] + [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]
            # If there are extra columns, name them as extra_1, extra_2, etc.
            if n_cols > 26:
                extra_cols = [f'extra_{i}' for i in range(1, n_cols - 25)]
                columns.extend(extra_cols)
        else:
            # If fewer columns, create generic names
            columns = ['unit_id', 'time_cycle'] + [f'col_{i}' for i in range(3, n_cols + 1)]
        
        # Assign column names (only use as many as we have)
        data.columns = columns[:n_cols]
        
        print(f"Columns assigned: {list(data.columns)}")
        return data
        
    except FileNotFoundError:
        print(f"File {filename} not found. Creating sample data for demonstration...")
        return create_sample_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating sample data for demonstration...")
        return create_sample_data()

def create_sample_data():
    """Create sample data that mimics CMAPSS structure"""
    np.random.seed(42)
    n_engines = 100
    max_cycles = 200
    
    data_list = []
    for engine_id in range(1, n_engines + 1):
        # Random lifecycle length for each engine
        lifecycle = np.random.randint(50, max_cycles)
        for cycle in range(1, lifecycle + 1):
            row = {
                'unit_id': engine_id,
                'time_cycle': cycle,
                'setting_1': np.random.normal(0, 0.1),
                'setting_2': np.random.normal(0, 0.1),
                'setting_3': np.random.normal(0, 0.1),
            }
            # Add sensor data with some degradation trend
            degradation_factor = cycle / lifecycle  # 0 to 1
            for sensor_num in range(1, 22):
                base_value = np.random.normal(500 + sensor_num * 50, 10)
                # Add degradation trend to some sensors
                if sensor_num in [1, 2, 3, 4, 7, 11, 12]:
                    trend = degradation_factor * np.random.normal(20, 5)
                    noise = np.random.normal(0, 2)
                    row[f'sensor_{sensor_num}'] = base_value + trend + noise
                else:
                    row[f'sensor_{sensor_num}'] = base_value + np.random.normal(0, 2)
            
            data_list.append(row)
    
    return pd.DataFrame(data_list)

# load the actual data
train_data = load_cmapss_data('train_FD001.txt')
test_data = load_cmapss_data('test_FD001.txt')

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print("\nFirst few rows:")
print(train_data.head())


# In[64]:


# Display basic information
print("DATASET OVERVIEW")
print(f"Number of engines: {train_data['unit_id'].nunique()}")
print(f"Total number of cycles: {len(train_data)}")
print(f"Average cycles per engine: {len(train_data) / train_data['unit_id'].nunique():.1f}")

# Get available sensor columns
sensor_cols = [col for col in train_data.columns if col.startswith('sensor_')]
setting_cols = [col for col in train_data.columns if col.startswith('setting_')]

print(f"Available sensors: {len(sensor_cols)}")
print(f"Available settings: {len(setting_cols)}")
print(f"Sensor columns: {sensor_cols[:5]}...")  # Show first 5

# Calculate engine lifecycles
engine_lifecycles = train_data.groupby('unit_id')['time_cycle'].max()
print(f"\nEngine lifecycle stats:")
print(f"  • Min cycles: {engine_lifecycles.min()}")
print(f"  • Max cycles: {engine_lifecycles.max()}")
print(f"  • Mean cycles: {engine_lifecycles.mean():.1f}")
print(f"  • Std cycles: {engine_lifecycles.std():.1f}")


# In[17]:


# Create the visualizations with flexible sensor handling
plt.style.use('default')  # More compatible style
fig = plt.figure(figsize=(20, 15))

# Plot 1: Engine Lifecycles Distribution
plt.subplot(2, 3, 1)
plt.hist(engine_lifecycles, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution of Engine Lifecycles', fontsize=14, fontweight='bold')
plt.xlabel('Number of Cycles to Failure')
plt.ylabel('Number of Engines')
plt.grid(True, alpha=0.3)

# Plot 2: Sensor readings over time for a single engine (using first available sensors)
plt.subplot(2, 3, 2)
engine_1 = train_data[train_data['unit_id'] == 1]
colors = ['blue', 'red', 'green', 'orange', 'purple']
for i, sensor in enumerate(sensor_cols[:5]):  # Plot first 5 sensors
    if sensor in engine_1.columns:
        plt.plot(engine_1['time_cycle'], engine_1[sensor], 
                label=sensor, linewidth=2, color=colors[i % len(colors)])
plt.title('Sensor Readings Over Time (Engine 1)', fontsize=14, fontweight='bold')
plt.xlabel('Time Cycle')
plt.ylabel('Sensor Value')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Sensor correlation heatmap (first 8 sensors for readability)
plt.subplot(2, 3, 3)
if len(sensor_cols) > 0:
    sensors_for_corr = sensor_cols[:8]  # Use first 8 sensors
    correlation_matrix = train_data[sensors_for_corr].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, cbar_kws={'shrink': 0.8}, fmt='.2f')
    plt.title('Sensor Correlation Matrix (First 8 Sensors)', fontsize=14, fontweight='bold')

# Plot 4: Add RUL and plot degradation
def add_rul(df):
    df_rul = df.copy()
    df_rul['RUL'] = df_rul.groupby('unit_id')['time_cycle'].transform('max') - df_rul['time_cycle']
    return df_rul

train_with_rul = add_rul(train_data)

plt.subplot(2, 3, 4)
if len(sensor_cols) > 0:
    fourth_sensor = sensor_cols[3]
    plt.scatter(train_with_rul['RUL'], train_with_rul[fourth_sensor], alpha=0.5, s=1)
    plt.title(f'{fourth_sensor} vs Remaining Useful Life', fontsize=14, fontweight='bold')
    plt.xlabel('Remaining Useful Life (cycles)')
    plt.ylabel(f'{fourth_sensor} Value')
    plt.grid(True, alpha=0.3)

# Plot 5: Multiple engines comparison
plt.subplot(2, 3, 5)
if len(sensor_cols) > 0:
    fourth_sensor = sensor_cols[3]
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    for i, engine_id in enumerate([1, 2, 3, 4, 5]):
        engine_data = train_data[train_data['unit_id'] == engine_id]
        if not engine_data.empty:
            plt.plot(engine_data['time_cycle'], engine_data[fourth_sensor], 
                     label=f'Engine {engine_id}', alpha=0.7, linewidth=1.5, color=colors[i])
    plt.title(f'{fourth_sensor} Degradation: Multiple Engines', fontsize=14, fontweight='bold')
    plt.xlabel('Time Cycle')
    plt.ylabel(f'{fourth_sensor} Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Plot 6: End-of-life patterns
plt.subplot(2, 3, 6)
if len(sensor_cols) > 0:
    fourth_sensor = sensor_cols[3]
    end_of_life_data = []
    for engine_id in train_data['unit_id'].unique()[:10]:
        engine_data = train_data[train_data['unit_id'] == engine_id]
        if len(engine_data) >= 30:  # Reduced from 50 to 30 for more data
            last_cycles = engine_data.tail(30).copy()
            last_cycles['cycles_to_failure'] = range(len(last_cycles), 0, -1)
            end_of_life_data.append(last_cycles)
    
    if end_of_life_data:
        combined_eol = pd.concat(end_of_life_data)
        avg_sensor_by_failure = combined_eol.groupby('cycles_to_failure')[fourth_sensor].mean()
        plt.plot(avg_sensor_by_failure.index, avg_sensor_by_failure.values, 
                 linewidth=3, color='red')
        plt.title(f'Avg {fourth_sensor} Approaching Failure', fontsize=14, fontweight='bold')
        plt.xlabel('Cycles to Failure')
        plt.ylabel(f'Average {fourth_sensor} Value')
        plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('NASA_CMAPSS_Data_Exploration.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n🎉 Analysis complete! Visualization saved as 'NASA_CMAPSS_Data_Exploration.png'")


# In[92]:


# Enhanced analysis focusing on engines near failure
print("\n" + "="*60)
print("DEEP DIVE: ENGINES APPROACHING FAILURE")
print("="*60)

# Look specifically at engines in their last 20% of life
def analyze_degradation_patterns(data):
    results = {}
    
    for engine_id in data['unit_id'].unique()[:10]:  # First 10 engines
        engine_data = data[data['unit_id'] == engine_id].copy()
        max_cycle = engine_data['time_cycle'].max()
        
        # Split into early life (first 50%) and late life (last 50%)
        early_life = engine_data[engine_data['time_cycle'] <= max_cycle * 0.5]
        late_life = engine_data[engine_data['time_cycle'] > max_cycle * 0.5]
        
        if len(early_life) > 0 and len(late_life) > 0:
            # Calculate changes in sensor readings for ALL 21 sensors
            sensor_changes = {}
            for sensor in sensor_cols:  # Changed from sensor_cols[:5] to sensor_cols (all 21)
                early_mean = early_life[sensor].mean()
                late_mean = late_life[sensor].mean()
                change_percent = ((late_mean - early_mean) / early_mean) * 100
                sensor_changes[sensor] = change_percent
            
            results[engine_id] = sensor_changes
    
    return results

degradation_analysis = analyze_degradation_patterns(train_data)

# Display the results
print("\nSensor value changes from early to late engine life:")
print("(Positive = increase, Negative = decrease)")
print("-" * 50)
if degradation_analysis:
    for sensor in sensor_cols:  # Changed from sensor_cols[:5] to sensor_cols (all 21)
        changes = [results[sensor] for results in degradation_analysis.values() if sensor in results]
        if changes:
            avg_change = np.mean(changes)
            print(f"{sensor:12}: {avg_change:+6.2f}% average change")

# Create an additional plot showing this analysis
plt.figure(figsize=(16, 10))  # Increased figure size for better visibility

# Plot 1: Sensor variance by engine lifecycle stage (using all sensors for variance calculation)
plt.subplot(2, 2, 1)
lifecycle_stages = []
sensor_variances = []

for engine_id in train_data['unit_id'].unique()[:20]:
    engine_data = train_data[train_data['unit_id'] == engine_id]
    max_cycle = engine_data['time_cycle'].max()
    
    # Divide lifecycle into quarters
    for stage, (start_pct, end_pct) in enumerate([(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]):
        stage_data = engine_data[
            (engine_data['time_cycle'] >= max_cycle * start_pct) & 
            (engine_data['time_cycle'] <= max_cycle * end_pct)
        ]
        if len(stage_data) > 5 and sensor_cols:  # Need enough data points
            # Calculate average variance across all sensors for this stage
            variance = np.mean([stage_data[sensor].var() for sensor in sensor_cols])
            lifecycle_stages.append(stage + 1)
            sensor_variances.append(variance)

if lifecycle_stages:
    plt.boxplot([sensor_variances[i::4] for i in range(4)], 
                labels=['Q1 (0-25%)', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Q4 (75-100%)'])
    plt.title('Average Sensor Variance by Lifecycle Stage (All 21 Sensors)')
    plt.ylabel('Average Sensor Variance')
    plt.xlabel('Engine Lifecycle Stage')

# Additional summary statistics for all sensors
print("SUMMARY STATISTICS FOR ALL 21 SENSORS")

if degradation_analysis:
    print("\nTop 10 sensors showing most significant changes:")
    print("-" * 50)
    sensor_avg_changes = {}
    for sensor in sensor_cols:
        changes = [results[sensor] for results in degradation_analysis.values() if sensor in results]
        if changes:
            sensor_avg_changes[sensor] = np.mean(changes)
    
    sorted_sensors = sorted(sensor_avg_changes.items(), key=lambda x: abs(x[1]), reverse=True)
    
    for i, (sensor, change) in enumerate(sorted_sensors[:10], 1):
        direction = "increases" if change > 0 else "decreases"
        print(f"{i:2d}. {sensor:12}: {direction} by {abs(change):6.2f}% on average")
        # Look at how the quarters were defined in the original code
for stage, (start_pct, end_pct) in enumerate([(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]):
    print(f"Stage {stage+1} (Q{stage+1}): {start_pct*100}% to {end_pct*100}% of max_cycle")
# Check a few engines to see what max_cycle means
for engine_id in train_data['unit_id'].unique()[:3]:
    engine_data = train_data[train_data['unit_id'] == engine_id]
    max_cycle = engine_data['time_cycle'].max()
    min_cycle = engine_data['time_cycle'].min()
    print(f"Engine {engine_id}: cycles {min_cycle} to {max_cycle}")
    
    # Check if there's RUL data to compare
    if 'RUL' in engine_data.columns:
        print(f"  RUL at start: {engine_data[engine_data['time_cycle']==min_cycle]['RUL'].iloc[0]}")
        print(f"  RUL at end: {engine_data[engine_data['time_cycle']==max_cycle]['RUL'].iloc[0]}")
# Check sensor_4 values in each quarter to see if they match the RUL plot pattern
for engine_id in train_data['unit_id'].unique()[:5]:
    engine_data = train_data[train_data['unit_id'] == engine_id]
    max_cycle = engine_data['time_cycle'].max()
    
    q1_data = engine_data[engine_data['time_cycle'] <= max_cycle * 0.25]
    q4_data = engine_data[engine_data['time_cycle'] > max_cycle * 0.75]
    
    q1_sensor4_mean = q1_data['sensor_4'].mean()
    q4_sensor4_mean = q4_data['sensor_4'].mean()
    
    print(f"Engine {engine_id}: Q1 sensor_4 = {q1_sensor4_mean:.1f}, Q4 sensor_4 = {q4_sensor4_mean:.1f}")


# In[93]:


print("INVESTIGATING THE FLAT LINE MYSTERY")
print("="*60)

# Check which sensors are actually flat vs. which show variation
sensor_analysis = {}
for sensor in sensor_cols:
    std_dev = train_data[sensor].std()
    min_val = train_data[sensor].min()
    max_val = train_data[sensor].max()
    range_val = max_val - min_val
    
    sensor_analysis[sensor] = {
        'std': std_dev,
        'range': range_val,
        'min': min_val,
        'max': max_val,
        'is_flat': std_dev < 0.01  # Very low variation
    }

print("Sensor Variation Analysis:")
print("-" * 40)
for sensor, stats in sensor_analysis.items():
    status = "FLAT" if stats['is_flat'] else "VARIES"
    print(f"{sensor:12}: {status} | Range: {stats['range']:.3f} | Std: {stats['std']:.3f}")

# Count how many are flat
flat_sensors = [s for s, stats in sensor_analysis.items() if stats['is_flat']]
varying_sensors = [s for s, stats in sensor_analysis.items() if not stats['is_flat']]

print(f"\nSUMMARY:")
print(f"   Flat sensors: {len(flat_sensors)}")
print(f"   Varying sensors: {len(varying_sensors)}")
print(f"   Flat sensor names: {flat_sensors}")


# In[94]:


print(f"\nFOCUSING ON SENSORS WITH ACTUAL VARIATION")
print("="*60)

if varying_sensors:
    # Re-plot using only the sensors that actually vary
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Distribution of varying sensors for one engine
    ax = axes[0, 0]
    engine_1 = train_data[train_data['unit_id'] == 1]
    for i, sensor in enumerate(varying_sensors[:4]):  # Plot first 4 varying sensors
        ax.plot(engine_1['time_cycle'], engine_1[sensor], 
                label=sensor, linewidth=2, alpha=0.8)
    ax.set_title('Varying Sensors Over Time (Engine 1)')
    ax.set_xlabel('Time Cycle')
    ax.set_ylabel('Sensor Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Compare multiple engines on most varying sensor
    ax = axes[0, 1]
    if varying_sensors:
        most_varying_sensor = max(varying_sensors, 
                                key=lambda s: sensor_analysis[s]['std'])
        print(f"Most varying sensor: {most_varying_sensor}")
        
        for engine_id in [1, 2, 3, 4, 5]:
            engine_data = train_data[train_data['unit_id'] == engine_id]
            ax.plot(engine_data['time_cycle'], engine_data[most_varying_sensor], 
                   label=f'Engine {engine_id}', alpha=0.7)
        ax.set_title(f'{most_varying_sensor}: Multiple Engines')
        ax.set_xlabel('Time Cycle')
        ax.set_ylabel('Sensor Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Sensor value distribution
    ax = axes[1, 0]
    if len(varying_sensors) >= 2:
        ax.hist(train_data[varying_sensors[0]], bins=30, alpha=0.7, 
               label=varying_sensors[0])
        ax.hist(train_data[varying_sensors[1]], bins=30, alpha=0.7, 
               label=varying_sensors[1])
        ax.set_title('Distribution of Sensor Values')
        ax.set_xlabel('Sensor Value')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    # Plot 4: Correlation between varying sensors
    ax = axes[1, 1]
    if len(varying_sensors) >= 2:
        ax.scatter(train_data[varying_sensors[2]], train_data[varying_sensors[3]], 
                  alpha=0.5, s=1)
        ax.set_xlabel(varying_sensors[2])
        ax.set_ylabel(varying_sensors[3])
        ax.set_title(f'Correlation: {varying_sensors[2]} vs {varying_sensors[3]}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Varying_Sensors_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
else:
    print("⚠️  WARNING: No sensors show significant variation!")
    print("   This suggests we might be looking at normalized or processed data")


# In[95]:


print(f"\nDATA QUALITY CHECK")
print("="*40)

# Check for suspicious patterns
print("Checking for data quality issues:")

# Are there any sensors that are EXACTLY the same across all readings?
exactly_constant = []
for sensor in sensor_cols:
    unique_values = train_data[sensor].nunique()
    if unique_values == 1:
        exactly_constant.append(sensor)

print(f"Exactly constant sensors: {exactly_constant}")

# Check if data is integer vs float (might indicate processing)
for sensor in sensor_cols[:5]:
    sample_values = train_data[sensor].head()
    has_decimals = any(val != int(val) for val in sample_values)
    print(f"{sensor}: {'Float values' if has_decimals else 'Integer values'}")

# Look at a few raw values
print(f"\nSample raw values from {sensor_cols[0]}:")
print(train_data[sensor_cols[0]].head(10).tolist())


# In[100]:


print("VARIATION SUMMARY")

# Categorize sensors by variation level
flat_sensors = []
subtle_sensors = []
varying_sensors = []

sensor_categories = {
    'sensor_1': 0.000000, 'sensor_2': 0.500053, 'sensor_3': 6.131150,
    'sensor_4': 9.000605, 'sensor_5': 0.000000, 'sensor_6': 0.001389,
    'sensor_7': 0.885092, 'sensor_8': 0.070985, 'sensor_9': 22.082880,
    'sensor_10': 0.000000, 'sensor_11': 0.267087, 'sensor_12': 0.737553,
    'sensor_13': 0.071919, 'sensor_14': 19.076176, 'sensor_15': 0.037505,
    'sensor_16': 0.000000, 'sensor_17': 1.548763, 'sensor_18': 0.000000,
    'sensor_19': 0.000000, 'sensor_20': 0.180746, 'sensor_21': 0.108251
}

for sensor, std in sensor_categories.items():
    if std == 0.0:
        flat_sensors.append(sensor)
    elif std < 1.0:
        subtle_sensors.append(sensor)
    else:
        varying_sensors.append(sensor)

print(f"Flat sensors (6): {flat_sensors}")
print(f"Subtle variation (9): {subtle_sensors}")
print(f"Clear variation (6): {varying_sensors}")


# In[101]:


# Define all sensors except the flat ones (1,5,10,16,18,19)
all_sensor_cols = [f'sensor_{i}' for i in range(1, 22)]  # sensor_1 to sensor_21
flat_sensors = ['sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
target_sensors = [sensor for sensor in all_sensor_cols if sensor not in flat_sensors]

print(f"Analyzing {len(target_sensors)} sensors:")
print(f"Target sensors: {target_sensors}")

# Calculate RUL for the dataset
def add_rul(df):
    df_rul = df.copy()
    df_rul['RUL'] = df_rul.groupby('unit_id')['time_cycle'].transform('max') - df_rul['time_cycle']
    return df_rul

train_with_rul = add_rul(train_data)

# Calculate number of rows and columns needed for subplot grid
n_sensors = len(target_sensors)
n_cols = 3  # 3 plot types per sensor
n_rows = n_sensors  # One row per sensor

# Create the large subplot grid
fig = plt.figure(figsize=(20, 4*n_sensors))  # Height scales with number of sensors

for sensor_idx, sensor in enumerate(target_sensors):
    
    # Plot 1: Sensor vs RUL scatter plot
    plt.subplot(n_rows, n_cols, sensor_idx*n_cols + 1)
    plt.scatter(train_with_rul['RUL'], train_with_rul[sensor], alpha=0.5, s=1)
    plt.title(f'{sensor} vs Remaining Useful Life', fontsize=10, fontweight='bold')
    plt.xlabel('Remaining Useful Life (cycles)')
    plt.ylabel(f'{sensor} Value')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Multiple engines over time (smoothed)
    plt.subplot(n_rows, n_cols, sensor_idx*n_cols + 2)
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    window_size = 10  # Adjust this for more/less smoothing

    for i, engine_id in enumerate([36,37,38,39,72]):
        engine_data = train_data[train_data['unit_id'] == engine_id]
        if not engine_data.empty and len(engine_data) > window_size:
            # Calculate rolling average to smooth the line
            smoothed_values = engine_data[sensor].rolling(window=window_size, center=True).mean()
            plt.plot(engine_data['time_cycle'], smoothed_values, 
                    label=f'Engine {engine_id}', alpha=0.7, linewidth=1.5, color=colors[i])
                 
    plt.title(f'{sensor} Degradation: Multiple Engines (Smoothed)', fontsize=10, fontweight='bold')
    plt.xlabel('Time Cycle')
    plt.ylabel(f'{sensor} Value')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Average approaching failure
    plt.subplot(n_rows, n_cols, sensor_idx*n_cols + 3)
    end_of_life_data = []
    for engine_id in train_data['unit_id'].unique()[:10]:  # First 10 engines
        engine_data = train_data[train_data['unit_id'] == engine_id]
        if len(engine_data) >= 30:  # Need at least 30 cycles
            last_cycles = engine_data.tail(300).copy()
            last_cycles['cycles_to_failure'] = range(len(last_cycles), 0, -1)
            end_of_life_data.append(last_cycles)
    
    if end_of_life_data:
        combined_eol = pd.concat(end_of_life_data)
        avg_sensor_by_failure = combined_eol.groupby('cycles_to_failure')[sensor].mean()
        plt.plot(avg_sensor_by_failure.index, avg_sensor_by_failure.values, 
                 linewidth=3, color='red')
        plt.title(f'Avg {sensor} Approaching Failure', fontsize=10, fontweight='bold')
        plt.xlabel('Cycles to Failure')
        plt.ylabel(f'Average {sensor} Value')
        plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('All_Varying_Sensors_Analysis.png', dpi=300, bbox_inches='tight')
plt.show()


# In[104]:


def classify_engine_patterns(sensor_name):
    """Classify engines by their degradation patterns"""
    engine_classifications = []
    
    for engine_id in train_data['unit_id'].unique()[:40]:  # First 20 engines
        engine_data = train_data[train_data['unit_id'] == engine_id]
        
        if len(engine_data) > 50:  # Need enough data
            # Calculate trend characteristics
            sensor_values = engine_data[sensor_name].values
            cycles = engine_data['time_cycle'].values
            
            # Linear trend
            slope = np.polyfit(cycles, sensor_values, 1)[0]
            
            # Variability (how erratic)
            smoothed = pd.Series(sensor_values).rolling(10, center=True).mean()
            variability = np.std(sensor_values - smoothed.fillna(method='bfill').fillna(method='ffill'))
            
            # Sudden change detection (last 20% vs first 80%)
            split_point = int(len(sensor_values) * 0.8)
            early_avg = np.mean(sensor_values[:split_point])
            late_avg = np.mean(sensor_values[split_point:])
            sudden_change = abs(late_avg - early_avg) / np.std(sensor_values)
            
            # Classify based on characteristics
            if sudden_change > 2.0:
                pattern = "Sudden Failure"
            elif variability > np.std(sensor_values) * 0.5:
                pattern = "Erratic"
            elif abs(slope) > np.std(sensor_values) / len(cycles) * 10:
                pattern = "Gradual Degrader"
            else:
                pattern = "Stable"
            
            engine_classifications.append({
                'engine_id': engine_id,
                'pattern': pattern,
                'lifecycle': len(engine_data),
                'slope': slope,
                'variability': variability,
                'sudden_change': sudden_change
            })
    
    return pd.DataFrame(engine_classifications)

# Run classification for most variable sensor
classifications = classify_engine_patterns('sensor_14')

# Display results
print("ENGINE PATTERN CLASSIFICATION:")
print("="*50)
pattern_counts = classifications['pattern'].value_counts()
for pattern, count in pattern_counts.items():
    avg_life = classifications[classifications['pattern'] == pattern]['lifecycle'].mean()
    print(f"{pattern:15}: {count:2d} engines | Avg lifecycle: {avg_life:.0f} cycles")

# Show examples of each pattern
print(f"\nEXAMPLES BY PATTERN:")
for pattern in pattern_counts.index:
    examples = classifications[classifications['pattern'] == pattern]['engine_id'].head(300).tolist()
    print(f"{pattern:15}: Engines {examples}")


# In[98]:


# Analyze engine differences
def analyze_engine_differences():
    engine_comparison = []
    
    for engine_id in train_data['unit_id'].unique()[:100]:
        engine_data = train_data[train_data['unit_id'] == engine_id]
        
        if len(engine_data) > 100:
            lifecycle = len(engine_data)
            
            # Initial sensor readings (health baseline)
            initial_readings = engine_data.head(10).mean()
            
            # Final sensor readings (failure signature) 
            final_readings = engine_data.tail(10).mean()
            
            # Calculate degradation rate for key sensors
            sensor_14_change = final_readings['sensor_14'] - initial_readings['sensor_14']
            sensor_9_change = final_readings['sensor_9'] - initial_readings['sensor_9']
            
            engine_comparison.append({
                'engine_id': engine_id,
                'lifecycle': lifecycle,
                'initial_s14': initial_readings['sensor_14'],
                'final_s14': final_readings['sensor_14'],
                's14_change': sensor_14_change,
                'initial_s9': initial_readings['sensor_9'], 
                'final_s9': final_readings['sensor_9'],
                's9_change': sensor_9_change
            })
    
    return pd.DataFrame(engine_comparison)

engine_diffs = analyze_engine_differences()

print("ENGINE DIFFERENCES ANALYSIS:")
print("="*60)
print(f"Lifecycle variation:")
print(f"  • Shortest life: {engine_diffs['lifecycle'].min()} cycles")
print(f"  • Longest life: {engine_diffs['lifecycle'].max()} cycles") 
print(f"  • Difference: {engine_diffs['lifecycle'].max() - engine_diffs['lifecycle'].min()} cycles!")

print(f"\nInitial sensor_14 readings (brand new engines):")
print(f"  • Lowest: {engine_diffs['initial_s14'].min():.1f}")
print(f"  • Highest: {engine_diffs['initial_s14'].max():.1f}")
print(f"  • Manufacturing variation: {engine_diffs['initial_s14'].max() - engine_diffs['initial_s14'].min():.1f}")

print(f"\nDegradation rates (sensor_14 change):")
print(f"  • Smallest change: {engine_diffs['s14_change'].min():.1f}")
print(f"  • Largest change: {engine_diffs['s14_change'].max():.1f}") 
print(f"  • Some engines degrade {abs(engine_diffs['s14_change'].max() / engine_diffs['s14_change'].min()):.1f}x faster!")


# In[99]:


def create_health_index(df, degradation_start=200):
    """Create HI based on your 200-cycle finding"""
    df_hi = df.copy()
    
    for unit in df['unit_id'].unique():  # Changed from 'unit' to 'unit_id'
        unit_data = df[df['unit_id'] == unit]
        max_cycle = unit_data['time_cycle'].max()  # Changed from 'cycle' to 'time_cycle'
        
        # Create RUL
        df_hi.loc[df_hi['unit_id'] == unit, 'RUL'] = max_cycle - df_hi.loc[df_hi['unit_id'] == unit, 'time_cycle']
        
        # Create HI based on findings
        df_hi.loc[df_hi['unit_id'] == unit, 'HI'] = np.where(
            df_hi.loc[df_hi['unit_id'] == unit, 'RUL'] >= degradation_start,
            1.0,  # Healthy
            df_hi.loc[df_hi['unit_id'] == unit, 'RUL'] / degradation_start  # Degrading
        )
    
    return df_hi

# Use it with current dataset
train_with_hi = create_health_index(train_data, degradation_start=200)

# Check the results
print("Health Index Created!")
print(f"Dataset shape: {train_with_hi.shape}")
print(f"New columns added: RUL, HI")
print("\nSample data:")
print(train_with_hi[['unit_id', 'time_cycle', 'RUL', 'HI']].head(10))

# Verify the health index logic
print(f"\nHealth Index Distribution:")
print(f"Healthy engines (HI = 1.0): {len(train_with_hi[train_with_hi['HI'] == 1.0]):,} data points")
print(f"Degrading engines (HI < 1.0): {len(train_with_hi[train_with_hi['HI'] < 1.0]):,} data points")

# Show example for one engine
engine_1 = train_with_hi[train_with_hi['unit_id'] == 1][['time_cycle', 'RUL', 'HI']].head(10)
print(f"\nExample - Engine 1:")
print(engine_1)


# In[87]:


if varying_sensors:
    # Re-plot using only the sensors that actually vary
    import numpy as np
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    if len(varying_sensors) >= 2:
        x_data = train_data[varying_sensors[0]]
        y_data = train_data[varying_sensors[1]]
        
        # Remove any NaN values
        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_clean = x_data[mask]
        y_clean = y_data[mask]
        
        # Create scatter plot
        ax.scatter(x_clean, y_clean, alpha=0.5, s=1, label='Data points')
        
        # Calculate correlation metrics
        correlation_coeff, p_value = stats.pearsonr(x_clean, y_clean)
        
        # Fit line of best fit
        X_reshaped = x_clean.values.reshape(-1, 1)
        reg = LinearRegression()
        reg.fit(X_reshaped, y_clean)
        y_pred = reg.predict(X_reshaped)
        
        # Calculate R-squared
        r_squared = r2_score(y_clean, y_pred)
        
        # Plot line of best fit
        # Sort for smooth line plotting
        sort_idx = np.argsort(x_clean)
        ax.plot(x_clean.iloc[sort_idx], y_pred[sort_idx], 
                color='red', linewidth=2, label='Line of best fit')
        
        # Set labels and title with correlation info
        ax.set_xlabel(varying_sensors[0])
        ax.set_ylabel(varying_sensors[1])
        ax.set_title(f'{varying_sensors[0]} vs {varying_sensors[1]}\n'
                    f'Correlation (r): {correlation_coeff:.3f}, '
                    f'R²: {r_squared:.3f}, '
                    f'p-value: {p_value:.3e}')
        
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add text box with interpretation
        interpretation = ""
        if abs(correlation_coeff) > 0.8:
            strength = "Very strong"
        elif abs(correlation_coeff) > 0.6:
            strength = "Strong"
        elif abs(correlation_coeff) > 0.4:
            strength = "Moderate"
        elif abs(correlation_coeff) > 0.2:
            strength = "Weak"
        else:
            strength = "Very weak"
            
        direction = "positive" if correlation_coeff > 0 else "negative"
        significance = "significant" if p_value < 0.05 else "not significant"
        
        textstr = f'{strength} {direction} correlation\n'
        textstr += f'Statistically {significance} (α=0.05)\n'
        textstr += f'{r_squared*100:.1f}% of variance explained'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('Varying_Sensors_Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


# In[89]:


if varying_sensors:
    import numpy as np
    from scipy import stats
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    import seaborn as sns
    
    # Option 1: Correlation matrix heatmap for all varying sensors
    if len(varying_sensors) > 2:
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Heatmap of all correlations
        correlation_matrix = train_data[varying_sensors].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[0,0], fmt='.2f', square=True)
        axes[0,0].set_title('Correlation Matrix - All Varying Sensors')
        
        # Function to plot correlation with stats
        def plot_correlation(ax, x_data, y_data, x_label, y_label, title_suffix=""):
            # Remove NaN values
            mask = ~(np.isnan(x_data) | np.isnan(y_data))
            x_clean = x_data[mask]
            y_clean = y_data[mask]
            
            if len(x_clean) < 2:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
                return
            
            # Scatter plot
            ax.scatter(x_clean, y_clean, alpha=0.6, s=2)
            
            # Calculate correlation metrics
            correlation_coeff, p_value = stats.pearsonr(x_clean, y_clean)
            
            # Fit line of best fit
            X_reshaped = x_clean.values.reshape(-1, 1)
            reg = LinearRegression()
            reg.fit(X_reshaped, y_clean)
            y_pred = reg.predict(X_reshaped)
            r_squared = r2_score(y_clean, y_pred)
            
            # Plot line
            sort_idx = np.argsort(x_clean)
            ax.plot(x_clean.iloc[sort_idx], y_pred[sort_idx], 
                   color='red', linewidth=2, alpha=0.8)
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f'{x_label} vs {y_label}{title_suffix}\nr={correlation_coeff:.3f}, R²={r_squared:.3f}, p={p_value:.2e}')
            ax.grid(True, alpha=0.3)
        
        # Plot pairwise correlations: 0vs1, 2vs3, 4vs5, etc.
        pair_idx = 0
        plot_positions = [(0,1), (1,0), (1,1)]
        
        for i in range(0, len(varying_sensors)-1, 2):
            if pair_idx >= len(plot_positions):
                break
                
            sensor1 = varying_sensors[i]
            sensor2 = varying_sensors[i+1] if i+1 < len(varying_sensors) else varying_sensors[i-1]
            
            pos = plot_positions[pair_idx]
            plot_correlation(axes[pos[0], pos[1]], 
                           train_data[sensor1], train_data[sensor2],
                           sensor1, sensor2, f" (Pair {pair_idx+1})")
            pair_idx += 1
        
        plt.tight_layout()
        plt.savefig('Comprehensive_Sensor_Correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    # Option 2: Create a summary table of ALL possible pairwise correlations
    print("\n=== ALL PAIRWISE CORRELATION SUMMARY ===")
    print(f"{'Sensor 1':<15} {'Sensor 2':<15} {'Correlation':<12} {'R-squared':<12} {'P-value':<12} {'Strength'}")
    print("-" * 80)
    
    correlation_results = []
    
    # Compare EVERY sensor with EVERY other sensor (avoiding duplicates)
    for i in range(len(varying_sensors)):
        for j in range(i+1, len(varying_sensors)):  # j starts at i+1 to avoid duplicates and self-correlation
            sensor1 = varying_sensors[i]
            sensor2 = varying_sensors[j]
            
            x_data = train_data[sensor1]
            y_data = train_data[sensor2]
            
            # Remove NaN values
            mask = ~(np.isnan(x_data) | np.isnan(y_data))
            x_clean = x_data[mask]
            y_clean = y_data[mask]
            
            if len(x_clean) >= 2:
                corr_coeff, p_val = stats.pearsonr(x_clean, y_clean)
                
                # Calculate R²
                X_reshaped = x_clean.values.reshape(-1, 1)
                reg = LinearRegression()
                reg.fit(X_reshaped, y_clean)
                y_pred = reg.predict(X_reshaped)
                r_squared = r2_score(y_clean, y_pred)
                
                # Determine strength
                if abs(corr_coeff) > 0.8:
                    strength = "Very Strong"
                elif abs(corr_coeff) > 0.6:
                    strength = "Strong"
                elif abs(corr_coeff) > 0.4:
                    strength = "Moderate"
                elif abs(corr_coeff) > 0.2:
                    strength = "Weak"
                else:
                    strength = "Very Weak"
                    
                correlation_results.append({
                    'sensor1': sensor1, 'sensor2': sensor2, 'correlation': corr_coeff, 
                    'r_squared': r_squared, 'p_value': p_val, 'strength': strength
                })
    
    # Sort by absolute correlation strength (strongest first)
    correlation_results.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    # Print top 20 strongest correlations (or all if fewer than 20)
    top_n = min(20, len(correlation_results))
    print(f"\nTOP {top_n} STRONGEST CORRELATIONS:")
    print("-" * 80)
    
    for result in correlation_results[:top_n]:
        print(f"{result['sensor1']:<15} {result['sensor2']:<15} {result['correlation']:<12.3f} "
              f"{result['r_squared']:<12.3f} {result['p_value']:<12.2e} {result['strength']}")
    
    # Summary statistics
    if correlation_results:
        strongest = correlation_results[0]  # Already sorted by strength
        total_pairs = len(correlation_results)
        
        # Count by strength categories
        very_strong = sum(1 for r in correlation_results if abs(r['correlation']) > 0.8)
        strong = sum(1 for r in correlation_results if 0.6 < abs(r['correlation']) <= 0.8)
        moderate = sum(1 for r in correlation_results if 0.4 < abs(r['correlation']) <= 0.6)
        weak = sum(1 for r in correlation_results if 0.2 < abs(r['correlation']) <= 0.4)
        very_weak = sum(1 for r in correlation_results if abs(r['correlation']) <= 0.2)
        
        print(f"\n=== CORRELATION SUMMARY ===")
        print(f"Strongest correlation: {strongest['sensor1']} vs {strongest['sensor2']} (r = {strongest['correlation']:.3f})")
        print(f"Total unique sensor pairs analyzed: {total_pairs}")
        print(f"Very Strong (|r| > 0.8): {very_strong}")
        print(f"Strong (0.6 < |r| ≤ 0.8): {strong}")
        print(f"Moderate (0.4 < |r| ≤ 0.6): {moderate}")
        print(f"Weak (0.2 < |r| ≤ 0.4): {weak}")
        print(f"Very Weak (|r| ≤ 0.2): {very_weak}")
        
        # Calculate total possible pairs
        n_sensors = len(varying_sensors)
        total_possible = (n_sensors * (n_sensors - 1)) // 2
        print(f"Total possible unique pairs from {n_sensors} sensors: {total_possible}")
        
        if total_pairs < total_possible:
            print(f"Note: {total_possible - total_pairs} pairs had insufficient data for analysis")
    
else:
    print("No varying sensors found for correlation analysis.")


# In[ ]:




