"""
Complete RUL Prediction Pipeline with Improved LLM Integration
- Random Forest Baseline
- LLM Hybrid Adjustment (recommended approach)
- Comprehensive comparison and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import time
import pickle
import os
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = "."
WINDOW_SIZE = 30
CONSTANT_THRESHOLD = 0.01
RANDOM_STATE = 42

# OpenAI Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_and_preprocess_data(filepath):
    """Load and preprocess C-MAPSS data"""
    print(f"Loading data from {filepath}...")
    data = pd.read_csv(filepath, sep=r'\s+', header=None)
    
    if data.shape[1] > 26:
        data = data.iloc[:, :26]
    
    column_names = ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + \
                   [f'sensor_{i}' for i in range(1, 22)]
    data.columns = column_names
    
    print(f"  Loaded {len(data)} rows, {data['engine_id'].nunique()} engines")
    return data


def find_constant_sensors(data, threshold=0.01):
    """Find sensors that have very little variation"""
    sensor_cols = [col for col in data.columns if col.startswith('sensor_')]
    constant_sensors = []
    
    print("\nAnalyzing sensor variability...")
    for sensor in sensor_cols:
        sensor_std = data[sensor].std()
        sensor_range = data[sensor].max() - data[sensor].min()
        
        if sensor_std < threshold or sensor_range < threshold:
            constant_sensors.append(sensor)
            print(f"  {sensor}: std={sensor_std:.6f}, range={sensor_range:.6f} -> REMOVING")
    
    return constant_sensors


def create_rul_windows(engine_data, window_size=30, sensor_cols=None):
    """Create sliding windows for an engine"""
    windows = []
    targets = []
    
    if len(engine_data) < window_size:
        return np.array([]), np.array([])
    
    if sensor_cols is None:
        sensor_cols = [col for col in engine_data.columns if col.startswith('sensor_')]
    
    for i in range(len(engine_data) - window_size + 1):
        window = engine_data[sensor_cols].iloc[i:i+window_size].values
        windows.append(window.flatten())
        target = engine_data['RUL'].iloc[i+window_size-1]
        targets.append(target)
    
    return np.array(windows), np.array(targets)


def compute_s_score(rul_true, rul_pred):
    """Calculate S-score metric for RUL prediction"""
    diff = rul_pred - rul_true
    return np.sum(np.where(diff < 0, np.exp(-diff/13)-1, np.exp(diff/10)-1))


def slope_1d(y):
    """Calculate slope per cycle using least squares"""
    n = len(y)
    x = np.arange(n)
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return m


def summarize_block(block, cols):
    """Compute summary statistics per column over the window"""
    block = np.asarray(block)
    means = np.nanmean(block, axis=0)
    stds = np.nanstd(block, axis=0)
    mins = np.nanmin(block, axis=0)
    maxs = np.nanmax(block, axis=0)
    slopes = np.array([slope_1d(block[:, j]) for j in range(block.shape[1])])
    
    summary = pd.DataFrame({
        "feature": cols,
        "mean": means,
        "std": stds,
        "min": mins,
        "max": maxs,
        "slope": slopes
    })
    return summary

def make_rich_llm_summary(engine_data, remaining_sensors, rf_prediction):
    """Generate rich summary text for LLM adjustment with more context"""
    # Last 30 cycles
    last_block = engine_data[remaining_sensors].iloc[-WINDOW_SIZE:].values
    summary = summarize_block(last_block, remaining_sensors)

    # Health Index (simple heuristic)
    rul_est = engine_data['RUL'].iloc[-1] if 'RUL' in engine_data else np.nan
    HI = 1.0 if pd.notnull(rul_est) and rul_est >= 200 else (rul_est / 200.0 if pd.notnull(rul_est) else 0.5)

    # Settings (last known values)
    settings = engine_data[['setting_1','setting_2','setting_3']].iloc[-1].to_dict()

    # Trend comparison: last 10 vs previous 20 cycles
    if len(engine_data) >= 30:
        recent = engine_data[remaining_sensors].iloc[-10:].mean()
        prior = engine_data[remaining_sensors].iloc[-30:-10].mean()
        pct_change = ((recent - prior) / (prior.abs() + 1e-6)) * 100
        top_changes = pct_change.abs().sort_values(ascending=False).head(3)
    else:
        top_changes = {}

    # Correlation between key sensors (example: S14 & S9)
    corr = None
    if 'sensor_14' in engine_data.columns and 'sensor_9' in engine_data.columns:
        corr = engine_data[['sensor_14','sensor_9']].corr().iloc[0,1]

    # Build structured text
    max_cycle = engine_data['cycle'].max()
    summary_text = (
        f"Engine {engine_data['engine_id'].iloc[0]} | "
        f"Cycle {max_cycle} | HI={HI:.2f}\n"
        f"RF predicts {rf_prediction:.1f} cycles\n\n"
        f"Settings: s1={settings['setting_1']:.2f}, "
        f"s2={settings['setting_2']:.2f}, "
        f"s3={settings['setting_3']:.2f}\n"
        f"Trends (last 30 cycles):\n"
    )
    for sensor, change in top_changes.items():
        slope_val = summary.loc[summary['feature']==sensor,'slope'].values[0]
        summary_text += f"  - {sensor}: slope={slope_val:+.3f}, Δ={change:+.1f}%\n"
    if corr is not None:
        summary_text += f"Cross-signals: Corr(S14,S9)={corr:+.2f}\n"

    return summary_text

def make_llm_summary_text(summary_df, top_k=6):
    """Create human-readable summary emphasizing largest anomalies/trends"""
    s = summary_df.copy()
    s["z_abs_mean"] = s["mean"].abs()
    s["abs_slope"] = s["slope"].abs()
    s["rank_score"] = 0.6 * s["z_abs_mean"] + 0.4 * s["abs_slope"]
    s = s.sort_values("rank_score", ascending=False)
    picks = s.head(top_k)
    
    parts = []
    for _, r in picks.iterrows():
        name = r["feature"].replace("setting_", "Set").replace("sensor_", "S")
        parts.append(
            f"{name}: z-mean={r['mean']:+.2f}, slope={r['slope']:+.03f}/cycle, σ={r['std']:.2f}"
        )
    return "; ".join(parts)


# =============================================================================
# BASELINE RANDOM FOREST PIPELINE
# =============================================================================

def run_baseline_pipeline():
    """Execute the complete Random Forest baseline pipeline"""
    
    print("=" * 60)
    print("C-MAPSS BASELINE PIPELINE")
    print("=" * 60)
    
    # 1. Load training data
    print("\n[1/9] Loading training data...")
    train_data = load_and_preprocess_data('train_FD001.txt')
    
    # 2. Remove constant sensors
    print("\n[2/9] Removing constant sensors...")
    constant_sensors = find_constant_sensors(train_data, threshold=CONSTANT_THRESHOLD)
    
    if constant_sensors:
        print(f"  Removing {len(constant_sensors)} sensors")
        train_data = train_data.drop(columns=constant_sensors)
    
    remaining_sensors = [col for col in train_data.columns if col.startswith('sensor_')]
    print(f"  Remaining: {len(remaining_sensors)} sensors")
    
    # 3. Create RUL labels
    print("\n[3/9] Creating RUL labels...")
    max_life = train_data.groupby('engine_id')['cycle'].max().reset_index()
    max_life.columns = ['engine_id', 'max_life']
    train_data = train_data.merge(max_life, on='engine_id')
    train_data['RUL'] = train_data['max_life'] - train_data['cycle']
    train_data['RUL'] = train_data['RUL'].clip(upper=125)
    print(f"  RUL range: {train_data['RUL'].min()} to {train_data['RUL'].max()}")
    
    # 4. Load test data
    print("\n[4/9] Loading test data...")
    test_data = load_and_preprocess_data('test_FD001.txt')
    if constant_sensors:
        test_data = test_data.drop(columns=constant_sensors)
    
    true_rul = pd.read_csv('RUL_FD001.txt', header=None, names=['true_RUL'])
    true_rul['engine_id'] = range(1, len(true_rul) + 1)
    
    # 5. Scale sensor data
    print("\n[5/9] Scaling sensor data...")
    scale_columns = ['setting_1', 'setting_2', 'setting_3'] + remaining_sensors
    scaler = StandardScaler()
    
    train_data[scale_columns] = scaler.fit_transform(train_data[scale_columns])
    test_data[scale_columns] = scaler.transform(test_data[scale_columns])
    
    # 6. Create training windows
    print("\n[6/9] Creating training windows...")
    train_windows = []
    train_targets = []
    
    for engine_id in train_data['engine_id'].unique():
        engine_data = train_data[train_data['engine_id'] == engine_id]
        windows, targets = create_rul_windows(
            engine_data, window_size=WINDOW_SIZE, sensor_cols=remaining_sensors
        )
        if len(windows) > 0:
            train_windows.append(windows)
            train_targets.append(targets)
    
    X_train = np.vstack(train_windows)
    y_train = np.concatenate(train_targets)
    print(f"  Training windows: {X_train.shape}")
    
    # 7. Create test windows
    print("\n[7/9] Creating test windows...")
    test_windows = []
    test_engine_ids = []
    
    for engine_id in test_data['engine_id'].unique():
        engine_data = test_data[test_data['engine_id'] == engine_id]
        
        if len(engine_data) >= WINDOW_SIZE:
            last_window = engine_data[remaining_sensors].iloc[-WINDOW_SIZE:].values
            test_windows.append(last_window.flatten())
            test_engine_ids.append(engine_id)
    
    X_test = np.array(test_windows)
    test_engine_ids = np.array(test_engine_ids)
    y_test = true_rul[true_rul['engine_id'].isin(test_engine_ids)]['true_RUL'].values
    
    print(f"  Test windows: {X_test.shape}")
    
    # 8. Train Random Forest
    print("\n[8/9] Training Random Forest...")
    start_time = time.time()
    
    rf_model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=1,
        max_features='sqrt',
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    
    rf_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"  Training completed in {train_time:.2f}s")
    
    # 9. Evaluate
    print("\n[9/9] Evaluating model...")
    y_pred = rf_model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    s_score = compute_s_score(y_test, y_pred)
    
    # Print results
    print("\n" + "=" * 60)
    print("RANDOM FOREST BASELINE RESULTS")
    print("=" * 60)
    print(f"R² Score: {r2:.4f} ({r2*100:.2f}%)")
    print(f"RMSE: {rmse:.2f} cycles")
    print(f"MAE: {mae:.2f} cycles")
    print(f"S-Score: {s_score:.2f}")
    
    # Save results
    print("\nSaving results...")
    np.save('X_train_improved.npy', X_train)
    np.save('y_train_improved.npy', y_train)
    np.save('X_test_improved.npy', X_test)
    np.save('y_test_improved.npy', y_test)
    np.save('y_pred_improved.npy', y_pred)
    np.save('test_engine_ids_improved.npy', test_engine_ids)
    np.save('remaining_sensors.npy', np.array(remaining_sensors))
    
    with open('sensor_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return {
        'train_data': train_data,
        'test_data': test_data,
        'scaler': scaler,
        'remaining_sensors': remaining_sensors,
        'rf_model': rf_model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'test_engine_ids': test_engine_ids,
        'metrics': {'r2': r2, 'rmse': rmse, 'mae': mae, 's_score': s_score}
    }


# =============================================================================
# LLM SUMMARY GENERATION
# =============================================================================

def generate_llm_summaries(test_data, scaler, remaining_sensors, rf_predictions):
    """Generate richer sensor summaries for each test engine"""
    
    print("\n" + "=" * 60)
    print("GENERATING LLM SUMMARIES (RICH CONTEXT)")
    print("=" * 60)
    
    # Scale test data
    test_data_scaled = test_data.copy()
    scale_cols = ['setting_1', 'setting_2', 'setting_3'] + remaining_sensors
    test_data_scaled[scale_cols] = scaler.transform(test_data[scale_cols])
    
    # Generate summaries
    rows = []
    for i, eng in enumerate(sorted(test_data_scaled['engine_id'].unique())):
        engine_data = test_data_scaled[test_data_scaled['engine_id'] == eng]
        
        if len(engine_data) < WINDOW_SIZE:
            continue
        
        text = make_rich_llm_summary(engine_data, remaining_sensors, rf_predictions[i])
        
        rows.append({
            'engine_id': eng,
            'llm_summary': text
        })
    
    summ_df = pd.DataFrame(rows).set_index('engine_id').sort_index()
    print(f"Generated summaries for {len(summ_df)} engines")
    
    return summ_df


# =============================================================================
# IMPROVED LLM INTEGRATION - HYBRID APPROACH
# =============================================================================

def setup_openai_client():
    """Setup OpenAI client with error checking"""
    
    if not OPENAI_API_KEY:
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
        )
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("✓ OpenAI client initialized successfully")
        return client
    except ImportError:
        raise ImportError("OpenAI package not installed. Install it with: pip install openai>=1.40.0")


def get_llm_adjustment(client, engine_id, summary_text, rf_prediction, max_retries=3):
    """
    LLM provides adjustment factor to RF prediction
    Much more reliable than absolute prediction
    """
    
    adjustment_schema = {
        "name": "rul_adjustment",
        "schema": {
            "type": "object",
            "properties": {
                "adjustment_factor": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 1.5,
                    "description": "Multiplier for RF prediction (0.5-1.5). <1 = higher risk, >1 = lower risk"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in this adjustment"
                },
                "rationale": {
                    "type": "string",
                    "maxLength": 400,
                    "description": "Brief explanation focusing on anomalies"
                }
            },
            "required": ["adjustment_factor", "confidence", "rationale"],
            "additionalProperties": False
        },
        "strict": True
    }
    
    system_prompt = """You are an expert in turbofan engine diagnostics.

Your task: Review sensor trends and determine if the Random Forest prediction needs adjustment.

Guidelines:
- adjustment_factor: 
  * 1.0 = agree with RF (no adjustment)
  * 0.7-0.9 = RF underestimates risk (failure sooner)
  * 1.1-1.3 = RF overestimates risk (failure later)
  * Stay conservative: only adjust if you see clear evidence
  
- Look for:
  * Accelerating degradation (slopes increasing)
  * Multi-sensor correlations RF might miss
  * Abnormal patterns in critical sensors (temp, vibration)
  
- FD001 context: Single condition, High Pressure Compressor (HPC) degradation
- Be conservative: if uncertain, use adjustment_factor close to 1.0"""
    
    user_msg = f"""Engine {engine_id}
Random Forest prediction: {rf_prediction:.1f} cycles RUL

Sensor trends (last 30 cycles, z-scored):
{summary_text}

Should we adjust the RF prediction? Focus on degradation patterns."""
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                response_format={"type": "json_schema", "json_schema": adjustment_schema},
                temperature=0.2,
                max_tokens=600
            )
            
            content = response.choices[0].message.content
            payload = json.loads(content)
            
            return {
                "adjustment_factor": float(payload["adjustment_factor"]),
                "confidence": float(payload["confidence"]),
                "rationale": payload["rationale"],
                "adjusted_rul": rf_prediction * float(payload["adjustment_factor"]),
                "api_success": True
            }
            
        except Exception as e:
            if attempt == max_retries - 1:
                # Default: no adjustment
                return {
                    "adjustment_factor": 1.0,
                    "confidence": 0.0,
                    "rationale": f"API_ERROR: {str(e)[:100]}",
                    "adjusted_rul": rf_prediction,
                    "api_success": False
                }
            time.sleep(1.0)


def run_hybrid_llm_analysis(summ_df, rf_predictions, batch_size=10):
    """
    Run hybrid analysis: LLM adjusts RF predictions
    Processes in batches for better progress tracking
    """
    
    print("\n" + "="*60)
    print("HYBRID LLM ANALYSIS (LLM refines RF predictions)")
    print("="*60)
    
    client = setup_openai_client()
    results = []
    
    total = len(summ_df)
    print(f"Processing {total} engines in batches of {batch_size}...")
    
    start_time = time.time()
    
    for i, (eng_id, row) in enumerate(summ_df.iterrows()):
        if (i + 1) % batch_size == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate if rate > 0 else 0
            print(f"  Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%) - Est. {remaining:.0f}s remaining")
        
        result = get_llm_adjustment(
            client=client,
            engine_id=eng_id,
            summary_text=row['llm_summary'],
            rf_prediction=rf_predictions[i]
        )
        
        result['engine_id'] = eng_id
        result['rf_prediction'] = rf_predictions[i]
        results.append(result)
        
        time.sleep(0.15)  # Rate limiting
    
    llm_df = pd.DataFrame(results).set_index('engine_id').sort_index()
    
    success_rate = llm_df['api_success'].sum() / len(llm_df) * 100
    total_time = time.time() - start_time
    print(f"\n✓ Completed in {total_time:.1f}s! Success rate: {success_rate:.1f}%")
    
    # Show adjustment statistics
    adj_factors = llm_df['adjustment_factor'].values
    print(f"\nAdjustment Statistics:")
    print(f"  Mean adjustment: {adj_factors.mean():.3f}")
    print(f"  Std adjustment:  {adj_factors.std():.3f}")
    print(f"  Range: [{adj_factors.min():.3f}, {adj_factors.max():.3f}]")
    print(f"  Adjusted down (<1.0): {(adj_factors < 1.0).sum()} engines")
    print(f"  No adjustment (≈1.0): {(np.abs(adj_factors - 1.0) < 0.1).sum()} engines")
    print(f"  Adjusted up (>1.0):   {(adj_factors > 1.0).sum()} engines")
    
    return llm_df


# =============================================================================
# COMPARISON AND VISUALIZATION
# =============================================================================

def compare_results(y_test, y_pred_rf, y_pred_llm):
    """Compare RF baseline with LLM predictions"""
    
    print("\n" + "=" * 60)
    print("COMPARISON: RF BASELINE vs HYBRID LLM")
    print("=" * 60)
    
    # RF metrics
    rf_rmse = np.sqrt(np.mean((y_pred_rf - y_test)**2))
    rf_mae = np.mean(np.abs(y_pred_rf - y_test))
    rf_s_score = compute_s_score(y_test, y_pred_rf)
    
    # LLM metrics
    llm_rmse = np.sqrt(np.mean((y_pred_llm - y_test)**2))
    llm_mae = np.mean(np.abs(y_pred_llm - y_test))
    llm_s_score = compute_s_score(y_test, y_pred_llm)
    
    print(f"\n{'Metric':<15} {'RF':<12} {'Hybrid LLM':<12} {'Improvement':<15}")
    print("-" * 60)
    print(f"{'RMSE':<15} {rf_rmse:<12.2f} {llm_rmse:<12.2f} {(rf_rmse-llm_rmse)/rf_rmse*100:>+6.1f}%")
    print(f"{'MAE':<15} {rf_mae:<12.2f} {llm_mae:<12.2f} {(rf_mae-llm_mae)/rf_mae*100:>+6.1f}%")
    print(f"{'S-Score':<15} {rf_s_score:<12.2f} {llm_s_score:<12.2f} {(rf_s_score-llm_s_score)/rf_s_score*100:>+6.1f}%")
    
    # Visualization
    plot_results(y_test, y_pred_rf, y_pred_llm)
    
    return {
        'rf': {'rmse': rf_rmse, 'mae': rf_mae, 's_score': rf_s_score},
        'llm': {'rmse': llm_rmse, 'mae': llm_mae, 's_score': llm_s_score}
    }


def plot_results(y_test, y_pred_rf, y_pred_llm):
    """Create comparison visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: Predictions vs True
    # RF
    axes[0, 0].scatter(y_test, y_pred_rf, alpha=0.6, color='blue', s=30)
    axes[0, 0].plot([0, 130], [0, 130], 'r--', lw=2)
    axes[0, 0].set_xlabel('True RUL')
    axes[0, 0].set_ylabel('Predicted RUL')
    axes[0, 0].set_title('RF Baseline Predictions')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Hybrid LLM
    axes[0, 1].scatter(y_test, y_pred_llm, alpha=0.6, color='green', s=30)
    axes[0, 1].plot([0, 130], [0, 130], 'r--', lw=2)
    axes[0, 1].set_xlabel('True RUL')
    axes[0, 1].set_ylabel('Predicted RUL')
    axes[0, 1].set_title('Hybrid LLM Predictions')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Direct comparison
    axes[0, 2].scatter(y_pred_rf, y_pred_llm, alpha=0.6, color='purple', s=30)
    axes[0, 2].plot([0, 130], [0, 130], 'r--', lw=2)
    axes[0, 2].set_xlabel('RF Prediction')
    axes[0, 2].set_ylabel('Hybrid LLM Prediction')
    axes[0, 2].set_title('RF vs Hybrid LLM')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Error distributions
    rf_errors = y_pred_rf - y_test
    llm_errors = y_pred_llm - y_test
    
    # RF errors
    axes[1, 0].hist(rf_errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].axvline(0, color='red', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Error (Predicted - True)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'RF Error Distribution\nMAE={np.mean(np.abs(rf_errors)):.2f}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # LLM errors
    axes[1, 1].hist(llm_errors, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].axvline(0, color='red', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Error (Predicted - True)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Hybrid LLM Error Distribution\nMAE={np.mean(np.abs(llm_errors)):.2f}')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Absolute errors comparison
    indices = np.arange(len(y_test))
    axes[1, 2].scatter(indices, np.abs(rf_errors), alpha=0.5, color='blue', s=20, label='RF')
    axes[1, 2].scatter(indices, np.abs(llm_errors), alpha=0.5, color='green', s=20, label='Hybrid LLM')
    axes[1, 2].set_xlabel('Engine Index')
    axes[1, 2].set_ylabel('Absolute Error')
    axes[1, 2].set_title('Absolute Errors by Engine')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rul_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'rul_comparison.png'")
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("RUL PREDICTION PIPELINE WITH HYBRID LLM INTEGRATION")
    print("="*80)
    
    # Step 1: Run baseline
    print("\n" + "="*80)
    print("STEP 1: RANDOM FOREST BASELINE")
    print("="*80)
    baseline_results = run_baseline_pipeline()
    
    # Step 2: Generate LLM summaries
    print("\n" + "="*80)
    print("STEP 2: GENERATE SENSOR SUMMARIES")
    print("="*80)
    summ_df = generate_llm_summaries(
    baseline_results['test_data'],
    baseline_results['scaler'],
    baseline_results['remaining_sensors'],
    baseline_results['y_pred']  # pass RF predictions
)

    
    # Step 3: Run LLM analysis (optional)
    print("\n" + "="*80)
    print("STEP 3: HYBRID LLM ANALYSIS (OPTIONAL)")
    print("="*80)
    
    run_llm = input("\nRun Hybrid LLM analysis? (requires OpenAI API key) [y/N]: ").lower() == 'y'
    
    if run_llm:
        try:
            llm_df = run_hybrid_llm_analysis(summ_df, baseline_results['y_pred'], batch_size=10)
            
            if llm_df is not None and len(llm_df) > 0:
                y_pred_llm = llm_df['adjusted_rul'].values
                metrics = compare_results(
                    baseline_results['y_test'],
                    baseline_results['y_pred'],
                    y_pred_llm
                )
                
                # Save LLM results
                llm_df.to_csv('llm_results.csv')
                print("\n✓ LLM results saved to 'llm_results.csv'")
                
        except Exception as e:
            print(f"\n❌ LLM analysis failed: {e}")
            print("Continuing with RF baseline only...")
    
    else:
        print("\nSkipping LLM analysis. RF baseline complete!")
    
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - X_train_improved.npy, y_train_improved.npy")
    print("  - X_test_improved.npy, y_test_improved.npy")
    print("  - y_pred_improved.npy (RF predictions)")
    print("  - sensor_scaler.pkl")
    if run_llm:
        print("  - llm_results.csv")
        print("  - rul_comparison.png")
    
    print("\n" + "="*80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*80)
    
    if run_llm and 'metrics' in locals():
        rf_metrics = metrics['rf']
        llm_metrics = metrics['llm']
        
        print("\nPerformance Summary:")
        print(f"  RF Baseline:  RMSE={rf_metrics['rmse']:.2f}, MAE={rf_metrics['mae']:.2f}, S-Score={rf_metrics['s_score']:.2f}")
        print(f"  Hybrid LLM:   RMSE={llm_metrics['rmse']:.2f}, MAE={llm_metrics['mae']:.2f}, S-Score={llm_metrics['s_score']:.2f}")
        
        # Determine if LLM helped
        improvement = (rf_metrics['rmse'] - llm_metrics['rmse']) / rf_metrics['rmse'] * 100
        
        if improvement > 5:
            print(f"\n✓ SUCCESS! Hybrid LLM improved RMSE by {improvement:.1f}%")
            print("  Recommendation: Use Hybrid LLM approach for production")
        elif improvement > 0:
            print(f"\n→ MARGINAL: Hybrid LLM improved RMSE by {improvement:.1f}%")
            print("  Recommendation: Consider cost/benefit - RF alone may be sufficient")
        else:
            print(f"\n⚠ LIMITED BENEFIT: Hybrid LLM changed RMSE by {improvement:.1f}%")
            print("  Recommendation: Stick with RF baseline or try alternative approaches:")
            print("    1. Selective LLM (only uncertain cases)")
            print("    2. Risk classification instead of regression")
            print("    3. Better prompt engineering")
    else:
        print("\nRF Baseline Results:")
        print(f"  RMSE: {baseline_results['metrics']['rmse']:.2f} cycles")
        print(f"  MAE:  {baseline_results['metrics']['mae']:.2f} cycles")
        print(f"  S-Score: {baseline_results['metrics']['s_score']:.2f}")
        print("\nTo enable LLM integration:")
        print("  1. Set OPENAI_API_KEY environment variable")
        print("  2. Re-run and select 'y' for LLM analysis")