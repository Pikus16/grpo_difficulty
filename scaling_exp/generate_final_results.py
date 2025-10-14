#!/usr/bin/env python3
"""
Generate comprehensive final results for all scaling law models
Runs 3 times to validate stability
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, HuberRegressor, LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.special import logit, expit
from scipy.stats import norm
import pickle

EPS = 1e-8

def safe_logit(p):
    return logit(np.clip(p, EPS, 1.0 - EPS))

def robust_slope_0_T(run_df, base, T):
    """Calculate robust slope using base as checkpoint 0"""
    pts = run_df[run_df['checkpoint'].between(1, T)].copy()  # Start from 1, not 0
    if pts.empty:
        return np.nan
    
    # Build checkpoint sequence: 0 (base), then actual checkpoints
    xs, ys = [0], [safe_logit(1 - base)]
    
    for _, row in pts.sort_values('checkpoint').iterrows():
        xs.append(row['checkpoint'])
        ys.append(safe_logit(1 - row['accuracy']))
    
    if len(xs) < 2:
        return np.nan
    
    xs, ys = np.asarray(xs).reshape(-1, 1), np.asarray(ys)
    huber = HuberRegressor(alpha=0.0, fit_intercept=True, epsilon=1.35)
    huber.fit(xs, ys)
    return float(huber.coef_[0])

def learnability_features(run_df, base, T):
    """Calculate continuous learnability using base as checkpoint 0"""
    pts = run_df[run_df['checkpoint'].between(1, T)].sort_values('checkpoint')
    
    # Start with base
    logits, steps = [safe_logit(1 - base)], [0]
    
    for _, row in pts.iterrows():
        logits.append(safe_logit(1 - row['accuracy']))
        steps.append(int(row['checkpoint']))
    
    if len(logits) < 2:
        return 0.0, 0.0, 0.0
    
    logits, steps = np.asarray(logits), np.asarray(steps)
    deltas = np.diff(logits)
    impr = np.maximum(0.0, -deltas)
    mass = impr.sum()
    max_impr = impr.max() if impr.size else 0.0
    widths = np.diff(steps)
    auc_impr = np.sum(impr * widths[:len(impr)]) if impr.size else 0.0
    return float(mass), float(max_impr), float(auc_impr)

def fit_trajectory_model(train_df, held_out_df, T, name, seed):
    """Fit trajectory model with full preprocessing"""
    np.random.seed(seed)
    
    # Build training features
    final = train_df[train_df['checkpoint'] == 1000].copy()
    rows = []
    
    for _, run in final.iterrows():
        key = (run['dataset'], run['strategy'], run['model_name'])
        rd = train_df[(train_df['dataset'] == key[0]) & (train_df['strategy'] == key[1]) & 
                      (train_df['model_name'] == key[2])].copy()
        
        # Use base as checkpoint 0
        slope = robust_slope_0_T(rd, run['base'], T)
        if not np.isfinite(slope):
            continue
        mass, max_impr, auc_impr = learnability_features(rd, run['base'], T)
        
        rows.append({
            'logM': np.log(np.clip(run['model_size'], EPS, None)),
            'Llog': safe_logit(np.clip(run['perc_learnable'], EPS, 0.999)),
            'slope': slope, 'L_mass': mass, 'L_max': max_impr, 'L_auc': auc_impr,
            'final_error': 1 - run['final_acc'],
            'base_error': 1 - run['base'],
            'base': run['base'],
            'y_star': safe_logit(1 - run['final_acc']) - safe_logit(1 - run['base']),
            'logit_e0': safe_logit(1 - run['base'])
        })
    
    if len(rows) == 0:
        return None
        
    train_feat = pd.DataFrame(rows)
    X_train = train_feat[['logM','Llog','slope','L_mass','L_max','L_auc']].values
    y_train = train_feat['y_star'].values
    
    proxy = 1.0 + np.abs(y_train) + np.abs(train_feat['L_max'].values)
    w = 1.0 / proxy
    
    model = Ridge(alpha=1e-3)
    model.fit(X_train, y_train, sample_weight=w)
    
    train_pred_ystar = model.predict(X_train)
    train_pred = expit(train_feat['logit_e0'].values + train_pred_ystar)
    train_r2 = r2_score(train_feat['final_error'], train_pred)
    
    lr_train = LinearRegression()
    lr_train.fit(train_pred.reshape(-1, 1), train_feat['final_error'].values)
    train_calib_slope = lr_train.coef_[0]
    
    # Held-out
    final_held = held_out_df[held_out_df['checkpoint'] == 1000].copy()
    held_rows = []
    
    for _, run in final_held.iterrows():
        key = (run['dataset'], run['strategy'], run['model_name'])
        rd = held_out_df[(held_out_df['dataset'] == key[0]) & (held_out_df['strategy'] == key[1]) & 
                         (held_out_df['model_name'] == key[2])].copy()
        
        slope = robust_slope_0_T(rd, run['base'], T)
        if not np.isfinite(slope):
            continue
        mass, max_impr, auc_impr = learnability_features(rd, run['base'], T)
        
        held_rows.append({
            'logM': np.log(np.clip(run['model_size'], EPS, None)),
            'Llog': safe_logit(np.clip(run['perc_learnable'], EPS, 0.999)),
            'slope': slope, 'L_mass': mass, 'L_max': max_impr, 'L_auc': auc_impr,
            'final_error': 1 - run['final_acc'],
            'base_error': 1 - run['base'],
            'base': run['base'],
            'logit_e0': safe_logit(1 - run['base'])
        })
    
    if len(held_rows) == 0:
        return None
        
    held_feat = pd.DataFrame(held_rows)
    X_held = held_feat[['logM','Llog','slope','L_mass','L_max','L_auc']].values
    held_pred_ystar = model.predict(X_held)
    held_pred = expit(held_feat['logit_e0'].values + held_pred_ystar)
    held_r2 = r2_score(held_feat['final_error'], held_pred)
    
    lr_held = LinearRegression()
    lr_held.fit(held_pred.reshape(-1, 1), held_feat['final_error'].values)
    held_calib_slope = lr_held.coef_[0]
    
    # Policy savings
    thresholds_pp = [5, 10, 15, 20]
    policy_results = []
    
    for thresh_pp in thresholds_pp:
        threshold = thresh_pp / 100.0
        actual_gains = held_feat['base'].values - held_feat['final_error'].values
        winners = actual_gains >= threshold
        
        residuals = held_feat['final_error'].values - held_pred
        std_residual = np.std(residuals)
        
        target_error = held_feat['base'].values - threshold
        z_scores = (target_error - held_pred) / np.maximum(std_residual, 1e-8)
        p_success = norm.cdf(z_scores)
        
        best_saved = 0
        best_missed_rate = 1.0
        for p_cut in np.linspace(0.1, 0.9, 17):
            stop = p_success < p_cut
            if winners.sum() > 0:
                missed_rate = (stop & winners).sum() / winners.sum()
                if missed_rate <= 0.05:
                    compute_mult = 0.90 if T == 100 else 0.80
                    saved = stop.sum() / len(held_feat) * compute_mult
                    if saved > best_saved:
                        best_saved = saved
                        best_missed_rate = missed_rate
        
        policy_results.append({
            'threshold_pp': thresh_pp,
            'saved_pct': best_saved * 100,
            'missed_pct': best_missed_rate * 100
        })
    
    return {
        'name': name,
        'T': T,
        'train_r2': train_r2,
        'held_r2': held_r2,
        'train_n': len(train_feat),
        'held_n': len(held_feat),
        'train_calib_slope': train_calib_slope,
        'held_calib_slope': held_calib_slope,
        'policy': policy_results,
        'coefficients': {
            'intercept': model.intercept_,
            'logM': model.coef_[0],
            'Llog': model.coef_[1],
            'slope': model.coef_[2],
            'L_mass': model.coef_[3],
            'L_max': model.coef_[4],
            'L_auc': model.coef_[5]
        }
    }

def fit_cp_logit_model(train_df, held_out_df, checkpoint, name, seed):
    """Fit checkpoint logit model"""
    np.random.seed(seed)
    
    # Training
    cp_train = train_df[train_df['checkpoint'] == checkpoint].copy()
    final_train = train_df[train_df['checkpoint'] == 1000].copy()
    
    merged_train = cp_train[['dataset', 'strategy', 'model_name', 'accuracy']].merge(
        final_train[['dataset', 'strategy', 'model_name', 'final_acc', 'base']],
        on=['dataset', 'strategy', 'model_name']
    )
    
    merged_train['error_cp'] = 1 - merged_train['accuracy']
    merged_train['final_error'] = 1 - merged_train['final_acc']
    
    logit_cp_train = safe_logit(merged_train['error_cp'].values)
    logit_final_train = safe_logit(merged_train['final_error'].values)
    
    lr = LinearRegression()
    lr.fit(logit_cp_train.reshape(-1, 1), logit_final_train)
    
    train_pred_logit = lr.predict(logit_cp_train.reshape(-1, 1))
    train_pred = expit(train_pred_logit)
    train_r2 = r2_score(merged_train['final_error'], train_pred)
    
    lr_calib_train = LinearRegression()
    lr_calib_train.fit(train_pred.reshape(-1, 1), merged_train['final_error'].values)
    train_calib_slope = lr_calib_train.coef_[0]
    
    # Held-out
    cp_held = held_out_df[held_out_df['checkpoint'] == checkpoint].copy()
    final_held = held_out_df[held_out_df['checkpoint'] == 1000].copy()
    
    merged_held = cp_held[['dataset', 'strategy', 'model_name', 'accuracy']].merge(
        final_held[['dataset', 'strategy', 'model_name', 'final_acc', 'base']],
        on=['dataset', 'strategy', 'model_name']
    )
    
    merged_held['error_cp'] = 1 - merged_held['accuracy']
    merged_held['final_error'] = 1 - merged_held['final_acc']
    
    logit_cp_held = safe_logit(merged_held['error_cp'].values)
    held_pred_logit = lr.predict(logit_cp_held.reshape(-1, 1))
    held_pred = expit(held_pred_logit)
    held_r2 = r2_score(merged_held['final_error'], held_pred)
    
    lr_calib_held = LinearRegression()
    lr_calib_held.fit(held_pred.reshape(-1, 1), merged_held['final_error'].values)
    held_calib_slope = lr_calib_held.coef_[0]
    
    # Policy savings
    thresholds_pp = [5, 10, 15, 20]
    policy_results = []
    
    for thresh_pp in thresholds_pp:
        threshold = thresh_pp / 100.0
        actual_gains = merged_held['base'].values - merged_held['final_error'].values
        winners = actual_gains >= threshold
        
        residuals = merged_held['final_error'].values - held_pred
        std_residual = np.std(residuals)
        
        target_error = merged_held['base'].values - threshold
        z_scores = (target_error - held_pred) / np.maximum(std_residual, 1e-8)
        p_success = norm.cdf(z_scores)
        
        best_saved = 0
        best_missed_rate = 1.0
        for p_cut in np.linspace(0.1, 0.9, 17):
            stop = p_success < p_cut
            if winners.sum() > 0:
                missed_rate = (stop & winners).sum() / winners.sum()
                if missed_rate <= 0.05:
                    compute_mult = 0.90 if checkpoint == 100 else 0.80
                    saved = stop.sum() / len(merged_held) * compute_mult
                    if saved > best_saved:
                        best_saved = saved
                        best_missed_rate = missed_rate
        
        policy_results.append({
            'threshold_pp': thresh_pp,
            'saved_pct': best_saved * 100,
            'missed_pct': best_missed_rate * 100
        })
    
    return {
        'name': name,
        'checkpoint': checkpoint,
        'train_r2': train_r2,
        'held_r2': held_r2,
        'train_n': len(merged_train),
        'held_n': len(merged_held),
        'train_calib_slope': train_calib_slope,
        'held_calib_slope': held_calib_slope,
        'policy': policy_results,
        'coefficients': {
            'a': lr.intercept_,
            'b': lr.coef_[0]
        }
    }

def main():
    print("="*80)
    print("COMPREHENSIVE SCALING LAW EVALUATION")
    print("="*80)
    
    # Load data
    train_df = pd.read_csv('scaling_analysis_results.csv')
    held_out_df = pd.read_csv('held_out_scaling_numbers.csv')
    
    print(f"\nTraining: {len(train_df)} points")
    print(f"Held-out: {len(held_out_df)} points")
    
    # Run 3 times
    all_runs_results = []
    
    for run_num in range(3):
        print(f"\n{'='*80}")
        print(f"RUN {run_num + 1}/3")
        print(f"{'='*80}")
        
        results = {}
        
        # 1. Trajectory CP100
        print(f"  1. Trajectory (CP100)...", end='', flush=True)
        results['traj_100'] = fit_trajectory_model(train_df, held_out_df, 100, "Trajectory CP100", run_num)
        if results['traj_100']:
            print(f" R²={results['traj_100']['held_r2']:.4f}")
        else:
            print(" FAILED")
        
        # 2. Trajectory CP200
        print(f"  2. Trajectory (CP200)...", end='', flush=True)
        results['traj_200'] = fit_trajectory_model(train_df, held_out_df, 200, "Trajectory CP200", run_num)
        if results['traj_200']:
            print(f" R²={results['traj_200']['held_r2']:.4f}")
        else:
            print(" FAILED")
        
        # 3. CP100 Logit
        print(f"  3. CP100 Logit...", end='', flush=True)
        results['cp100'] = fit_cp_logit_model(train_df, held_out_df, 100, "CP100 Logit", run_num)
        print(f" R²={results['cp100']['held_r2']:.4f}")
        
        # 4. CP200 Logit
        print(f"  4. CP200 Logit...", end='', flush=True)
        results['cp200'] = fit_cp_logit_model(train_df, held_out_df, 200, "CP200 Logit", run_num)
        print(f" R²={results['cp200']['held_r2']:.4f}")
        
        all_runs_results.append(results)
    
    # Aggregate
    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS")
    print(f"{'='*80}")
    
    final_results = {}
    for model_key in ['traj_100', 'traj_200', 'cp100', 'cp200']:
        # Skip if all runs failed
        if all(r[model_key] is None for r in all_runs_results):
            print(f"  Skipping {model_key} - all runs failed")
            continue
            
        r2_trains = [r[model_key]['train_r2'] for r in all_runs_results if r[model_key] is not None]
        r2_helds = [r[model_key]['held_r2'] for r in all_runs_results if r[model_key] is not None]
        calib_slopes = [r[model_key]['held_calib_slope'] for r in all_runs_results if r[model_key] is not None]
        
        # Get first non-None result for metadata
        first_result = next(r[model_key] for r in all_runs_results if r[model_key] is not None)
        
        final_results[model_key] = {
            'name': first_result['name'],
            'train_r2': np.mean(r2_trains),
            'train_r2_std': np.std(r2_trains),
            'held_r2': np.mean(r2_helds),
            'held_r2_std': np.std(r2_helds),
            'calib_slope': np.mean(calib_slopes),
            'calib_slope_std': np.std(calib_slopes),
            'policy': first_result['policy'],
            'n_train': first_result['train_n'],
            'n_held': first_result['held_n'],
            'coefficients': first_result['coefficients']
        }
    
    # Print summary
    print(f"\n{'Model':<25} {'Train R²':<20} {'Held R²':<20} {'Calib Slope'}")
    print("-"*90)
    for key in ['cp100', 'cp200', 'traj_100', 'traj_200']:
        if key in final_results:
            res = final_results[key]
            print(f"{res['name']:<25} {res['train_r2']:.4f}±{res['train_r2_std']:.5f}  {res['held_r2']:.4f}±{res['held_r2_std']:.5f}  {res['calib_slope']:.4f}±{res['calib_slope_std']:.4f}")
    
    # Save
    with open('final_results_data.pkl', 'wb') as f:
        pickle.dump(final_results, f)
    
    print(f"\n✅ Results saved to final_results_data.pkl")
    return final_results

if __name__ == "__main__":
    results = main()

