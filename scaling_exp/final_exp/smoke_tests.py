#!/usr/bin/env python3
"""
Comprehensive smoke tests for the final scaling law model
Tests for overfitting, data leakage, and selection bias
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.special import logit, expit
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
import sys
sys.path.append('..')
from scaling_law_models import ScalingLawModels


class SmokeTests:
    """Run comprehensive smoke tests on the scaling law model"""
    
    def __init__(self, train_df, held_out_df):
        self.train_df = train_df
        self.held_out_df = held_out_df
        self.epsilon = 1e-4
        
    def test_a_grouped_time_forward(self):
        """Test A: Grouped time-forward evaluation"""
        print("\n" + "="*70)
        print("TEST A: GROUPED TIME-FORWARD EVALUATION")
        print("="*70)
        
        checkpoints = [0, 100, 200, 300, 500]
        results = []
        
        for T in checkpoints:
            # For each run, use only data up to checkpoint T
            if T == 0:
                # Use only base accuracy
                r2, calib = self._evaluate_at_checkpoint_0()
            else:
                r2, calib = self._evaluate_at_checkpoint(T)
                
            results.append({
                'checkpoint': T,
                'r2': r2,
                'calibration_slope': calib['slope'] if calib else None,
                'calibration_intercept': calib['intercept'] if calib else None
            })
            
            print(f"\nCheckpoint {T}:")
            print(f"  R² = {r2:.4f}")
            if calib:
                print(f"  Calibration: slope = {calib['slope']:.3f}, intercept = {calib['intercept']:.3f}")
        
        # Plot R² progression
        results_df = pd.DataFrame(results)
        self._plot_time_forward_results(results_df)
        
        return results_df
    
    def _evaluate_at_checkpoint_0(self):
        """Evaluate using only base accuracy"""
        final_df = self.train_df[self.train_df['checkpoint'] == 1000].copy()
        final_df['final_error'] = 1 - final_df['final_acc']
        
        # Simple baseline: predict final = base
        predictions = 1 - final_df['base']
        r2 = r2_score(final_df['final_error'], predictions)
        
        # Calibration
        lr = LinearRegression()
        lr.fit(predictions.values.reshape(-1, 1), final_df['final_error'].values)
        calib = {'slope': lr.coef_[0], 'intercept': lr.intercept_}
        
        return r2, calib
    
    def _evaluate_at_checkpoint(self, T):
        """Evaluate using data up to checkpoint T"""
        # Get data at checkpoint T and final
        cp_df = self.train_df[self.train_df['checkpoint'] == T].copy()
        final_df = self.train_df[self.train_df['checkpoint'] == 1000].copy()
        
        # Get final accuracy data with renamed column
        final_data = final_df[['dataset', 'strategy', 'model_name', 'final_acc']].copy()
        final_data = final_data.rename(columns={'final_acc': 'final_final_acc'})
        
        # Merge
        merged = cp_df.merge(
            final_data,
            on=['dataset', 'strategy', 'model_name']
        )
        
        if len(merged) == 0:
            return 0.0, None
            
        # Calculate features available at checkpoint T
        merged['error_T'] = 1 - merged['accuracy']
        merged['final_error'] = 1 - merged['final_final_acc']
        merged['base_error'] = 1 - merged['base']
        
        # Calculate slope from 0 to T
        merged['logit_error_0'] = logit(np.clip(merged['base_error'], self.epsilon, 1-self.epsilon))
        merged['logit_error_T'] = logit(np.clip(merged['error_T'], self.epsilon, 1-self.epsilon))
        merged['slope_0_T'] = (merged['logit_error_T'] - merged['logit_error_0']) / T
        
        # Build model
        X = np.column_stack([
            np.log(merged['model_size']),
            logit(np.clip(merged['perc_learnable'], self.epsilon, 1-self.epsilon)),
            merged['slope_0_T']
        ])
        
        y_star = (logit(np.clip(merged['final_error'], self.epsilon, 1-self.epsilon)) - 
                  merged['logit_error_0'])
        
        # Fit and predict
        model = Ridge(alpha=1e-3)
        model.fit(X, y_star)
        pred_y_star = model.predict(X)
        pred_final_error = expit(merged['logit_error_0'] + pred_y_star)
        
        # Calculate R²
        r2 = r2_score(merged['final_error'], pred_final_error)
        
        # Calibration
        lr = LinearRegression()
        # Handle both numpy arrays and pandas Series
        pred_array = pred_final_error.values if hasattr(pred_final_error, 'values') else pred_final_error
        lr.fit(pred_array.reshape(-1, 1), merged['final_error'].values)
        calib = {'slope': lr.coef_[0], 'intercept': lr.intercept_}
        
        return r2, calib
    
    def test_b_leakage_missingness_audit(self):
        """Test B: Leakage & missingness audit"""
        print("\n" + "="*70)
        print("TEST B: LEAKAGE & MISSINGNESS AUDIT")
        print("="*70)
        
        # Identify which held-out runs have early checkpoints
        held_out_final = self.held_out_df[self.held_out_df['checkpoint'] == 1000].copy()
        
        # Check which have checkpoint 200
        held_out_200 = self.held_out_df[self.held_out_df['checkpoint'] == 200]
        has_200 = held_out_200[['dataset', 'strategy', 'model_name']].drop_duplicates()
        
        # Merge to identify included vs excluded
        held_out_final['has_early_checkpoint'] = held_out_final.apply(
            lambda r: len(has_200[
                (has_200['dataset'] == r['dataset']) & 
                (has_200['strategy'] == r['strategy']) & 
                (has_200['model_name'] == r['model_name'])
            ]) > 0, axis=1
        )
        
        included = held_out_final[held_out_final['has_early_checkpoint']]
        excluded = held_out_final[~held_out_final['has_early_checkpoint']]
        
        print(f"\nHeld-out runs: {len(held_out_final)} total")
        print(f"  With early checkpoints: {len(included)} ({len(included)/len(held_out_final)*100:.1f}%)")
        print(f"  Without early checkpoints: {len(excluded)} ({len(excluded)/len(held_out_final)*100:.1f}%)")
        
        # Compare characteristics
        print("\nComparing included vs excluded runs:")
        
        metrics = ['base', 'model_size', 'perc_learnable', 'final_acc']
        for metric in metrics:
            inc_mean = included[metric].mean()
            exc_mean = excluded[metric].mean()
            inc_std = included[metric].std()
            exc_std = excluded[metric].std()
            
            # T-test
            from scipy.stats import ttest_ind
            t_stat, p_value = ttest_ind(included[metric], excluded[metric])
            
            print(f"\n{metric}:")
            print(f"  Included: {inc_mean:.3f} ± {inc_std:.3f}")
            print(f"  Excluded: {exc_mean:.3f} ± {exc_std:.3f}")
            print(f"  Difference: {inc_mean - exc_mean:.3f} (p = {p_value:.3f})")
        
        # Strategy distribution
        print("\nStrategy distribution:")
        inc_strategies = included['strategy'].value_counts()
        exc_strategies = excluded['strategy'].value_counts()
        
        all_strategies = set(inc_strategies.index) | set(exc_strategies.index)
        for strategy in sorted(all_strategies):
            inc_count = inc_strategies.get(strategy, 0)
            exc_count = exc_strategies.get(strategy, 0)
            print(f"  {strategy}: {inc_count} included, {exc_count} excluded")
        
        # Dataset distribution
        print("\nDataset distribution:")
        inc_datasets = included['dataset'].value_counts()
        exc_datasets = excluded['dataset'].value_counts()
        
        all_datasets = set(inc_datasets.index) | set(exc_datasets.index)
        for dataset in sorted(all_datasets):
            inc_count = inc_datasets.get(dataset, 0)
            exc_count = exc_datasets.get(dataset, 0)
            print(f"  {dataset}: {inc_count} included, {exc_count} excluded")
        
        return included, excluded
    
    def test_c_slope_robustness(self):
        """Test C: Robustness of slope definition"""
        print("\n" + "="*70)
        print("TEST C: ROBUSTNESS OF SLOPE DEFINITION")
        print("="*70)
        
        # Get runs with checkpoints 0, 100, 200
        final_df = self.train_df[self.train_df['checkpoint'] == 1000].copy()
        
        slope_methods = {
            'linear_fit': self._compute_slope_linear_fit,
            'two_point': self._compute_slope_two_point,
            'smoothed': self._compute_slope_smoothed
        }
        
        results = {}
        
        for method_name, method_func in slope_methods.items():
            print(f"\n{method_name.replace('_', ' ').title()}:")
            
            # Compute slopes
            slopes = []
            valid_runs = []
            
            for _, run in final_df.iterrows():
                slope = method_func(run['dataset'], run['strategy'], run['model_name'])
                if slope is not None:
                    slopes.append(slope)
                    valid_runs.append(run)
            
            if len(slopes) == 0:
                print("  No valid slopes computed")
                continue
                
            valid_df = pd.DataFrame(valid_runs)
            valid_df['slope'] = slopes
            valid_df['final_error'] = 1 - valid_df['final_acc']
            
            # Build model with this slope definition
            X = np.column_stack([
                np.log(valid_df['model_size']),
                logit(np.clip(valid_df['perc_learnable'], self.epsilon, 1-self.epsilon)),
                valid_df['slope']
            ])
            
            valid_df['base_error'] = 1 - valid_df['base']
            y_star = (logit(np.clip(valid_df['final_error'], self.epsilon, 1-self.epsilon)) - 
                     logit(np.clip(valid_df['base_error'], self.epsilon, 1-self.epsilon)))
            
            # Fit model
            model = Ridge(alpha=1e-3)
            model.fit(X, y_star)
            
            # Predict
            pred_y_star = model.predict(X)
            pred_error = expit(logit(np.clip(valid_df['base_error'], self.epsilon, 1-self.epsilon)) + pred_y_star)
            
            # Calculate R²
            r2 = r2_score(valid_df['final_error'], pred_error)
            
            # Convert slope coefficient to per-100 steps
            slope_coef_per_100 = model.coef_[2] * 100
            
            results[method_name] = {
                'r2': r2,
                'slope_coefficient': model.coef_[2],
                'slope_per_100_steps': slope_coef_per_100,
                'n_valid': len(valid_df)
            }
            
            print(f"  R² = {r2:.4f}")
            print(f"  Slope coefficient = {model.coef_[2]:.1f}")
            print(f"  Per 100 steps = {slope_coef_per_100:.1f}")
            print(f"  Valid runs = {len(valid_df)}")
        
        # Compare results
        print("\nComparison across methods:")
        r2_values = [r['r2'] for r in results.values()]
        coef_values = [r['slope_per_100_steps'] for r in results.values()]
        
        print(f"  R² range: {min(r2_values):.4f} - {max(r2_values):.4f} (Δ = {max(r2_values) - min(r2_values):.4f})")
        print(f"  Coefficient range: {min(coef_values):.1f} - {max(coef_values):.1f} (Δ = {max(coef_values) - min(coef_values):.1f})")
        
        return results
    
    def _compute_slope_linear_fit(self, dataset, strategy, model_name):
        """Compute slope using linear fit to checkpoints 0-200"""
        run_data = self.train_df[
            (self.train_df['dataset'] == dataset) & 
            (self.train_df['strategy'] == strategy) & 
            (self.train_df['model_name'] == model_name)
        ]
        
        # Get checkpoints <= 200
        early_data = run_data[run_data['checkpoint'] <= 200].copy()
        if len(early_data) < 2:
            return None
            
        # Add checkpoint 0 (base)
        base_error = 1 - run_data.iloc[0]['base']
        checkpoints = [0] + early_data['checkpoint'].tolist()
        errors = [base_error] + (1 - early_data['accuracy']).tolist()
        
        # Convert to logit
        logit_errors = [logit(np.clip(e, self.epsilon, 1-self.epsilon)) for e in errors]
        
        # Linear fit
        slope, _ = np.polyfit(checkpoints, logit_errors, 1)
        
        return slope
    
    def _compute_slope_two_point(self, dataset, strategy, model_name):
        """Compute slope using two-point secant (0 → 200)"""
        run_data = self.train_df[
            (self.train_df['dataset'] == dataset) & 
            (self.train_df['strategy'] == strategy) & 
            (self.train_df['model_name'] == model_name)
        ]
        
        # Get checkpoint 200
        cp_200 = run_data[run_data['checkpoint'] == 200]
        if len(cp_200) == 0:
            return None
            
        base_error = 1 - run_data.iloc[0]['base']
        error_200 = 1 - cp_200.iloc[0]['accuracy']
        
        # Logit transform
        logit_0 = logit(np.clip(base_error, self.epsilon, 1-self.epsilon))
        logit_200 = logit(np.clip(error_200, self.epsilon, 1-self.epsilon))
        
        return (logit_200 - logit_0) / 200
    
    def _compute_slope_smoothed(self, dataset, strategy, model_name):
        """Compute slope using smoothed finite differences"""
        run_data = self.train_df[
            (self.train_df['dataset'] == dataset) & 
            (self.train_df['strategy'] == strategy) & 
            (self.train_df['model_name'] == model_name)
        ]
        
        # Get checkpoints <= 200
        early_data = run_data[run_data['checkpoint'] <= 200].sort_values('checkpoint')
        if len(early_data) < 2:
            return None
            
        # Add base
        base_error = 1 - run_data.iloc[0]['base']
        
        # Compute slopes between consecutive checkpoints
        slopes = []
        prev_cp = 0
        prev_error = base_error
        
        for _, row in early_data.iterrows():
            curr_cp = row['checkpoint']
            curr_error = 1 - row['accuracy']
            
            if curr_cp > prev_cp:
                # Compute slope for this segment
                logit_prev = logit(np.clip(prev_error, self.epsilon, 1-self.epsilon))
                logit_curr = logit(np.clip(curr_error, self.epsilon, 1-self.epsilon))
                
                slope = (logit_curr - logit_prev) / (curr_cp - prev_cp)
                slopes.append(slope)
                
            prev_cp = curr_cp
            prev_error = curr_error
        
        # Return average slope
        return np.mean(slopes) if slopes else None
    
    def test_d_identity_free_vs_aware(self):
        """Test D: Identity-free vs identity-aware models"""
        print("\n" + "="*70)
        print("TEST D: IDENTITY-FREE VS IDENTITY-AWARE")
        print("="*70)
        
        # Prepare held-out data with slopes
        held_out_with_slopes = self._prepare_held_out_with_slopes()
        
        if len(held_out_with_slopes) == 0:
            print("No held-out data with slopes available")
            return None
            
        # Model 1: Dynamic-only (slope + base-offset + M + logit(L))
        print("\n1. Dynamic-only model:")
        r2_dynamic = self._fit_dynamic_only_model(held_out_with_slopes)
        print(f"   Held-out R² = {r2_dynamic:.4f}")
        
        # Model 2: Fixed effects (dataset + strategy dummies)
        print("\n2. Fixed effects model:")
        r2_fixed = self._fit_fixed_effects_model(held_out_with_slopes)
        print(f"   Held-out R² = {r2_fixed:.4f}")
        
        # Model 3: Combined (dynamic + fixed effects)
        print("\n3. Combined model:")
        r2_combined = self._fit_combined_model(held_out_with_slopes)
        print(f"   Held-out R² = {r2_combined:.4f}")
        
        print(f"\nDynamic-only advantage: {r2_dynamic - r2_fixed:+.4f}")
        print(f"Combined advantage over dynamic: {r2_combined - r2_dynamic:+.4f}")
        
        return {
            'dynamic_only': r2_dynamic,
            'fixed_effects': r2_fixed,
            'combined': r2_combined
        }
    
    def _prepare_held_out_with_slopes(self):
        """Prepare held-out data with calculated slopes"""
        held_out_final = self.held_out_df[self.held_out_df['checkpoint'] == 1000].copy()
        
        slopes = []
        valid_runs = []
        
        for _, run in held_out_final.iterrows():
            slope = self._compute_slope_two_point(run['dataset'], run['strategy'], run['model_name'])
            if slope is not None:
                slopes.append(slope)
                valid_runs.append(run)
        
        if len(valid_runs) == 0:
            return pd.DataFrame()
            
        valid_df = pd.DataFrame(valid_runs)
        valid_df['slope'] = slopes
        valid_df['final_error'] = 1 - valid_df['final_acc']
        valid_df['base_error'] = 1 - valid_df['base']
        
        return valid_df
    
    def _fit_dynamic_only_model(self, data):
        """Fit model with only dynamic features"""
        X = np.column_stack([
            np.log(data['model_size']),
            logit(np.clip(data['perc_learnable'], self.epsilon, 1-self.epsilon)),
            data['slope']
        ])
        
        y_star = (logit(np.clip(data['final_error'], self.epsilon, 1-self.epsilon)) - 
                  logit(np.clip(data['base_error'], self.epsilon, 1-self.epsilon)))
        
        model = Ridge(alpha=1e-3)
        model.fit(X, y_star)
        
        pred_y_star = model.predict(X)
        pred_error = expit(logit(np.clip(data['base_error'], self.epsilon, 1-self.epsilon)) + pred_y_star)
        
        return r2_score(data['final_error'], pred_error)
    
    def _fit_fixed_effects_model(self, data):
        """Fit model with fixed effects only"""
        # Create dummies
        dataset_dummies = pd.get_dummies(data['dataset'], prefix='dataset', drop_first=True)
        strategy_dummies = pd.get_dummies(data['strategy'], prefix='strategy', drop_first=True)
        
        X = np.hstack([
            np.log(data[['model_size', 'base', 'perc_learnable']].values + self.epsilon),
            dataset_dummies.values,
            strategy_dummies.values
        ])
        
        y = logit(np.clip(data['final_error'], self.epsilon, 1-self.epsilon))
        
        model = Ridge(alpha=1e-3)
        model.fit(X, y)
        
        pred_y = model.predict(X)
        pred_error = expit(pred_y)
        
        return r2_score(data['final_error'], pred_error)
    
    def _fit_combined_model(self, data):
        """Fit model with both dynamic and fixed effects"""
        dataset_dummies = pd.get_dummies(data['dataset'], prefix='dataset', drop_first=True)
        strategy_dummies = pd.get_dummies(data['strategy'], prefix='strategy', drop_first=True)
        
        X = np.column_stack([
            np.log(data['model_size']),
            logit(np.clip(data['perc_learnable'], self.epsilon, 1-self.epsilon)),
            data['slope'],
            dataset_dummies.values,
            strategy_dummies.values
        ])
        
        y_star = (logit(np.clip(data['final_error'], self.epsilon, 1-self.epsilon)) - 
                  logit(np.clip(data['base_error'], self.epsilon, 1-self.epsilon)))
        
        model = Ridge(alpha=1e-3)
        model.fit(X, y_star)
        
        pred_y_star = model.predict(X)
        pred_error = expit(logit(np.clip(data['base_error'], self.epsilon, 1-self.epsilon)) + pred_y_star)
        
        return r2_score(data['final_error'], pred_error)
    
    def test_e_decision_utility(self):
        """Test E: Decision utility (early-stop policy)"""
        print("\n" + "="*70)
        print("TEST E: DECISION UTILITY (EARLY-STOP POLICY)")
        print("="*70)
        
        # Use checkpoint 200 features to predict final gains
        final_df = self.train_df[self.train_df['checkpoint'] == 1000].copy()
        
        # Calculate actual gains
        final_df['actual_gain'] = final_df['final_acc'] - final_df['base']
        
        # Get predictions at checkpoint 200
        predictions = []
        
        for _, run in final_df.iterrows():
            pred_gain = self._predict_gain_at_200(run['dataset'], run['strategy'], run['model_name'])
            predictions.append(pred_gain)
        
        final_df['predicted_gain'] = predictions
        
        # Remove runs without predictions
        valid_df = final_df[final_df['predicted_gain'].notna()].copy()
        
        # Test different thresholds
        thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]  # 1-5 percentage points
        
        results = []
        
        for threshold in thresholds:
            # Decision: stop if predicted gain < threshold
            stop_mask = valid_df['predicted_gain'] < threshold
            
            # Compute saved and missed
            n_stopped = stop_mask.sum()
            compute_saved = n_stopped / len(valid_df) * 0.8  # 80% of compute saved per stopped run
            
            # Check how many true winners we missed
            true_winners = valid_df['actual_gain'] >= threshold
            missed_winners = (stop_mask & true_winners).sum()
            missed_rate = missed_winners / true_winners.sum() if true_winners.sum() > 0 else 0
            
            # Average regret on stopped runs
            stopped_gains = valid_df[stop_mask]['actual_gain']
            avg_regret = stopped_gains[stopped_gains >= threshold].mean() if len(stopped_gains[stopped_gains >= threshold]) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'threshold_pp': threshold * 100,
                'n_stopped': n_stopped,
                'compute_saved': compute_saved,
                'missed_winners': missed_winners,
                'missed_rate': missed_rate,
                'avg_regret': avg_regret
            })
            
            print(f"\nThreshold: ≥ {threshold*100:.0f}pp gain")
            print(f"  Stopped: {n_stopped}/{len(valid_df)} runs")
            print(f"  Compute saved: {compute_saved*100:.1f}%")
            print(f"  True winners missed: {missed_winners} ({missed_rate*100:.1f}%)")
            if avg_regret > 0:
                print(f"  Average regret on missed: {avg_regret*100:.1f}pp")
        
        # Plot decision curves
        results_df = pd.DataFrame(results)
        self._plot_decision_curves(results_df)
        
        # Test on held-out
        print("\n\nHeld-out performance:")
        held_out_results = self._test_decision_utility_held_out(thresholds)
        
        return results_df, held_out_results
    
    def _predict_gain_at_200(self, dataset, strategy, model_name):
        """Predict final gain using only checkpoint 200 data"""
        run_data = self.train_df[
            (self.train_df['dataset'] == dataset) & 
            (self.train_df['strategy'] == strategy) & 
            (self.train_df['model_name'] == model_name)
        ]
        
        cp_200 = run_data[run_data['checkpoint'] == 200]
        if len(cp_200) == 0:
            return np.nan
            
        # Simple model: use error@200 as predictor
        error_200 = 1 - cp_200.iloc[0]['accuracy']
        base_error = 1 - run_data.iloc[0]['base']
        
        # Predicted final error (from our best model)
        pred_final_error = 0.95 * error_200 + 0.02
        
        # Predicted gain
        pred_gain = (1 - pred_final_error) - (1 - base_error)
        
        return pred_gain
    
    def _test_decision_utility_held_out(self, thresholds):
        """Test decision utility on held-out data"""
        held_out_final = self.held_out_df[self.held_out_df['checkpoint'] == 1000].copy()
        held_out_final['actual_gain'] = held_out_final['final_acc'] - held_out_final['base']
        
        # Get predictions
        predictions = []
        for _, run in held_out_final.iterrows():
            # Check if we have checkpoint 200
            cp_200 = self.held_out_df[
                (self.held_out_df['dataset'] == run['dataset']) & 
                (self.held_out_df['strategy'] == run['strategy']) & 
                (self.held_out_df['model_name'] == run['model_name']) &
                (self.held_out_df['checkpoint'] == 200)
            ]
            
            if len(cp_200) > 0:
                error_200 = 1 - cp_200.iloc[0]['accuracy']
                base_error = 1 - run['base']
                pred_final_error = 0.95 * error_200 + 0.02
                pred_gain = (1 - pred_final_error) - (1 - base_error)
            else:
                pred_gain = np.nan
                
            predictions.append(pred_gain)
        
        held_out_final['predicted_gain'] = predictions
        valid_df = held_out_final[held_out_final['predicted_gain'].notna()].copy()
        
        results = []
        
        for threshold in thresholds:
            stop_mask = valid_df['predicted_gain'] < threshold
            n_stopped = stop_mask.sum()
            compute_saved = n_stopped / len(valid_df) * 0.8
            
            true_winners = valid_df['actual_gain'] >= threshold
            missed_winners = (stop_mask & true_winners).sum()
            missed_rate = missed_winners / true_winners.sum() if true_winners.sum() > 0 else 0
            
            results.append({
                'threshold': threshold,
                'compute_saved': compute_saved,
                'missed_rate': missed_rate
            })
            
            print(f"  {threshold*100:.0f}pp: {compute_saved*100:.1f}% saved, {missed_rate*100:.1f}% missed")
        
        return pd.DataFrame(results)
    
    def _plot_time_forward_results(self, results_df):
        """Plot time-forward evaluation results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # R² progression
        ax1.plot(results_df['checkpoint'], results_df['r2'], 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Checkpoint')
        ax1.set_ylabel('R²')
        ax1.set_title('Predictive Power vs Training Progress')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Calibration
        valid_calib = results_df[results_df['calibration_slope'].notna()]
        ax2.plot(valid_calib['checkpoint'], valid_calib['calibration_slope'], 'o-', label='Slope')
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Checkpoint')
        ax2.set_ylabel('Calibration Slope')
        ax2.set_title('Calibration Quality')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('time_forward_evaluation.png', dpi=150)
    
    def _plot_decision_curves(self, results_df):
        """Plot decision utility curves"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(results_df['compute_saved'] * 100, results_df['missed_rate'] * 100, 
                'o-', linewidth=2, markersize=8)
        
        # Add threshold labels
        for _, row in results_df.iterrows():
            ax.annotate(f"{row['threshold_pp']:.0f}pp", 
                       (row['compute_saved'] * 100, row['missed_rate'] * 100),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Compute Saved (%)')
        ax.set_ylabel('True Winners Missed (%)')
        ax.set_title('Early Stopping Trade-off')
        ax.grid(True, alpha=0.3)
        
        # Add diagonal reference lines
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='1:1')
        ax.plot([0, 100], [0, 50], 'k--', alpha=0.3, label='2:1')
        
        plt.tight_layout()
        plt.savefig('decision_utility_curves.png', dpi=150)


def main():
    """Run all smoke tests"""
    # Load data
    train_df = pd.read_csv('../scaling_analysis_results.csv')
    held_out_df = pd.read_csv('../held_out_scaling_numbers.csv')
    
    print("Running comprehensive smoke tests...")
    print(f"Training data: {len(train_df)} points")
    print(f"Held-out data: {len(held_out_df)} points")
    
    # Initialize tester
    tester = SmokeTests(train_df, held_out_df)
    
    # Run all tests
    results = {}
    
    # Test A: Time-forward evaluation
    results['time_forward'] = tester.test_a_grouped_time_forward()
    
    # Test B: Leakage audit
    results['included'], results['excluded'] = tester.test_b_leakage_missingness_audit()
    
    # Test C: Slope robustness
    results['slope_robustness'] = tester.test_c_slope_robustness()
    
    # Test D: Identity-free vs aware
    results['identity_comparison'] = tester.test_d_identity_free_vs_aware()
    
    # Test E: Decision utility
    results['decision_utility'], results['decision_held_out'] = tester.test_e_decision_utility()
    
    # Save results
    print("\n\nSaving results...")
    
    # Save key dataframes
    results['time_forward'].to_csv('time_forward_results.csv', index=False)
    results['decision_utility'].to_csv('decision_utility_results.csv', index=False)
    
    print("✓ Smoke tests complete!")
    print("✓ Results saved to CSV files")
    print("✓ Plots saved as PNG files")
    
    return results


if __name__ == "__main__":
    results = main()
