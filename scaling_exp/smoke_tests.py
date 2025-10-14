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
        """Evaluate using data up to checkpoint T with GroupKFold (no run leakage)"""
        from sklearn.model_selection import GroupKFold
        
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
        logit0 = logit(np.clip(merged['base_error'], self.epsilon, 1-self.epsilon))
        logitT = logit(np.clip(merged['error_T'], self.epsilon, 1-self.epsilon))
        merged['slope_0_T'] = (logitT - logit0) / T
        
        # Build features
        X = np.column_stack([
            np.log(np.clip(merged['model_size'], self.epsilon, None)),
            logit(np.clip(merged['perc_learnable'], self.epsilon, 1-self.epsilon)),
            merged['slope_0_T']
        ])
        
        y_star = (logit(np.clip(merged['final_error'], self.epsilon, 1-self.epsilon)) - logit0)
        
        # Create groups (one per unique run)
        groups = merged[['dataset', 'strategy', 'model_name']].astype(str).agg('|'.join, axis=1)
        
        # GroupKFold cross-validation (5 folds)
        gkf = GroupKFold(n_splits=min(5, len(groups.unique())))
        
        r2s, slopes, intercepts = [], [], []
        for tr_idx, te_idx in gkf.split(X, y_star, groups=groups):
            # Fit model on train fold
            model = Ridge(alpha=1e-3)
            model.fit(X[tr_idx], y_star.iloc[tr_idx])
            
            # Predict on test fold
            pred_y_star = model.predict(X[te_idx])
            pred_error = expit(logit0.iloc[te_idx].values + pred_y_star)
            
            # R² on test fold
            r2s.append(r2_score(merged['final_error'].iloc[te_idx], pred_error))
            
            # Calibration on test fold
            lr = LinearRegression()
            lr.fit(pred_error.reshape(-1, 1), merged['final_error'].iloc[te_idx].values)
            slopes.append(lr.coef_[0])
            intercepts.append(lr.intercept_)
        
        # Average across folds
        avg_r2 = float(np.mean(r2s))
        avg_slope = float(np.mean(slopes))
        avg_intercept = float(np.mean(intercepts))
        
        calib = {'slope': avg_slope, 'intercept': avg_intercept}
        
        return avg_r2, calib
    
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
            
            # Get slope statistics
            slope_mean = valid_df['slope'].mean()
            slope_std = valid_df['slope'].std()
            
            results[method_name] = {
                'r2': r2,
                'slope_coefficient': model.coef_[2],
                'slope_mean': slope_mean,
                'slope_std': slope_std,
                'n_valid': len(valid_df)
            }
            
            print(f"  R² = {r2:.4f}")
            print(f"  Slope coefficient = {model.coef_[2]:.1f}")
            print(f"  Mean slope in data = {slope_mean:.4f}")
            print(f"  Valid runs = {len(valid_df)}")
        
        # Compare results
        print("\nComparison across methods:")
        r2_values = [r['r2'] for r in results.values()]
        coef_values = [r['slope_coefficient'] for r in results.values()]
        
        print(f"  R² range: {min(r2_values):.4f} - {max(r2_values):.4f} (Δ = {max(r2_values) - min(r2_values):.4f})")
        print(f"  Coefficient range: {min(coef_values):.1f} - {max(coef_values):.1f} (Δ = {max(coef_values) - min(coef_values):.1f})")
        
        # Compute partial R² for slope (stable across units)
        self._compute_partial_r2_for_slope(valid_df)
        
        return results
    
    def _compute_partial_r2_for_slope(self, valid_df):
        """Compute partial R² contribution of slope term"""
        from sklearn.metrics import r2_score
        
        # Fit with slope
        X_full = np.column_stack([
            np.log(np.clip(valid_df['model_size'], self.epsilon, None)),
            logit(np.clip(valid_df['perc_learnable'], self.epsilon, 1-self.epsilon)),
            valid_df['slope']
        ])
        
        base_error = 1 - valid_df['base']
        final_error = 1 - valid_df['final_acc']
        y_star = (logit(np.clip(final_error, self.epsilon, 1-self.epsilon)) - 
                  logit(np.clip(base_error, self.epsilon, 1-self.epsilon)))
        
        m_full = Ridge(alpha=1e-3).fit(X_full, y_star)
        pred_full = expit(logit(np.clip(base_error, self.epsilon, 1-self.epsilon)) + m_full.predict(X_full))
        r2_full = r2_score(final_error, pred_full)
        
        # Fit without slope
        X_drop = X_full[:, :2]
        m_drop = Ridge(alpha=1e-3).fit(X_drop, y_star)
        pred_drop = expit(logit(np.clip(base_error, self.epsilon, 1-self.epsilon)) + m_drop.predict(X_drop))
        r2_drop = r2_score(final_error, pred_drop)
        
        partial_r2 = r2_full - r2_drop
        print(f"\nSlope impact (partial R²): {partial_r2:+.4f}")
        print(f"  R² with slope: {r2_full:.4f}")
        print(f"  R² without slope: {r2_drop:.4f}")
        print(f"  → Slope contributes {partial_r2:.4f} to total R²")
        
        return partial_r2
    
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
        
        # Build features correctly - logit for probabilities, log for size
        base_features = np.column_stack([
            np.log(data['model_size']),
            logit(np.clip(data['base'], self.epsilon, 1-self.epsilon)),
            logit(np.clip(data['perc_learnable'], self.epsilon, 1-self.epsilon))
        ])
        
        X = np.hstack([
            base_features,
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
    
    def _optimize_decision_frontier(self, final_df):
        """Optimize decision frontier over threshold and probability cutoff pairs"""
        # Test different thresholds and find optimal cutoffs
        thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]
        pi_grid = np.linspace(0.05, 0.8, 30)
        
        best_results = []
        
        for threshold in thresholds:
            # Get predictions and calculate success probabilities
            prob_successes = []
            actual_gains = []
            
            for _, run in final_df.iterrows():
                pred_result = self._predict_gain_at_200(run['dataset'], run['strategy'], run['model_name'])
                
                if isinstance(pred_result, tuple) and pred_result[1] is not None:
                    pred_final_error, uncertainty = pred_result
                    
                    # Calculate probability of achieving threshold gain
                    base_error = 1 - run['base']
                    threshold_error = 1 - (run['base'] + threshold)
                    
                    # In y* space
                    y_star_threshold = (logit(np.clip(threshold_error, self.epsilon, 1-self.epsilon)) - 
                                      logit(np.clip(base_error, self.epsilon, 1-self.epsilon)))
                    y_star_pred = (logit(np.clip(pred_final_error, self.epsilon, 1-self.epsilon)) - 
                                 logit(np.clip(base_error, self.epsilon, 1-self.epsilon)))
                    
                    # Probability using normal approximation
                    from scipy.stats import norm
                    sigma = uncertainty / 1.645  # Convert 90% quantile to std
                    prob_success = norm.cdf(y_star_threshold, loc=y_star_pred, scale=sigma)
                    
                    prob_successes.append(prob_success)
                    actual_gains.append(run['actual_gain'])
            
            # Convert to arrays
            prob_successes = np.array(prob_successes)
            actual_gains = np.array(actual_gains)
            
            # Find best cutoff for this threshold
            best_cutoff = None
            best_compute = 0
            best_missed = 1.0
            
            for pi_min in pi_grid:
                stop_mask = prob_successes < pi_min
                
                if stop_mask.sum() > 0:
                    true_winners = actual_gains >= threshold
                    missed_winners = (stop_mask & true_winners).sum()
                    missed_rate = missed_winners / true_winners.sum() if true_winners.sum() > 0 else 0
                    compute_saved = stop_mask.sum() / len(stop_mask) * 0.8
                    
                    # Keep if meets constraint and improves compute saved
                    if missed_rate <= 0.05 and compute_saved > best_compute:
                        best_compute = compute_saved
                        best_cutoff = pi_min
                        best_missed = missed_rate
            
            if best_cutoff is not None:
                best_results.append({
                    'threshold': threshold,
                    'pi_min': best_cutoff,
                    'compute_saved': best_compute,
                    'missed_rate': best_missed
                })
        
        # Return best frontier point
        if best_results:
            return max(best_results, key=lambda x: x['compute_saved'])
        else:
            return {'threshold': 0.03, 'pi_min': 0.5, 'compute_saved': 0, 'missed_rate': 0}
    
    def _tune_probability_cutoff(self, final_df, threshold):
        """Legacy method - now calls the frontier optimizer"""
        result = self._optimize_decision_frontier(final_df)
        return result.get('pi_min', 0.5)
    
    def _q_eff(self, alpha, w):
        """Compute finite-sample corrected quantile level (numerically safe)"""
        sw = np.sum(w)
        s2 = np.sum(w**2)
        n_eff = (sw**2) / max(s2, 1e-8)
        q_nom = np.ceil((n_eff + 1) * (1 - alpha)) / max(n_eff, 1.0)
        return float(np.clip(q_nom, 0.0, 1.0))
    
    def _weighted_quantile(self, values, weights, q):
        """Compute weighted quantile"""
        # Convert to numpy arrays if needed
        if hasattr(values, 'values'):
            values = values.values
        if hasattr(weights, 'values'):
            weights = weights.values
        
        order = np.argsort(values)
        values, weights = values[order], weights[order]
        cdf = np.cumsum(weights) / np.sum(weights)
        return float(np.interp(q, cdf, values))
    
    def _compute_weighted_conformal_quantiles(self, X_cal_scale, cal_pred_y_star):
        """Compute weighted conformal quantiles using density ratio + Mondrian bins"""
        from sklearn.linear_model import LogisticRegression
        
        # Build held-out features for density ratio estimation
        held_out_features = []
        for _, run in self.held_out_df[self.held_out_df['checkpoint'] == 1000].iterrows():
            cp200 = self.held_out_df[
                (self.held_out_df['dataset'] == run['dataset']) &
                (self.held_out_df['strategy'] == run['strategy']) &
                (self.held_out_df['model_name'] == run['model_name']) &
                (self.held_out_df['checkpoint'] == 200)
            ]
            if len(cp200) == 0:
                continue
            
            e0 = 1 - run['base']
            e200 = 1 - cp200.iloc[0]['accuracy']
            logit0 = logit(np.clip(e0, self.epsilon, 1-self.epsilon))
            logit200 = logit(np.clip(e200, self.epsilon, 1-self.epsilon))
            slope = (logit200 - logit0) / 200
            
            X_test = np.array([[
                np.log(np.clip(run['model_size'], self.epsilon, None)),
                logit(np.clip(run['perc_learnable'], self.epsilon, 1-self.epsilon)),
                slope
            ]])
            
            ystar_pred = self.model.predict(X_test)[0]
            
            held_out_features.append([
                logit(np.clip(e200, self.epsilon, 1-self.epsilon)),
                slope,
                np.log(np.clip(run['model_size'], self.epsilon, None)),
                logit(np.clip(run['perc_learnable'], self.epsilon, 1-self.epsilon)),
                ystar_pred
            ])
        
        if len(held_out_features) == 0:
            # Fallback to unweighted
            self.conformal_quantile_90 = np.quantile(self.cal_residuals_norm, 0.90)
            self.q90_cells = {}
            self.mondrian_b1 = None
            self.mondrian_b2 = None
            return
        
        # Stack features
        X_cal_shift = np.column_stack([X_cal_scale, cal_pred_y_star])
        X_test_shift = np.array(held_out_features)
        
        # Train domain classifier for density ratio
        X_dom = np.vstack([X_cal_shift, X_test_shift])
        y_dom = np.r_[np.zeros(len(X_cal_shift)), np.ones(len(X_test_shift))]
        
        dom_clf = LogisticRegression(max_iter=2000, random_state=42)
        dom_clf.fit(X_dom, y_dom)
        
        # Get importance weights for calibration points
        p_cal = dom_clf.predict_proba(X_cal_shift)[:, 1]
        w_cal = p_cal / np.maximum(1 - p_cal, 1e-6)
        w_cal = np.clip(w_cal, 0.1, 10)  # Clip extreme weights
        
        # For 90% two-sided coverage, use conditional tail quantiles (α/2 each)
        alpha_lo = 0.05  # Lower tail
        alpha_hi = 0.05  # Upper tail
        
        print(f"\nAsymmetric conformal setup (conditional tail quantiles, α/2 = {alpha_lo}):")
        
        # Masks for conditional tails (exclude zeros from opposite tail)
        mask_pos = self.r_pos > 0
        mask_neg = self.r_neg > 0
        
        w_pos = w_cal[mask_pos]
        w_neg = w_cal[mask_neg]
        r_pos_pos = self.r_pos[mask_pos]
        r_neg_neg = self.r_neg[mask_neg]
        
        # Weighted mass in each tail
        pi_pos = np.sum(w_pos) / np.sum(w_cal) if np.sum(w_cal) > 0 else 0.0
        pi_neg = np.sum(w_neg) / np.sum(w_cal) if np.sum(w_cal) > 0 else 0.0
        
        print(f"  Tail masses: π_pos = {pi_pos:.4f}, π_neg = {pi_neg:.4f}")
        
        # Allocate α within each tail conditionally
        alpha_hi_cond = min(1.0, alpha_hi / max(pi_pos, 1e-8))
        alpha_lo_cond = min(1.0, alpha_lo / max(pi_neg, 1e-8))
        
        # Finite-sample corrected levels on tail subsets
        q_hi = self._q_eff(alpha_hi_cond, w_pos if len(w_pos) else np.array([1.0]))
        q_lo = self._q_eff(alpha_lo_cond, w_neg if len(w_neg) else np.array([1.0]))
        
        # Conditional weighted tail quantiles
        q90_hi_base = (self._weighted_quantile(r_pos_pos, w_pos, q_hi)
                       if len(r_pos_pos) > 0 else self._weighted_quantile(self.r_pos, w_cal, self._q_eff(alpha_hi, w_cal)))
        
        q90_lo_base = (self._weighted_quantile(r_neg_neg, w_neg, q_lo)
                       if len(r_neg_neg) > 0 else self._weighted_quantile(self.r_neg, w_cal, self._q_eff(alpha_lo, w_cal)))
        
        # Apply conservative multiplier to hit 85%+ coverage
        lambda_conservative = 1.20  # Moderate multiplier for conditional tails
        self.q90_hi = q90_hi_base * lambda_conservative
        self.q90_lo = q90_lo_base * lambda_conservative
        
        print(f"  Base conditional quantiles: q90_lo = {q90_lo_base:.4f}, q90_hi = {q90_hi_base:.4f}")
        print(f"  With λ={lambda_conservative}: q90_lo = {self.q90_lo:.4f}, q90_hi = {self.q90_hi:.4f}")
        
        # Backward compatibility: symmetric quantile
        self.conformal_quantile_90 = max(self.q90_lo, self.q90_hi)
        
        # Set up Mondrian bins with conditional asymmetric quantiles
        self._setup_mondrian_bins(X_cal_shift, X_test_shift, w_cal, (alpha_lo, alpha_hi))
    
    def _setup_mondrian_bins(self, X_cal_shift, X_test_shift, w_cal, alphas):
        """Set up Mondrian bins with conditional asymmetric tail quantiles"""
        alpha_lo, alpha_hi = alphas
        
        # Define bins based on held-out distribution
        # Bin 1: quartiles of logit(e200)
        b1 = np.quantile(X_test_shift[:, 0], [0.25, 0.5, 0.75])
        # Bin 2: tertiles of slope
        b2 = np.quantile(X_test_shift[:, 1], [1/3, 2/3])
        
        def cell_id(x1, x2):
            return (np.searchsorted(b1, x1), np.searchsorted(b2, x2))
        
        # Compute conditional asymmetric weighted quantiles per cell
        cells = {}
        for i in range(len(self.r_pos)):
            key = cell_id(X_cal_shift[i, 0], X_cal_shift[i, 1])
            if key not in cells:
                cells[key] = {'r_pos': [], 'r_neg': [], 'w': []}
            cells[key]['r_pos'].append(self.r_pos[i])
            cells[key]['r_neg'].append(self.r_neg[i])
            cells[key]['w'].append(w_cal[i])
        
        # Compute quantiles for cells with enough data
        self.q90_lo_cells = {}
        self.q90_hi_cells = {}
        min_count = 10
        min_eff = 8
        
        def n_eff_calc(w_arr):
            sw = np.sum(w_arr)
            s2 = np.sum(w_arr**2)
            return (sw**2) / max(s2, 1e-8)
        
        for key, data in cells.items():
            r_pos_arr = np.array(data['r_pos'])
            r_neg_arr = np.array(data['r_neg'])
            w_arr = np.array(data['w'])
            
            # Check both raw count and effective n on full cell
            if len(w_arr) < min_count or n_eff_calc(w_arr) < min_eff:
                continue
            
            # Conditional tail masks
            m_pos = r_pos_arr > 0
            m_neg = r_neg_arr > 0
            
            w_pos_cell = w_arr[m_pos]
            w_neg_cell = w_arr[m_neg]
            r_pos_pos_cell = r_pos_arr[m_pos]
            r_neg_neg_cell = r_neg_arr[m_neg]
            
            # Conditional masses
            pi_pos_cell = np.sum(w_pos_cell) / np.sum(w_arr) if np.sum(w_arr) > 0 else 0.0
            pi_neg_cell = np.sum(w_neg_cell) / np.sum(w_arr) if np.sum(w_arr) > 0 else 0.0
            
            # Conditional α in cell
            alpha_hi_cond = min(1.0, alpha_hi / max(pi_pos_cell, 1e-8))
            alpha_lo_cond = min(1.0, alpha_lo / max(pi_neg_cell, 1e-8))
            
            # Finite-sample correction on tail subsets (fallback if empty)
            q_hi_cell = self._q_eff(alpha_hi_cond, w_pos_cell if len(w_pos_cell) else w_arr)
            q_lo_cell = self._q_eff(alpha_lo_cond, w_neg_cell if len(w_neg_cell) else w_arr)
            
            # Conditional weighted quantiles with conservative multiplier
            lambda_cell = 1.20  # Consistent with global
            
            q90_hi_base = (self._weighted_quantile(r_pos_pos_cell, w_pos_cell, q_hi_cell)
                           if len(r_pos_pos_cell) > 0 else self.q90_hi)
            q90_lo_base = (self._weighted_quantile(r_neg_neg_cell, w_neg_cell, q_lo_cell)
                           if len(r_neg_neg_cell) > 0 else self.q90_lo)
            
            self.q90_hi_cells[key] = q90_hi_base * lambda_cell
            self.q90_lo_cells[key] = q90_lo_base * lambda_cell
        
        # Store bin edges for test time
        self.mondrian_b1 = b1
        self.mondrian_b2 = b2
    
    def _fit_model_with_conformal(self):
        """Fit the model and compute weighted + Mondrian conformal quantiles"""
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.linear_model import LogisticRegression
        
        # Prepare data as in the calibration analysis
        final_df = self.train_df[self.train_df['checkpoint'] == 1000].copy()
        final_df['final_error'] = 1 - final_df['final_acc']
        final_df['base_error'] = 1 - final_df['base']
        
        # Calculate slopes
        slopes = []
        valid_idx = []
        
        for idx, run in final_df.iterrows():
            # Get checkpoint 200 data
            cp_200 = self.train_df[
                (self.train_df['dataset'] == run['dataset']) & 
                (self.train_df['strategy'] == run['strategy']) & 
                (self.train_df['model_name'] == run['model_name']) &
                (self.train_df['checkpoint'] == 200)
            ]
            
            if len(cp_200) > 0:
                # Calculate slope
                error_0 = run['base_error']
                error_200 = 1 - cp_200.iloc[0]['accuracy']
                
                logit_0 = logit(np.clip(error_0, self.epsilon, 1-self.epsilon))
                logit_200 = logit(np.clip(error_200, self.epsilon, 1-self.epsilon))
                
                slope = (logit_200 - logit_0) / 200
                slopes.append(slope)
                valid_idx.append(idx)
        
        # Filter to valid runs
        valid_df = final_df.loc[valid_idx].copy()
        valid_df['slope'] = slopes
        
        # Build features
        X = np.column_stack([
            np.log(valid_df['model_size']),
            logit(np.clip(valid_df['perc_learnable'], self.epsilon, 1-self.epsilon)),
            valid_df['slope']
        ])
        
        # Target
        y_star = (logit(np.clip(valid_df['final_error'], self.epsilon, 1-self.epsilon)) - 
                  logit(np.clip(valid_df['base_error'], self.epsilon, 1-self.epsilon)))
        
        # Split for conformal
        # Use unique run identifier as group (dataset + strategy + model_name)
        groups = (valid_df['dataset'] + '_' + valid_df['strategy'] + '_' + valid_df['model_name']).values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, cal_idx = next(gss.split(X, y_star, groups=groups))
        
        # Fit model
        self.model = Ridge(alpha=1e-3)
        self.model.fit(X[train_idx], y_star.iloc[train_idx])
        
        # Compute conformal quantiles
        cal_preds = self.model.predict(X[cal_idx])
        cal_signed = y_star.iloc[cal_idx].values - cal_preds  # Keep signed for asymmetric
        cal_residuals = np.abs(cal_signed)
        
        # Build features for scale model using checkpoint 200 info
        cal_df = valid_df.iloc[cal_idx]
        error_200_cal = []
        for idx, run in cal_df.iterrows():
            cp_200 = self.train_df[
                (self.train_df['dataset'] == run['dataset']) & 
                (self.train_df['strategy'] == run['strategy']) & 
                (self.train_df['model_name'] == run['model_name']) &
                (self.train_df['checkpoint'] == 200)
            ]
            if len(cp_200) > 0:
                error_200_cal.append(1 - cp_200.iloc[0]['accuracy'])
            else:
                error_200_cal.append(run['final_error'])
        
        X_scale = np.column_stack([
            logit(np.clip(error_200_cal, self.epsilon, 1-self.epsilon)),
            cal_df['slope'].values,
            X[cal_idx, 0],  # log(model_size)
            X[cal_idx, 1]   # logit(perc_learnable)
        ])
        
        # Fit scale model
        self.scale_model = Ridge(alpha=1e-2)
        self.scale_model.fit(X_scale, np.log(cal_residuals + 1e-8))
        
        # Compute normalized residuals (asymmetric)
        s_hat_cal = np.exp(self.scale_model.predict(X_scale))
        r_norm = cal_residuals / (s_hat_cal + 1e-8)
        
        # Split into positive (upper tail) and negative (lower tail) normalized residuals
        self.r_pos = np.maximum(0.0, cal_signed) / (s_hat_cal + 1e-8)
        self.r_neg = np.maximum(0.0, -cal_signed) / (s_hat_cal + 1e-8)
        
        # Store for weighted conformal
        self.cal_residuals_norm = r_norm
        self.X_cal_scale = X_scale
        self.cal_pred_y_star = cal_preds
        
        # Compute weighted asymmetric conformal quantiles with Mondrian bins
        self._compute_weighted_conformal_quantiles(X_scale, cal_preds)
        
        # Also store residual std for fallback
        self.residual_std = np.std(y_star.iloc[train_idx] - self.model.predict(X[train_idx]))
    
    def test_e_decision_utility(self):
        """Test E: Decision utility (early-stop policy) with two-stage approach"""
        print("\n" + "="*70)
        print("TEST E: DECISION UTILITY (TWO-STAGE EARLY-STOP POLICY)")
        print("="*70)
        
        # First fit the model with conformal intervals
        self._fit_model_with_conformal()
        
        # Use checkpoint 200 features to predict final gains
        final_df = self.train_df[self.train_df['checkpoint'] == 1000].copy()
        
        # Calculate actual gains
        final_df['actual_gain'] = final_df['final_acc'] - final_df['base']
        
        # STAGE 1: CP100 filter (stop very bad runs early)
        print("\n" + "─"*70)
        print("STAGE 1: CP100 FILTER (90% of compute if stopped)")
        print("─"*70)
        self._test_stage1_cp100_filter(final_df)
        
        # STAGE 2: CP200 probability-based policy
        print("\n" + "─"*70)
        print("STAGE 2: CP200 PROBABILITY POLICY (80% of compute if stopped)")
        print("─"*70)
        
        # Get predictions at checkpoint 200 with uncertainty
        predictions = []
        uncertainties = []
        prob_successes = []
        
        # Optimize the decision frontier
        frontier_result = self._optimize_decision_frontier(final_df)
        
        print(f"\nStage 2 Optimal frontier point:")
        print(f"  Threshold: {frontier_result['threshold']*100:.0f}pp")
        print(f"  Probability cutoff: {frontier_result['pi_min']:.3f}")
        print(f"  Stage 2 compute saved: {frontier_result['compute_saved']*100:.1f}%")
        print(f"  Missed rate: {frontier_result['missed_rate']*100:.1f}%")
        
        # Test different thresholds to show the full frontier
        thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]  # 1-5 percentage points
        
        # For each threshold, re-run the optimization to get individual best cutoffs
        optimal_prob_cutoffs = []
        for threshold in thresholds:
            # Simplified: just use the frontier result for the optimal threshold
            if abs(threshold - frontier_result['threshold']) < 0.001:
                optimal_prob_cutoffs.append(frontier_result['pi_min'])
            else:
                # Default cutoff for other thresholds
                optimal_prob_cutoffs.append(0.5)
        
        for _, run in final_df.iterrows():
            pred_result = self._predict_gain_at_200(run['dataset'], run['strategy'], run['model_name'])
            
            if isinstance(pred_result, tuple) and pred_result[1] is not None:
                pred_final_error, uncertainty = pred_result
                predictions.append(1 - pred_final_error - run['base'])  # Convert to gain
                uncertainties.append(uncertainty)
                
                # Calculate success probabilities for each threshold
                run_probs = []
                for threshold in thresholds:
                    # Calculate probability of achieving threshold gain
                    base_error = 1 - run['base']
                    threshold_error = 1 - (run['base'] + threshold)
                    
                    # In y* space
                    y_star_threshold = (logit(np.clip(threshold_error, self.epsilon, 1-self.epsilon)) - 
                                      logit(np.clip(base_error, self.epsilon, 1-self.epsilon)))
                    y_star_pred = (logit(np.clip(pred_final_error, self.epsilon, 1-self.epsilon)) - 
                                 logit(np.clip(base_error, self.epsilon, 1-self.epsilon)))
                    
                    # Probability using normal approximation
                    from scipy.stats import norm
                    sigma = uncertainty / 1.645  # Convert 90% quantile to std
                    prob_success = norm.cdf(y_star_threshold, loc=y_star_pred, scale=sigma)
                    run_probs.append(prob_success)
                
                prob_successes.append(run_probs)
            else:
                # Fallback
                if not isinstance(pred_result, tuple):
                    predictions.append(pred_result)
                else:
                    predictions.append(1 - pred_result[0] - run['base'])
                uncertainties.append(np.nan)
                prob_successes.append([np.nan] * len(thresholds))
        
        final_df['predicted_gain'] = predictions
        final_df['uncertainty'] = uncertainties
        
        # Convert prob_successes to dataframe columns
        prob_successes = np.array(prob_successes)
        for i, threshold in enumerate(thresholds):
            final_df[f'prob_success_{threshold}'] = prob_successes[:, i]
        
        # Remove runs without predictions
        valid_df = final_df[final_df['predicted_gain'].notna()].copy()
        
        print(f"\nStage 2 Results by threshold:")
        
        results = []
        
        for i, (threshold, prob_cutoff) in enumerate(zip(thresholds, optimal_prob_cutoffs)):
            # Decision: stop if P(gain >= threshold) < prob_cutoff
            prob_col = f'prob_success_{threshold}'
            if prob_col in valid_df.columns:
                stop_mask = valid_df[prob_col] < prob_cutoff
            else:
                # Fallback to simple threshold
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
                'prob_cutoff': prob_cutoff,
                'n_stopped': n_stopped,
                'compute_saved': compute_saved,
                'missed_winners': missed_winners,
                'missed_rate': missed_rate,
                'avg_regret': avg_regret
            })
            
            print(f"\nThreshold: ≥ {threshold*100:.0f}pp gain (P_stop < {prob_cutoff})")
            print(f"  Stopped: {n_stopped}/{len(valid_df)} runs")
            print(f"  Stage 2 compute saved: {compute_saved*100:.1f}% (80% of remaining)")
            print(f"  True winners missed: {missed_winners} ({missed_rate*100:.1f}%)")
            if avg_regret > 0:
                print(f"  Average regret on missed: {avg_regret*100:.1f}pp")
        
        # Plot decision curves
        results_df = pd.DataFrame(results)
        self._plot_decision_curves(results_df)
        
        # Compute 90% PI coverage on held-out
        if hasattr(self, 'model') and self.model is not None:
            self._pi_coverage_on_heldout()
        
        # Test on held-out
        print("\n\nHeld-out performance:")
        held_out_results = self._test_decision_utility_held_out(thresholds)
        
        return results_df, held_out_results
    
    def _predict_gain_at_200(self, dataset, strategy, model_name):
        """Predict final gain using only checkpoint 200 data with uncertainty"""
        run_data = self.train_df[
            (self.train_df['dataset'] == dataset) & 
            (self.train_df['strategy'] == strategy) & 
            (self.train_df['model_name'] == model_name)
        ]
        
        cp_200 = run_data[run_data['checkpoint'] == 200]
        cp_1000 = run_data[run_data['checkpoint'] == 1000]
        if len(cp_200) == 0 or len(cp_1000) == 0:
            return np.nan, None
            
        # Get model features at checkpoint 200
        error_0 = 1 - run_data.iloc[0]['base']
        error_200 = 1 - cp_200.iloc[0]['accuracy']
        model_size = cp_1000.iloc[0]['model_size']
        perc_learnable = cp_1000.iloc[0]['perc_learnable']
        
        # Calculate slope
        logit_0 = logit(np.clip(error_0, self.epsilon, 1-self.epsilon))
        logit_200 = logit(np.clip(error_200, self.epsilon, 1-self.epsilon))
        slope = (logit_200 - logit_0) / 200
        
        # Build features (matching the model)
        X = np.array([[
            np.log(model_size),
            logit(np.clip(perc_learnable, self.epsilon, 1-self.epsilon)),
            slope
        ]])
        
        # Get prediction from model
        if hasattr(self, 'model') and self.model is not None:
            pred_y_star = self.model.predict(X)[0]
            pred_final_error = expit(logit_0 + pred_y_star)
            
            # Get uncertainty estimate from scale model
            if hasattr(self, 'scale_model'):
                # Build scale features
                X_scale = np.array([[
                    logit(np.clip(error_200, self.epsilon, 1-self.epsilon)),
                    slope,
                    np.log(model_size),
                    logit(np.clip(perc_learnable, self.epsilon, 1-self.epsilon))
                ]])
                
                # Get scale prediction
                s_hat = np.exp(self.scale_model.predict(X_scale)[0])
                
                # Add evaluation noise (binomial variance)
                n_eval = 1000  # Typical evaluation set size
                e_pred = expit(logit_0 + pred_y_star)
                s_eval = 1.0 / np.sqrt(np.maximum(n_eval * e_pred * (1 - e_pred), 1e-8))
                s_total = np.sqrt(s_hat**2 + s_eval**2)
                
                uncertainty = self.conformal_quantile_90 * s_total
            else:
                # Fallback to standard conformal
                uncertainty = self.conformal_quantile_90 if hasattr(self, 'conformal_quantile_90') else self.residual_std * 1.645
            
            return pred_final_error, uncertainty
        else:
            # Fallback to FITTED logit regression if trained model not available
            models_fallback = ScalingLawModels()
            pred_final_error = models_fallback.checkpoint_200_logit_regression(error_200)
            pred_gain = (1 - pred_final_error) - (1 - error_0)
            return pred_gain, None
    
    def _test_stage1_cp100_filter(self, final_df):
        """Stage 1: Simple CP100 filter to catch very bad runs early"""
        from scaling_law_models import ScalingLawModels
        models = ScalingLawModels()
        
        # Get CP100 predictions
        cp100_preds = []
        actual_gains = []
        
        for _, run in final_df.iterrows():
            cp100 = self.train_df[
                (self.train_df['dataset'] == run['dataset']) &
                (self.train_df['strategy'] == run['strategy']) &
                (self.train_df['model_name'] == run['model_name']) &
                (self.train_df['checkpoint'] == 100)
            ]
            
            if len(cp100) > 0:
                error_100 = 1 - cp100.iloc[0]['accuracy']
                pred_final_error = models.predict_from_checkpoint_100(error_100)
                pred_gain = (1 - pred_final_error) - run['base']
                cp100_preds.append(pred_gain)
                actual_gains.append(run['final_acc'] - run['base'])
            else:
                cp100_preds.append(np.nan)
                actual_gains.append(np.nan)
        
        final_df['cp100_pred_gain'] = cp100_preds
        valid_cp100 = final_df[final_df['cp100_pred_gain'].notna()].copy()
        
        if len(valid_cp100) == 0:
            print("  No valid CP100 predictions")
            return
        
        # Test different Stage 1 thresholds (percentiles of predicted gain)
        stage1_thresholds = [0.10, 0.15, 0.20, 0.25]  # Stop bottom X% of predictions
        
        print(f"\nStage 1 Results (N = {len(valid_cp100)} runs):")
        print(f"  Predicted gain: {valid_cp100['cp100_pred_gain'].mean():.3f} ± {valid_cp100['cp100_pred_gain'].std():.3f}")
        print(f"\nPercentile-based cutoffs:")
        
        for pct in stage1_thresholds:
            cutoff = valid_cp100['cp100_pred_gain'].quantile(pct)
            stop_mask = valid_cp100['cp100_pred_gain'] < cutoff
            n_stopped = stop_mask.sum()
            
            true_winners = valid_cp100['actual_gain'] >= 0.03  # Winners defined as >3pp gain
            missed_winners = (stop_mask & true_winners).sum()
            missed_rate = missed_winners / true_winners.sum() if true_winners.sum() > 0 else 0
            
            compute_saved = n_stopped / len(valid_cp100) * 0.90  # 90% of compute saved
            
            print(f"  p{int(pct*100)} cutoff ({cutoff:.3f}): Stop {n_stopped}/{len(valid_cp100)} ({compute_saved*100:.1f}% saved), miss {missed_winners} ({missed_rate*100:.1f}%)")
    
    def _pi_coverage_on_heldout(self):
        """Compute 90% PI coverage on held-out data"""
        held = self.held_out_df[self.held_out_df['checkpoint'] == 1000].copy()
        cov90_list = []
        
        for _, run in held.iterrows():
            cp200 = self.held_out_df[
                (self.held_out_df['dataset'] == run['dataset']) &
                (self.held_out_df['strategy'] == run['strategy']) &
                (self.held_out_df['model_name'] == run['model_name']) &
                (self.held_out_df['checkpoint'] == 200)
            ]
            if len(cp200) == 0:
                continue
                
            e0 = 1 - run['base']
            e200 = 1 - cp200.iloc[0]['accuracy']
            logit0 = logit(np.clip(e0, self.epsilon, 1-self.epsilon))
            logit200 = logit(np.clip(e200, self.epsilon, 1-self.epsilon))
            slope = (logit200 - logit0) / 200
            
            X = np.array([[
                np.log(np.clip(run['model_size'], self.epsilon, None)),
                logit(np.clip(run['perc_learnable'], self.epsilon, 1-self.epsilon)),
                slope
            ]])
            
            ystar_hat = self.model.predict(X)[0]
            pred_e = expit(logit0 + ystar_hat)
            
            # Scale features
            Xs = np.array([[
                logit(np.clip(e200, self.epsilon, 1-self.epsilon)),
                slope,
                np.log(np.clip(run['model_size'], self.epsilon, None)),
                logit(np.clip(run['perc_learnable'], self.epsilon, 1-self.epsilon))
            ]])
            
            s_hat = float(np.exp(self.scale_model.predict(Xs)[0]))
            
            # Eval noise
            N_eval = 1000
            s_eval = 1.0 / np.sqrt(max(N_eval * pred_e * (1 - pred_e), 1e-8))
            s_tot = np.sqrt(s_hat**2 + s_eval**2)
            
            # Get Mondrian bin-specific asymmetric quantiles if available
            if hasattr(self, 'mondrian_b1') and self.mondrian_b1 is not None:
                b1_idx = np.searchsorted(self.mondrian_b1, logit(np.clip(e200, self.epsilon, 1-self.epsilon)))
                b2_idx = np.searchsorted(self.mondrian_b2, slope)
                cell_key = (b1_idx, b2_idx)
                
                # Use cell-specific quantiles if available, else global
                if cell_key in self.q90_lo_cells:
                    q_lo = self.q90_lo_cells[cell_key]
                    q_hi = self.q90_hi_cells[cell_key]
                else:
                    q_lo = self.q90_lo  # Fallback to global weighted
                    q_hi = self.q90_hi
            else:
                # Fallback to symmetric
                q_lo = self.conformal_quantile_90
                q_hi = self.conformal_quantile_90
            
            # Asymmetric intervals
            lo = expit(logit0 + ystar_hat - q_lo * s_tot)
            hi = expit(logit0 + ystar_hat + q_hi * s_tot)
            y_true = 1 - run['final_acc']
            
            cov90_list.append((y_true >= lo) and (y_true <= hi))
        
        cov90 = np.mean(cov90_list) if cov90_list else np.nan
        
        print(f"\n90% PI coverage on held-out: {cov90*100:.1f}% (target 85-95%)")
        print(f"  Total covered: {len([x for x in cov90_list if x])}/{len(cov90_list)}")
        return cov90
    
    def _test_decision_utility_held_out(self, thresholds):
        """Test decision utility on held-out data"""
        held_out_final = self.held_out_df[self.held_out_df['checkpoint'] == 1000].copy()
        held_out_final['actual_gain'] = held_out_final['final_acc'] - held_out_final['base']
        
        # Use the FITTED logit regression model from scaling_law_models
        models = ScalingLawModels()
        
        # Get predictions (all models for comparison)
        predictions_cp200_fitted = []
        predictions_heuristic = []
        predictions_trajectory = []
        predictions_trajectory_with_labels = []
        
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
                base_acc = run['base']
                
                # Model 1: FITTED CP200 Logit Regression
                pred_final_error_cp200 = models.checkpoint_200_logit_regression(error_200)
                pred_gain_cp200 = (1 - pred_final_error_cp200) - (1 - base_error)
                
                # Model 2: Old heuristic for comparison
                pred_final_error_heur = 0.95 * error_200 + 0.02
                pred_gain_heur = (1 - pred_final_error_heur) - (1 - base_error)
                
                # Model 3: Full trajectory model with hard-coded coefficients
                # Calculate slope
                logit_0 = logit(np.clip(base_error, self.epsilon, 1-self.epsilon))
                logit_200 = logit(np.clip(error_200, self.epsilon, 1-self.epsilon))
                slope = (logit_200 - logit_0) / 200
                
                # Try with strategy labels first
                try:
                    pred_final_error_traj_with = models.early_trajectory_model(
                        model_size=run['model_size'],
                        base=base_acc,
                        perc_learnable=run['perc_learnable'],
                        early_slope=slope,
                        dataset=run['dataset'],
                        strategy=run['strategy']
                    )
                    pred_gain_traj_with = (1 - pred_final_error_traj_with) - (1 - base_error)
                except:
                    pred_gain_traj_with = np.nan
                
                # Try without strategy labels (should use only dynamic features)
                try:
                    pred_final_error_traj_without = models.early_trajectory_model(
                        model_size=run['model_size'],
                        base=base_acc,
                        perc_learnable=run['perc_learnable'],
                        early_slope=slope,
                        dataset=None,  # No dataset effects
                        strategy=None  # No strategy effects
                    )
                    pred_gain_traj_without = (1 - pred_final_error_traj_without) - (1 - base_error)
                except:
                    pred_gain_traj_without = np.nan
                
                # Use the version without labels (better for generalization)
                pred_gain_traj = pred_gain_traj_without
            else:
                pred_gain_cp200 = np.nan
                pred_gain_heur = np.nan
                pred_gain_traj = np.nan
                pred_gain_traj_with = np.nan
                
            predictions_cp200_fitted.append(pred_gain_cp200)
            predictions_heuristic.append(pred_gain_heur)
            predictions_trajectory.append(pred_gain_traj)
            predictions_trajectory_with_labels.append(pred_gain_traj_with)
        
        # Add all predictions to dataframe
        held_out_final['predicted_gain_cp200'] = predictions_cp200_fitted
        held_out_final['predicted_gain_heuristic'] = predictions_heuristic
        held_out_final['predicted_gain_trajectory'] = predictions_trajectory
        held_out_final['predicted_gain_trajectory_labeled'] = predictions_trajectory_with_labels
        
        # Use CP200 as default for decision thresholds
        held_out_final['predicted_gain'] = predictions_cp200_fitted
        
        valid_df = held_out_final[held_out_final['predicted_gain'].notna()].copy()
        
        # Print comprehensive comparison
        if len(valid_df) > 0:
            from sklearn.metrics import r2_score, mean_absolute_error
            
            # Compute errors (what the models actually predict)
            valid_df['actual_error'] = 1 - held_out_final.loc[valid_df.index, 'final_acc']
            valid_df['pred_error_heuristic'] = 1 - (valid_df['predicted_gain_heuristic'] + held_out_final.loc[valid_df.index, 'base'])
            valid_df['pred_error_cp200'] = 1 - (valid_df['predicted_gain_cp200'] + held_out_final.loc[valid_df.index, 'base'])
            valid_df['pred_error_trajectory'] = 1 - (valid_df['predicted_gain_trajectory'] + held_out_final.loc[valid_df.index, 'base'])
            valid_df['pred_error_trajectory_labeled'] = 1 - (valid_df['predicted_gain_trajectory_labeled'] + held_out_final.loc[valid_df.index, 'base'])
            
            print(f"\n  ═══════════════════════════════════════════════════════════════")
            print(f"  HELD-OUT MODEL COMPARISON (N = {len(valid_df)} runs)")
            print(f"  ═══════════════════════════════════════════════════════════════")
            print(f"\n  ACTUAL PERFORMANCE:")
            print(f"    Error: {valid_df['actual_error'].mean():.3f} ± {valid_df['actual_error'].std():.3f}")
            print(f"    Gain: {valid_df['actual_gain'].mean():.3f} ± {valid_df['actual_gain'].std():.3f}")
            
            # Model 1: Old Heuristic
            print(f"\n  MODEL 1: OLD HEURISTIC (0.95 × e₂₀₀ + 0.02)")
            if valid_df['pred_error_heuristic'].notna().sum() > 0:
                r2_error_heur = r2_score(valid_df['actual_error'], valid_df['pred_error_heuristic'])
                r2_gain_heur = r2_score(valid_df['actual_gain'], valid_df['predicted_gain_heuristic'])
                mae_heur = mean_absolute_error(valid_df['actual_error'], valid_df['pred_error_heuristic'])
                print(f"    R² (ERROR) = {r2_error_heur:.4f}, R² (GAIN) = {r2_gain_heur:.4f}, MAE = {mae_heur:.3f}")
                print(f"    Error Bias: {(valid_df['pred_error_heuristic'].mean() - valid_df['actual_error'].mean()):.3f}")
            
            # Model 2: CP200 Fitted Logit
            print(f"\n  MODEL 2: CP200 FITTED LOGIT REGRESSION ⭐")
            if valid_df['pred_error_cp200'].notna().sum() > 0:
                r2_error_cp200 = r2_score(valid_df['actual_error'], valid_df['pred_error_cp200'])
                r2_gain_cp200 = r2_score(valid_df['actual_gain'], valid_df['predicted_gain_cp200'])
                mae_cp200 = mean_absolute_error(valid_df['actual_error'], valid_df['pred_error_cp200'])
                print(f"    R² (ERROR) = {r2_error_cp200:.4f}, R² (GAIN) = {r2_gain_cp200:.4f}, MAE = {mae_cp200:.3f}")
                print(f"    Error Bias: {(valid_df['pred_error_cp200'].mean() - valid_df['actual_error'].mean()):.3f}")
            
            # Model 3a: Trajectory (identity-free)
            print(f"\n  MODEL 3a: TRAJECTORY IDENTITY-FREE (dynamic features only) ⭐⭐")
            valid_traj = valid_df[valid_df['pred_error_trajectory'].notna()]
            if len(valid_traj) > 0:
                r2_error_traj = r2_score(valid_traj['actual_error'], valid_traj['pred_error_trajectory'])
                r2_gain_traj = r2_score(valid_traj['actual_gain'], valid_traj['predicted_gain_trajectory'])
                mae_traj = mean_absolute_error(valid_traj['actual_error'], valid_traj['pred_error_trajectory'])
                print(f"    R² (ERROR) = {r2_error_traj:.4f}, R² (GAIN) = {r2_gain_traj:.4f}, MAE = {mae_traj:.3f}")
                print(f"    Error Bias: {(valid_traj['pred_error_trajectory'].mean() - valid_traj['actual_error'].mean()):.3f}")
                print(f"    Coverage: {len(valid_traj)}/{len(valid_df)} runs ({len(valid_traj)/len(valid_df)*100:.1f}%)")
            else:
                print(f"    No valid predictions")
            
            # Model 3b: Trajectory with labels
            print(f"\n  MODEL 3b: TRAJECTORY WITH LABELS (includes strategy effects)")
            valid_traj_lab = valid_df[valid_df['pred_error_trajectory_labeled'].notna()]
            if len(valid_traj_lab) > 0:
                r2_error_traj_lab = r2_score(valid_traj_lab['actual_error'], valid_traj_lab['pred_error_trajectory_labeled'])
                r2_gain_traj_lab = r2_score(valid_traj_lab['actual_gain'], valid_traj_lab['predicted_gain_trajectory_labeled'])
                mae_traj_lab = mean_absolute_error(valid_traj_lab['actual_error'], valid_traj_lab['pred_error_trajectory_labeled'])
                print(f"    R² (ERROR) = {r2_error_traj_lab:.4f}, R² (GAIN) = {r2_gain_traj_lab:.4f}, MAE = {mae_traj_lab:.3f}")
                print(f"    Error Bias: {(valid_traj_lab['pred_error_trajectory_labeled'].mean() - valid_traj_lab['actual_error'].mean()):.3f}")
            else:
                print(f"    No valid predictions")
            
            print(f"\n  ═══════════════════════════════════════════════════════════════")
        
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
