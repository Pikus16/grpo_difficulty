#!/usr/bin/env python3
"""
AIME Results Analysis Script
Compares performance across different training strategies
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

def extract_boxed(text: str) -> Optional[str]:
    """Extract content from \\boxed{} format"""
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1].strip() if matches else None

def extract_number_from_boxed(boxed_text: str) -> Optional[int]:
    """Extract integer from boxed content"""
    if not boxed_text:
        return None
    numbers = re.findall(r'\d+', boxed_text)
    return int(numbers[0]) if numbers else None

def analyze_responses(responses_file: str) -> Dict:
    """Analyze a single model's responses"""
    if not os.path.exists(responses_file):
        return {"error": f"File not found: {responses_file}"}
    
    with open(responses_file, 'r') as f:
        data = json.load(f)
    
    total_problems = len(data)
    total_responses = sum(len(prob_responses) for prob_responses in data)
    
    # Statistics
    stats = {
        "total_problems": total_problems,
        "total_responses": total_responses,
        "responses_per_problem": len(data[0]) if data else 0,
        "boxed_answers": 0,
        "complete_responses": 0,
        "avg_response_length": 0,
        "predictions": []
    }
    
    all_lengths = []
    
    for i, prob_responses in enumerate(data):
        prob_predictions = []
        
        for response in prob_responses:
            all_lengths.append(len(response))
            
            # Check for boxed answers
            boxed = extract_boxed(response)
            if boxed:
                stats["boxed_answers"] += 1
            
            # Check if response appears complete
            if len(response) > 1000 and any(response.strip().endswith(end) for end in ['}', '.', '$$']):
                stats["complete_responses"] += 1
            
            # Extract prediction
            number = extract_number_from_boxed(boxed) if boxed else None
            prob_predictions.append(number)
        
        # Store first prediction for this problem
        stats["predictions"].append(prob_predictions[0] if prob_predictions else None)
    
    stats["avg_response_length"] = sum(all_lengths) / len(all_lengths) if all_lengths else 0
    stats["boxed_percentage"] = (stats["boxed_answers"] / total_responses) * 100
    stats["complete_percentage"] = (stats["complete_responses"] / total_responses) * 100
    
    return stats

def main():
    """Main analysis function"""
    base_dir = "/home/pratyush/grpo_difficulty"
    results_dir = f"{base_dir}/dsets/AIME2025"
    
    # Model configurations
    models = {
        "base": f"{results_dir}/unsloth-Qwen3-4B-unsloth-bnb-4bit",
        "easiest": f"{results_dir}/8gen_1000steps_unsloth-Qwen3-4B-unsloth-bnb-4bit_strategyeasiest_subsetperc0.1-final",
        "hardest": f"{results_dir}/8gen_1000steps_unsloth-Qwen3-4B-unsloth-bnb-4bit_strategyhardest_subsetperc0.1-final", 
        "middle": f"{results_dir}/8gen_1000steps_unsloth-Qwen3-4B-unsloth-bnb-4bit_strategymiddle_subsetperc0.1-final",
        "random": f"{results_dir}/8gen_1000steps_unsloth-Qwen3-4B-unsloth-bnb-4bit_strategyrandom_subsetperc0.1-final"
    }
    
    print("🔍 AIME Results Analysis")
    print("=" * 50)
    
    all_results = {}
    
    for strategy, model_dir in models.items():
        responses_file = f"{model_dir}/eval_responses.json"
        print(f"\n📊 Analyzing: {strategy}")
        print(f"   File: {responses_file}")
        
        results = analyze_responses(responses_file)
        all_results[strategy] = results
        
        if "error" in results:
            print(f"   ❌ {results['error']}")
            continue
        
        print(f"   ✅ Problems: {results['total_problems']}")
        print(f"   📝 Avg length: {results['avg_response_length']:.0f} chars")
        print(f"   📦 Boxed answers: {results['boxed_answers']}/{results['total_responses']} ({results['boxed_percentage']:.1f}%)")
        print(f"   ✨ Complete responses: {results['complete_responses']}/{results['total_responses']} ({results['complete_percentage']:.1f}%)")
    
    # Summary comparison
    print(f"\n📈 PERFORMANCE COMPARISON")
    print("=" * 50)
    print(f"{'Strategy':<10} {'Boxed %':<10} {'Complete %':<12} {'Avg Length':<12}")
    print("-" * 50)
    
    for strategy, results in all_results.items():
        if "error" not in results:
            print(f"{strategy:<10} {results['boxed_percentage']:<10.1f} {results['complete_percentage']:<12.1f} {results['avg_response_length']:<12.0f}")
    
    # Problem-by-problem comparison
    print(f"\n🎯 PROBLEM-BY-PROBLEM PREDICTIONS")
    print("=" * 60)
    print(f"{'Problem':<8}", end="")
    for strategy in models.keys():
        if strategy in all_results and "error" not in all_results[strategy]:
            print(f"{strategy:<10}", end="")
    print()
    print("-" * 60)
    
    max_problems = max(len(results.get('predictions', [])) for results in all_results.values() if 'error' not in results)
    
    for i in range(max_problems):
        print(f"{i+1:<8}", end="")
        for strategy in models.keys():
            if strategy in all_results and "error" not in all_results[strategy]:
                predictions = all_results[strategy].get('predictions', [])
                pred = predictions[i] if i < len(predictions) else None
                pred_str = str(pred) if pred is not None else "None"
                print(f"{pred_str:<10}", end="")
        print()
    
    # Save detailed results
    output_file = f"{base_dir}/aime_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main() 