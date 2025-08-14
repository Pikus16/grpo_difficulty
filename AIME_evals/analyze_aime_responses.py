#!/usr/bin/env python3
"""
AIME 2025 Response Analysis Script
Analyzes actual model responses compared to ground truth answers
"""
import json
import re
from typing import List, Dict, Optional, Tuple

# AIME 2025-I Ground Truth Answers (from opencompass/AIME2025 HuggingFace dataset)
GROUND_TRUTH = [
    70,   # Problem 1
    588,  # Problem 2  
    16,   # Problem 3
    117,  # Problem 4
    279,  # Problem 5
    504,  # Problem 6
    821,  # Problem 7
    77,   # Problem 8
    62,   # Problem 9
    81,   # Problem 10
    259,  # Problem 11
    510,  # Problem 12
    204,  # Problem 13
    60,   # Problem 14
    735   # Problem 15
]

def extract_boxed(text: str) -> Optional[str]:
    """Extract content from \\boxed{} format"""
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1].strip() if matches else None

def extract_number_from_boxed(boxed_text: str) -> Optional[int]:
    """Extract integer from boxed content"""
    if not boxed_text:
        return None
    # Look for numbers in the boxed content
    numbers = re.findall(r'\d+', boxed_text)
    if numbers:
        return int(numbers[0])
    return None

def analyze_single_response(response: str, problem_idx: int) -> Dict:
    """Analyze a single response"""
    ground_truth = GROUND_TRUTH[problem_idx]
    boxed_content = extract_boxed(response)
    extracted_number = extract_number_from_boxed(boxed_content)
    
    return {
        'response_length': len(response),
        'has_boxed': boxed_content is not None,
        'boxed_content': boxed_content,
        'extracted_number': extracted_number,
        'is_correct': extracted_number == ground_truth,
        'ground_truth': ground_truth,
        'response_preview': response[:200] + "..." if len(response) > 200 else response
    }

def calculate_pass_at_k(responses: List[str], problem_idx: int) -> Tuple[bool, List[Dict]]:
    """Calculate if any response in the list is correct (pass@k)"""
    ground_truth = GROUND_TRUTH[problem_idx]
    analyses = []
    
    for i, response in enumerate(responses):
        analysis = analyze_single_response(response, problem_idx)
        analysis['response_idx'] = i
        analyses.append(analysis)
    
    # Check if any response is correct
    pass_at_k = any(analysis['is_correct'] for analysis in analyses)
    
    return pass_at_k, analyses

def analyze_model_responses(responses_file: str, model_name: str) -> Dict:
    """Analyze all responses from a model"""
    try:
        with open(responses_file, 'r') as f:
            all_responses = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {responses_file}")
        return {}
    
    results = {
        'model_name': model_name,
        'total_problems': len(all_responses),
        'pass_at_k_results': [],
        'pass_at_k_score': 0,
        'boxed_rate': 0,
        'correct_boxed_rate': 0,
        'problem_analyses': []
    }
    
    total_responses = 0
    total_boxed = 0
    total_correct_boxed = 0
    
    for problem_idx, problem_responses in enumerate(all_responses):
        if problem_idx >= len(GROUND_TRUTH):
            break
            
        pass_at_k, response_analyses = calculate_pass_at_k(problem_responses, problem_idx)
        results['pass_at_k_results'].append(pass_at_k)
        results['problem_analyses'].append({
            'problem_idx': problem_idx + 1,
            'ground_truth': GROUND_TRUTH[problem_idx],
            'pass_at_k': pass_at_k,
            'responses': response_analyses
        })
        
        # Update statistics
        for analysis in response_analyses:
            total_responses += 1
            if analysis['has_boxed']:
                total_boxed += 1
                if analysis['is_correct']:
                    total_correct_boxed += 1
    
    results['pass_at_k_score'] = sum(results['pass_at_k_results']) / len(results['pass_at_k_results'])
    results['boxed_rate'] = total_boxed / total_responses if total_responses > 0 else 0
    results['correct_boxed_rate'] = total_correct_boxed / total_boxed if total_boxed > 0 else 0
    
    return results

def print_model_summary(results: Dict):
    """Print a summary for a single model"""
    if not results:
        return
        
    print(f"\n📊 {results['model_name'].upper()} MODEL ANALYSIS")
    print("=" * 60)
    print(f"Pass@8 Score: {results['pass_at_k_score']:.1%} ({sum(results['pass_at_k_results'])}/{len(results['pass_at_k_results'])} problems)")
    print(f"Boxed Rate: {results['boxed_rate']:.1%}")
    print(f"Accuracy when boxed: {results['correct_boxed_rate']:.1%}")
    
    # Show which problems were solved
    solved_problems = [i+1 for i, solved in enumerate(results['pass_at_k_results']) if solved]
    print(f"Solved Problems: {solved_problems}")

def print_detailed_comparison(all_results: Dict):
    """Print detailed problem-by-problem comparison"""
    print(f"\n🔍 DETAILED PROBLEM-BY-PROBLEM ANALYSIS")
    print("=" * 80)
    
    models = list(all_results.keys())
    
    for problem_idx in range(15):
        print(f"\n📝 Problem {problem_idx + 1} (Answer: {GROUND_TRUTH[problem_idx]})")
        print("-" * 40)
        
        for model_name in models:
            if model_name in all_results and problem_idx < len(all_results[model_name]['problem_analyses']):
                analysis = all_results[model_name]['problem_analyses'][problem_idx]
                pass_symbol = "✅" if analysis['pass_at_k'] else "❌"
                print(f"{pass_symbol} {model_name}: {pass_symbol}")
                
                # Show first response details
                if analysis['responses']:
                    first_resp = analysis['responses'][0]
                    if first_resp['has_boxed']:
                        print(f"   Boxed: {first_resp['boxed_content']} → {first_resp['extracted_number']}")
                    else:
                        print(f"   No boxed answer found")

def show_example_responses(all_results: Dict, problem_idx: int):
    """Show example responses for a specific problem"""
    print(f"\n📄 EXAMPLE RESPONSES FOR PROBLEM {problem_idx + 1}")
    print(f"Ground Truth: {GROUND_TRUTH[problem_idx]}")
    print("=" * 80)
    
    for model_name, results in all_results.items():
        if problem_idx < len(results['problem_analyses']):
            problem_analysis = results['problem_analyses'][problem_idx]
            first_response = problem_analysis['responses'][0]
            
            print(f"\n🤖 {model_name.upper()}:")
            print(f"Status: {'✅ CORRECT' if first_response['is_correct'] else '❌ INCORRECT'}")
            if first_response['has_boxed']:
                print(f"Boxed: {first_response['boxed_content']} → {first_response['extracted_number']}")
            else:
                print("No boxed answer")
            print(f"Response preview: {first_response['response_preview']}")
            print("-" * 40)

def main():
    """Main analysis function"""
    models = {
        'Base': 'base_responses.json',
        'Easiest': 'easiest_responses.json', 
        'Hardest': 'hardest_responses.json',
        'Middle': 'middle_responses.json',
        'Random': 'random_responses.json'
    }
    
    print("🎯 AIME 2025-I COMPREHENSIVE RESPONSE ANALYSIS")
    print("=" * 80)
    
    all_results = {}
    
    # Analyze each model
    for model_name, filename in models.items():
        print(f"\n🔄 Analyzing {model_name} model...")
        results = analyze_model_responses(filename, model_name)
        if results:
            all_results[model_name] = results
            print_model_summary(results)
    
    if len(all_results) >= 2:
        print_detailed_comparison(all_results)
        
        # Show examples for a few interesting problems
        print(f"\n🎯 EXAMPLE RESPONSES")
        print("=" * 80)
        
        # Show examples for problems where models disagree
        for problem_idx in [0, 1, 4, 10]:  # Problems 1, 2, 5, 11
            if problem_idx < 15:
                show_example_responses(all_results, problem_idx)
    
    # Final summary
    print(f"\n📈 FINAL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<10} {'Pass@8':<8} {'Problems':<12} {'Boxed Rate':<12}")
    print("-" * 60)
    
    # Sort models by pass@8 performance
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['pass_at_k_score'], reverse=True)
    
    for model_name, results in sorted_results:
        pass_at_k_pct = results['pass_at_k_score'] * 100
        solved_count = sum(results['pass_at_k_results'])
        boxed_rate = results['boxed_rate'] * 100
        print(f"{model_name:<10} {pass_at_k_pct:5.1f}%   {solved_count:2d}/15       {boxed_rate:5.1f}%")

if __name__ == "__main__":
    main() 