#!/usr/bin/env python3

import json
import sys
import argparse
import re
from collections import namedtuple

def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate (WER) using edit distance
    WER = (S + D + I) / N
    where:
    S = substitutions
    D = deletions  
    I = insertions
    N = total words in reference
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Initialize matrix for edit distance calculation
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    
    # Initialize first row and column
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    # Fill the matrix
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]  # No operation needed
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    
    # Total edit distance
    edit_distance = d[len(ref_words)][len(hyp_words)]
    
    # Calculate detailed error counts by backtracking
    i, j = len(ref_words), len(hyp_words)
    substitutions = 0
    deletions = 0
    insertions = 0
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            if ref_words[i-1] == hyp_words[j-1]:
                i -= 1
                j -= 1
            elif d[i][j] == d[i-1][j-1] + 1:  # Substitution
                substitutions += 1
                i -= 1
                j -= 1
            elif d[i][j] == d[i-1][j] + 1:  # Deletion
                deletions += 1
                i -= 1
            else:  # Insertion
                insertions += 1
                j -= 1
        elif i > 0:  # Deletion
            deletions += 1
            i -= 1
        else:  # Insertion
            insertions += 1
            j -= 1
    
    total_ref_words = len(ref_words)
    wer = edit_distance / total_ref_words if total_ref_words > 0 else 0
    
    return {
        'wer': wer,
        'total_errors': edit_distance,
        'substitutions': substitutions,
        'deletions': deletions,
        'insertions': insertions,
        'total_reference_words': total_ref_words,
        'total_hypothesis_words': len(hyp_words)
    }

def extract_text_from_json(json_file):
    """Extract reference text from JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    words = []
    for segment in data['segments']:
        text = segment.get('text', '').strip()
        if text:
            # Extract words and convert to lowercase
            segment_words = re.findall(r'\b\w+\b', text.lower())
            words.extend(segment_words)
    
    return ' '.join(words)

def extract_text_from_ctm(ctm_file):
    """Extract hypothesis text from CTM file"""
    words = []
    with open(ctm_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                word = parts[4].lower()
                # Remove punctuation for comparison
                word = re.sub(r'[^\w]', '', word)
                if word:
                    words.append(word)
    
    return ' '.join(words)

def format_percentage(value):
    """Format percentage with 2 decimal places"""
    return f"{value * 100:.2f}%"

def main():
    parser = argparse.ArgumentParser(description='Calculate WER for CTM files vs JSON reference')
    parser.add_argument('--reference', required=True, help='Reference JSON file')
    parser.add_argument('--hypothesis-original', required=True, help='Original CTM file (with transformations)')
    parser.add_argument('--hypothesis-reversed', required=True, help='Reversed CTM file (transformations undone)')
    
    args = parser.parse_args()
    
    print("=== WER (Word Error Rate) Comparison Analysis ===\n")
    
    # Extract reference text
    print("Extracting reference text from JSON...")
    reference_text = extract_text_from_json(args.reference)
    print(f"Reference contains {len(reference_text.split())} words\n")
    
    # Extract hypothesis texts
    print("Extracting hypothesis text from original CTM...")
    original_text = extract_text_from_ctm(args.hypothesis_original)
    print(f"Original CTM contains {len(original_text.split())} words")
    
    print("Extracting hypothesis text from reversed CTM...")
    reversed_text = extract_text_from_ctm(args.hypothesis_reversed)
    print(f"Reversed CTM contains {len(reversed_text.split())} words\n")
    
    # Calculate WER for both versions
    print("Calculating WER for original CTM...")
    original_wer = calculate_wer(reference_text, original_text)
    
    print("Calculating WER for reversed CTM...")
    reversed_wer = calculate_wer(reference_text, reversed_text)
    
    # Print detailed comparison
    print("=== DETAILED WER COMPARISON ===\n")
    
    print(f"Reference words: {original_wer['total_reference_words']}")
    print()
    
    print("ORIGINAL CTM (with linguistic transformations):")
    print(f"  Total hypothesis words: {original_wer['total_hypothesis_words']}")
    print(f"  WER: {format_percentage(original_wer['wer'])}")
    print(f"  Total errors: {original_wer['total_errors']}")
    print(f"  - Substitutions: {original_wer['substitutions']}")
    print(f"  - Deletions: {original_wer['deletions']}")
    print(f"  - Insertions: {original_wer['insertions']}")
    print()
    
    print("REVERSED CTM (transformations undone):")
    print(f"  Total hypothesis words: {reversed_wer['total_hypothesis_words']}")
    print(f"  WER: {format_percentage(reversed_wer['wer'])}")
    print(f"  Total errors: {reversed_wer['total_errors']}")
    print(f"  - Substitutions: {reversed_wer['substitutions']}")
    print(f"  - Deletions: {reversed_wer['deletions']}")
    print(f"  - Insertions: {reversed_wer['insertions']}")
    print()
    
    # Calculate improvement/degradation
    wer_difference = reversed_wer['wer'] - original_wer['wer']
    error_difference = reversed_wer['total_errors'] - original_wer['total_errors']
    
    print("=== IMPACT ANALYSIS ===")
    print()
    
    if wer_difference < 0:
        print(f"✅ IMPROVEMENT: Reversing transformations REDUCES WER by {format_percentage(-wer_difference)}")
        print(f"   ({-error_difference} fewer errors)")
    elif wer_difference > 0:
        print(f"❌ DEGRADATION: Reversing transformations INCREASES WER by {format_percentage(wer_difference)}")
        print(f"   ({error_difference} more errors)")
    else:
        print("⚖️ NEUTRAL: No change in WER")
    
    print()
    print("=== RECOMMENDATIONS ===")
    print()
    
    if wer_difference < -0.01:  # More than 1% improvement
        print("🎯 Use REVERSED CTM for downstream tasks requiring exact word matching")
        print("   The linguistic transformations appear to hurt alignment accuracy")
    elif wer_difference > 0.01:  # More than 1% degradation
        print("🎯 Use ORIGINAL CTM for downstream tasks")
        print("   The linguistic transformations improve overall quality")
    else:
        print("🎯 Either version can be used - minimal difference")
        print("   Choose based on your specific requirements:")
        print("   - Original: Better linguistic quality, compound words")
        print("   - Reversed: Closer to source transcription")
    
    print()
    print("=== TECHNICAL DETAILS ===")
    print()
    print(f"Original CTM file: {args.hypothesis_original}")
    print(f"Reversed CTM file: {args.hypothesis_reversed}")
    print(f"Reference JSON file: {args.reference}")
    
    # Save numerical results for potential further processing
    results = {
        'reference_words': original_wer['total_reference_words'],
        'original': original_wer,
        'reversed': reversed_wer,
        'difference': {
            'wer': wer_difference,
            'errors': error_difference,
            'percentage_change': (wer_difference / original_wer['wer'] * 100) if original_wer['wer'] > 0 else 0
        }
    }
    
    print()
    print("=== JSON SUMMARY ===")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()