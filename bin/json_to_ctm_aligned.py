#!/usr/bin/env python3
"""
Convert text alignment JSON results to CTM format.
Generates CTM with adjusted word timings from text-based alignment.
"""

import json
import sys
import argparse
from typing import List, Dict, Any


def generate_ctm_from_alignment(alignment_data: Dict[str, Any]) -> List[str]:
    """
    Convert alignment JSON to CTM format lines.
    
    CTM format: <file> <channel> <start_time> <duration> <word> [confidence]
    """
    ctm_lines = []
    
    for alignment in alignment_data.get("alignments", []):
        segment_id = alignment["segment_id"]
        
        for word_info in alignment["words"]:
            word = word_info["word"]
            timing = word_info.get("timing")
            
            if timing and len(timing) == 2:
                start_time = timing[0]
                end_time = timing[1]
                duration = end_time - start_time
                
                # Use confidence if available, otherwise omit
                confidence = word_info.get("confidence")
                
                if confidence is not None:
                    ctm_line = f"audio 1 {start_time:.2f} {duration:.2f} {word} {confidence:.3f}"
                else:
                    ctm_line = f"audio 1 {start_time:.2f} {duration:.2f} {word}"
                
                ctm_lines.append(ctm_line)
            else:
                # Handle missing timing with warning
                print(f"Warning: Missing timing for word '{word}' in segment {segment_id}", file=sys.stderr)
                # Assign minimal timing as fallback
                ctm_line = f"audio 1 0.00 0.10 {word}"
                ctm_lines.append(ctm_line)
    
    return ctm_lines


def main():
    parser = argparse.ArgumentParser(description="Convert alignment JSON to CTM format")
    parser.add_argument("alignment_json", help="Input alignment JSON file")
    parser.add_argument("--output", "-o", help="Output CTM file (default: stdout)")
    parser.add_argument("--sort-by-time", action="store_true", help="Sort output by start time")
    
    args = parser.parse_args()
    
    try:
        # Load alignment data
        with open(args.alignment_json, 'r', encoding='utf-8') as f:
            alignment_data = json.load(f)
        
        # Generate CTM lines
        ctm_lines = generate_ctm_from_alignment(alignment_data)
        
        # Sort by start time if requested
        if args.sort_by_time:
            def extract_start_time(line: str) -> float:
                parts = line.split()
                return float(parts[2]) if len(parts) >= 4 else 0.0
            
            ctm_lines.sort(key=extract_start_time)
        
        # Write output
        output_file = sys.stdout
        if args.output:
            output_file = open(args.output, 'w', encoding='utf-8')
        
        try:
            for line in ctm_lines:
                print(line, file=output_file)
        finally:
            if args.output:
                output_file.close()
        
        # Print statistics to stderr
        total_words = len(ctm_lines)
        print(f"Generated CTM with {total_words} words", file=sys.stderr)
        
        # Print timing statistics
        if ctm_lines and args.sort_by_time:
            first_time = float(ctm_lines[0].split()[2])
            last_line_parts = ctm_lines[-1].split()
            last_start = float(last_line_parts[2])
            last_duration = float(last_line_parts[3])
            last_time = last_start + last_duration
            
            print(f"Time range: {first_time:.2f}s - {last_time:.2f}s", file=sys.stderr)
    
    except FileNotFoundError:
        print(f"Error: Could not find input file '{args.alignment_json}'", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()