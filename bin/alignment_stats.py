#!/usr/bin/env python3
"""
Generate alignment statistics report from alignment stats JSON.
Provides detailed breakdown of D/I/S operations and coverage metrics.
"""

import json
import sys
import argparse
from typing import Dict, Any
from datetime import datetime


def format_stats_report(stats_data: Dict[str, Any]) -> str:
    """
    Format alignment statistics into a human-readable report.
    """
    report_lines = [
        "=" * 60,
        "TEXT-BASED ALIGNMENT STATISTICS REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    
    # Total statistics
    total_operations = (
        stats_data.get("deletions", 0) + 
        stats_data.get("insertions", 0) + 
        stats_data.get("substitutions", 0) + 
        stats_data.get("matches", 0)
    )
    
    report_lines.extend([
        "SUMMARY",
        "-" * 20,
        f"Total operations: {total_operations}",
        f"  - Matches:       {stats_data.get('matches', 0):6d} ({stats_data.get('matches', 0)/total_operations*100:5.1f}%)" if total_operations > 0 else "  - Matches:       0",
        f"  - Substitutions: {stats_data.get('substitutions', 0):6d} ({stats_data.get('substitutions', 0)/total_operations*100:5.1f}%)" if total_operations > 0 else "  - Substitutions: 0", 
        f"  - Insertions:    {stats_data.get('insertions', 0):6d} ({stats_data.get('insertions', 0)/total_operations*100:5.1f}%)" if total_operations > 0 else "  - Insertions:    0",
        f"  - Deletions:     {stats_data.get('deletions', 0):6d} ({stats_data.get('deletions', 0)/total_operations*100:5.1f}%)" if total_operations > 0 else "  - Deletions:     0",
        "",
    ])
    
    # Edit distance and accuracy metrics
    edit_distance = stats_data.get("substitutions", 0) + stats_data.get("insertions", 0) + stats_data.get("deletions", 0)
    accuracy = stats_data.get("matches", 0) / total_operations * 100 if total_operations > 0 else 0
    
    report_lines.extend([
        "ACCURACY METRICS",
        "-" * 20,
        f"Total edit distance: {edit_distance}",
        f"Word accuracy:       {accuracy:.2f}%",
        f"Error rate:          {100-accuracy:.2f}%",
        "",
    ])
    
    # Word coverage information
    reference_words = stats_data.get("matches", 0) + stats_data.get("substitutions", 0) + stats_data.get("insertions", 0)
    asr_words = stats_data.get("matches", 0) + stats_data.get("substitutions", 0) + stats_data.get("deletions", 0)
    
    report_lines.extend([
        "WORD COVERAGE",
        "-" * 20,
        f"Reference words: {reference_words}",
        f"ASR words:       {asr_words}",
        f"Coverage ratio:  {reference_words/asr_words:.3f}" if asr_words > 0 else "Coverage ratio:  N/A",
        "",
    ])
    
    # Operation breakdown
    if edit_distance > 0:
        report_lines.extend([
            "EDIT OPERATION BREAKDOWN",
            "-" * 28,
            f"Substitutions: {stats_data.get('substitutions', 0)/edit_distance*100:.1f}% of errors" if edit_distance > 0 else "Substitutions: 0% of errors",
            f"Insertions:    {stats_data.get('insertions', 0)/edit_distance*100:.1f}% of errors" if edit_distance > 0 else "Insertions:    0% of errors",
            f"Deletions:     {stats_data.get('deletions', 0)/edit_distance*100:.1f}% of errors" if edit_distance > 0 else "Deletions:     0% of errors",
            "",
        ])
    
    # Quality assessment
    quality_assessment = "EXCELLENT" if accuracy >= 95 else "GOOD" if accuracy >= 85 else "FAIR" if accuracy >= 70 else "POOR"
    
    report_lines.extend([
        "ALIGNMENT QUALITY ASSESSMENT",
        "-" * 29,
        f"Overall quality: {quality_assessment}",
        "",
    ])
    
    # Recommendations based on error patterns
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("-" * 15)
    
    if stats_data.get("substitutions", 0) > stats_data.get("insertions", 0) + stats_data.get("deletions", 0):
        report_lines.append("• High substitution rate suggests pronunciation/recognition differences")
        report_lines.append("• Consider acoustic model adaptation or vocabulary tuning")
    
    if stats_data.get("insertions", 0) > stats_data.get("substitutions", 0) + stats_data.get("deletions", 0):
        report_lines.append("• High insertion rate suggests ASR is adding extra words")
        report_lines.append("• Review language model or decoding parameters")
        
    if stats_data.get("deletions", 0) > stats_data.get("substitutions", 0) + stats_data.get("insertions", 0):
        report_lines.append("• High deletion rate suggests ASR is missing words")
        report_lines.append("• Check for audio quality issues or vocabulary coverage")
    
    if edit_distance == 0:
        report_lines.append("• Perfect alignment achieved - ASR output matches reference exactly!")
    
    report_lines.extend([
        "",
        "=" * 60,
    ])
    
    return "\\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(description="Generate alignment statistics report")
    parser.add_argument("stats_json", help="Input alignment statistics JSON file")
    parser.add_argument("--output", "-o", help="Output report file (default: stdout)")
    parser.add_argument("--format", choices=["text", "json", "csv"], default="text", help="Output format")
    
    args = parser.parse_args()
    
    try:
        # Load statistics data
        with open(args.stats_json, 'r', encoding='utf-8') as f:
            stats_data = json.load(f)
        
        # Generate report based on format
        if args.format == "text":
            report_content = format_stats_report(stats_data)
        elif args.format == "json":
            # Add computed metrics to JSON
            total_operations = sum(stats_data.get(k, 0) for k in ["matches", "substitutions", "insertions", "deletions"])
            edit_distance = sum(stats_data.get(k, 0) for k in ["substitutions", "insertions", "deletions"])
            accuracy = stats_data.get("matches", 0) / total_operations * 100 if total_operations > 0 else 0
            
            enhanced_stats = dict(stats_data)
            enhanced_stats.update({
                "total_operations": total_operations,
                "edit_distance": edit_distance,
                "accuracy_percent": accuracy,
                "error_rate_percent": 100 - accuracy
            })
            report_content = json.dumps(enhanced_stats, indent=2)
        elif args.format == "csv":
            # CSV format for spreadsheet analysis
            report_content = "metric,value\\n"
            report_content += f"matches,{stats_data.get('matches', 0)}\\n"
            report_content += f"substitutions,{stats_data.get('substitutions', 0)}\\n"
            report_content += f"insertions,{stats_data.get('insertions', 0)}\\n"
            report_content += f"deletions,{stats_data.get('deletions', 0)}\\n"
            
            total_operations = sum(stats_data.get(k, 0) for k in ["matches", "substitutions", "insertions", "deletions"])
            edit_distance = sum(stats_data.get(k, 0) for k in ["substitutions", "insertions", "deletions"])
            accuracy = stats_data.get("matches", 0) / total_operations * 100 if total_operations > 0 else 0
            
            report_content += f"total_operations,{total_operations}\\n"
            report_content += f"edit_distance,{edit_distance}\\n"
            report_content += f"accuracy_percent,{accuracy:.2f}\\n"
            report_content += f"error_rate_percent,{100-accuracy:.2f}\\n"
        
        # Write output
        output_file = sys.stdout
        if args.output:
            output_file = open(args.output, 'w', encoding='utf-8')
        
        try:
            print(report_content, file=output_file)
        finally:
            if args.output:
                output_file.close()
    
    except FileNotFoundError:
        print(f"Error: Could not find input file '{args.stats_json}'", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()