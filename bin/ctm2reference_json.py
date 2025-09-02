#!/usr/bin/env python3

import json
import sys
import argparse
from typing import List, Dict, Any

def parse_ctm_file(ctm_file):
    """Parse CTM file and extract word-level alignments."""
    words = []
    
    with open(ctm_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if len(parts) < 4:
                print(f"Warning: Line {line_num} has insufficient fields: {line}", file=sys.stderr)
                continue
            
            try:
                word_data = {
                    'utterance_id': parts[0],
                    'channel': parts[1],
                    'start': float(parts[2]),
                    'duration': float(parts[3]),
                    'end': float(parts[2]) + float(parts[3]),
                    'word': parts[4] if len(parts) > 4 else '<unk>',
                    'confidence': float(parts[5]) if len(parts) > 5 else 1.0,
                    'line_number': line_num
                }
                words.append(word_data)
            except ValueError as e:
                print(f"Warning: Could not parse line {line_num}: {line} - {e}", file=sys.stderr)
                continue
    
    return sorted(words, key=lambda x: x['start'])

def group_words_into_segments(words, max_segment_duration=30.0, min_segment_duration=5.0):
    """Group words into segments for better alignment processing."""
    if not words:
        return []
    
    segments = []
    current_segment = {
        'words': [],
        'start': None,
        'end': None,
        'text': []
    }
    
    for word in words:
        # Skip non-word tokens
        if word['word'] in ['<unk>', '<s>', '</s>', '<silence>', '<sil>']:
            continue
            
        # Initialize first segment
        if current_segment['start'] is None:
            current_segment['start'] = word['start']
            current_segment['end'] = word['end']
            current_segment['words'].append(word)
            current_segment['text'].append(word['word'])
            continue
        
        segment_duration = current_segment['end'] - current_segment['start']
        gap_to_next = word['start'] - current_segment['end']
        
        # Decide whether to end current segment
        should_end_segment = False
        
        # End if segment would be too long
        if word['end'] - current_segment['start'] > max_segment_duration:
            should_end_segment = True
        
        # End if there's a significant gap (>2 seconds)
        elif gap_to_next > 2.0:
            should_end_segment = True
        
        # End if current segment has good duration and there's a pause
        elif segment_duration >= min_segment_duration and gap_to_next > 0.5:
            should_end_segment = True
        
        if should_end_segment and current_segment['words']:
            # Finalize current segment
            segments.append({
                'id': f'seg_{len(segments)+1:04d}',
                'start': current_segment['start'],
                'end': current_segment['end'],
                'text': ' '.join(current_segment['text']),
                'words': current_segment['words'],
                'word_count': len(current_segment['words']),
                'duration': current_segment['end'] - current_segment['start']
            })
            
            # Start new segment
            current_segment = {
                'words': [word],
                'start': word['start'],
                'end': word['end'],
                'text': [word['word']]
            }
        else:
            # Add to current segment
            current_segment['words'].append(word)
            current_segment['text'].append(word['word'])
            current_segment['end'] = word['end']
    
    # Don't forget the last segment
    if current_segment['words']:
        segments.append({
            'id': f'seg_{len(segments)+1:04d}',
            'start': current_segment['start'],
            'end': current_segment['end'],
            'text': ' '.join(current_segment['text']),
            'words': current_segment['words'],
            'word_count': len(current_segment['words']),
            'duration': current_segment['end'] - current_segment['start']
        })
    
    return segments

def create_reference_json(words, segments):
    """Create JSON reference structure compatible with alignment pipeline."""
    
    # Create word-level segments (for precise anchoring)
    word_segments = []
    for word in words:
        if word['word'] not in ['<unk>', '<s>', '</s>', '<silence>', '<sil>']:
            word_segments.append({
                'start': word['start'],
                'end': word['end'],
                'text': word['word'],
                'confidence': word.get('confidence', 1.0),
                'source': 'ctm'
            })
    
    # Create best_hypothesis structure (compatible with pipeline)
    best_hypothesis = {
        'monologues': [
            {
                'speaker': 'speaker_1',
                'turns': [
                    {
                        'start': segment['start'],
                        'end': segment['end'],
                        'words': [
                            {
                                'word': w['word'],
                                'word_with_punctuation': w['word'],
                                'start': w['start'],
                                'end': w['end'],
                                'confidence': w.get('confidence', 1.0)
                            } for w in segment['words']
                        ]
                    } for segment in segments
                ]
            }
        ]
    }
    
    # Main JSON structure
    reference_json = {
        'segments': word_segments,  # Word-level for precise anchoring
        'best_hypothesis': best_hypothesis,  # Pipeline-compatible structure
        'grouped_segments': segments,  # Segment-level groupings
        'metadata': {
            'source_format': 'ctm',
            'total_words': len([w for w in words if w['word'] not in ['<unk>', '<s>', '</s>', '<silence>', '<sil>']]),
            'total_segments': len(segments),
            'total_duration': max(w['end'] for w in words) if words else 0,
            'average_segment_duration': sum(s['duration'] for s in segments) / len(segments) if segments else 0,
            'conversion_info': {
                'max_segment_duration': 30.0,
                'min_segment_duration': 5.0,
                'gap_threshold': 2.0
            }
        },
        'speakers': {
            'speaker_1': {
                'name': 'Unknown Speaker'
            }
        }
    }
    
    return reference_json

def validate_json_structure(json_data):
    """Validate that the JSON structure is compatible with the alignment pipeline."""
    issues = []
    
    # Check required top-level keys
    required_keys = ['segments', 'best_hypothesis', 'metadata']
    for key in required_keys:
        if key not in json_data:
            issues.append(f"Missing required key: {key}")
    
    # Check segments structure
    if 'segments' in json_data:
        for i, segment in enumerate(json_data['segments'][:5]):  # Check first 5
            if 'start' not in segment or 'end' not in segment or 'text' not in segment:
                issues.append(f"Segment {i} missing required fields")
    
    # Check best_hypothesis structure
    if 'best_hypothesis' in json_data:
        hypothesis = json_data['best_hypothesis']
        if 'monologues' not in hypothesis:
            issues.append("best_hypothesis missing 'monologues' key")
        else:
            for i, monologue in enumerate(hypothesis['monologues'][:3]):  # Check first 3
                if 'turns' not in monologue:
                    issues.append(f"Monologue {i} missing 'turns' key")
    
    # Check timing consistency
    if 'segments' in json_data:
        segments = json_data['segments']
        for i, segment in enumerate(segments):
            if segment.get('end', 0) <= segment.get('start', 0):
                issues.append(f"Segment {i} has invalid timing: {segment.get('start')} -> {segment.get('end')}")
    
    return issues

def main():
    parser = argparse.ArgumentParser(description="Convert CTM file to reference JSON format")
    parser.add_argument('ctm_file', help='Input CTM file')
    parser.add_argument('--output', '-o', help='Output JSON file (default: stdout)')
    parser.add_argument('--max-segment-duration', type=float, default=30.0, 
                       help='Maximum segment duration in seconds')
    parser.add_argument('--min-segment-duration', type=float, default=5.0,
                       help='Minimum segment duration in seconds')
    parser.add_argument('--validate', action='store_true', 
                       help='Validate output JSON structure')
    
    args = parser.parse_args()
    
    # Parse CTM file
    print(f"Parsing CTM file: {args.ctm_file}", file=sys.stderr)
    words = parse_ctm_file(args.ctm_file)
    
    if not words:
        print("Error: No valid words found in CTM file", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(words)} words", file=sys.stderr)
    
    # Group words into segments
    segments = group_words_into_segments(
        words, 
        args.max_segment_duration, 
        args.min_segment_duration
    )
    
    print(f"Created {len(segments)} segments", file=sys.stderr)
    
    # Create reference JSON
    reference_json = create_reference_json(words, segments)
    
    # Validate structure if requested
    if args.validate:
        issues = validate_json_structure(reference_json)
        if issues:
            print("Validation issues found:", file=sys.stderr)
            for issue in issues:
                print(f"  - {issue}", file=sys.stderr)
        else:
            print("JSON structure validation passed", file=sys.stderr)
    
    # Output JSON
    json_output = json.dumps(reference_json, indent=2, ensure_ascii=False)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(json_output)
        print(f"Reference JSON written to: {args.output}", file=sys.stderr)
    else:
        print(json_output)

if __name__ == '__main__':
    main()