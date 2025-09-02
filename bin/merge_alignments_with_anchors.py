#!/usr/bin/env python3

import json
import sys
import argparse
from typing import Dict, List, Any, Optional

def parse_ctm(ctm_file):
    """Parse CTM file and return word alignments."""
    words = []
    with open(ctm_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                words.append({
                    'utterance': parts[0],
                    'channel': parts[1] if len(parts) > 1 else '1',
                    'start': float(parts[2]),
                    'duration': float(parts[3]),
                    'end': float(parts[2]) + float(parts[3]),
                    'word': parts[4] if len(parts) > 4 else '',
                    'confidence': float(parts[5]) if len(parts) > 5 else 1.0
                })
    return words

def parse_alignment_json(json_file):
    """Parse alignment JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_reference_timings(reference_data):
    """Extract word timings from reference data."""
    ref_timings = {}
    
    # Handle different reference formats
    if 'segments' in reference_data:
        for segment in reference_data['segments']:
            if 'text' in segment:
                # Word-level segment
                word = segment['text'].strip()
                if word:
                    ref_timings[word.lower()] = {
                        'start': segment['start'],
                        'end': segment['end'],
                        'original_text': word
                    }
    
    elif 'best_hypothesis' in reference_data:
        # Pipeline format with word-level timings
        hypothesis = reference_data['best_hypothesis']
        if 'sections' in hypothesis:
            for section in hypothesis['sections']:
                if section.get('type') == 'speech':
                    for turn in section.get('turns', []):
                        for word_data in turn.get('words', []):
                            word = word_data.get('word_with_punctuation', word_data.get('word', ''))
                            if word:
                                ref_timings[word.lower()] = {
                                    'start': word_data['start'],
                                    'end': word_data['end'],
                                    'original_text': word
                                }
        elif 'monologues' in hypothesis:
            for monologue in hypothesis['monologues']:
                for turn in monologue.get('turns', []):
                    for word_data in turn.get('words', []):
                        word = word_data.get('word_with_punctuation', word_data.get('word', ''))
                        if word:
                            ref_timings[word.lower()] = {
                                'start': word_data['start'],
                                'end': word_data['end'],
                                'original_text': word
                            }
    
    return ref_timings

def merge_alignments(new_ctm_words, new_json_data, anchor_mapping, reference_data, confidence_threshold):
    """
    Merge new alignments with reference anchors to create improved alignments.
    """
    
    # Extract reference timings
    ref_timings = extract_reference_timings(reference_data)
    
    # Get high-confidence anchors
    high_conf_anchors = [a for a in anchor_mapping.get('anchors', []) 
                        if a['confidence'] >= confidence_threshold]
    
    print(f"Using {len(high_conf_anchors)} high-confidence anchors", file=sys.stderr)
    print(f"Reference has {len(ref_timings)} word timings", file=sys.stderr)
    
    # Create merged CTM
    merged_ctm_words = []
    merged_json_words = []
    
    anchor_used_count = 0
    new_alignment_count = 0
    
    # Process each word from new alignment
    for word_data in new_ctm_words:
        word = word_data['word'].lower()
        
        # Check if this word has a high-confidence anchor
        matching_anchor = None
        for anchor in high_conf_anchors:
            anchor_text = anchor['text'].lower()
            if anchor_text == word:
                matching_anchor = anchor
                break
        
        # Also check reference timings directly
        ref_timing = ref_timings.get(word)
        
        if matching_anchor and ref_timing:
            # Use reference timing for anchored word
            merged_word = {
                **word_data,
                'start': matching_anchor['reference_timing']['start'],
                'duration': (matching_anchor['reference_timing']['end'] - 
                           matching_anchor['reference_timing']['start']),
                'end': matching_anchor['reference_timing']['end'],
                'source': 'anchor',
                'confidence': matching_anchor['confidence']
            }
            anchor_used_count += 1
        elif ref_timing and abs(ref_timing['start'] - word_data['start']) < 1.0:
            # Use reference timing if it's close to new alignment (within 1 second)
            merged_word = {
                **word_data,
                'start': ref_timing['start'],
                'duration': ref_timing['end'] - ref_timing['start'],
                'end': ref_timing['end'],
                'source': 'reference_close',
                'confidence': 0.9
            }
            anchor_used_count += 1
        else:
            # Use new alignment
            merged_word = {
                **word_data,
                'source': 'new_alignment',
                'confidence': word_data.get('confidence', 0.8)
            }
            new_alignment_count += 1
        
        merged_ctm_words.append(merged_word)
        
        # Create JSON word entry
        json_word = {
            'word': word_data['word'],
            'start': merged_word['start'],
            'end': merged_word['end'],
            'confidence': merged_word['confidence'],
            'source': merged_word['source']
        }
        merged_json_words.append(json_word)
    
    print(f"Merged: {anchor_used_count} anchored words, {new_alignment_count} new alignments", file=sys.stderr)
    
    return merged_ctm_words, merged_json_words, {
        'anchor_used_count': anchor_used_count,
        'new_alignment_count': new_alignment_count,
        'total_words': len(merged_ctm_words),
        'anchor_usage_rate': anchor_used_count / len(merged_ctm_words) if merged_ctm_words else 0
    }

def create_merged_ctm(merged_words, output_file):
    """Write merged words to CTM format."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in merged_words:
            # CTM format: utterance channel start duration word [confidence]
            line = f"{word['utterance']} {word['channel']} {word['start']:.3f} {word['duration']:.3f} {word['word']}"
            if 'confidence' in word:
                line += f" {word['confidence']:.3f}"
            f.write(line + '\n')

def create_merged_json(merged_json_words, original_json_data, output_file):
    """Create merged JSON with improved word-level alignments."""
    
    # Create a new JSON structure with merged alignments
    merged_data = {
        'words': merged_json_words,
        'metadata': {
            'source': 'merged_with_anchors',
            'total_words': len(merged_json_words),
            'anchor_sources': len([w for w in merged_json_words if w['source'] in ['anchor', 'reference_close']]),
            'new_alignments': len([w for w in merged_json_words if w['source'] == 'new_alignment'])
        }
    }
    
    # Try to preserve original structure if possible
    if isinstance(original_json_data, dict):
        if 'segments' in original_json_data:
            # Create segments from merged words
            segments = []
            current_segment = None
            segment_id = 1
            
            for word in merged_json_words:
                if current_segment is None:
                    current_segment = {
                        'id': f'seg_{segment_id:04d}',
                        'start': word['start'],
                        'end': word['end'],
                        'words': [word]
                    }
                elif word['start'] - current_segment['end'] > 1.0:  # Gap > 1 second
                    # Finalize current segment
                    current_segment['text'] = ' '.join(w['word'] for w in current_segment['words'])
                    segments.append(current_segment)
                    
                    # Start new segment
                    segment_id += 1
                    current_segment = {
                        'id': f'seg_{segment_id:04d}',
                        'start': word['start'],
                        'end': word['end'],
                        'words': [word]
                    }
                else:
                    # Add to current segment
                    current_segment['end'] = word['end']
                    current_segment['words'].append(word)
            
            # Don't forget the last segment
            if current_segment:
                current_segment['text'] = ' '.join(w['word'] for w in current_segment['words'])
                segments.append(current_segment)
            
            merged_data['segments'] = segments
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Merge new alignments with reference anchors")
    parser.add_argument('--new-alignments-ctm', required=True, help='New alignments CTM file')
    parser.add_argument('--new-alignments-json', required=True, help='New alignments JSON file')
    parser.add_argument('--reference', required=True, help='Reference JSON with original timings')
    parser.add_argument('--anchor-mapping', required=True, help='Anchor mapping JSON file')
    parser.add_argument('--confidence-threshold', type=float, default=0.8, help='Minimum confidence for using anchors')
    parser.add_argument('--output-ctm', required=True, help='Output merged CTM file')
    parser.add_argument('--output-json', required=True, help='Output merged JSON file')
    parser.add_argument('--output-report', required=True, help='Output merge report JSON')
    
    args = parser.parse_args()
    
    # Load input data
    new_ctm_words = parse_ctm(args.new_alignments_ctm)
    new_json_data = parse_alignment_json(args.new_alignments_json)
    
    with open(args.reference, 'r', encoding='utf-8') as f:
        reference_data = json.load(f)
    
    with open(args.anchor_mapping, 'r', encoding='utf-8') as f:
        anchor_mapping = json.load(f)
    
    print(f"Loaded {len(new_ctm_words)} words from new CTM", file=sys.stderr)
    
    # Merge alignments
    merged_ctm_words, merged_json_words, merge_stats = merge_alignments(
        new_ctm_words, new_json_data, anchor_mapping, reference_data, args.confidence_threshold
    )
    
    # Write outputs
    create_merged_ctm(merged_ctm_words, args.output_ctm)
    create_merged_json(merged_json_words, new_json_data, args.output_json)
    
    # Create merge report
    report = {
        'merge_statistics': merge_stats,
        'input_summary': {
            'new_ctm_words': len(new_ctm_words),
            'available_anchors': len(anchor_mapping.get('anchors', [])),
            'high_confidence_anchors': len([a for a in anchor_mapping.get('anchors', []) 
                                          if a['confidence'] >= args.confidence_threshold])
        },
        'output_summary': {
            'merged_ctm_words': len(merged_ctm_words),
            'merged_json_words': len(merged_json_words)
        },
        'quality_metrics': {
            'anchor_usage_percentage': merge_stats['anchor_usage_rate'] * 100,
            'confidence_threshold_used': args.confidence_threshold,
            'recommended_quality': 'high' if merge_stats['anchor_usage_rate'] > 0.5 else 
                                  'medium' if merge_stats['anchor_usage_rate'] > 0.2 else 'low'
        }
    }
    
    with open(args.output_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Merge completed successfully", file=sys.stderr)
    print(f"Anchor usage: {merge_stats['anchor_usage_rate']*100:.1f}%", file=sys.stderr)

if __name__ == '__main__':
    main()