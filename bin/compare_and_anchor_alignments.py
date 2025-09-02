#!/usr/bin/env python3

import json
import sys
import argparse
import difflib
from typing import Dict, List, Tuple, Any
import re

def normalize_text(text):
    """Normalize text for comparison by removing punctuation and converting to lowercase."""
    return re.sub(r'[^\w\s]', '', text.lower()).strip()

def extract_words_from_json(data):
    """Extract word list with timing info from JSON data."""
    words = []
    
    # Handle different JSON formats
    if 'segments' in data:
        # Direct segments format or word-level segments
        for segment in data['segments']:
            if 'text' in segment:
                # Word-level segment
                words.append({
                    'text': segment['text'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'source': 'segment'
                })
            elif 'alternatives' in segment:
                # Segment with alternatives, use best
                best_text = segment['alternatives'][0]['text'] if segment['alternatives'] else ''
                if best_text.strip():
                    # Split into words and estimate timings
                    segment_words = best_text.split()
                    duration = segment['end'] - segment['start']
                    word_duration = duration / len(segment_words) if segment_words else 0
                    
                    for i, word in enumerate(segment_words):
                        word_start = segment['start'] + (i * word_duration)
                        word_end = segment['start'] + ((i + 1) * word_duration)
                        words.append({
                            'text': word,
                            'start': word_start,
                            'end': min(word_end, segment['end']),
                            'source': 'segment_split'
                        })
    
    elif 'best_hypothesis' in data:
        # Pipeline format with best hypothesis
        hypothesis = data['best_hypothesis']
        if 'sections' in hypothesis:
            # New format with sections
            for section in hypothesis['sections']:
                if section.get('type') == 'speech':
                    for turn in section.get('turns', []):
                        for word in turn.get('words', []):
                            words.append({
                                'text': word.get('word_with_punctuation', word.get('word', '')),
                                'start': word['start'],
                                'end': word['end'],
                                'source': 'word'
                            })
        elif 'monologues' in hypothesis:
            # Old format with monologues
            for monologue in hypothesis['monologues']:
                for turn in monologue.get('turns', []):
                    for word in turn.get('words', []):
                        words.append({
                            'text': word.get('word_with_punctuation', word.get('word', '')),
                            'start': word['start'],
                            'end': word['end'],
                            'source': 'word'
                        })
    
    elif 'sections' in data:
        # Direct sections format (like from TRS)
        for section in data['sections']:
            for turn in section.get('turns', []):
                for word in turn.get('words', []):
                    words.append({
                        'text': word['text'],
                        'start': word['start'],
                        'end': word['end'],
                        'source': 'word'
                    })
    
    return sorted(words, key=lambda x: x['start'])

def compare_texts(reference_words, edited_words, confidence_threshold=0.8):
    """Compare reference and edited text, identify anchors and changes."""
    
    # Extract just the text for comparison
    ref_texts = [normalize_text(w['text']) for w in reference_words]
    edited_texts = [normalize_text(w['text']) for w in edited_words]
    
    # Use difflib to find matching sequences
    matcher = difflib.SequenceMatcher(None, ref_texts, edited_texts)
    matches = matcher.get_matching_blocks()
    
    anchors = []
    edits = []
    anchor_count = 0
    
    for match in matches:
        # match: (ref_start, edited_start, length)
        if match.size == 0:
            continue
            
        for i in range(match.size):
            ref_idx = match.a + i
            edited_idx = match.b + i
            
            if ref_idx < len(reference_words) and edited_idx < len(edited_words):
                ref_word = reference_words[ref_idx]
                edited_word = edited_words[edited_idx]
                
                # Calculate confidence based on exact text match
                ref_norm = normalize_text(ref_word['text'])
                edited_norm = normalize_text(edited_word['text'])
                
                if ref_norm == edited_norm and ref_norm.strip():
                    confidence = 1.0 if ref_word['text'] == edited_word['text'] else 0.9
                    
                    anchors.append({
                        'reference_index': ref_idx,
                        'edited_index': edited_idx,
                        'text': ref_word['text'],
                        'reference_timing': {
                            'start': ref_word['start'],
                            'end': ref_word['end']
                        },
                        'confidence': confidence,
                        'anchor_id': f"anchor_{anchor_count:04d}"
                    })
                    anchor_count += 1
    
    # Identify edit operations
    opcodes = matcher.get_opcodes()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'delete':
            edits.append({
                'type': 'delete',
                'reference_range': [i1, i2],
                'words': [reference_words[i]['text'] for i in range(i1, min(i2, len(reference_words)))]
            })
        elif tag == 'insert':
            edits.append({
                'type': 'insert', 
                'edited_range': [j1, j2],
                'words': [edited_words[j]['text'] for j in range(j1, min(j2, len(edited_words)))]
            })
        elif tag == 'replace':
            edits.append({
                'type': 'replace',
                'reference_range': [i1, i2],
                'edited_range': [j1, j2],
                'reference_words': [reference_words[i]['text'] for i in range(i1, min(i2, len(reference_words)))],
                'edited_words': [edited_words[j]['text'] for j in range(j1, min(j2, len(edited_words)))]
            })
    
    return anchors, edits

def create_anchored_segments(edited_data, anchors, confidence_threshold):
    """Create segments for alignment that preserve high-confidence anchors."""
    
    # Extract words from edited data
    edited_words = extract_words_from_json(edited_data)
    
    # Filter high-confidence anchors
    high_conf_anchors = [a for a in anchors if a['confidence'] >= confidence_threshold]
    
    # Create segments with anchor constraints
    segments = []
    current_segment_words = []
    current_start = None
    segment_id = 1
    
    # Group words into segments while respecting anchors
    for i, word in enumerate(edited_words):
        # Check if this word is a high-confidence anchor
        anchor = next((a for a in high_conf_anchors if a['edited_index'] == i), None)
        
        if current_start is None:
            current_start = word.get('start', 0)
            
        current_segment_words.append(word['text'])
        
        # Decide when to end a segment
        should_end_segment = False
        
        # End segment if it would be too long (30 seconds)
        if word.get('end', 0) - current_start > 30.0:
            should_end_segment = True
            
        # End segment if there's a significant gap to next word (>2 seconds)
        if i + 1 < len(edited_words):
            next_word = edited_words[i + 1]
            gap = next_word.get('start', 0) - word.get('end', 0)
            if gap > 2.0:
                should_end_segment = True
        
        # End at last word
        if i == len(edited_words) - 1:
            should_end_segment = True
            
        # End if we have enough content (>15 seconds) and next word is an anchor
        if (word.get('end', 0) - current_start > 15.0 and 
            i + 1 < len(edited_words) and
            any(a['edited_index'] == i + 1 for a in high_conf_anchors)):
            should_end_segment = True
            
        if should_end_segment and current_segment_words:
            segment_end = word.get('end', current_start + 1.0)
            
            segments.append({
                'id': f'seg_{segment_id:04d}',
                'start': current_start,
                'end': segment_end,
                'text': ' '.join(current_segment_words),
                'anchor_constraints': [a for a in high_conf_anchors 
                                     if current_start <= a.get('reference_timing', {}).get('start', 0) <= segment_end]
            })
            
            segment_id += 1
            current_segment_words = []
            current_start = None
    
    return segments

def main():
    parser = argparse.ArgumentParser(description="Compare edited and reference transcripts, identify anchors")
    parser.add_argument('--edited', required=True, help='Edited JSON file')
    parser.add_argument('--reference', required=True, help='Reference JSON file')  
    parser.add_argument('--confidence-threshold', type=float, default=0.8, help='Minimum confidence for anchors')
    parser.add_argument('--output-mapping', required=True, help='Output anchor mapping JSON')
    parser.add_argument('--output-segments', required=True, help='Output anchored segments JSON')
    parser.add_argument('--output-report', required=True, help='Output analysis report JSON')
    
    args = parser.parse_args()
    
    # Load data
    with open(args.edited, 'r', encoding='utf-8') as f:
        edited_data = json.load(f)
    
    with open(args.reference, 'r', encoding='utf-8') as f:
        reference_data = json.load(f)
    
    # Extract words
    reference_words = extract_words_from_json(reference_data)
    edited_words = extract_words_from_json(edited_data)
    
    print(f"Reference words: {len(reference_words)}", file=sys.stderr)
    print(f"Edited words: {len(edited_words)}", file=sys.stderr)
    
    # Compare and find anchors
    anchors, edits = compare_texts(reference_words, edited_words, args.confidence_threshold)
    
    high_conf_anchors = [a for a in anchors if a['confidence'] >= args.confidence_threshold]
    
    print(f"Found {len(anchors)} total anchors, {len(high_conf_anchors)} high-confidence", file=sys.stderr)
    print(f"Found {len(edits)} edit operations", file=sys.stderr)
    
    # Create anchored segments
    anchored_segments = create_anchored_segments(edited_data, anchors, args.confidence_threshold)
    
    print(f"Created {len(anchored_segments)} anchored segments", file=sys.stderr)
    
    # Output anchor mapping
    anchor_mapping = {
        'anchors': anchors,
        'high_confidence_anchors': high_conf_anchors,
        'edit_operations': edits,
        'statistics': {
            'total_anchors': len(anchors),
            'high_confidence_count': len(high_conf_anchors),
            'reference_words': len(reference_words),
            'edited_words': len(edited_words),
            'anchor_ratio': len(high_conf_anchors) / len(edited_words) if edited_words else 0,
            'edit_operations': len(edits)
        }
    }
    
    with open(args.output_mapping, 'w', encoding='utf-8') as f:
        json.dump(anchor_mapping, f, indent=2, ensure_ascii=False)
    
    # Output anchored segments
    segments_output = {
        'segments': anchored_segments,
        'metadata': {
            'anchor_count': len(high_conf_anchors),
            'confidence_threshold': args.confidence_threshold,
            'source': 'anchored_alignment'
        }
    }
    
    with open(args.output_segments, 'w', encoding='utf-8') as f:
        json.dump(segments_output, f, indent=2, ensure_ascii=False)
    
    # Output analysis report
    report = {
        'comparison_summary': {
            'reference_words': len(reference_words),
            'edited_words': len(edited_words),
            'total_anchors': len(anchors),
            'high_confidence_anchors': len(high_conf_anchors),
            'anchor_percentage': (len(high_conf_anchors) / len(edited_words) * 100) if edited_words else 0,
            'edit_operations': len(edits)
        },
        'edit_analysis': {
            'insertions': len([e for e in edits if e['type'] == 'insert']),
            'deletions': len([e for e in edits if e['type'] == 'delete']),
            'replacements': len([e for e in edits if e['type'] == 'replace'])
        },
        'segmentation': {
            'segments_created': len(anchored_segments),
            'average_segment_duration': sum(s['end'] - s['start'] for s in anchored_segments) / len(anchored_segments) if anchored_segments else 0
        },
        'confidence_threshold': args.confidence_threshold,
        'recommended_for_anchoring': len(high_conf_anchors) > len(edited_words) * 0.3  # At least 30% anchors
    }
    
    with open(args.output_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()