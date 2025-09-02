#!/usr/bin/env python3

import json
import sys
import argparse
import os

def create_kaldi_datadir(segments_data, output_dir):
    """
    Create Kaldi datadir from anchored segments data with optimized segmentation
    for forced alignment.
    """
    
    segments = segments_data.get('segments', [])
    
    if not segments:
        print("No segments found in input data", file=sys.stderr)
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    segments_lines = []
    text_lines = []
    utt2spk_lines = []
    
    # Process each segment
    for segment in segments:
        segment_id = segment.get('id', f"seg_{len(segments_lines):04d}")
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text'].strip()
        
        # Skip empty segments
        if not text:
            continue
            
        # Validate timing
        if end_time <= start_time:
            print(f"Warning: Invalid timing for segment {segment_id}: {start_time} -> {end_time}", file=sys.stderr)
            end_time = start_time + 0.1  # Minimum duration
            
        # Clean text for Kaldi (remove excessive punctuation, normalize)
        cleaned_text = clean_text_for_kaldi(text)
        
        if not cleaned_text.strip():
            print(f"Warning: Segment {segment_id} has no valid text after cleaning", file=sys.stderr)
            continue
            
        # Add to Kaldi format
        segments_lines.append(f"{segment_id} audio {start_time:.3f} {end_time:.3f}")
        text_lines.append(f"{segment_id} {cleaned_text}")
        utt2spk_lines.append(f"{segment_id} {segment_id}")  # Each segment is its own speaker for alignment
    
    if not segments_lines:
        print("Error: No valid segments created", file=sys.stderr)
        return False
    
    # Write Kaldi format files
    with open(os.path.join(output_dir, 'segments'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(segments_lines) + '\n')
        
    with open(os.path.join(output_dir, 'text'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_lines) + '\n')
        
    with open(os.path.join(output_dir, 'utt2spk'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(utt2spk_lines) + '\n')
        
    with open(os.path.join(output_dir, 'spk2utt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(utt2spk_lines) + '\n')  # Same as utt2spk for single-speaker segments
    
    print(f"Created Kaldi datadir with {len(segments_lines)} segments in {output_dir}", file=sys.stderr)
    
    # Write segment metadata for later use
    metadata = {
        'segment_count': len(segments_lines),
        'total_duration': sum(seg['end'] - seg['start'] for seg in segments),
        'average_duration': sum(seg['end'] - seg['start'] for seg in segments) / len(segments) if segments else 0,
        'anchor_constraints': sum(1 for seg in segments if seg.get('anchor_constraints', [])),
        'processing_info': {
            'text_cleaning_applied': True,
            'timing_validation_applied': True,
            'source': 'smart_segment_for_alignment'
        }
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return True

def clean_text_for_kaldi(text):
    """
    Clean text for Kaldi alignment by normalizing punctuation and formatting.
    """
    import re
    
    # Remove or normalize problematic characters
    text = re.sub(r'[""„"]', '"', text)  # Normalize quotes
    text = re.sub(r'[''`]', "'", text)   # Normalize apostrophes
    text = re.sub(r'[–—]', '-', text)    # Normalize dashes
    text = re.sub(r'\.{2,}', '...', text)  # Normalize ellipsis
    
    # Remove excessive punctuation at word boundaries
    text = re.sub(r'([.!?]){2,}', r'\1', text)  # Multiple sentence endings
    text = re.sub(r'([,;:]){2,}', r'\1', text)   # Multiple commas/colons
    
    # Ensure spaces around punctuation for better processing
    text = re.sub(r'([.!?])([A-ZÄÖÜÕŠŽ])', r'\1 \2', text)  # Space after sentence end
    text = re.sub(r'([,;:])([A-ZÄÖÜÕŠŽa-zäöüõšž])', r'\1 \2', text)  # Space after comma/colon
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert to lowercase for alignment (Kaldi typically works better with lowercase)
    # Note: Keep original case if the acoustic model expects it
    # text = text.lower()  # Uncomment if needed
    
    return text

def optimize_segmentation(segments_data, max_duration=30.0, min_duration=5.0):
    """
    Optimize segment boundaries for better alignment while preserving anchors.
    """
    
    segments = segments_data.get('segments', [])
    optimized_segments = []
    
    current_segment = None
    
    for segment in segments:
        duration = segment['end'] - segment['start']
        anchor_constraints = segment.get('anchor_constraints', [])
        
        # If segment is too short, try to merge with next
        if duration < min_duration and not anchor_constraints:
            if current_segment is None:
                current_segment = segment.copy()
            else:
                # Merge with current segment
                current_segment['end'] = segment['end']
                current_segment['text'] += ' ' + segment['text']
                # Combine anchor constraints
                current_constraints = current_segment.get('anchor_constraints', [])
                current_constraints.extend(anchor_constraints)
                current_segment['anchor_constraints'] = current_constraints
        else:
            # Finalize current segment if exists
            if current_segment is not None:
                optimized_segments.append(current_segment)
                current_segment = None
                
            # If current segment is too long, try to split at natural boundaries
            if duration > max_duration:
                split_segments = split_long_segment(segment, max_duration)
                optimized_segments.extend(split_segments)
            else:
                optimized_segments.append(segment)
    
    # Don't forget the last segment
    if current_segment is not None:
        optimized_segments.append(current_segment)
    
    # Update segment IDs
    for i, segment in enumerate(optimized_segments):
        segment['id'] = f"seg_{i+1:04d}"
    
    return {
        **segments_data,
        'segments': optimized_segments,
        'metadata': {
            **segments_data.get('metadata', {}),
            'optimization_applied': True,
            'original_segment_count': len(segments),
            'optimized_segment_count': len(optimized_segments)
        }
    }

def split_long_segment(segment, max_duration):
    """
    Split a long segment at natural boundaries (sentence breaks, pauses).
    """
    text = segment['text']
    start_time = segment['start']
    end_time = segment['end']
    total_duration = end_time - start_time
    
    # Find sentence boundaries
    import re
    sentences = re.split(r'([.!?]\s+)', text)
    if len(sentences) <= 2:  # No good split points
        # Split roughly in the middle
        mid_time = start_time + (total_duration / 2)
        words = text.split()
        mid_word = len(words) // 2
        
        return [
            {
                'id': segment['id'] + '_a',
                'start': start_time,
                'end': mid_time,
                'text': ' '.join(words[:mid_word]),
                'anchor_constraints': []
            },
            {
                'id': segment['id'] + '_b', 
                'start': mid_time,
                'end': end_time,
                'text': ' '.join(words[mid_word:]),
                'anchor_constraints': segment.get('anchor_constraints', [])
            }
        ]
    
    # Split at sentence boundaries
    splits = []
    current_text = ""
    current_start = start_time
    
    for i, part in enumerate(sentences):
        current_text += part
        
        # Estimate timing based on text length
        text_progress = len(current_text) / len(text)
        current_end = start_time + (text_progress * total_duration)
        
        # If we have a complete sentence and reasonable duration
        if ('.' in part or '!' in part or '?' in part) and (current_end - current_start) > 5.0:
            splits.append({
                'id': f"{segment['id']}_{len(splits)+1}",
                'start': current_start,
                'end': current_end,
                'text': current_text.strip(),
                'anchor_constraints': []
            })
            current_text = ""
            current_start = current_end
    
    # Handle remaining text
    if current_text.strip():
        splits.append({
            'id': f"{segment['id']}_{len(splits)+1}",
            'start': current_start,
            'end': end_time,
            'text': current_text.strip(),
            'anchor_constraints': segment.get('anchor_constraints', [])
        })
    
    return splits if splits else [segment]

def main():
    parser = argparse.ArgumentParser(description="Create optimized Kaldi datadir from anchored segments")
    parser.add_argument('segments_json', help='Input anchored segments JSON file')
    parser.add_argument('output_dir', help='Output Kaldi datadir directory')
    parser.add_argument('--optimize', action='store_true', help='Apply segmentation optimization')
    parser.add_argument('--max-duration', type=float, default=30.0, help='Maximum segment duration')
    parser.add_argument('--min-duration', type=float, default=5.0, help='Minimum segment duration')
    
    args = parser.parse_args()
    
    # Load segments data
    with open(args.segments_json, 'r', encoding='utf-8') as f:
        segments_data = json.load(f)
    
    # Apply optimization if requested
    if args.optimize:
        print("Applying segmentation optimization...", file=sys.stderr)
        segments_data = optimize_segmentation(
            segments_data, 
            args.max_duration, 
            args.min_duration
        )
    
    # Create Kaldi datadir
    success = create_kaldi_datadir(segments_data, args.output_dir)
    
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main()