#!/usr/bin/env python3

import json
import sys
import argparse
import os

def main():
    parser = argparse.ArgumentParser("Convert JSON to Kaldi datadir format")
    parser.add_argument('json_file', help='Input JSON file')
    parser.add_argument('output_dir', help='Output datadir directory')
    args = parser.parse_args()

    # Load JSON file
    with open(args.json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    segments_lines = []
    text_lines = []

    # Handle different JSON formats
    if 'segments' in data:
        # Direct segments format
        segments = data['segments']
    elif 'best_hypothesis' in data and 'sections' in data['best_hypothesis']:
        # New n-best pipeline format - extract from best hypothesis sections
        segments = []
        for section in data['best_hypothesis']['sections']:
            if section.get('type') == 'speech':
                for turn in section.get('turns', []):
                    for word in turn.get('words', []):
                        segments.append({
                            'start': word['start'],
                            'end': word['end'],
                            'text': word['word_with_punctuation']
                        })
    elif 'best_hypothesis' in data and 'monologues' in data['best_hypothesis']:
        # Old pipeline format - extract from best hypothesis monologues
        segments = []
        for monologue in data['best_hypothesis']['monologues']:
            for turn in monologue.get('turns', []):
                for word in turn.get('words', []):
                    segments.append({
                        'start': word['start'],
                        'end': word['end'],
                        'text': word['word_with_punctuation']
                    })
    else:
        print('Error: Unsupported JSON format', file=sys.stderr)
        sys.exit(1)

    # Sort segments by start time and validate timestamps
    valid_segments = []
    for item in segments:
        if 'text' in item:
            # Word-level data or direct segments with 'text' field
            text = item['text']
            start = item['start']
            end = item['end']
        else:
            # Segment-level data with alternatives
            text = item.get('alternatives', [{}])[0].get('text', '') if 'alternatives' in item else item.get('text', '')
            start = item['start']
            end = item['end']
        
        if not text.strip():
            continue
            
        # Skip segments with invalid timestamps
        if end <= start:
            print(f"Warning: Skipping segment with invalid timestamps: start={start}, end={end}", file=sys.stderr)
            continue
            
        valid_segments.append({'text': text, 'start': start, 'end': end})
    
    # Sort by start time
    valid_segments.sort(key=lambda x: x['start'])

    # Group words into segments (20-30 second target duration for better alignment)
    current_segment = []
    current_start = None
    current_end = None
    segment_id = 1
    min_duration = 20.0  # Minimum 20 seconds
    max_duration = 30.0  # Maximum 30 seconds

    for item in valid_segments:
        text = item['text']
        start = item['start']
        end = item['end']
        
        if current_start is None:
            current_start = start
            current_end = end
            current_segment = [text]
        else:
            new_duration = end - current_start
            current_duration = current_end - current_start
            
            # If adding this segment would exceed max duration (30s), finalize current segment
            if new_duration > max_duration:
                # Output current segment (ensure end >= start)
                utt_id = f'{segment_id:04d}'
                segments_lines.append(f'{utt_id} audio {current_start:.3f} {max(current_end, current_start + 0.1):.3f}')
                text_lines.append(f'{utt_id} {" ".join(current_segment)}')
                
                # Start new segment
                segment_id += 1
                current_start = start
                current_end = end
                current_segment = [text]
            else:
                # Check for gaps - if there's a significant gap (>3 seconds) and we have enough content (>20s), finalize
                gap = start - current_end if current_end else 0
                has_long_gap = gap > 3.0
                has_min_duration = current_duration >= min_duration
                
                if has_long_gap and has_min_duration:
                    # Finalize current segment
                    utt_id = f'{segment_id:04d}'
                    segments_lines.append(f'{utt_id} audio {current_start:.3f} {max(current_end, current_start + 0.1):.3f}')
                    text_lines.append(f'{utt_id} {" ".join(current_segment)}')
                    
                    # Start new segment
                    segment_id += 1
                    current_start = start
                    current_end = end
                    current_segment = [text]
                else:
                    # Add to current segment - extend even across small gaps to reach target duration
                    current_end = max(current_end, end)  # Extend end time
                    current_segment.append(text)

    # Output final segment if exists
    if current_segment:
        utt_id = f'{segment_id:04d}'
        segments_lines.append(f'{utt_id} audio {current_start:.3f} {max(current_end, current_start + 0.1):.3f}')
        text_lines.append(f'{utt_id} {" ".join(current_segment)}')

    # Write Kaldi format files
    with open(os.path.join(args.output_dir, 'segments'), 'w') as f:
        f.write('\n'.join(segments_lines) + '\n')

    with open(os.path.join(args.output_dir, 'text'), 'w') as f:
        f.write('\n'.join(text_lines) + '\n')

    # Create utt2spk and spk2utt (each utterance is its own speaker for alignment)
    utt2spk_lines = []
    for i in range(len(segments_lines)):
        utt_id = f'{i+1:04d}'
        utt2spk_lines.append(f'{utt_id} {utt_id}')

    with open(os.path.join(args.output_dir, 'utt2spk'), 'w') as f:
        f.write('\n'.join(utt2spk_lines) + '\n')

    with open(os.path.join(args.output_dir, 'spk2utt'), 'w') as f:
        f.write('\n'.join(utt2spk_lines) + '\n')

    print(f"Created Kaldi datadir with {len(segments_lines)} segments", file=sys.stderr)

if __name__ == '__main__':
    main()