#!/usr/bin/env python3

import json
import sys
import argparse
from lxml import etree

def parse_trs_to_json(trs_file):
    """
    Parse TRS (Transcriber XML) file and convert to JSON format
    compatible with the alignment pipeline.
    """
    try:
        tree = etree.parse(trs_file)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing TRS file: {e}", file=sys.stderr)
        return None

    result = {
        'segments': [],
        'speakers': {},
        'sections': [],
        'metadata': {
            'source_format': 'trs',
            'audio_filename': root.get('audio_filename', '')
        }
    }

    # Parse speakers
    speakers_elem = root.find('Speakers')
    if speakers_elem is not None:
        for speaker in speakers_elem.findall('Speaker'):
            speaker_id = speaker.get('id', '')
            speaker_name = speaker.get('name', 'Unknown')
            result['speakers'][speaker_id] = {
                'name': speaker_name
            }

    # Parse episodes and sections  
    episode = root.find('Episode')
    if episode is not None:
        for section in episode.findall('Section'):
            section_start = float(section.get('startTime', 0))
            section_end = float(section.get('endTime', 0))
            section_type = section.get('type', 'report')
            
            section_data = {
                'start': section_start,
                'end': section_end,
                'type': section_type,
                'turns': []
            }

            # Parse turns within section
            for turn in section.findall('Turn'):
                turn_start = float(turn.get('startTime', 0))
                turn_end = float(turn.get('endTime', 0))
                speaker = turn.get('speaker', 'unknown')
                
                # Extract text content including sync tags
                text_parts = []
                
                # Get initial text
                if turn.text:
                    text_parts.append(turn.text.strip())
                
                # Process sync tags and their tail text
                current_time = turn_start
                word_segments = []
                
                for sync in turn.findall('Sync'):
                    sync_time = float(sync.get('time', current_time))
                    if sync.tail:
                        sync_text = sync.tail.strip()
                        if sync_text:
                            text_parts.append(sync_text)
                            # Create word-level segments for sync portions
                            words = sync_text.split()
                            if words:
                                duration_per_word = (turn_end - sync_time) / len(words)
                                for i, word in enumerate(words):
                                    word_start = sync_time + (i * duration_per_word)
                                    word_end = sync_time + ((i + 1) * duration_per_word)
                                    word_segments.append({
                                        'start': word_start,
                                        'end': min(word_end, turn_end),
                                        'text': word,
                                        'speaker': speaker
                                    })
                    current_time = sync_time

                # Combine all text
                full_text = ' '.join(text_parts).strip()
                
                # If no sync tags, treat whole turn as one segment
                if not word_segments and full_text:
                    words = full_text.split()
                    if words:
                        duration_per_word = (turn_end - turn_start) / len(words)
                        for i, word in enumerate(words):
                            word_start = turn_start + (i * duration_per_word)
                            word_end = turn_start + ((i + 1) * duration_per_word)
                            word_segments.append({
                                'start': word_start,
                                'end': min(word_end, turn_end),
                                'text': word,
                                'speaker': speaker
                            })

                turn_data = {
                    'start': turn_start,
                    'end': turn_end,
                    'speaker': speaker,
                    'text': full_text,
                    'words': word_segments
                }
                
                section_data['turns'].append(turn_data)
                
                # Add word segments to main segments list
                result['segments'].extend(word_segments)

            result['sections'].append(section_data)

    # Sort segments by start time
    result['segments'].sort(key=lambda x: x['start'])

    # Add segment IDs
    for i, segment in enumerate(result['segments']):
        segment['id'] = f"seg_{i:04d}"

    return result

def main():
    parser = argparse.ArgumentParser(description="Convert TRS XML to JSON format for alignment")
    parser.add_argument('trs_file', help='Input TRS XML file')
    parser.add_argument('--output', '-o', help='Output JSON file (default: stdout)')
    
    args = parser.parse_args()
    
    result = parse_trs_to_json(args.trs_file)
    
    if result is None:
        sys.exit(1)
    
    json_output = json.dumps(result, indent=2, ensure_ascii=False)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(json_output)
    else:
        print(json_output)

if __name__ == '__main__':
    main()