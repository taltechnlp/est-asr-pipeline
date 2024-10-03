#!/usr/bin/env python

import json
import argparse
import sys
from lxml import etree

def create_transcriber_trs(json_data, file_id):
    # Create the root Trans element and add the audio filename attribute
    root = etree.Element('Trans')
    if file_id:
        root.set('audio_filename', file_id)
    
    # Speakers section (separate from Episode)
    speakers_elem = etree.SubElement(root, 'Speakers')
    
    speakers = json_data['speakers']
    
    # Create a mapping from original speaker IDs to "spk%d" format
    speaker_map = {}
    for i, speaker_id in enumerate(speakers.keys(), 1):
        speaker_map[speaker_id] = f"spk{i}"
    
    # Create speakers in the <Speakers> section
    for speaker_id, speaker_info in speakers.items():
        speaker_elem = etree.SubElement(speakers_elem, 'Speaker')
        speaker_elem.set('id', speaker_map[speaker_id])
        speaker_elem.set('name', speaker_info.get('name', 'Unknown'))
    
    # Create the Episode section (without Version)
    episode = etree.SubElement(root, 'Episode')
    
    # Create sections and turns
    for section in json_data['sections']:
        section_elem = etree.SubElement(episode, 'Section')
        section_elem.set('type', "report")
        section_elem.set('startTime', str(section['start']))
        section_elem.set('endTime', str(section['end']))

        current_turn_elem = None
        
        for i, turn in enumerate(section.get('turns', [])):                
            start = turn['start']
            end = turn['end']
            if i == 0:
              section_elem.set('startTime', str(start))
            speaker = speaker_map[turn['speaker']]  # Use the mapped speaker ID
            unnormalized_transcript = turn.get('unnormalized_transcript', '')
            
            # Check if the current turn can be merged with the previous one
            #breakpoint()
            if current_turn_elem is not None and abs(float(current_turn_elem.get('endTime')) - start) < 0.0001 and current_turn_elem.get('speaker') == speaker:
                # Insert a Sync tag and continue the same turn
                sync = etree.SubElement(current_turn_elem, 'Sync')
                sync.set('time', str(start))
                sync.tail = unnormalized_transcript
                # Update the endTime of the current turn
                current_turn_elem.set('endTime', str(end))
            else:
                # Create a new turn
                current_turn_elem = etree.SubElement(section_elem, 'Turn')
                current_turn_elem.set('startTime', str(start))
                current_turn_elem.set('endTime', str(end))
                current_turn_elem.set('speaker', speaker)
                
                # Add the transcript to the current turn
                current_turn_elem.text = unnormalized_transcript
        
        
        if current_turn_elem is not None:
            section_elem.set('endTime', current_turn_elem.get("endTime"))
    # Return the formatted XML with DOCTYPE
    return root

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert JSON transcription to TRS XML format")
    parser.add_argument("input_json", help="Path to the input JSON file containing transcription data")
    parser.add_argument("--fid", type=str, default="", help="File ID to use as the audio_filename attribute in the TRS file")

    args = parser.parse_args()

    # Load the JSON file
    with open(args.input_json, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Create the TRS XML using the provided file ID
    trs_xml_tree = create_transcriber_trs(json_data, args.fid)

    # Create the DOCTYPE element and serialize XML with DOCTYPE
    doctype = '<!DOCTYPE Trans SYSTEM "trans-14.dtd">'
    trs_xml_with_doctype = etree.tostring(
        trs_xml_tree, 
        pretty_print=True, 
        xml_declaration=True, 
        encoding="UTF-8", 
        doctype=doctype
    ).decode('utf-8')

    # Print the output to stdout
    sys.stdout.write(trs_xml_with_doctype)

if __name__ == "__main__":
    main()
