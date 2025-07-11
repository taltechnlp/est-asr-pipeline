#! /usr/bin/env python3
import json
import sys
import argparse

def get_word(word):
    return word.get("word_with_punctuation", word["word"])

def extract_text_from_json(json_data):
    """Extract plain text from JSON transcription format"""
    text_parts = []
    
    sections = json_data.get("sections", [])
    for section in sections:
        if section["type"] == "speech":
            turns = section.get("turns", [])
            for turn in turns:
                words = turn.get("words", [])
                if words:
                    # Join words with spaces
                    turn_text = " ".join([get_word(word) for word in words])
                    text_parts.append(turn_text)
    
    # Join all turns with line breaks
    return "\n".join(text_parts)

def main():
    parser = argparse.ArgumentParser("Converts JSON format to plain text")
    parser.add_argument('json', help="JSON input file")
    
    args = parser.parse_args()
    
    try:
        trans = json.load(open(args.json))
        text = extract_text_from_json(trans)
        print(text)
    except Exception as e:
        print(f"Error processing JSON file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 