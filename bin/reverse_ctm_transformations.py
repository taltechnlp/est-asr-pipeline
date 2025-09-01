#!/usr/bin/env python3

import json
import sys
import argparse
import re
from collections import defaultdict, namedtuple

# Define transformation patterns for Estonian
TRANSFORMATION_PATTERNS = [
    # Common Estonian compounds
    ('võib-olla', ['võib', 'olla']),
    ('nii-öelda', ['nii', 'öelda']),
    ('võibolla', ['võib', 'olla']),
    ('niiöelda', ['nii', 'öelda']),
    ('enamvähem', ['enam', 'vähem']),
    ('naisedmehed', ['naised', 'mehed']),
    ('poisssõbra', ['poiss', 'sõbra']),
    ('pooleet', ['poole', 'et']),
    
    # Repetition patterns
    ('naanaanaa', ['naa', 'naa', 'naa']),
    
    # Phonetic variants (these need fuzzy matching)
    ('fakboyd', ['fakboy']),
    ('fakboyde', ['fakboy']),
    ('notai', ['no', 'tai']),
    ('pikeet', ['pike', 'et']),
    ('podcasti', ['podcast']),
    ('podcastiga', ['podcast']),
]

CTMEntry = namedtuple('CTMEntry', ['channel', 'start_time', 'duration', 'word', 'confidence'])

class CTMReverser:
    def __init__(self):
        self.transformations = {}
        self.stats = {
            'total_transformations': 0,
            'compound_splits': 0,
            'phonetic_reversals': 0,
            'repetition_expansions': 0,
            'unknown_removals': 0,
            'examples': []
        }
    
    def load_original_words(self, json_file):
        """Load original words from JSON with timing information"""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        original_words = []
        for segment in data['segments']:
            text = segment.get('text', '').strip()
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            
            if text:
                # Split into words and estimate timing
                words = re.findall(r'\b\w+\b', text.lower())
                if words:
                    word_duration = (end_time - start_time) / len(words)
                    for i, word in enumerate(words):
                        word_start = start_time + i * word_duration
                        original_words.append({
                            'word': word,
                            'start': word_start,
                            'end': word_start + word_duration,
                            'segment_start': start_time,
                            'segment_end': end_time
                        })
        
        return original_words
    
    def load_ctm(self, ctm_file):
        """Load CTM file"""
        ctm_entries = []
        with open(ctm_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    entry = CTMEntry(
                        channel=parts[1],
                        start_time=float(parts[2]),
                        duration=float(parts[3]),
                        word=parts[4],
                        confidence=parts[5] if len(parts) > 5 else '1.0'
                    )
                    ctm_entries.append(entry)
        return ctm_entries
    
    def find_transformation(self, word):
        """Find if a word matches any transformation pattern"""
        word_lower = word.lower()
        
        # Check exact matches first
        for compound, original_parts in TRANSFORMATION_PATTERNS:
            if word_lower == compound.lower():
                return original_parts, 'exact'
        
        # Check for phonetic similarities (edit distance)
        for compound, original_parts in TRANSFORMATION_PATTERNS:
            if self.is_phonetic_variant(word_lower, compound.lower()):
                return original_parts, 'phonetic'
        
        # Check if it's an unknown word marker
        if word_lower in ['unk', '<unk>']:
            return [], 'unknown'
        
        return None, None
    
    def is_phonetic_variant(self, word1, word2, threshold=2):
        """Check if two words are phonetic variants using edit distance"""
        def edit_distance(s1, s2):
            if len(s1) < len(s2):
                return edit_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        return edit_distance(word1, word2) <= threshold and len(word1) > 3
    
    def find_original_word_in_timeframe(self, original_words, start_time, duration):
        """Find original word(s) that overlap with the given timeframe"""
        end_time = start_time + duration
        candidates = []
        
        for orig_word in original_words:
            # Check for temporal overlap
            overlap_start = max(start_time, orig_word['start'])
            overlap_end = min(end_time, orig_word['end'])
            
            if overlap_start < overlap_end:  # There is overlap
                overlap_duration = overlap_end - overlap_start
                overlap_ratio = overlap_duration / duration
                candidates.append((orig_word, overlap_ratio))
        
        # Sort by overlap ratio and return best matches
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates]
    
    def distribute_timing(self, original_parts, total_duration):
        """Distribute timing proportionally among original word parts"""
        if not original_parts:
            return []
        
        # Simple proportional distribution based on word length
        total_chars = sum(len(word) for word in original_parts)
        if total_chars == 0:
            word_duration = total_duration / len(original_parts)
            return [(word, word_duration) for word in original_parts]
        
        result = []
        for word in original_parts:
            word_duration = (len(word) / total_chars) * total_duration
            result.append((word, max(word_duration, 0.1)))  # Minimum 0.1s
        
        return result
    
    def reverse_ctm_entry(self, entry, original_words):
        """Reverse a single CTM entry if it matches transformation patterns"""
        transformation, trans_type = self.find_transformation(entry.word)
        
        if transformation is None:
            # No transformation needed
            return [entry]
        
        if trans_type == 'unknown':
            # Try to find original word in timeframe
            candidates = self.find_original_word_in_timeframe(
                original_words, entry.start_time, entry.duration
            )
            if candidates:
                # Use the best candidate
                original_word = candidates[0]['word']
                self.stats['unknown_removals'] += 1
                self.stats['examples'].append({
                    'type': 'unknown_replacement',
                    'original': entry.word,
                    'reversed': original_word,
                    'time': entry.start_time
                })
                return [CTMEntry(
                    channel=entry.channel,
                    start_time=entry.start_time,
                    duration=entry.duration,
                    word=original_word,
                    confidence=entry.confidence
                )]
            else:
                # Remove unknown words
                self.stats['unknown_removals'] += 1
                return []
        
        # Handle word splits (compounds, repetitions, etc.)
        if len(transformation) > 1:
            timing_parts = self.distribute_timing(transformation, entry.duration)
            result = []
            current_time = entry.start_time
            
            for word, word_duration in timing_parts:
                result.append(CTMEntry(
                    channel=entry.channel,
                    start_time=current_time,
                    duration=word_duration,
                    word=word,
                    confidence=entry.confidence
                ))
                current_time += word_duration
            
            # Update statistics
            if trans_type == 'exact':
                if any(pattern in entry.word.lower() for pattern in ['naa', 'aa']):
                    self.stats['repetition_expansions'] += 1
                else:
                    self.stats['compound_splits'] += 1
            elif trans_type == 'phonetic':
                self.stats['phonetic_reversals'] += 1
            
            self.stats['total_transformations'] += 1
            self.stats['examples'].append({
                'type': trans_type,
                'original': entry.word,
                'reversed': ' '.join(transformation),
                'time': entry.start_time
            })
            
            return result
        
        elif len(transformation) == 1:
            # Single word replacement (phonetic correction)
            self.stats['phonetic_reversals'] += 1
            self.stats['total_transformations'] += 1
            self.stats['examples'].append({
                'type': 'phonetic',
                'original': entry.word,
                'reversed': transformation[0],
                'time': entry.start_time
            })
            
            return [CTMEntry(
                channel=entry.channel,
                start_time=entry.start_time,
                duration=entry.duration,
                word=transformation[0],
                confidence=entry.confidence
            )]
        
        # Default: no change
        return [entry]
    
    def process_ctm(self, ctm_file, json_file, output_ctm, output_report):
        """Process the CTM file and reverse transformations"""
        print(f"Loading original words from {json_file}...")
        original_words = self.load_original_words(json_file)
        print(f"Loaded {len(original_words)} original words")
        
        print(f"Loading CTM from {ctm_file}...")
        ctm_entries = self.load_ctm(ctm_file)
        print(f"Loaded {len(ctm_entries)} CTM entries")
        
        print("Processing transformations...")
        reversed_entries = []
        
        for entry in ctm_entries:
            reversed_parts = self.reverse_ctm_entry(entry, original_words)
            reversed_entries.extend(reversed_parts)
        
        print(f"Writing reversed CTM to {output_ctm}...")
        with open(output_ctm, 'w', encoding='utf-8') as f:
            for entry in reversed_entries:
                f.write(f"audio {entry.channel} {entry.start_time:.3f} {entry.duration:.3f} {entry.word} {entry.confidence}\n")
        
        print(f"Writing transformation report to {output_report}...")
        with open(output_report, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        print(f"Completed! Processed {len(ctm_entries)} entries -> {len(reversed_entries)} entries")
        print(f"Total transformations: {self.stats['total_transformations']}")
        print(f"- Compound splits: {self.stats['compound_splits']}")
        print(f"- Phonetic reversals: {self.stats['phonetic_reversals']}")
        print(f"- Repetition expansions: {self.stats['repetition_expansions']}")
        print(f"- Unknown removals: {self.stats['unknown_removals']}")

def main():
    parser = argparse.ArgumentParser(description='Reverse CTM transformations for WER comparison')
    parser.add_argument('ctm_file', help='Input CTM file')
    parser.add_argument('json_file', help='Original JSON file')
    parser.add_argument('text_file', help='Kaldi text file (optional)', nargs='?')
    parser.add_argument('--output-ctm', required=True, help='Output reversed CTM file')
    parser.add_argument('--output-report', required=True, help='Output transformation report JSON')
    
    args = parser.parse_args()
    
    reverser = CTMReverser()
    reverser.process_ctm(args.ctm_file, args.json_file, args.output_ctm, args.output_report)

if __name__ == '__main__':
    main()