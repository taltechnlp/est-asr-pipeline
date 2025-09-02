#!/usr/bin/env python3
"""
TRS-CTM Comparison Tool

This script compares TRS segments with CTM word alignments from three different sources,
providing detailed analysis of timing alignment and content differences.

Usage:
    python trs_ctm_compare.py <trs_file> <ctm0_file> <ctm1_file> <ctm2_file> <output_json>

Example:
    python trs_ctm_compare.py reference.trs alt_asr.ctm main_asr.ctm agent.ctm comparison.json
    
Arguments:
    trs_file: TRS reference file
    ctm0_file: Alternative ASR CTM file
    ctm1_file: Main ASR CTM file 
    ctm2_file: Agent CTM file
"""

import xml.etree.ElementTree as ET
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class CTMWord:
    """Represents a single word from a CTM file"""
    file_id: str
    channel: str
    start_time: float
    duration: float
    word: str
    confidence: Optional[float] = None

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    def overlaps_with(self, start: float, end: float, tolerance: float = 0.1) -> bool:
        """Check if this word overlaps with a time segment"""
        return not (self.end_time < start - tolerance or self.start_time > end + tolerance)


@dataclass
class TRSSegment:
    """Represents a Turn segment from TRS file"""
    segment_id: int
    start_time: float
    end_time: float
    speaker: str
    text: str
    language_events: List[Dict[str, Any]]


class TRSCTMComparator:
    """Main class for comparing TRS segments with CTM words"""

    def __init__(self, time_tolerance: float = 0.1):
        self.time_tolerance = time_tolerance
        self.trs_segments = []
        self.ctm0_words = []  # Alternative ASR CTM
        self.ctm1_words = []  # Main ASR CTM
        self.ctm2_words = []  # Agent CTM

    def parse_trs_file(self, trs_file: Path) -> List[TRSSegment]:
        """Parse TRS XML file and extract Turn segments"""
        print(f"Parsing TRS file: {trs_file}")
        
        tree = ET.parse(trs_file)
        root = tree.getroot()
        
        segments = []
        segment_id = 0
        
        # Extract speaker mappings
        speaker_map = {}
        for speaker in root.findall(".//Speaker"):
            speaker_id = speaker.attrib.get("id", "")
            speaker_name = speaker.attrib.get("name", speaker_id)
            speaker_map[speaker_id] = speaker_name
        
        # Process each Turn
        for turn in root.findall(".//Turn"):
            segment_id += 1
            
            speaker_id = turn.attrib.get("speaker", "").strip()
            speaker_name = speaker_map.get(speaker_id, speaker_id)
            
            start_time = float(turn.attrib["startTime"])
            end_time = float(turn.attrib["endTime"])
            
            # Extract text content and language events
            text_parts = []
            language_events = []
            
            # Process initial text
            if turn.text:
                text_parts.append(turn.text.strip())
            
            # Process child elements
            current_lang = "et"  # Default to Estonian
            for elem in turn:
                if elem.tag == "Event":
                    event_type = elem.attrib.get("type", "")
                    if event_type == "language":
                        lang_desc = elem.attrib.get("desc", "")
                        current_lang = lang_desc if lang_desc else "et"
                        language_events.append({
                            "type": "language",
                            "language": current_lang,
                            "position": len(" ".join(text_parts))
                        })
                elif elem.tag == "Sync":
                    # Handle sync elements - they can contain timing info
                    pass
                
                # Add element text if it exists
                if elem.text:
                    text_parts.append(elem.text.strip())
                if elem.tail:
                    text_parts.append(elem.tail.strip())
            
            # Clean and join text
            full_text = " ".join(filter(None, text_parts))
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            
            if full_text or language_events:  # Include segments with text or language events
                segment = TRSSegment(
                    segment_id=segment_id,
                    start_time=start_time,
                    end_time=end_time,
                    speaker=speaker_name,
                    text=full_text,
                    language_events=language_events
                )
                segments.append(segment)
        
        print(f"Found {len(segments)} TRS segments")
        return segments

    def parse_ctm_file(self, ctm_file: Path) -> List[CTMWord]:
        """Parse CTM file and extract word timings"""
        print(f"Parsing CTM file: {ctm_file}")
        
        words = []
        with open(ctm_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    print(f"Warning: Malformed CTM line {line_num}: {line}")
                    continue
                
                try:
                    file_id = parts[0]
                    channel = parts[1]
                    start_time = float(parts[2])
                    duration = float(parts[3])
                    word = parts[4]
                    confidence = float(parts[5]) if len(parts) > 5 else None
                    
                    ctm_word = CTMWord(
                        file_id=file_id,
                        channel=channel,
                        start_time=start_time,
                        duration=duration,
                        word=word,
                        confidence=confidence
                    )
                    words.append(ctm_word)
                
                except ValueError as e:
                    print(f"Warning: Error parsing CTM line {line_num}: {e}")
                    continue
        
        print(f"Found {len(words)} CTM words")
        return words

    def align_words_to_segments(self, ctm_words: List[CTMWord], 
                               trs_segments: List[TRSSegment]) -> Tuple[Dict[int, List[CTMWord]], Dict[str, List[CTMWord]]]:
        """Align CTM words to TRS segments and identify orphans"""
        
        # Sort segments and words by time for efficient processing
        sorted_segments = sorted(trs_segments, key=lambda s: s.start_time)
        sorted_words = sorted(ctm_words, key=lambda w: w.start_time)
        
        aligned_words = defaultdict(list)
        orphan_words = {
            "before_first_segment": [],
            "between_segments": [],
            "after_last_segment": []
        }
        
        word_idx = 0
        segment_idx = 0
        
        if not sorted_segments:
            orphan_words["before_first_segment"] = sorted_words
            return aligned_words, orphan_words
        
        first_segment_start = sorted_segments[0].start_time
        last_segment_end = sorted_segments[-1].end_time
        
        # Process words before first segment
        while word_idx < len(sorted_words) and sorted_words[word_idx].end_time < first_segment_start:
            orphan_words["before_first_segment"].append(sorted_words[word_idx])
            word_idx += 1
        
        # Process words that align with segments
        for segment in sorted_segments:
            segment_words = []
            
            # Find words that overlap with this segment
            temp_idx = word_idx
            while temp_idx < len(sorted_words):
                word = sorted_words[temp_idx]
                
                if word.start_time > segment.end_time + self.time_tolerance:
                    break
                
                if word.overlaps_with(segment.start_time, segment.end_time, self.time_tolerance):
                    segment_words.append(word)
                
                temp_idx += 1
            
            aligned_words[segment.segment_id] = segment_words
            
            # Move word_idx past processed words
            while word_idx < len(sorted_words) and sorted_words[word_idx].end_time <= segment.end_time + self.time_tolerance:
                word_idx += 1
        
        # Process words between segments and after last segment
        current_segment_idx = 0
        while word_idx < len(sorted_words):
            word = sorted_words[word_idx]
            
            # Check if word is between segments
            between_segments = False
            for i in range(len(sorted_segments) - 1):
                seg1_end = sorted_segments[i].end_time
                seg2_start = sorted_segments[i + 1].start_time
                
                if seg1_end <= word.start_time <= seg2_start:
                    # Find or create between_segments entry
                    between_entry = None
                    for entry in orphan_words["between_segments"]:
                        if (entry["after_segment"] == sorted_segments[i].segment_id and 
                            entry["before_segment"] == sorted_segments[i + 1].segment_id):
                            between_entry = entry
                            break
                    
                    if not between_entry:
                        between_entry = {
                            "after_segment": sorted_segments[i].segment_id,
                            "before_segment": sorted_segments[i + 1].segment_id,
                            "gap_start": seg1_end,
                            "gap_end": seg2_start,
                            "words": []
                        }
                        orphan_words["between_segments"].append(between_entry)
                    
                    between_entry["words"].append(word)
                    between_segments = True
                    break
            
            if not between_segments:
                # Word is after last segment
                if word.start_time > last_segment_end:
                    orphan_words["after_last_segment"].append(word)
            
            word_idx += 1
        
        return aligned_words, orphan_words

    def create_comparison_data(self, trs_file: Path, ctm0_file: Path, ctm1_file: Path, ctm2_file: Path) -> Dict[str, Any]:
        """Create comprehensive comparison data structure with 3 CTM files
        
        Args:
            trs_file: TRS reference file
            ctm0_file: Alternative ASR CTM file
            ctm1_file: Main ASR CTM file 
            ctm2_file: Agent CTM file
        """
        
        # Parse files
        self.trs_segments = self.parse_trs_file(trs_file)
        self.ctm0_words = self.parse_ctm_file(ctm0_file)
        self.ctm1_words = self.parse_ctm_file(ctm1_file)
        self.ctm2_words = self.parse_ctm_file(ctm2_file)
        
        # Align words to segments
        ctm0_aligned, ctm0_orphans = self.align_words_to_segments(self.ctm0_words, self.trs_segments)
        ctm1_aligned, ctm1_orphans = self.align_words_to_segments(self.ctm1_words, self.trs_segments)
        ctm2_aligned, ctm2_orphans = self.align_words_to_segments(self.ctm2_words, self.trs_segments)
        
        # Build comparison data
        comparison_data = {
            "metadata": {
                "trs_file": str(trs_file),
                "ctm0_file": str(ctm0_file),
                "ctm1_file": str(ctm1_file),
                "ctm2_file": str(ctm2_file),
                "total_segments": len(self.trs_segments),
                "ctm0_total_words": len(self.ctm0_words),
                "ctm1_total_words": len(self.ctm1_words),
                "ctm2_total_words": len(self.ctm2_words),
                "time_tolerance": self.time_tolerance,
                "generated_by": "trs_ctm_compare.py"
            },
            "segments": [],
            "orphan_words": {
                "ctm0": self._format_orphan_words(ctm0_orphans),
                "ctm1": self._format_orphan_words(ctm1_orphans),
                "ctm2": self._format_orphan_words(ctm2_orphans)
            }
        }
        
        # Process each segment
        for segment in self.trs_segments:
            ctm0_segment_words = ctm0_aligned.get(segment.segment_id, [])
            ctm1_segment_words = ctm1_aligned.get(segment.segment_id, [])
            ctm2_segment_words = ctm2_aligned.get(segment.segment_id, [])
            
            segment_data = {
                "segment_id": segment.segment_id,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "duration": segment.end_time - segment.start_time,
                "speaker": segment.speaker,
                "trs_text": segment.text,
                "language_events": segment.language_events,
                "ctm0_words": [w.word for w in ctm0_segment_words],
                "ctm0_text": " ".join(w.word for w in ctm0_segment_words),
                "ctm0_timing": [{"word": w.word, "start": w.start_time, "duration": w.duration} for w in ctm0_segment_words],
                "ctm1_words": [w.word for w in ctm1_segment_words],
                "ctm1_text": " ".join(w.word for w in ctm1_segment_words),
                "ctm1_timing": [{"word": w.word, "start": w.start_time, "duration": w.duration} for w in ctm1_segment_words],
                "ctm2_words": [w.word for w in ctm2_segment_words],
                "ctm2_text": " ".join(w.word for w in ctm2_segment_words),
                "ctm2_timing": [{"word": w.word, "start": w.start_time, "duration": w.duration} for w in ctm2_segment_words],
                "word_counts": {
                    "trs": len(segment.text.split()),
                    "ctm0": len(ctm0_segment_words),
                    "ctm1": len(ctm1_segment_words),
                    "ctm2": len(ctm2_segment_words)
                }
            }
            
            comparison_data["segments"].append(segment_data)
        
        return comparison_data

    def _format_orphan_words(self, orphan_words: Dict[str, Any]) -> Dict[str, Any]:
        """Format orphan words for JSON output"""
        formatted = {}
        
        # Before first segment
        formatted["before_first_segment"] = [
            {"word": w.word, "start": w.start_time, "duration": w.duration, "end": w.end_time}
            for w in orphan_words["before_first_segment"]
        ]
        
        # Between segments
        formatted["between_segments"] = []
        for entry in orphan_words["between_segments"]:
            formatted_entry = {
                "after_segment": entry["after_segment"],
                "before_segment": entry["before_segment"],
                "gap_start": entry["gap_start"],
                "gap_end": entry["gap_end"],
                "words": [
                    {"word": w.word, "start": w.start_time, "duration": w.duration, "end": w.end_time}
                    for w in entry["words"]
                ]
            }
            formatted["between_segments"].append(formatted_entry)
        
        # After last segment
        formatted["after_last_segment"] = [
            {"word": w.word, "start": w.start_time, "duration": w.duration, "end": w.end_time}
            for w in orphan_words["after_last_segment"]
        ]
        
        return formatted

    def save_comparison_json(self, comparison_data: Dict[str, Any], output_file: Path) -> None:
        """Save comparison data to JSON file"""
        print(f"Saving comparison data to: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        
        print(f"Comparison complete! Saved {len(comparison_data['segments'])} segments")
        print(f"CTM0 orphan words: {len(comparison_data['orphan_words']['ctm0']['before_first_segment']) + len(comparison_data['orphan_words']['ctm0']['after_last_segment']) + sum(len(entry['words']) for entry in comparison_data['orphan_words']['ctm0']['between_segments'])}")
        print(f"CTM1 orphan words: {len(comparison_data['orphan_words']['ctm1']['before_first_segment']) + len(comparison_data['orphan_words']['ctm1']['after_last_segment']) + sum(len(entry['words']) for entry in comparison_data['orphan_words']['ctm1']['between_segments'])}")
        print(f"CTM2 orphan words: {len(comparison_data['orphan_words']['ctm2']['before_first_segment']) + len(comparison_data['orphan_words']['ctm2']['after_last_segment']) + sum(len(entry['words']) for entry in comparison_data['orphan_words']['ctm2']['between_segments'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare TRS segments with three CTM files and export detailed analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trs_ctm_compare.py reference.trs alt_asr.ctm main_asr.ctm agent.ctm comparison.json
  python trs_ctm_compare.py --tolerance 0.2 ref.trs alt.ctm main.ctm agent.ctm output.json
        """
    )
    
    parser.add_argument("trs_file", help="Input TRS (Transcriber XML) file")
    parser.add_argument("ctm0_file", help="Alternative ASR CTM file")
    parser.add_argument("ctm1_file", help="Main ASR CTM file")
    parser.add_argument("ctm2_file", help="Agent CTM file")
    parser.add_argument("output_json", help="Output JSON file for comparison results")
    parser.add_argument("--tolerance", "-t", type=float, default=0.1,
                       help="Time tolerance for word-segment alignment (default: 0.1s)")
    
    args = parser.parse_args()
    
    # Validate input files
    trs_file = Path(args.trs_file)
    ctm0_file = Path(args.ctm0_file)
    ctm1_file = Path(args.ctm1_file)
    ctm2_file = Path(args.ctm2_file)
    output_file = Path(args.output_json)
    
    for file_path, name in [(trs_file, "TRS"), (ctm0_file, "CTM0"), (ctm1_file, "CTM1"), (ctm2_file, "CTM2")]:
        if not file_path.exists():
            print(f"Error: {name} file not found: {file_path}")
            return 1
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Run comparison
    try:
        comparator = TRSCTMComparator(time_tolerance=args.tolerance)
        comparison_data = comparator.create_comparison_data(trs_file, ctm0_file, ctm1_file, ctm2_file)
        comparator.save_comparison_json(comparison_data, output_file)
        
        print(f"\nComparison completed successfully!")
        print(f"Results saved to: {output_file}")
        print(f"\nNext step: Open trs_ctm_viewer.html in a web browser and load {output_file}")
        
        return 0
    
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())