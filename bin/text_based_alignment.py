#!/usr/bin/env python3
"""
Text-based alignment using edit distance with intelligent timing adjustment.
Aligns reference text with ASR output and adjusts word timings based on D/I/S operations.
"""

import json
import argparse
import sys
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class EditOp(Enum):
    MATCH = "M"
    SUBSTITUTE = "S"
    INSERT = "I" 
    DELETE = "D"

@dataclass
class WordTiming:
    word: str
    start: float
    end: float
    confidence: Optional[float] = None

@dataclass  
class AlignmentOperation:
    op: EditOp
    asr_word: Optional[WordTiming]
    ref_word: Optional[str]
    adjusted_timing: Optional[Tuple[float, float]] = None

@dataclass
class SegmentAlignment:
    segment_id: str
    asr_start: float
    asr_end: float
    ref_start: float
    ref_end: float
    operations: List[AlignmentOperation]
    stats: Dict[str, int]

class TextAligner:
    def __init__(self, debug_file=None):
        self.debug_file = debug_file
        self.total_stats = {"deletions": 0, "insertions": 0, "substitutions": 0, "matches": 0}
        
    def log_debug(self, message: str):
        """Log debug messages to file if debug_file is provided"""
        if self.debug_file:
            print(message, file=self.debug_file)
    
    def normalize_word(self, word: str) -> str:
        """Normalize word for comparison (remove punctuation, lowercase)"""
        # Remove punctuation and convert to lowercase
        normalized = re.sub(r'[^\w]', '', word.lower())
        return normalized
    
    def compute_edit_distance(self, asr_words: List[str], ref_words: List[str]) -> Tuple[int, List[Tuple[EditOp, int, int]]]:
        """
        Compute edit distance and return alignment path.
        Returns (distance, path) where path is list of (operation, asr_idx, ref_idx)
        """
        m, n = len(asr_words), len(ref_words)
        
        # DP table: dp[i][j] = edit distance between asr_words[:i] and ref_words[:j]
        dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
        
        # Backtrack table to reconstruct alignment
        backtrack = [[None] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
            if i > 0:
                backtrack[i][0] = (EditOp.DELETE, i-1, -1)
                
        for j in range(n + 1):
            dp[0][j] = j  
            if j > 0:
                backtrack[0][j] = (EditOp.INSERT, -1, j-1)
        
        backtrack[0][0] = None
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                asr_norm = self.normalize_word(asr_words[i-1])
                ref_norm = self.normalize_word(ref_words[j-1])
                
                # Match/substitute
                if asr_norm == ref_norm:
                    cost = dp[i-1][j-1]  # Match
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        backtrack[i][j] = (EditOp.MATCH, i-1, j-1)
                else:
                    cost = dp[i-1][j-1] + 1  # Substitute
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        backtrack[i][j] = (EditOp.SUBSTITUTE, i-1, j-1)
                
                # Delete from ASR
                cost = dp[i-1][j] + 1
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    backtrack[i][j] = (EditOp.DELETE, i-1, j)
                    
                # Insert into ASR  
                cost = dp[i][j-1] + 1
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    backtrack[i][j] = (EditOp.INSERT, i, j-1)
        
        # Reconstruct path
        path = []
        i, j = m, n
        while backtrack[i][j] is not None:
            op, prev_i, prev_j = backtrack[i][j]
            if op == EditOp.DELETE:
                path.append((EditOp.DELETE, prev_i, -1))
                i = prev_i
            elif op == EditOp.INSERT:
                path.append((EditOp.INSERT, -1, prev_j))  
                j = prev_j
            else:  # MATCH or SUBSTITUTE
                path.append((op, prev_i, prev_j))
                i, j = prev_i, prev_j
        
        path.reverse()
        return dp[m][n], path
    
    def adjust_timing_for_insertions(self, operations: List[AlignmentOperation], asr_words: List[WordTiming]) -> None:
        """
        Adjust timings for insertion operations by finding gaps or reducing adjacent word durations
        """
        for i, op in enumerate(operations):
            if op.op != EditOp.INSERT:
                continue
                
            # Find previous and next non-insertion operations for timing context
            prev_end = None
            next_start = None
            
            # Look backward for last timed word
            for j in range(i-1, -1, -1):
                if operations[j].adjusted_timing:
                    prev_end = operations[j].adjusted_timing[1]
                    break
                    
            # Look forward for next timed word  
            for j in range(i+1, len(operations)):
                if operations[j].adjusted_timing:
                    next_start = operations[j].adjusted_timing[0]
                    break
            
            # Count consecutive insertions to divide space proportionally
            insertion_group = [op]
            # Look backward for consecutive insertions
            for j in range(i-1, -1, -1):
                if operations[j].op == EditOp.INSERT and not operations[j].adjusted_timing:
                    insertion_group.insert(0, operations[j])
                else:
                    break
            # Look forward for consecutive insertions  
            for j in range(i+1, len(operations)):
                if operations[j].op == EditOp.INSERT and not operations[j].adjusted_timing:
                    insertion_group.append(operations[j])
                else:
                    break
            
            # Calculate available time slot
            if prev_end is not None and next_start is not None:
                available_time = next_start - prev_end
                if available_time > 0:
                    # Divide available time proportionally among insertions
                    time_per_insertion = available_time / len(insertion_group)
                    for k, ins_op in enumerate(insertion_group):
                        start_time = prev_end + k * time_per_insertion
                        end_time = prev_end + (k + 1) * time_per_insertion
                        ins_op.adjusted_timing = (start_time, end_time)
                else:
                    # No gap available, reduce adjacent words by 33% each
                    self.reduce_adjacent_words_timing(operations, insertion_group, i)
            else:
                # Handle edge cases (beginning or end of segment)
                self.handle_edge_insertions(operations, insertion_group, asr_words, i)
    
    def reduce_adjacent_words_timing(self, operations: List[AlignmentOperation], insertion_group: List[AlignmentOperation], center_idx: int):
        """Reduce timing of adjacent words by 33% each to make space for insertions"""
        reduction_factor = 0.33
        
        # Find adjacent timed operations
        prev_op = None
        next_op = None
        
        for j in range(center_idx-1, -1, -1):
            if operations[j].adjusted_timing:
                prev_op = operations[j]
                break
                
        for j in range(center_idx+1, len(operations)):
            if operations[j].adjusted_timing:
                next_op = operations[j]
                break
        
        available_time = 0
        
        # Reduce previous word duration
        if prev_op and prev_op.adjusted_timing:
            duration = prev_op.adjusted_timing[1] - prev_op.adjusted_timing[0]
            reduction = duration * reduction_factor
            new_end = prev_op.adjusted_timing[1] - reduction
            prev_op.adjusted_timing = (prev_op.adjusted_timing[0], new_end)
            available_time += reduction
        
        # Reduce next word duration  
        if next_op and next_op.adjusted_timing:
            duration = next_op.adjusted_timing[1] - next_op.adjusted_timing[0]
            reduction = duration * reduction_factor  
            new_start = next_op.adjusted_timing[0] + reduction
            next_op.adjusted_timing = (new_start, next_op.adjusted_timing[1])
            available_time += reduction
        
        # Distribute available time among insertions
        if available_time > 0:
            time_per_insertion = available_time / len(insertion_group)
            start_time = prev_op.adjusted_timing[1] if prev_op else 0
            
            for k, ins_op in enumerate(insertion_group):
                ins_start = start_time + k * time_per_insertion  
                ins_end = start_time + (k + 1) * time_per_insertion
                ins_op.adjusted_timing = (ins_start, ins_end)
    
    def handle_edge_insertions(self, operations: List[AlignmentOperation], insertion_group: List[AlignmentOperation], asr_words: List[WordTiming], center_idx: int):
        """Handle insertions at the beginning or end of segments"""
        # Simple fallback: assign minimal duration (0.1s per word) 
        base_duration = 0.1
        
        if center_idx == 0:  # Beginning of segment
            # Place before first timed word
            next_start = None
            for op in operations:
                if op.adjusted_timing:
                    next_start = op.adjusted_timing[0]
                    break
            
            if next_start is not None:
                start_time = next_start - len(insertion_group) * base_duration
            else:
                start_time = 0
                
            for k, ins_op in enumerate(insertion_group):
                ins_start = start_time + k * base_duration
                ins_end = start_time + (k + 1) * base_duration
                ins_op.adjusted_timing = (ins_start, ins_end)
        else:
            # End of segment - place after last timed word
            prev_end = None
            for j in range(center_idx-1, -1, -1):
                if operations[j].adjusted_timing:
                    prev_end = operations[j].adjusted_timing[1]
                    break
            
            if prev_end is None:
                prev_end = 0
                
            for k, ins_op in enumerate(insertion_group):
                ins_start = prev_end + k * base_duration  
                ins_end = prev_end + (k + 1) * base_duration
                ins_op.adjusted_timing = (ins_start, ins_end)
    
    def align_segment(self, asr_words: List[WordTiming], ref_words: List[str], segment_id: str, 
                     asr_start: float, asr_end: float, ref_start: float, ref_end: float) -> SegmentAlignment:
        """Align a single segment and adjust timings"""
        
        self.log_debug(f"\\nAligning segment {segment_id}")
        self.log_debug(f"ASR words: {[w.word for w in asr_words]}")
        self.log_debug(f"Ref words: {ref_words}")
        
        asr_word_strings = [w.word for w in asr_words]
        edit_distance, alignment_path = self.compute_edit_distance(asr_word_strings, ref_words)
        
        self.log_debug(f"Edit distance: {edit_distance}")
        
        # Create alignment operations
        operations = []
        stats = {"deletions": 0, "insertions": 0, "substitutions": 0, "matches": 0}
        
        for op_type, asr_idx, ref_idx in alignment_path:
            asr_word = asr_words[asr_idx] if asr_idx >= 0 else None
            ref_word = ref_words[ref_idx] if ref_idx >= 0 else None
            
            # Initialize with original timing for matches/substitutions
            adjusted_timing = None
            if asr_word:
                adjusted_timing = (asr_word.start, asr_word.end)
            
            operation = AlignmentOperation(
                op=op_type,
                asr_word=asr_word,
                ref_word=ref_word,
                adjusted_timing=adjusted_timing
            )
            operations.append(operation)
            
            # Update statistics
            if op_type == EditOp.DELETE:
                stats["deletions"] += 1
            elif op_type == EditOp.INSERT:
                stats["insertions"] += 1
            elif op_type == EditOp.SUBSTITUTE:
                stats["substitutions"] += 1
            else:  # MATCH
                stats["matches"] += 1
        
        # Handle timing adjustments for insertions
        self.adjust_timing_for_insertions(operations, asr_words)
        
        # Handle 1-to-many and many-to-1 substitutions
        self.adjust_substitution_timings(operations)
        
        # Update global statistics
        for key in stats:
            self.total_stats[key] += stats[key]
        
        return SegmentAlignment(
            segment_id=segment_id,
            asr_start=asr_start,
            asr_end=asr_end, 
            ref_start=ref_start,
            ref_end=ref_end,
            operations=operations,
            stats=stats
        )
    
    def adjust_substitution_timings(self, operations: List[AlignmentOperation]):
        """Handle timing adjustments for substitution sequences"""
        i = 0
        while i < len(operations):
            if operations[i].op == EditOp.SUBSTITUTE:
                # Find consecutive substitutions
                subst_group = [operations[i]]
                j = i + 1
                while j < len(operations) and operations[j].op == EditOp.SUBSTITUTE:
                    subst_group.append(operations[j])
                    j += 1
                
                # For now, keep 1-1 substitution timings as-is
                # More complex many-to-many substitutions could be handled here
                # by proportionally dividing the total time span
                
                i = j
            else:
                i += 1
    
    def find_overlapping_segments(self, asr_segments: List[Dict], ref_segments: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """Find ASR and reference segments that overlap in time"""
        pairs = []
        
        for ref_seg in ref_segments:
            ref_start, ref_end = ref_seg["start"], ref_seg["end"]
            
            best_overlap = 0
            best_asr_seg = None
            
            # Find ASR segments with maximum overlap
            for asr_seg in asr_segments:
                # Extract time range from ASR segment structure
                if "turns" in asr_seg:
                    asr_start = min(turn["start"] for turn in asr_seg["turns"])
                    asr_end = max(turn["end"] for turn in asr_seg["turns"])
                else:
                    asr_start, asr_end = asr_seg["start"], asr_seg["end"]
                
                # Calculate overlap
                overlap_start = max(ref_start, asr_start)
                overlap_end = min(ref_end, asr_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_asr_seg = asr_seg
            
            if best_overlap > 0:
                pairs.append((best_asr_seg, ref_seg))
            else:
                self.log_debug(f"WARNING: No overlapping ASR segment found for reference segment {ref_start}-{ref_end}")
        
        return pairs
    
    def extract_asr_words(self, asr_turn: Dict) -> Tuple[List[WordTiming], float, float]:
        """Extract word timings from ASR turn (now receives turns directly)"""
        words = []
        
        # Since we now pass turns directly, extract words from the turn
        if "words" in asr_turn and asr_turn["words"]:
            for word_info in asr_turn["words"]:
                word_timing = WordTiming(
                    word=word_info.get("word_with_punctuation", word_info["word"]),
                    start=word_info["start"],
                    end=word_info["end"],
                    confidence=word_info.get("confidence")
                )
                words.append(word_timing)
        else:
            # Fallback: split turn time proportionally using transcript
            text = asr_turn.get("transcript", asr_turn.get("text", ""))
            text_words = text.split()
            segment_start = asr_turn["start"] 
            segment_end = asr_turn["end"]
            duration = segment_end - segment_start
            word_duration = duration / len(text_words) if text_words else 0
            
            for i, word in enumerate(text_words):
                start_time = segment_start + i * word_duration
                end_time = segment_start + (i + 1) * word_duration
                words.append(WordTiming(word, start_time, end_time))
        
        return words, asr_turn["start"], asr_turn["end"]


def main():
    parser = argparse.ArgumentParser(description="Text-based alignment with timing adjustment")
    parser.add_argument("--asr-json", required=True, help="ASR output JSON file")
    parser.add_argument("--reference-json", required=True, help="Reference segments JSON file")
    parser.add_argument("--output", required=True, help="Output alignment JSON file")
    parser.add_argument("--stats", required=True, help="Output statistics JSON file") 
    parser.add_argument("--debug", help="Debug log file")
    
    args = parser.parse_args()
    
    # Load input files
    with open(args.asr_json, 'r', encoding='utf-8') as f:
        asr_data = json.load(f)
    
    with open(args.reference_json, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)
    
    debug_file = None
    if args.debug:
        debug_file = open(args.debug, 'w', encoding='utf-8')
    
    try:
        aligner = TextAligner(debug_file)
        
        # Extract segments from both inputs
        # For ASR data, extract turns from sections as individual segments
        asr_segments = []
        for section in asr_data.get("best_hypothesis", {}).get("sections", []):
            if "turns" in section:
                for turn in section["turns"]:
                    if "words" in turn and turn["words"]:
                        asr_segments.append(turn)
        
        ref_segments = ref_data.get("segments", [])
        
        aligner.log_debug(f"Found {len(asr_segments)} ASR segments and {len(ref_segments)} reference segments")
        
        # Find overlapping segments
        segment_pairs = aligner.find_overlapping_segments(asr_segments, ref_segments)
        aligner.log_debug(f"Found {len(segment_pairs)} overlapping segment pairs")
        
        # Perform alignment for each segment pair
        alignments = []
        
        for i, (asr_seg, ref_seg) in enumerate(segment_pairs):
            # Extract words and timings
            asr_words, asr_start, asr_end = aligner.extract_asr_words(asr_seg)
            ref_words = ref_seg["text"].split()
            ref_start, ref_end = ref_seg["start"], ref_seg["end"]
            
            segment_id = f"seg_{i:04d}"
            
            # Perform alignment
            alignment = aligner.align_segment(
                asr_words, ref_words, segment_id,
                asr_start, asr_end, ref_start, ref_end
            )
            alignments.append(alignment)
        
        # Generate output
        output_data = {
            "alignments": [],
            "total_statistics": aligner.total_stats
        }
        
        for alignment in alignments:
            alignment_dict = {
                "segment_id": alignment.segment_id,
                "asr_timing": {"start": alignment.asr_start, "end": alignment.asr_end},
                "ref_timing": {"start": alignment.ref_start, "end": alignment.ref_end},
                "words": [],
                "statistics": alignment.stats
            }
            
            for op in alignment.operations:
                if op.ref_word:  # Only include words that appear in reference
                    word_dict = {
                        "word": op.ref_word,
                        "operation": op.op.value,
                        "timing": op.adjusted_timing
                    }
                    if op.asr_word:
                        word_dict["asr_word"] = op.asr_word.word
                        word_dict["confidence"] = op.asr_word.confidence
                    
                    alignment_dict["words"].append(word_dict)
            
            output_data["alignments"].append(alignment_dict)
        
        # Save outputs
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        with open(args.stats, 'w', encoding='utf-8') as f:
            json.dump(aligner.total_stats, f, indent=2)
        
        print(f"Alignment completed successfully!")
        print(f"Total statistics: {aligner.total_stats}")
        
    finally:
        if debug_file:
            debug_file.close()


if __name__ == "__main__":
    main()