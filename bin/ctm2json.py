#! /usr/bin/env python3
import json
import argparse
import string
import sys
import pandas.io.json as json


parser = argparse.ArgumentParser("Converts CTM file with the corresponding punctuated transcripts file to flat alignemnts JSON")
parser.add_argument('ctm')
parser.add_argument('segments_file')
parser.add_argument('text_file')

args = parser.parse_args()

text = {}
with open(args.text_file, 'r') as f:
  for line in f:
    r = line.strip().split()
    text[r[0]] = r[1:]

segments = {}      
with open(args.segments_file, 'r') as f:
    for line in f:
        seg_id, _, start, end = line.strip().split()
        segments[seg_id] = ((float(start), float(end)))


segment_ctm_words = {}

for l in open(args.ctm):
    seg_id, _, start, length, word = l.split()
    segment_ctm_words.setdefault(seg_id, []).append((word, float(start), float(start) + float(length)))


words = []
    
for seg_id, (segment_start, segment_end) in sorted(segments.items(), key=lambda k: k[1][0]):
    if seg_id in segment_ctm_words:
      i = 0
      segment_trans = text[seg_id] 
      #breakpoint()
      for word_info in segment_ctm_words[seg_id]:
        assert word_info[0] == segment_trans[i]
      
        new_word_info = {"word": word_info[0], "start": word_info[1] + segment_start, "end": word_info[2] + segment_start}
        
        if len(segment_trans) > i + 1 and segment_trans[i+1] in list(string.punctuation):
          new_word_info["punctuation"] = segment_trans[i+1]
          i += 1
        else:
          new_word_info["punctuation"] = ""
        new_word_info["word_with_punctuation"] = new_word_info["word"] + new_word_info["punctuation"]
        
        words.append(new_word_info)
        i += 1   
    else:
      print(f"Warning: no alignment information for segment {seg_id} -- skipping it", file=sys.stderr)
    
print(json.dumps(words, indent=2, double_precision=3))
    
