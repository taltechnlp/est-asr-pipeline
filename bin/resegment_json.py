#! /usr/bin/env python3

import sys
import argparse
import simplejson as json
#import json
import re
import numpy as np
from collections import OrderedDict
from decimal import Decimal

def get_speaker_for_time(time, segments):
  distances = []
  for segment in segments:
    distances.append(min(abs(segment[0] - time),  abs(time - segment[1])))
    if segment[0] <= time and segment[1] >= time:      
      return segment[3]
  # Word is outside a segment. Let's just find the closest segment
  i = np.array(distances).argmin()
  return segments[i][3]

def get_turn(start, speaker_id, sections, speakers, new_turn_sil_length):
  if speaker_id not in speakers:
    speakers[speaker_id] = {}
    
  current_section = None
  for section in sections:
    if section["type"] == "speech":
      if section["start"] <= start and section["end"] > start:
        current_section = section
        break
  if current_section is None:
    print("Warning: No speech section for word starting at %.3f" % start, file=sys.stderr)
    return None
  turns = section.setdefault("turns", [])
  if len(turns) == 0 or turns[-1]["speaker"] != speaker_id or turns[-1]["end"] + new_turn_sil_length < start:
    turns.append({"speaker" : speaker_id, "start" : start, "end" : start, "transcript" : "", "words" : []})
    
  return turns[-1]

parser = argparse.ArgumentParser("Converts flat unsegmented word alignments to dedicated JSON format")
parser.add_argument('--new-turn-sil-length', default="2.0", type=float, help="Length of silence in seconds, from which a new turn is created")
parser.add_argument('--speech-padding', default="0.25", type=float, help="Speech segments are padded by this amount")
parser.add_argument('--speaker-names', help="File in JSON format that maps speaker IDs to speaker info (usually just name name)")
parser.add_argument('--pms-seg', help="The pms (speech/non-speech segmentation) file from diarization")
parser.add_argument('datadir')
parser.add_argument('alignments_json')

args = parser.parse_args()

result = OrderedDict()

sections = []

segments = []
utt2spk = {}

utt2spk = {}
with open(args.datadir + "/utt2spk", 'r') as f:
    for line in f:
        seg_id, spk_id = line.strip().split()
        utt2spk[seg_id] = spk_id


segments = []            
with open(args.datadir + "/segments", 'r') as f:
    for line in f:
        seg_id, _, start, end = line.strip().split()
        segments.append((float(start), float(end), seg_id, utt2spk[seg_id]))



if args.pms_seg:
  for l in open(args.pms_seg):
    ss = l.split()
    start = int(ss[2]) / 100.0
    end = start + int(ss[3]) /100.0
    if ss[7] == 'speech' or ss[7] == '1':
      kind = 'speech'
      if (start > 0.0):
        start -=  args.speech_padding
      end +=  args.speech_padding
    else:
      kind = 'non-speech'
      if (start > 0.0):
        start +=  args.speech_padding
      end -=  args.speech_padding
    
    if start < end:
      sections.append({"type" : kind, "start" : start, "end" : end})
else:
  sections.append({"type" : "speech", "start" : 0.0, "end" : 99999.})
  
sections = sorted(sections, key=lambda s: s["start"])
 
speakers = {}

words = json.load(open(args.alignments_json))

uppercase_next = True
last_speaker_id = None

for word in words:  
  speaker_id = get_speaker_for_time((word["end"] + word["start"]) / 2, segments)
  if speaker_id != last_speaker_id:
    uppercase_next = True
  turn = get_turn(word["start"], speaker_id, sections, speakers, args.new_turn_sil_length)
  if turn is None:
    continue
  #print(turn)
  if uppercase_next:
    if word["word_with_punctuation"] == word["word_with_punctuation"].lower():
      word["word_with_punctuation"] = word["word_with_punctuation"].capitalize()
  turn["words"].append(word)  
  turn["end"] = word["end"]
  if word["punctuation"] in ". ! ?".split():
    uppercase_next = True
  else:
    uppercase_next = False
  last_speaker_id = speaker_id
  

for section in sections:
  if "turns" in section:
    for (i, turn) in enumerate(section["turns"]):
      turn["transcript"] = " ".join([w["word_with_punctuation"] for w in turn["words"]])
      # extend turn end time to the next turn, to avoid gaps between turns
      if i < len(section["turns"]) - 1:
        turn["end"] = section["turns"][i+1]["start"]
      

if not args.pms_seg:
  if "turns" in sections[0]:
    sections[0]["end"] = sections[0]["turns"][-1]["end"]

if args.speaker_names:
  speakers.update(json.load(open(args.speaker_names)))
  

result['speakers'] = speakers
result['sections'] = sections

print(json.dumps(result, sort_keys=False, indent=4))
