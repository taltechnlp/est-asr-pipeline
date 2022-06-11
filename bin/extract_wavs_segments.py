#! /usr/bin/env python3

import sys
import argparse
import os.path
import os
import logging
from pydub import AudioSegment

if __name__ == '__main__':
  logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
  parser = argparse.ArgumentParser(description="Extract segments from wav files according to segments file and save the resulting wavs")
  parser.add_argument("wav_scp")
  parser.add_argument("segments")
  parser.add_argument("output_dir")
  
  args = parser.parse_args()
  
  wav_locations = {}
  for l in open(args.wav_scp):
    wav_id, loc = l.split()
    wav_locations[wav_id] = loc
    
  wav_segments = {}
  for l in open(args.segments):
    segment_id, wav_id, start, end = l.split()
    wav_segments.setdefault(wav_id, []).append((segment_id, float(start), float(end)))
    
  for wav_id in wav_segments.keys():
    wav = None
    for segment_id, start, end in wav_segments[wav_id]:
      filename = f"{args.output_dir}/{segment_id}.wav"
      if os.path.isfile(filename):
        logging.info(f"Skipping file {filename} -- file already exists")
        continue
      if wav is None:
        wav = AudioSegment.from_wav(wav_locations[wav_id])
      segment = wav[start * 1000 : end * 1000]      
      dirname = os.path.dirname(filename)
      os.makedirs(dirname, exist_ok=True)
      logging.info(f"Saving file {filename}")
      segment.export(filename, format="wav")
    
      
      
    
    
    
    
  
    
