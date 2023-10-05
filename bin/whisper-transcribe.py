#!/usr/bin/env python

import argparse
import logging
import sys
import numpy as np
import zlib
#import gc
import torch
from faster_whisper import WhisperModel
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.audio import decode_audio

def get_compression_ratio(text: str) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))

if __name__ == "__main__":
  logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
  parser = argparse.ArgumentParser(description='Transcribe speech using predefined segements using Whisper')
  parser.add_argument('--language', type=str, default="en", help='Language of the audio')
  parser.add_argument('--is_multilingual', action="store_true", default=False)
  parser.add_argument('--beam', default=5, type=int)
  parser.add_argument('--compression_ratio_threshold', default=2.4, type=float)
  parser.add_argument('model_path', type=str, help='Whisper model')  
  parser.add_argument('audio_file', type=str, help='Audio file')
  parser.add_argument('segments_file', type=str, help='Segments file, with lines: <segment_id> <dummy> <start> <end>')
  
  args = parser.parse_args()
  logging.info(f"Loading Whisper model from {args.model_path}")
  
  model = WhisperModel(args.model_path, device="cuda", compute_type="float32", cpu_threads=1, device_index=0)
  
  sampling_rate = model.feature_extractor.sampling_rate
  logging.info(f"Loading audio from {args.audio_file}")
  audio = decode_audio(args.audio_file, sampling_rate=sampling_rate)
  
  segments = []
  for l in open(args.segments_file):
    seg_id, _, start, end = l.split()
    start = float(start)
    end = float(end)
    segments.append((seg_id, start, end))
    
  logging.info(f"Loaded info about {len(segments)} segments")
  
  features = model.feature_extractor(audio)
  
  tokenizer = Tokenizer(model.hf_tokenizer, args.is_multilingual, task="transcribe", language=args.language)
  
  previous_tokens = []
  
  for segment_info in segments:
    start_frame = int(segment_info[1]/model.feature_extractor.time_per_frame)
    end_frame = int(segment_info[2]/model.feature_extractor.time_per_frame)
    segment = features[:, start_frame:end_frame]
    padded_segment = np.pad(segment, ((0, 0), (0, model.feature_extractor.nb_max_frames-segment.shape[1])))
    encoder_output = model.encode(padded_segment)
    prompt = model.get_prompt(tokenizer, previous_tokens, without_timestamps=True, prefix=None)
    for beam_delta in [0, 2, 4, 8]:
      res = model.model.generate(encoder_output, 
        [prompt], length_penalty=1, return_scores=True, return_no_speech_prob=True, beam_size=args.beam+beam_delta, patience=1,
        max_length=model.max_length,)[0]
      tokens = res.sequences_ids[0]
      text = tokenizer.decode(tokens).strip()
      compression_ratio = get_compression_ratio(text)
      if compression_ratio > args.compression_ratio_threshold:  
        # too repetitive
        logging.warning(f"Hypothesis very repetitive: [{text}], retrying?")
      else:
        break
        
    previous_tokens.extend(tokens)      
    print(segment_info[0], text)

  # avoids a hang at exit
  encoder_output = None
