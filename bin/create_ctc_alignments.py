#! /usr/bin/env python3

import string
import sys
import argparse
import ctc_segmentation
from transformers import  AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
import kaldiio
import numpy as np
from typing import List
import re
import torch
import logging
import pandas.io.json

def align_with_transcript(
    audio : np.ndarray,
    words : List,
    samplerate : int,
    model : AutoModelForCTC,
    processor : AutoFeatureExtractor,
    tokenizer : AutoTokenizer
):
    assert audio.ndim == 1
    # Run prediction, get logits and probabilities
    inputs = processor(audio, return_tensors="pt", sampling_rate=samplerate).to(model.device)
    with torch.no_grad():
        logits = model(inputs.input_values).logits.cpu()[0]
        probs = torch.nn.functional.softmax(logits,dim=-1)
        #breakpoint()
    
    # Tokenize transcripts
    vocab = tokenizer.get_vocab()
    inv_vocab = {v:k for k,v in vocab.items()}
    unk_id = vocab["<unk>"]
    
    tokens = []
    for word in words:
        assert len(word["word"]) > 0
        tok_ids = tokenizer(word["word"])['input_ids']
        tok_ids = np.array(tok_ids, dtype=int)
        tokens.append(tok_ids[tok_ids != unk_id])
    
    # Align
    char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    #config.index_duration = audio.shape[0] / probs.size()[0] / samplerate
    config.index_duration = 0.02
    config.space = "|"
    config.score_min_mean_over_L = 100
    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_token_list(config, tokens)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(config, probs.numpy(), ground_truth_mat)
    #breakpoint()
    segments = ctc_segmentation.determine_utterance_segments(config, utt_begin_indices, char_probs, timings, [word["word"] for word in words])
    for i, (t, p) in enumerate(zip([word["word"] for word in words], segments)):
        # return [{"text" : t, "start" : p[0], "end" : p[1], "conf" : p[2]} for t,p in zip(transcripts, segments)]
        assert t == words[i]["word"]
        words[i]["start"] = p[0]
        words[i]["end"] = p[1]
        words[i]["confidence"] = p[2]
    #breakpoint()
    
    

if __name__ == '__main__':
  logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

  parser = argparse.ArgumentParser(description='')  
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("model_name")
  parser.add_argument("datadir")
  parser.add_argument("output_json")
  
  args = parser.parse_args()

  processor = AutoFeatureExtractor.from_pretrained(args.model_name)
  tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  model = AutoModelForCTC.from_pretrained(args.model_name).to(args.device)
  
  texts = {}
  for l in open(f"{args.datadir}/text"):
      res = re.split("\s+", l.strip(), maxsplit=1)
      key = res[0]
      text = res[1] if len(res) > 1 else ""
      words = []
      for token in text.split():
        if token in string.punctuation:
          if len(words) > 0:
            words[-1]["punctuation"] = token
            words[-1]["word_with_punctuation"] = words[-1]["word"] + token
        else:
          words.append({"word": token, "word_with_punctuation": token, "punctuation": ""})
      texts[key] = words
  #breakpoint()
  
  segments = {}
  for l in open(f"{args.datadir}/segments"):
    seg_id, _, start, end = l.split()
    segments[seg_id] = (float(start), float(end))
  
  alignments = {}
  with kaldiio.ReadHelper(f'scp:{args.datadir}/wav.scp', segments=f'{args.datadir}/segments') as reader:
    for key, (rate, numpy_array) in reader:
      align_with_transcript(numpy_array.astype(np.double), texts[key], samplerate=rate, model=model, processor=processor, tokenizer=tokenizer)
      
  flattened_words = []
  for seg_id, (start, end) in sorted(segments.items(), key=lambda s: s[1][0], reverse=False):
    for word in texts[seg_id]:
      word["start"] += start
      word["end"] += start
      flattened_words.append(word)
        
    
  with open(args.output_json, 'w') as f:
    f.write(pandas.io.json.dumps(flattened_words, double_precision=3, indent=2))
  
