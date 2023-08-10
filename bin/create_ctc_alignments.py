#! /usr/bin/env python3

import string
import sys
import argparse
import ctc_segmentation
from transformers import  AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC, AutoProcessor
import kaldiio
import numpy as np
from typing import List
import re
import torch
import logging
import pandas.io.json
from dataclasses import dataclass


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]

# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments



def align_with_transcript(
    audio : np.ndarray,
    words : List,
    samplerate : int,
    model : AutoModelForCTC,
    processor : AutoFeatureExtractor,
    tokenizer : AutoTokenizer):

    assert audio.ndim == 1
    # Run prediction, get logits and probabilities
    inputs = processor(audio, return_tensors="pt", sampling_rate=samplerate).to(model.device)
    with torch.no_grad():
        logits = model(inputs.input_values).logits.cpu()[0]
        probs = torch.nn.functional.log_softmax(logits,dim=-1)
        
        #breakpoint()

    
    
    vocab = tokenizer.get_vocab()
    inv_vocab = {v:k for k,v in vocab.items()}
    unk_id = vocab["<unk>"]
    
    transcript = "|".join([word["word"] for word in words]) + "|"
    
    tokens = [vocab[c] for c in transcript]
        
    #print(list(zip(transcript, tokens)))      
    
    trellis = get_trellis(probs, tokens)
    path = backtrack(trellis, probs, tokens)
    segments = merge_repeats(path, transcript)
    for seg in segments:
        print(seg)    
    
    i = 0
    index_duration = 0.02
    
    
    char_seq = [inv_vocab[i] for i in probs.argmax(1).numpy()]
    hyp = "".join(char_seq).replace("<pad>", " ")
    
    #breakpoint()
    
    for word in words:
      word["start"] = segments[i].start * index_duration
      i += len(word["word"]) 
      word["end"] = segments[i-1].end * index_duration
      i += 1
      #print(word)
    
      


def __align_with_transcript(
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
        tokens.append(tok_ids[tok_ids != unk_id] + [vocab["|"]])
        #tokens.append(tok_ids[tok_ids != unk_id])
        
    
    # Align
    char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    #config.index_duration = audio.shape[0] / probs.size()[0] / samplerate
    config.index_duration = 0.02
    config.space = 4
    #config.blank_transition_cost_zero = True
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

  #processor = AutoFeatureExtractor.from_pretrained(args.model_name)
  processor = AutoProcessor.from_pretrained(args.model_name)
  #processor.tokenizer.set_target_lang("est")
  #tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  tokenizer = processor.tokenizer
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
  
