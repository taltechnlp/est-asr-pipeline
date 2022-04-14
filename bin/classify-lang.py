#! /usr/bin/env python
import logging
import sys
import argparse
import kaldiio
import torch
import pickle
from speechbrain.pretrained import EncoderClassifier

if __name__ == '__main__':
  logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
  parser = argparse.ArgumentParser(description="Apply LID to utterances")
  parser.add_argument("dir")

  args = parser.parse_args()

  lid_svc = pickle.load(open("local/lid_clf.pkl", "rb"))

  language_id = EncoderClassifier.from_hparams(source="TalTechNLP/voxlingua107-epaca-tdnn")

  with kaldiio.ReadHelper(f'scp:{args.dir}/wav.scp', segments=f'{args.dir}/segments') as reader:
    for key, (rate, numpy_array) in reader:
      torch_array = torch.from_numpy(numpy_array)
      
      prediction = language_id.classify_batch(torch_array)
      
      emb = language_id.encode_batch(torch_array)
      
      svc_result = lid_svc.predict([emb.squeeze().numpy()])
      print(key, svc_result, prediction[3])
      
      #breakpoint()
      #top5 = prediction[0][0].argsort(descending=True)[0:5]
      #lang_codes = [language_id.hparams.label_encoder.ind2lab[i.item()] for i in top5]
      #scores = prediction[0][0][top5]
      #print(key, " ".join([f"{l}:{s:0.2f}" for l, s in zip(lang_codes, scores)]))
