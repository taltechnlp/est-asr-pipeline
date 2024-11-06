#!/usr/bin/env python

import argparse
import os
from pyannote.audio import Pipeline
import torch
import torchaudio
from pyannote.audio.pipelines.utils.hook import ProgressHook

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diarize long audio file")
    parser.add_argument("--diarization-model", type=str, default="pyannote/speaker-diarization-3.1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("audio", type=str)
    parser.add_argument("output_rttm")

    args = parser.parse_args()

    access_token = os.environ.get('HF_TOKEN')
    
    diarization_model = Pipeline.from_pretrained(args.diarization_model, use_auth_token=access_token)
    diarization_model = diarization_model.to(torch.device("cuda:0"))
    waveform, sample_rate = torchaudio.load(args.audio)
    with ProgressHook() as hook:
      diarization_raw = diarization_model({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)

    # dump the diarization output to disk using RTTM format
    with open(args.output_rttm, "w") as rttm:
        diarization_raw.write_rttm(rttm)
