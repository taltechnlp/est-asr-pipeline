import logging
from torch.utils.data import Dataset
import json
from collections import Counter
import torchaudio
import torch
import random
from argparse import ArgumentParser
import math

def normalize(tensor):
    # Subtract the mean, and scale to the interval [-1,1]
    tensor -= tensor.mean()
    tensor /= torch.abs(tensor).max()
    return tensor

class WeaklySupervisedSpeakerDataset(Dataset):

    def __init__(self, datadir, labels=None, extract_segments=True, segment_length=5.0, segments_per_spk=0, min_speaker_occ=5, **kwargs):
        self.extract_segments = extract_segments
        self.segment_length = segment_length
        self.segments_per_spk = segments_per_spk
        logging.info(f"Loading dataset from {datadir}..., with segments_per_spk={segments_per_spk}")
        self.wav2file = {}

        for l in open(f"{datadir}/wav.scp"):
            id, wav = l.split()
            self.wav2file[id] = wav

        self.utt2spk = {}
        for l in open(f"{datadir}/utt2spk"):
            utt, spk = l.split()
            self.utt2spk[utt] = spk
        
        self.wav2spk2segments = {}
        for l in open(f"{datadir}/segments"):
            utt, wav_id, start, end = l.split()
            start = float(start)
            end = float(end)
            if self.extract_segments:
                length = end - start
                num_subsegments = max(int(length / segment_length), 1)
                for i in range(num_subsegments):                    
                    self.wav2spk2segments.setdefault(wav_id, {}).setdefault(self.utt2spk[utt], []).append((utt, start + i * segment_length, min(end, start + (i+1) * segment_length)))    
            else:
                self.wav2spk2segments.setdefault(wav_id, {}).setdefault(self.utt2spk[utt], []).append((utt, start, end))
        
        self.wavs = sorted(self.wav2spk2segments.keys())

        
        try:
            self.wav2names = json.load(open(f"{datadir}/wav2names.json"))
            counter = Counter()
            for wav, names in self.wav2names.items():
                counter.update(names)            

            logging.info(f"Top 5 names in {dir}: {counter.most_common(5)}")
            
            # Compute oracle coverage
            num_covered = 0
            total_occs = 0
            for name, count in counter.items():
                total_occs += count
                if name in self.label2id:
                    num_covered += count
            
            coverage = 1.0 * num_covered / total_occs
            print("Oracle name coverage: %f" % coverage)
            if labels is None:
                logging.info("Constructing speaker pool ...")
                freq_speakers = [name for name, count in counter.items() if count >= min_speaker_occ]
                logging.info(f"{len(freq_speakers)} speakers occured at least {min_speaker_occ} times")
                self.label2id = {"<unk>" : 0}
                self.labels = ["<unk>"]
                for name in freq_speakers:
                    self.label2id[name] = len(self.label2id)
                    self.labels.append(name)            
            else:
                self.labels = labels
                self.label2id = {}
                for i, label in enumerate(labels):
                    self.label2id[label] = i
            self.num_labels = len(self.label2id)    
        except:
            logging.warning("No wav2names.json file? Assuming it's unlabelled prediction data")
        


    def __len__(self):
        return len(self.wav2spk2segments)

    def __getitem__(self, idx):
        result = {}
        wav_id =self.wavs[idx]
        try:
          result["names"] = self.wav2names[wav_id]
          active_speaker_ids = []
          for name in self.wav2names[wav_id]:
              if name in self.label2id:
                  active_speaker_ids.append(self.label2id[name])
          result["active_speaker_ids"] = torch.tensor(active_speaker_ids).long()        
          result["active_speaker_names"] = [self.labels[i] for i in active_speaker_ids]          
          result["num_total_speakers"] = len(self.wav2names[wav_id])
        except:
          pass

        audio, sample_rate = torchaudio.load(self.wav2file[wav_id])
        assert sample_rate == 16000
        if audio.shape[0] != 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = audio[0]
        #audio = normalize(audio)

        chunk_length_in_samples = int(self.segment_length * sample_rate)
        segment_speakers = []
        segment_audios = []
        segment_lengths = []
        #breakpoint()
        spks = []
        for i, (spk, segments) in enumerate(self.wav2spk2segments[wav_id].items()):
            spks.append(spk)
            if self.segments_per_spk > 0:
                segments_iter = random.choices(segments, k=min(len(segments), self.segments_per_spk))
            else:
                segments_iter = segments
            for segment in segments_iter:
                segment_speakers.append(i)
                start = int(segment[1] * sample_rate)
                end = int(segment[2] * sample_rate)                 
                segment_audios.append(audio[start:end])
                segment_lengths.append(end - start)
        result["segments"] = torch.nn.utils.rnn.pad_sequence(segment_audios, batch_first=True)
        result["segment_lens"] = torch.tensor(segment_lengths)
        result["segment_speakers"] = torch.tensor(segment_speakers)
        result["num_spks"] = len(self.wav2spk2segments[wav_id])
        result["wav"] = wav_id
        result["audio_file"] = self.wav2file[wav_id]
        
        result["spks"] = spks   

        return result


    @staticmethod
    def add_data_specific_args(parent_parser, root_dir):  # pragma: no cover
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--min-speaker-occ', default=10, type=int)
        parser.add_argument('--segment-length', default=5.0, type=float),
        parser.add_argument('--segments-per-spk', default=3, type=int)        
        return parser
