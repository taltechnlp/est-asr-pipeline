#!/usr/bin/env python

import argparse
import re
import os

def load_segments(segments_file, utt2spk_file, text_file):
    utt2spk = {}
    with open(utt2spk_file, 'r') as f:
        for line in f:
            seg_id, spk_id = line.strip().split()
            utt2spk[seg_id] = spk_id

    text = {}
    if os.path.exists(text_file):
      with open(text_file, 'r') as f:
          for line in f:
              r = line.strip().split(maxsplit=1)
              if len(r) == 2:
                text[r[0]] = r[1]
              else:
                text[r[0]] = ""
    else:
      for key in utt2spk.keys():
        text[key] = ""
            
    segments = {}
    with open(segments_file, 'r') as f:
        for line in f:
            seg_id, rec_id, start, end = line.strip().split()
            if rec_id not in segments:
                segments[rec_id] = []
            segments[rec_id].append((float(start), float(end), seg_id, utt2spk[seg_id], text[seg_id]))
    return segments

def merge_segments(segments, max_duration):
    merged = {}
    for rec_id in segments:
        segments[rec_id].sort()
        current_segment = segments[rec_id][0]
        for seg in segments[rec_id][1:]:
            if seg[1] - current_segment[0] <= max_duration:
                spk = current_segment[3]
                if seg[3] not in current_segment[3]:
                    spk = current_segment[3] +  "+" + re.sub(".*###", "", seg[3])                    
                text = current_segment[4] + " " + seg[4]
                current_segment = (current_segment[0], seg[1], f"{spk}###{current_segment[0]:09.3f}-{seg[1]:09.3f}", spk, text)
            else:
                if rec_id not in merged:
                    merged[rec_id] = []
                merged[rec_id].append(current_segment)
                current_segment = seg
        if rec_id not in merged:
            merged[rec_id] = []
        merged[rec_id].append(current_segment)
    return merged

def write_segments(segments, segments_file, utt2spk_file, text_file):    
    utt2spk_f = open(utt2spk_file, 'w')
    text_f = open(text_file, 'w')
    segments_f = open(segments_file, 'w')
    for rec_id in segments:
        segments[rec_id].sort()
        for segment in segments[rec_id]:
            print(segment[2], segment[3], file=utt2spk_f)
            print(segment[2], segment[4], file=text_f)
            print(segment[2], rec_id, segment[0], segment[1], file=segments_f)

    utt2spk_f.close()
    text_f.close()
    segments_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge Kaldi segments.')
    parser.add_argument('data_dir', type=str, help='Data directory')
    parser.add_argument('out_data_dir', type=str, help='Data directory')
    parser.add_argument('--max_duration', type=float, default=30.0, help='Maximum duration of a segment')
    args = parser.parse_args()
    
    segments = load_segments(f"{args.data_dir}/segments", f"{args.data_dir}/utt2spk", f"{args.data_dir}/text")
    
    merged_segments = merge_segments(segments, args.max_duration)
    write_segments(merged_segments, f"{args.out_data_dir}/segments", f"{args.out_data_dir}/utt2spk", f"{args.out_data_dir}/text")
        
    
