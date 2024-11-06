#! /usr/bin/env python3

import argparse
import pandas as pd

def read_rttm(file_path):
    columns = ["Type", "File ID", "Channel ID", "Turn Onset", "Turn Duration", "Orthography Field", "Speaker Type", "Speaker Name", "Confidence Score", "Signal Lookahead Time"]
    return pd.read_csv(file_path, sep=' ', names=columns, comment='#', na_values=['<NA>', 'nan'])

def write_rttm(df, file_path):
    with open(file_path, 'w') as f:
        for _, row in df.iterrows():
            row["Turn Onset"] = f"{row['Turn Onset']:.3f}"
            row["Turn Duration"] = f"{row['Turn Duration']:.3f}"
            row = row.fillna('<NA>')
            f.write(' '.join(row.astype(str).values) + '\n')

def merge_segments(df, max_gap):
    merged_segments = []
    prev_segment = None

    for _, row in df.iterrows():
        if prev_segment is None:
            prev_segment = row
        else:
            if (row["Speaker Name"] == prev_segment["Speaker Name"]) and ((row["Turn Onset"] - (prev_segment["Turn Onset"] + prev_segment["Turn Duration"])) <= max_gap):
                prev_segment["Turn Duration"] = row["Turn Onset"] + row["Turn Duration"] - prev_segment["Turn Onset"]
            else:
                merged_segments.append(prev_segment)
                prev_segment = row

    if prev_segment is not None:
        merged_segments.append(prev_segment)

    return pd.DataFrame(merged_segments)

def main(input_file, output_file, max_gap):
    rttm_df = read_rttm(input_file)
    merged_df = merge_segments(rttm_df, max_gap)
    write_rttm(merged_df, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge RTTM segments where the speaker is the same and the gap between segments is less than a specified threshold.")
    parser.add_argument("input_file", type=str, help="Input RTTM file")
    parser.add_argument("output_file", type=str, help="Output RTTM file")
    parser.add_argument("--max_gap", type=float, default=0.5, help="Maximum gap between segments to merge (default: 0.5 seconds)")

    args = parser.parse_args()
    main(args.input_file, args.output_file, args.max_gap)
