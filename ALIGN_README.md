# Alignment-Only Pipeline

This is a standalone Nextflow script (`align.nf`) that performs forced alignment on Estonian JSON transcription data to produce CTM format output for WER calculation.

## Purpose

The script takes:
- A JSON file with segment-level transcription data (with timestamps and text)
- The corresponding audio file

And produces:
- CTM (Confusion Time Mark) format output suitable for WER evaluation

## Input JSON Format

The script supports two JSON input formats:

### Format 1: Direct segments
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Hello world"
    },
    {
      "start": 2.5,
      "end": 5.0,
      "text": "This is a test"
    }
  ]
}
```

### Format 2: Full pipeline output
```json
{
  "best_hypothesis": {
    "monologues": [
      {
        "turns": [
          {
            "words": [
              {
                "start": 0.0,
                "end": 1.0,
                "word_with_punctuation": "Hello"
              }
            ]
          }
        ]
      }
    ]
  }
}
```

## Usage

### With Docker (Recommended)
```bash
nextflow run align.nf -profile docker \
  --in_json /path/to/transcription.json \
  --in_audio /path/to/audio.wav \
  --out_dir results_align/
```

### Without Docker (requires local Kaldi installation)
```bash
nextflow run align.nf \
  --in_json /path/to/transcription.json \
  --in_audio /path/to/audio.wav \
  --out_dir results_align/
```

## Output

The script produces:
- `result.ctm` - CTM format file suitable for WER calculation tools

## Requirements

- Nextflow
- Docker (with NVIDIA Container Runtime) OR local Kaldi installation
- GPU support (same as main pipeline)
- Estonian acoustic models (included in Docker container)

## CTM Format

The output CTM format contains:
```
<utterance-id> <channel> <start-time> <duration> <word>
```

This format is compatible with standard WER evaluation tools like `sclite`.