#!/usr/bin/env python3

import argparse
import json
import logging
import sys
import numpy as np
import ctranslate2
from faster_whisper import WhisperModel
import soundfile as sf

def setup_logging(verbose=True):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )

def load_audio(audio_path, sampling_rate=16000):
    """Load audio file"""
    audio, sr = sf.read(audio_path)
    if sr != sampling_rate:
        # Simple resampling (for production, use proper resampling)
        import scipy.signal
        audio = scipy.signal.resample(audio, int(len(audio) * sampling_rate / sr))
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    return audio.astype(np.float32)

def load_segments_from_datadir(datadir_path):
    """Load segments from Kaldi datadir format"""
    segments = []
    segments_file = f"{datadir_path}/segments"
    utt2spk_file = f"{datadir_path}/utt2spk"
    
    # Load utt2spk mapping
    utt2spk = {}
    try:
        with open(utt2spk_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    utt2spk[parts[0]] = parts[1]
    except FileNotFoundError:
        logging.warning(f"utt2spk file not found: {utt2spk_file}")
    
    # Load segments
    try:
        with open(segments_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    seg_id, rec_id, start_time, end_time = parts[:4]
                    speaker = utt2spk.get(seg_id, 'unknown')
                    segments.append({
                        'id': seg_id,
                        'start': float(start_time),
                        'end': float(end_time),
                        'speaker': speaker
                    })
    except FileNotFoundError:
        logging.error(f"Segments file not found: {segments_file}")
        return []
    
    return sorted(segments, key=lambda x: x['start'])

def create_mel_features_for_segment(audio, start_time, end_time, feature_extractor):
    """Create mel spectrogram features for a specific audio segment"""
    sampling_rate = feature_extractor.sampling_rate
    start_sample = int(start_time * sampling_rate)
    end_sample = int(end_time * sampling_rate)
    
    # Extract segment audio
    segment_audio = audio[start_sample:end_sample]
    
    # Pad segment to minimum required length if too short
    min_length = sampling_rate * 0.1  # 0.1 seconds minimum
    if len(segment_audio) < min_length:
        segment_audio = np.pad(segment_audio, (0, int(min_length - len(segment_audio))))
    
    # For segments longer than 30 seconds, we need to handle them specially
    max_length = sampling_rate * 30  # 30 seconds max for Whisper
    if len(segment_audio) > max_length:
        # Split long segments into 30-second chunks
        chunks = []
        for i in range(0, len(segment_audio), max_length):
            chunk = segment_audio[i:i + max_length]
            if len(chunk) < max_length:
                chunk = np.pad(chunk, (0, max_length - len(chunk)))
            
            features = feature_extractor(chunk)
            target_frames = 3000
            if features.shape[-1] != target_frames:
                if features.shape[-1] < target_frames:
                    padding = target_frames - features.shape[-1]
                    features = np.pad(features, ((0, 0), (0, padding)), mode='constant', constant_values=-11.5129)
                else:
                    features = features[:, :target_frames]
            chunks.append(features)
        return chunks
    else:
        # Pad to 30 seconds for Whisper
        if len(segment_audio) < max_length:
            segment_audio = np.pad(segment_audio, (0, max_length - len(segment_audio)))
        
        features = feature_extractor(segment_audio)
        target_frames = 3000
        if features.shape[-1] != target_frames:
            if features.shape[-1] < target_frames:
                padding = target_frames - features.shape[-1]
                features = np.pad(features, ((0, 0), (0, padding)), mode='constant', constant_values=-11.5129)
            else:
                features = features[:, :target_frames]
        
        return [features]

def get_segment_based_nbest(model_dir, audio_path, segments_datadir, beam_size=5, num_hypotheses=5, 
                           language="et", device="auto"):
    """Generate n-best hypotheses for each speaker segment"""
    
    logging.info(f"Loading model components from {model_dir}")
    
    # Determine device
    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    
    # Load faster-whisper model to get tokenizer and feature extractor
    faster_model = WhisperModel(model_dir, device=device, compute_type="float16" if device == "cuda" else "float32")
    
    # Extract components
    ct2_model = faster_model.model
    tokenizer = faster_model.hf_tokenizer
    feature_extractor = faster_model.feature_extractor
    
    # Ensure tokenizer handles UTF-8 properly
    if hasattr(tokenizer, 'errors'):
        tokenizer.errors = 'ignore'
    
    logging.info(f"Using device: {device}")
    logging.info(f"Model loaded. Beam size: {beam_size}, Num hypotheses: {num_hypotheses}")
    
    # Load segments from datadir
    segments = load_segments_from_datadir(segments_datadir)
    if not segments:
        logging.error("No segments found in datadir")
        return {"text": "", "language": language, "segments": []}
    
    logging.info(f"Found {len(segments)} segments to process")
    
    # Load and process audio
    audio = load_audio(audio_path, feature_extractor.sampling_rate)
    logging.info(f"Audio loaded: {len(audio) / feature_extractor.sampling_rate:.1f} seconds")
    
    all_segments = []
    
    # Process each segment individually
    for seg_idx, segment in enumerate(segments):
        logging.info(f"Processing segment {seg_idx + 1}/{len(segments)}: {segment['id']} ({segment['start']:.3f}-{segment['end']:.3f}s, speaker: {segment['speaker']})")
        
        # Create mel features for this segment
        mel_features_list = create_mel_features_for_segment(
            audio, segment['start'], segment['end'], feature_extractor
        )
        
        # Combine alternatives from all chunks of this segment
        segment_alternatives = []
        chunk_texts = {i: [] for i in range(num_hypotheses)}
        chunk_scores = {i: [] for i in range(num_hypotheses)}
        
        # Process each chunk of the segment
        for chunk_idx, mel_features in enumerate(mel_features_list):
            logging.debug(f"Processing chunk {chunk_idx + 1}/{len(mel_features_list)} for segment {segment['id']}")
            
            # Create SOT sequence
            sot_sequence = [
                50258,  # <|startoftranscript|>
                50302,  # <|et|> for Estonian  
                50359,  # <|transcribe|>
                50363   # <|notimestamps|>
            ]
            
            # Convert to CTranslate2 format
            features_tensor = np.expand_dims(mel_features, axis=0)
            features_tensor = np.ascontiguousarray(features_tensor)
            features_storage = ctranslate2.StorageView.from_array(features_tensor)
            
            try:
                # Pure CTranslate2 beam search
                results = ct2_model.generate(
                    features=features_storage,
                    prompts=[sot_sequence],
                    beam_size=beam_size,
                    num_hypotheses=num_hypotheses,
                    length_penalty=1.0,
                    return_scores=True,
                    max_length=448
                )
                
                result = results[0]
                
                # Extract hypotheses from this chunk
                for hyp_idx in range(min(num_hypotheses, len(result.sequences))):
                    sequence = result.sequences[hyp_idx]
                    score = result.scores[hyp_idx] if result.scores else 0.0
                    
                    # Decode text
                    if isinstance(sequence[0], str):
                        text = "".join(sequence).replace("Ġ", " ").strip()
                        # Fix encoding
                        try:
                            fixed_text = text.encode('latin-1').decode('utf-8')
                            text = fixed_text
                        except (UnicodeEncodeError, UnicodeDecodeError):
                            try:
                                text = text.replace('Ã¤', 'ä').replace('Ã¶', 'ö').replace('Ã¼', 'ü').replace('Ãµ', 'õ').replace('Ã¾', 'ž').replace('Å¡', 'š').replace('Ãĸ', 'Ö').replace('Ãľ', 'Ü')
                            except:
                                pass
                    else:
                        token_ids = [int(token) for token in sequence]
                        text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                        # Fix encoding
                        try:
                            fixed_text = text.encode('latin-1').decode('utf-8')
                            text = fixed_text
                        except (UnicodeEncodeError, UnicodeDecodeError):
                            try:
                                text = text.replace('Ã¤', 'ä').replace('Ã¶', 'ö').replace('Ã¼', 'ü').replace('Ãµ', 'õ').replace('Ã¾', 'ž').replace('Å¡', 'š').replace('Ãĸ', 'Ö').replace('Ãľ', 'Ü')
                            except:
                                pass
                    
                    chunk_texts[hyp_idx].append(text)
                    chunk_scores[hyp_idx].append(float(score))
                
            except Exception as e:
                logging.error(f"CTranslate2 beam search failed for segment {segment['id']}, chunk {chunk_idx}: {e}")
                # Add empty results for this chunk
                for hyp_idx in range(num_hypotheses):
                    chunk_texts[hyp_idx].append("")
                    chunk_scores[hyp_idx].append(0.0)
        
        # Combine chunks into final alternatives for this segment
        for hyp_idx in range(num_hypotheses):
            combined_text = " ".join(chunk_texts[hyp_idx]).strip()
            avg_score = np.mean(chunk_scores[hyp_idx]) if chunk_scores[hyp_idx] else 0.0
            
            segment_alternatives.append({
                "rank": hyp_idx + 1,
                "text": combined_text,
                "avg_logprob": float(avg_score)
            })
        
        # Create segment data
        segment_data = {
            "start": segment['start'],
            "end": segment['end'],
            "speaker": segment['speaker'],
            "segment_id": segment['id'],
            "alternatives": segment_alternatives
        }
        all_segments.append(segment_data)
        
        logging.info(f"Generated {len(segment_alternatives)} alternatives for segment {segment['id']}")
    
    # Construct final result
    full_text = " ".join([seg["alternatives"][0]["text"] for seg in all_segments if seg["alternatives"]])
    
    return {
        "text": full_text,
        "language": language,
        "segments": all_segments
    }

def main():
    parser = argparse.ArgumentParser(description='Segment-aware CTranslate2 beam search for speaker turns')
    parser.add_argument('audio_file', help='Audio file to transcribe')
    parser.add_argument('--model_directory', required=True, help='Path to CTranslate2 Whisper model')
    parser.add_argument('--segments_datadir', required=True, help='Path to Kaldi datadir with segments')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for search')
    parser.add_argument('--num_hypotheses', type=int, default=5, help='Number of beam search hypotheses')
    parser.add_argument('--language', default='et', help='Language code')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto', help='Device to use')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    try:
        result = get_segment_based_nbest(
            model_dir=args.model_directory,
            audio_path=args.audio_file,
            segments_datadir=args.segments_datadir,
            beam_size=args.beam_size,
            num_hypotheses=args.num_hypotheses,
            language=args.language,
            device=args.device
        )
        
        # Output JSON result
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        logging.error(f"SEGMENT-AWARE BEAM SEARCH FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()