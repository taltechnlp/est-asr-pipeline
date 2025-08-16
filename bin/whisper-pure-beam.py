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

def create_mel_features(audio, feature_extractor):
    """Create mel spectrogram features with exactly 3000 frames"""
    # CTranslate2 expects exactly 3000 frames (30 seconds * 100 frames/sec)
    target_length = feature_extractor.sampling_rate * 30  # 30 seconds of audio
    target_frames = 3000  # Expected mel frames
    
    if len(audio) > target_length:
        # Split into 30-second chunks
        chunks = []
        for i in range(0, len(audio), target_length):
            chunk = audio[i:i + target_length]
            if len(chunk) < target_length:
                # Pad the last chunk
                chunk = np.pad(chunk, (0, target_length - len(chunk)))
            
            # Extract features and ensure exactly 3000 frames
            features = feature_extractor(chunk)
            if features.shape[-1] != target_frames:
                if features.shape[-1] < target_frames:
                    # Pad to target frames
                    padding = target_frames - features.shape[-1]
                    features = np.pad(features, ((0, 0), (0, padding)), mode='constant', constant_values=-11.5129)
                else:
                    # Truncate to target frames  
                    features = features[:, :target_frames]
            
            chunks.append(features)
        return chunks
    else:
        # Pad if shorter than 30 seconds
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        
        # Extract features and ensure exactly 3000 frames
        features = feature_extractor(audio)
        if features.shape[-1] != target_frames:
            if features.shape[-1] < target_frames:
                # Pad to target frames
                padding = target_frames - features.shape[-1]
                features = np.pad(features, ((0, 0), (0, padding)), mode='constant', constant_values=-11.5129)
            else:
                # Truncate to target frames
                features = features[:, :target_frames]
        
        return [features]

def get_pure_beam_search_nbest(model_dir, audio_path, beam_size=5, num_hypotheses=5, 
                              language="et", device="auto"):
    """Pure CTranslate2 beam search with no fallbacks"""
    
    logging.info(f"Loading model components from {model_dir}")
    
    # Determine device
    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    
    # Load faster-whisper model to get tokenizer and feature extractor
    faster_model = WhisperModel(model_dir, device=device, compute_type="float16" if device == "cuda" else "float32")
    
    # Extract components
    ct2_model = faster_model.model  # Get underlying CTranslate2 model
    tokenizer = faster_model.hf_tokenizer  # Get tokenizer
    feature_extractor = faster_model.feature_extractor  # Get feature extractor
    
    logging.info(f"Using device: {device}")
    logging.info(f"Model loaded. Beam size: {beam_size}, Num hypotheses: {num_hypotheses}")
    
    # Load and process audio
    audio = load_audio(audio_path, feature_extractor.sampling_rate)
    logging.info(f"Audio loaded: {len(audio) / feature_extractor.sampling_rate:.1f} seconds")
    
    # Create mel features
    mel_features_list = create_mel_features(audio, feature_extractor)
    logging.info(f"Created {len(mel_features_list)} feature chunks")
    
    all_segments = []
    
    # Process each 30-second chunk
    for chunk_idx, mel_features in enumerate(mel_features_list):
        logging.info(f"Processing chunk {chunk_idx + 1}/{len(mel_features_list)}")
        
        # Create proper SOT (Start of Transcript) prompt tokens
        # Use token IDs directly for CTranslate2
        sot_sequence = [
            50258,  # <|startoftranscript|>
            50302,  # <|et|> for Estonian  
            50359,  # <|transcribe|>
            50363   # <|notimestamps|>
        ]
        
        # Convert to CTranslate2 StorageView format
        features_tensor = np.expand_dims(mel_features, axis=0)  # Add batch dimension
        features_tensor = np.ascontiguousarray(features_tensor)  # Ensure contiguous memory
        features_storage = ctranslate2.StorageView.from_array(features_tensor)
        
        try:
            # PURE CTranslate2 BEAM SEARCH - NO FALLBACKS
            results = ct2_model.generate(
                features=features_storage,
                prompts=[sot_sequence],
                beam_size=beam_size,
                num_hypotheses=num_hypotheses,
                length_penalty=1.0,
                return_scores=True,
                max_length=448  # Max tokens for Whisper
            )
            
            # Extract all beam search hypotheses
            alternatives = []
            result = results[0]  # First (and only) batch item
            
            for hyp_idx in range(min(num_hypotheses, len(result.sequences))):
                sequence = result.sequences[hyp_idx]
                score = result.scores[hyp_idx] if result.scores else 0.0
                
                # Handle different sequence formats
                if isinstance(sequence[0], str):
                    # Already decoded tokens - join them and fix encoding
                    text = "".join(sequence).replace("Ġ", " ").strip()
                    # Fix UTF-8 encoding issues
                    text = text.encode('latin1').decode('utf-8', errors='ignore')
                else:
                    # Token IDs - decode normally
                    token_ids = [int(token) for token in sequence]
                    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                
                alternatives.append({
                    "rank": hyp_idx + 1,
                    "text": text,
                    "avg_logprob": float(score)
                })
            
            # Create segment data
            start_time = chunk_idx * 30.0
            end_time = min((chunk_idx + 1) * 30.0, len(audio) / feature_extractor.sampling_rate)
            
            segment_data = {
                "start": start_time,
                "end": end_time,
                "alternatives": alternatives
            }
            all_segments.append(segment_data)
            
            logging.info(f"Generated {len(alternatives)} beam search alternatives for chunk {chunk_idx}")
            
        except Exception as e:
            # NO FALLBACKS - FAIL COMPLETELY
            logging.error(f"CTranslate2 beam search failed for chunk {chunk_idx}: {e}")
            raise RuntimeError(f"Pure beam search failed: {e}") from e
    
    # Construct final result
    full_text = " ".join([seg["alternatives"][0]["text"] for seg in all_segments if seg["alternatives"]])
    
    return {
        "text": full_text,
        "language": language,
        "segments": all_segments
    }

def main():
    parser = argparse.ArgumentParser(description='Pure CTranslate2 beam search - no fallbacks')
    parser.add_argument('audio_file', help='Audio file to transcribe')
    parser.add_argument('--model_directory', required=True, help='Path to CTranslate2 Whisper model')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for search')
    parser.add_argument('--num_hypotheses', type=int, default=5, help='Number of beam search hypotheses')
    parser.add_argument('--language', default='et', help='Language code')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto', help='Device to use')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    try:
        result = get_pure_beam_search_nbest(
            model_dir=args.model_directory,
            audio_path=args.audio_file,
            beam_size=args.beam_size,
            num_hypotheses=args.num_hypotheses,
            language=args.language,
            device=args.device
        )
        
        # Output JSON result
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        logging.error(f"PURE BEAM SEARCH FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()