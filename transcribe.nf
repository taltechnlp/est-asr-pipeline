#!/usr/bin/env nextflow
nextflow.enable.dsl=2
def env = System.getenv()
params.in_file_list = ""
params.in = "/home/aivo_olevi/tmp/ao_1.wav"
params.out_dir = "results/"
params.do_speaker_id = true
params.do_punctuation = true
params.do_language_id = true
//audio_file = file(params.in)


process to_wav {
    memory '500MB'
    cpus 2

    input:
    path audio_file

    output:
    path 'audio.wav', glob:false, emit: audio
    val "${audio_file.baseName}", emit: basename
    
    shell:
        """
        ffmpeg -i !{audio_file} -f sox - | sox  -t sox - -c 1 -b 16 -t wav audio.wav rate -v 16k
        """

}

process diarization {
    memory '5GB'
    label 'with_gpu'
    
    input:
    path audio

    output: 
    path 'show.rttm', emit: show_rttm    

    script:        
        """
        diarize.py $audio show.0.rttm
        postprocess_rttm.py show.0.rttm show.rttm
        """
}

process prepare_initial_data_dir {
    memory '1GB'

    input:
      path show_rttm
      path audio

    output:
      path 'init_datadir', emit: init_datadir, optional: true

    shell:
    '''    
    if [ $(wc -l < !{show_rttm}) -gt 1 ]; then
        . !{projectDir}/bin/prepare_process.sh
        mkdir init_datadir
        echo audio audio A > init_datadir/reco2file_and_channel
        
        while read -r line; do
            # Skip lines that do not start with "SPEAKER"
            if [[ ! $line =~ ^SPEAKER ]]; then
                continue
            fi

            # Read fields from RTTM line
            IFS=' ' read -r -a fields <<< "$line"
            file_id=audio
            channel=${fields[2]}
            start_time=${fields[3]}
            duration=${fields[4]}
            speaker_id=${fields[7]}

            # Skip segments shorter than 0.3 seconds
            if (( $(echo "$duration < 0.3" | bc -l) )); then
                continue
            fi

            # Calculate end time
            end_time=$(echo "$start_time + $duration" | bc -l)

            # Format start and end times with zero-padding
            start_time_padded=$(LC_NUMERIC="C" printf "%08.3f" $start_time)
            end_time_padded=$(LC_NUMERIC="C" printf "%08.3f" $end_time)

            # Generate unique utterance ID
            utt_id="${file_id}_${speaker_id}_${start_time_padded}-${end_time_padded}"

            # Append to segments and utt2spk files
            echo "$utt_id $file_id $start_time $end_time" >> init_datadir/segments
            echo "$utt_id $speaker_id" >> init_datadir/utt2spk

        done < !{show_rttm}
          
        echo "audio !{audio}" > init_datadir/wav.scp  
        utils/utt2spk_to_spk2utt.pl init_datadir/utt2spk > init_datadir/spk2utt
        echo "audio !{audio}" > init_datadir/wav.scp
    fi      

    '''
}

process language_id {
    memory '4GB'
    label 'with_gpu'
    
    input:
      path init_datadir
      path audio    
    
    output:
      path 'datadir', emit:datadir, optional:true
    
    shell:
    if ( params.do_language_id )
      '''
      . !{projectDir}/bin/prepare_process.sh
      
      extract_lid_features_kaldi.py --use-gpu !{init_datadir} .
      cat !{init_datadir}/segments | awk '{print($1, "0")}' > trials
      threshold=`cat !{params.rootdir}/models/lid_et/threshold`
      ivector-subtract-global-mean !{params.rootdir}/models/lid_et/xvector.global.vec scp:xvector.scp ark:- | \
      ivector-normalize-length --scaleup=false ark:- ark:- | \
      logistic-regression-eval --apply-log=true --max-steps=20 --mix-up=0 \
        !{params.rootdir}/models/lid_et/lr.scale.model \
        ark:trials ark:- scores
      cat scores  | \
        awk '{print($1, $3 > '$threshold' ? "et" : "other")}' > utt2lang
      grep "et$" utt2lang | sort | awk '{print($1)}' > utts.filtered

      if [ -s utts.filtered ]; then
        mkdir datadir
        
        join <(sort !{init_datadir}/segments) utts.filtered | \
          awk '{print($1, $2, $3, $4)}' | LC_ALL=C sort > datadir/segments
        
        cp -r !{init_datadir}/{wav.scp,utt2spk,spk2utt} datadir
        utils/fix_data_dir.sh datadir
      fi
      '''      
    else
      '''
      mkdir datadir
      cp -r !{init_datadir}/{wav.scp,segments,utt2spk,spk2utt} datadir
      '''
}
    




process speaker_id {
    memory '2GB'
    label 'with_gpu'
    
   
    input:
      path datadir
      path audio    

    output:
      path 'sid-result.json', emit:sid_result

    shell:
        if (params.do_speaker_id)
          '''
          spk_id.sh --device 1 --accelerator gpu --load-checkpoint !{params.rootdir}/models/spk_id/version_295816/checkpoints/last.ckpt \
            --predict-datadir datadir --predictions-file spk_preds.txt --segments-per-spk 7
          
	        cat spk_preds.txt | grep -v "<unk>" | awk \'{if ($2 > !{params.spk_id_min_prosterior}) {print $0}}\' | \
	        python -c \'import json, sys; spks={s.split()[0]:{"name" : " ".join(s.split()[2:])} for s in sys.stdin}; json.dump(spks, sys.stdout);\' > sid-result.json
          '''
        else
          '''
          echo "{}" >  sid-result.json
          '''
}


process prepare_data_for_whisper {
    memory '2GB'
    cpus 1

    input:
      path datadir
      
    output:
      path 'datadir_25sec', emit: datadir_25sec, optional:true
      
    shell:
      '''
      . !{projectDir}/bin/prepare_process.sh
      mkdir datadir_25sec      
      merge_consecutive_segments.py --max_duration 25 datadir datadir_25sec
      cp datadir/wav.scp datadir_25sec/wav.scp
      utils/utt2spk_to_spk2utt.pl datadir_25sec/utt2spk > datadir_25sec/spk2utt
      utils/fix_data_dir.sh datadir_25sec
      
      '''
}


process decode_whisper {
    memory '8GB'
    cpus 1
    label 'with_gpu'

    input:
      path audio    
      
    output:
      path 'audio.json', emit: audio_json
      
    shell:
      '''
      #python -c "import torch; print(torch.cuda.is_available()); print(torch.backends.cudnn.version())"
      export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`:$LD_LIBRARY_PATH
      
      #whisper-transcribe.py --beam 1 --is_multilingual --language et !{params.rootdir}/models/whisper-large-et-v3-ct2 !{audio} datadir_25sec/segments | tee trans.hyp
      #whisper-transcribe.py --beam 1 --is_multilingual --language et !{params.rootdir}/models/whisper-medium-et-v4-ct2 !{audio} datadir_25sec/segments | cut -f 2- -d " " | perl -npe \'s/(\\S)([,.!?:])/\\1 \\2/g\' | paste <(cut -f 1 -d " " datadir_25sec/segments) - | tee trans.hyp
      
      whisper-pure-beam.py !{audio} --model_directory !{params.rootdir}/models/whisper-medium-et-v5-ct2 --beam_size 5 --num_hypotheses 5 --language et --device auto > audio.json
      #whisper-ctranslate2 --device cuda --language et --model_directory  !{params.rootdir}/models/whisper-large-et-orth-v2-ct2 !{audio} --beam_size  1 --vad_filter True
      #whisper-ctranslate2 --device cuda --language et --model_directory  !{params.rootdir}/models/whisper-large-et-orth-20240510-ct2 !{audio} --beam_size  3 --vad_filter True
      
      '''
}


process postprocess_whisper {
    memory '2GB'
    cpus 1

    input:
      path audio_tsv
      
    output:
      path 'datadir_whisper', emit: datadir_whisper, optional:true
      
    shell:
      '''
      if [ $(wc -l < !{audio_tsv}) -gt 1 ]; then
          . !{projectDir}/bin/prepare_process.sh
          mkdir datadir_whisper     
          echo audio audio A > datadir_whisper/reco2file_and_channel
          echo "audio audio.wav" > datadir_whisper/wav.scp      
          tail -n +2 !{audio_tsv} | awk '{printf("%04d", NR); print(" audio " $1/1000 " " $2/1000);}' > datadir_whisper/segments
          tail -n +2 !{audio_tsv} | awk '{printf("%04d", NR); $1=""; $2=""; print }' > datadir_whisper/text
          awk '{print($1, $1)}' datadir_whisper/segments > datadir_whisper/utt2spk
          awk '{print($1, $1)}' datadir_whisper/segments > datadir_whisper/spk2utt
      fi      
      '''
}

process postprocess_whisper_nbest {
    memory '2GB'
    cpus 1

    input:
      path audio_json
      
    output:
      path 'datadir_whisper', emit: datadir_whisper, optional:true
      path 'alternatives.json', emit: alternatives_json, optional:true
      
    shell:
      '''
      . !{projectDir}/bin/prepare_process.sh
      
      # Check if JSON file exists and has content
      if [ -s !{audio_json} ]; then
          # Parse JSON from our custom faster-whisper script
          python3 -c "
import json
import sys

# Load our custom whisper JSON output
with open('!{audio_json}', 'r') as f:
    whisper_data = json.load(f)

# Check if segments exist
if 'segments' not in whisper_data or len(whisper_data['segments']) == 0:
    sys.exit(0)

# Create Kaldi datadir structure using best hypothesis (rank 1)
import os
os.makedirs('datadir_whisper', exist_ok=True)

with open('datadir_whisper/reco2file_and_channel', 'w') as f:
    f.write('audio audio A\\n')

with open('datadir_whisper/wav.scp', 'w') as f:
    f.write('audio audio.wav\\n')

segments_lines = []
text_lines = []

for i, segment in enumerate(whisper_data['segments']):
    utt_id = f'{i+1:04d}'
    start_time = segment['start']
    end_time = segment['end']
    
    # Use best hypothesis (rank 1) for alignment
    best_text = ''
    if 'alternatives' in segment and segment['alternatives']:
        best_text = segment['alternatives'][0]['text'].strip()
    else:
        # Fallback if no alternatives structure
        best_text = segment.get('text', '').strip()
    
    segments_lines.append(f'{utt_id} audio {start_time:.3f} {end_time:.3f}')
    text_lines.append(f'{utt_id} {best_text}')

with open('datadir_whisper/segments', 'w') as f:
    f.write('\\n'.join(segments_lines) + '\\n')

with open('datadir_whisper/text', 'w') as f:
    f.write('\\n'.join(text_lines) + '\\n')

# Create utt2spk and spk2utt (each utterance is its own speaker for now)
utt2spk_lines = []
for i in range(len(whisper_data['segments'])):
    utt_id = f'{i+1:04d}'
    utt2spk_lines.append(f'{utt_id} {utt_id}')

with open('datadir_whisper/utt2spk', 'w') as f:
    f.write('\\n'.join(utt2spk_lines) + '\\n')

with open('datadir_whisper/spk2utt', 'w') as f:
    f.write('\\n'.join(utt2spk_lines) + '\\n')

# Save alternatives data (already in correct format from our script)
alternatives_data = {
    'text': whisper_data.get('text', ''),
    'language': whisper_data.get('language', 'et'),
    'segments': whisper_data.get('segments', [])
}

with open('alternatives.json', 'w') as f:
    json.dump(alternatives_data, f, indent=2, ensure_ascii=False)
"
      fi
      '''
}



/*

process align {
    memory '8GB'
    cpus 1
    label 'with_gpu'

    input:
      path datadir_25sec
      file trans_hyp    
      file audio    
      
    output:
      file 'alignments.json' into alignments_json
      
    shell:
      '''
      mkdir -p datadir_25sec_transcribed      
      cp -r datadir_25sec/{segments,wav.scp,utt2spk} datadir_25sec_transcribed
      cat !{trans_hyp}  > datadir_25sec_transcribed/text
      create_ctc_alignments.py !{params.rootdir}/models/hf/xls-r-300m-et datadir_25sec_transcribed alignments.json
      '''
}

*/

process align {
    memory '4GB'
    cpus 1
    label 'with_gpu'
    
    input:
      path datadir_whisper
      path audio    

    output:
      path 'alignments.json', emit: alignments_json
      path 'alignments.ctm', emit: alignments_ctm

    shell:
      '''
      . !{projectDir}/bin/prepare_process.sh

      utils/copy_data_dir.sh !{datadir_whisper} datadir_whisper_hires
      #cat !{datadir_whisper}/text | perl -npe 's/ [[:punct:]]/ /g;' >  datadir_whisper_hires/text      
      steps/make_mfcc.sh --nj 1 \
          --mfcc-config !{params.rootdir}/kaldi-data/!{params.acoustic_model}/conf/mfcc.conf \
          datadir_whisper_hires || echo "Warning: steps/make_mfcc.sh resulted in error"
      steps/compute_cmvn_stats.sh datadir_whisper_hires || exit 1
      utils/fix_data_dir.sh datadir_whisper_hires

      steps/online/nnet2/extract_ivectors_online.sh  --nj 1 \
        datadir_whisper_hires !{params.rootdir}/kaldi-data/!{params.acoustic_model}/ivector_extractor ivectors || exit 1;

      mkdir dict
      cp -r !{params.rootdir}/kaldi-data/dict .
      rm -f dict/lexicon*.txt
      echo "<unk>   UNK" > dict/lexicon.txt
      echo "<sil>   SIL" >> dict/lexicon.txt

      cat datadir_whisper_hires/text | perl -npe 's/\\S+\\s//; s/\\s+/\\n/g;' | grep -v "^\\s*$" | sort | uniq | ${ET_G2P_ROOT}/run.sh |  perl -npe 's/\\(\\d\\)(\\s)/\\1/' | \
        perl -npe 's/\\b(\\w+) \\1\\b/\\1\\1 /g; s/(\\s)jj\\b/\\1j/g; s/(\\s)tttt\\b/\\1tt/g; s/(\\s)pppp\\b/\\1pp/g;  ' | uniq >> dict/lexicon.txt

      utils/prepare_lang.sh --phone-symbol-table !{params.rootdir}/kaldi-data/!{params.acoustic_model}/phones.txt --sil-prob 0.01 dict '<unk>' dict/tmp lang
      cp -r !{params.rootdir}/kaldi-data/!{params.acoustic_model}/conf .      

      steps/nnet3/align.sh --retry-beam 300 --use-gpu true --nj 1 --online-ivector-dir ivectors \
        --scale-opts '--transition-scale=1.0 --self-loop-scale=1.0 --acoustic-scale=1.0' \
        datadir_whisper_hires lang !{params.rootdir}/kaldi-data/!{params.acoustic_model} ali
        
      steps/get_train_ctm.sh --frame-shift 0.03 --use-segments false \
        datadir_whisper_hires lang ali .

      ctm2json.py ctm datadir_whisper_hires/segments !{datadir_whisper}/text > alignments.json
      utils/convert_ctm.pl datadir_whisper_hires/segments <(echo audio audio 1) ctm > alignments.ctm
              
      '''
}



process final_output {
    memory '500MB'
    
    publishDir "${params.out_dir}", mode: 'copy', overwrite: true

    input:
      val basename
      path datadir
      path show_rttm
      path sid_result
      path alignments_json
      path alignments_ctm
      path empty_result_txt

    output:
      path "result.json"
      path "result.srt"
      path "result.trs"
      path "result.ctm"
      path "result.txt"

    shell:
      '''
      resegment_json.py --new-turn-sil-length 1.0 --speaker-names !{sid_result} --rttm !{show_rttm} !{datadir} !{alignments_json} > punctuated.json
      
      normalize_json.py words2numbers.py punctuated.json > result.json

      json2trs.py --fid trs result.json > result.trs
      json2srt.py result.json > result.srt
      cat !{alignments_ctm} | perl -npe 's/[.,!?;:]$//' >  result.ctm
      json2text.py < result.json > result.txt

      '''
}

process final_output_nbest {
    memory '500MB'
    
    publishDir "${params.out_dir}", mode: 'copy', overwrite: true

    input:
      val basename
      path datadir
      path show_rttm
      path sid_result
      path alignments_json
      path alternatives_json
      path empty_result_txt

    output:
      path "result.json"

    shell:
      '''
      # First create the standard result with word-level alignments for best hypothesis
      resegment_json.py --new-turn-sil-length 1.0 --speaker-names !{sid_result} --rttm !{show_rttm} !{datadir} !{alignments_json} > punctuated.json
      
      normalize_json.py words2numbers.py punctuated.json > best_result.json

      # Combine best result with alternatives
      python3 -c "
import json
import sys

# Load best result with word-level alignments
try:
    with open('best_result.json', 'r') as f:
        best_result = json.load(f)
except:
    best_result = {'monologues': []}

# Load alternatives
try:
    with open('!{alternatives_json}', 'r') as f:
        alternatives_data = json.load(f)
except:
    alternatives_data = {'segments': []}

# Create final result structure
final_result = {
    'best_hypothesis': best_result,
    'alternatives': {
        'language': alternatives_data.get('language', 'et'),
        'segments': []
    },
    'metadata': {
        'n_best': 5,
        'has_word_alignments': True,
        'basename': '!{basename}'
    }
}

# Add segment-level alternatives
for segment in alternatives_data.get('segments', []):
    segment_alts = {
        'start': segment['start'],
        'end': segment['end'],
        'alternatives': segment.get('alternatives', [])
    }
    final_result['alternatives']['segments'].append(segment_alts)

# Output final result
with open('result.json', 'w') as f:
    json.dump(final_result, f, indent=2, ensure_ascii=False)
"
      '''
}


process empty_output {

    publishDir "${params.out_dir}", mode: 'copy', overwrite: true

    input:
      val basename

    output:
      path "result.json", emit: empty_result_json
      path "result.srt", emit: empty_result_srt
      path "result.trs", emit: empty_result_trs
      path "result.ctm", emit: empty_result_ctm
      path "result.txt", emit: empty_result_txt
      

    shell:      
      '''
      json=!{params.rootdir}/assets/empty.json
      cp $json result.json
      json2trs.py $json > result.trs      
      touch result.srt result.ctm result.txt
      '''
}


workflow {
  files = params.in_file_list != "" ? Channel.fromPath(params.in_file_list).splitText() : Channel.fromPath(params.in)
  
  to_wav(files)
  empty_output(to_wav.out.basename)
  diarization(to_wav.out.audio)
  prepare_initial_data_dir(diarization.out.show_rttm, to_wav.out.audio)
  language_id(prepare_initial_data_dir.out.init_datadir, to_wav.out.audio)
  speaker_id(language_id.out.datadir, to_wav.out.audio)
  decode_whisper(to_wav.out.audio)
  postprocess_whisper_nbest(decode_whisper.out.audio_json)
  align(postprocess_whisper_nbest.out.datadir_whisper, to_wav.out.audio)
  final_output_nbest(to_wav.out.basename, language_id.out.datadir, diarization.out.show_rttm, speaker_id.out.sid_result, align.out.alignments_json, postprocess_whisper_nbest.out.alternatives_json, empty_output.out.empty_result_txt)
}
