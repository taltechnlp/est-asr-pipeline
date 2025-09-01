#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.in_json = ""
params.in_audio = ""
params.out_dir = "align_results/"

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
        ffmpeg -i !{audio_file} -ac 1 -ar 16000 audio.wav
        """
}

process create_datadir_from_json {
    memory '2GB'
    cpus 1

    input:
    path json_file
    val basename

    output:
    path 'datadir_whisper', emit: datadir_whisper
    
    shell:
    '''
    export LANG=C.UTF-8
    export LC_ALL=C.UTF-8
    
    . !{projectDir}/bin/prepare_process.sh
    
    mkdir -p datadir_whisper

    # Create basic Kaldi datadir structure
    echo "audio audio A" > datadir_whisper/reco2file_and_channel
    echo "audio audio.wav" > datadir_whisper/wav.scp
    
    # Parse JSON and create Kaldi datadir using separate Python script
    json_to_kaldi_datadir.py !{json_file} datadir_whisper
    '''
}

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

process output_ctm {
    memory '500MB'
    
    publishDir "${params.out_dir}", mode: 'copy', overwrite: true

    input:
    val basename
    path alignments_ctm

    output:
    path "result.ctm"

    shell:
    '''
    export LANG=C.UTF-8
    export LC_ALL=C.UTF-8
    
    # Clean CTM by removing punctuation marks at end of words
    cat !{alignments_ctm} | perl -npe 's/[.,!?;:]$//' > result.ctm
    '''
}

process reverse_transformations {
    memory '1GB'
    cpus 1
    
    publishDir "${params.out_dir}", mode: 'copy', overwrite: true
    
    input:
    path alignments_ctm
    path original_json
    path datadir_whisper
    
    output:
    path "result_reversed.ctm", emit: reversed_ctm
    path "transformation_report.json", emit: report
    
    shell:
    '''
    export LANG=C.UTF-8
    export LC_ALL=C.UTF-8
    
    # Clean CTM first (same as output_ctm process)
    cat !{alignments_ctm} | perl -npe 's/[.,!?;:]$//' > cleaned.ctm
    
    # Apply transformation reversal
    reverse_ctm_transformations.py \
        cleaned.ctm \
        !{original_json} \
        !{datadir_whisper}/text \
        --output-ctm result_reversed.ctm \
        --output-report transformation_report.json
    '''
}

process calculate_wer {
    memory '1GB'
    cpus 1
    
    publishDir "${params.out_dir}", mode: 'copy', overwrite: true
    
    input:
    path original_ctm
    path reversed_ctm
    path original_json
    
    output:
    path "wer_comparison.txt"
    
    shell:
    '''
    export LANG=C.UTF-8
    export LC_ALL=C.UTF-8
    
    # Calculate WER for both versions
    calculate_wer.py \
        --reference !{original_json} \
        --hypothesis-original !{original_ctm} \
        --hypothesis-reversed !{reversed_ctm} \
        > wer_comparison.txt
    '''
}

workflow {
    json_file = Channel.fromPath(params.in_json)
    audio_file = Channel.fromPath(params.in_audio)
    
    to_wav(audio_file)
    create_datadir_from_json(json_file, to_wav.out.basename)
    align(create_datadir_from_json.out.datadir_whisper, to_wav.out.audio)
    output_ctm(to_wav.out.basename, align.out.alignments_ctm)
    
    // New processes for transformation reversal and WER comparison
    reverse_transformations(
        align.out.alignments_ctm,
        json_file,
        create_datadir_from_json.out.datadir_whisper
    )
    
    calculate_wer(
        output_ctm.out,
        reverse_transformations.out.reversed_ctm,
        json_file
    )
}