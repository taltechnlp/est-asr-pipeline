#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.in_json = ""
params.in_audio = ""
params.out_dir = "align_results/"
params.reference_trs = ""  // Optional TRS file for WER calculation
params.main_asr_ctm = ""   // Optional Main ASR CTM file
params.alt_asr_ctm = ""    // Optional Alternative ASR CTM file
params.calculate_wer = false  // Flag to enable WER calculation
params.export_comparison = false  // Flag to enable JSON comparison export

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

process calculate_wer {
    memory '4GB'
    cpus 2
    
    publishDir "${params.out_dir}", mode: 'copy', overwrite: true
    
    when:
    params.reference_trs != "" && params.calculate_wer
    
    input:
    path reference_trs
    path alt_asr_ctm
    path main_asr_ctm
    path agent_ctm
    val basename
    
    output:
    path "wer_report.txt"
    
    shell:
    '''
    export LANG=C.UTF-8
    export LC_ALL=C.UTF-8
    
    mkdir -p trs_ref_dir
    mkdir -p stm_output_dir
    
    # Copy TRS file to directory (script expects directory with .trs files)
    cp !{reference_trs} trs_ref_dir/
    TRS_BASENAME=$(basename !{reference_trs} .trs)
    
    echo "=== WER Analysis Results ===" > wer_report.txt
    echo "" >> wer_report.txt
    
    # Function to calculate WER for a CTM file
    calculate_single_wer() {
        local ctm_file="$1"
        local system_name="$2"
        local ctm_filename="$3"
        
        if [ "$ctm_file" != "OPTIONAL_FILE" ] && [ -f "$ctm_file" ]; then
            echo "=== $system_name ===" >> wer_report.txt
            
            local ctm_dir="ctm_${ctm_filename}_dir"
            mkdir -p "$ctm_dir"
            cp "$ctm_file" "$ctm_dir/${TRS_BASENAME}.ctm"
            
            # Fix CTM file ID to match TRS file for SCLITE compatibility
            if grep -q "^audio" "$ctm_dir/${TRS_BASENAME}.ctm"; then
                sed "s/^audio/${TRS_BASENAME}/g" "$ctm_dir/${TRS_BASENAME}.ctm" > "$ctm_dir/${TRS_BASENAME}_fixed.ctm"
                mv "$ctm_dir/${TRS_BASENAME}_fixed.ctm" "$ctm_dir/${TRS_BASENAME}.ctm"
            fi
            
            # Run WER calculation
            python3 /opt/est-asr-pipeline/bin/code_switching_wer.py \
                --trs-ref-dir-path trs_ref_dir \
                --ctm-hyp-dir-path "$ctm_dir" \
                --stm-ref-dir-path stm_output_dir >> wer_report.txt
            
            echo "" >> wer_report.txt
        else
            echo "=== $system_name: Not provided ===" >> wer_report.txt
            echo "" >> wer_report.txt
        fi
    }
    
    # Calculate WER for all three systems
    calculate_single_wer "!{alt_asr_ctm}" "Alternative ASR" "alt"
    calculate_single_wer "!{main_asr_ctm}" "Main ASR" "main"  
    calculate_single_wer "!{agent_ctm}" "Agent CTM (align.nf output)" "agent"
    '''
}

process export_comparison_json {
    memory '2GB'
    cpus 1
    
    publishDir "${params.out_dir}", mode: 'copy', overwrite: true
    
    when:
    params.reference_trs != "" && params.export_comparison
    
    input:
    path reference_trs
    path alt_asr_ctm
    path main_asr_ctm
    path agent_ctm
    val basename
    
    output:
    path "comparison.json"
    
    shell:
    '''
    export LANG=C.UTF-8
    export LC_ALL=C.UTF-8
    
    # Handle optional CTM files by creating empty ones if not provided
    ALT_CTM="!{alt_asr_ctm}"
    MAIN_CTM="!{main_asr_ctm}"
    
    # Create empty CTM files for missing inputs
    if [ "$ALT_CTM" = "OPTIONAL_FILE" ]; then
        touch empty_alt.ctm
        ALT_CTM="empty_alt.ctm"
    fi
    
    if [ "$MAIN_CTM" = "OPTIONAL_FILE" ]; then
        touch empty_main.ctm
        MAIN_CTM="empty_main.ctm"
    fi
    
    # Run comparison using trs_ctm_compare.py
    python3 /opt/est-asr-pipeline/bin/trs_ctm_compare.py !{reference_trs} "$ALT_CTM" "$MAIN_CTM" !{agent_ctm} comparison.json
    '''
}


workflow {
    json_file = Channel.fromPath(params.in_json)
    audio_file = Channel.fromPath(params.in_audio)
    
    to_wav(audio_file)
    create_datadir_from_json(json_file, to_wav.out.basename)
    align(create_datadir_from_json.out.datadir_whisper, to_wav.out.audio)
    output_ctm(to_wav.out.basename, align.out.alignments_ctm)
    
    // Optional WER calculation
    if (params.reference_trs != "" && params.calculate_wer) {
        reference_trs_file = Channel.fromPath(params.reference_trs)
        
        // Handle optional CTM files for WER calculation
        alt_ctm_file = params.alt_asr_ctm != "" ? Channel.fromPath(params.alt_asr_ctm) : Channel.value("OPTIONAL_FILE")
        main_ctm_file = params.main_asr_ctm != "" ? Channel.fromPath(params.main_asr_ctm) : Channel.value("OPTIONAL_FILE")
        
        calculate_wer(
            reference_trs_file, 
            alt_ctm_file, 
            main_ctm_file, 
            align.out.alignments_ctm, 
            to_wav.out.basename
        )
    }
    
    // Optional JSON comparison export
    if (params.reference_trs != "" && params.export_comparison) {
        reference_trs_file = Channel.fromPath(params.reference_trs)
        
        // Handle optional CTM files
        alt_ctm_file = params.alt_asr_ctm != "" ? Channel.fromPath(params.alt_asr_ctm) : Channel.value("OPTIONAL_FILE")
        main_ctm_file = params.main_asr_ctm != "" ? Channel.fromPath(params.main_asr_ctm) : Channel.value("OPTIONAL_FILE")
        
        export_comparison_json(
            reference_trs_file, 
            alt_ctm_file, 
            main_ctm_file, 
            align.out.alignments_ctm, 
            to_wav.out.basename
        )
    }
}