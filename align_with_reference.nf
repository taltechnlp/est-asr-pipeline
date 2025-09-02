#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.edited_json = ""
params.reference_json = ""
params.reference_ctm = ""
params.reference_trs = ""
params.in_audio = ""
params.out_dir = "align_with_ref_results/"
params.confidence_threshold = 0.8
params.use_anchoring = true

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

process convert_reference_trs {
    memory '1GB'
    cpus 1
    
    when:
    params.reference_trs != ""
    
    input:
    path trs_file
    
    output:
    path 'reference.json', emit: reference_json
    
    shell:
    '''
    export LANG=C.UTF-8
    export LC_ALL=C.UTF-8
    
    trs2json.py !{trs_file} > reference.json
    '''
}

process convert_reference_ctm {
    memory '1GB' 
    cpus 1
    
    when:
    params.reference_ctm != ""
    
    input:
    path ctm_file
    
    output:
    path 'reference.json', emit: reference_json
    
    shell:
    '''
    export LANG=C.UTF-8
    export LC_ALL=C.UTF-8
    
    ctm2reference_json.py !{ctm_file} > reference.json
    '''
}

process compare_and_anchor {
    memory '2GB'
    cpus 1

    input:
    path edited_json
    path reference_json

    output:
    path 'anchor_mapping.json', emit: anchor_mapping
    path 'anchored_segments.json', emit: anchored_segments
    path 'edit_report.json', emit: edit_report
    
    shell:
    '''
    export LANG=C.UTF-8
    export LC_ALL=C.UTF-8
    
    compare_and_anchor_alignments.py \
        --edited !{edited_json} \
        --reference !{reference_json} \
        --confidence-threshold !{params.confidence_threshold} \
        --output-mapping anchor_mapping.json \
        --output-segments anchored_segments.json \
        --output-report edit_report.json
    '''
}

process create_smart_segments {
    memory '2GB'
    cpus 1

    input:
    path anchored_segments
    val basename

    output:
    path 'datadir_anchored', emit: datadir_anchored
    
    shell:
    '''
    export LANG=C.UTF-8
    export LC_ALL=C.UTF-8
    
    . !{projectDir}/bin/prepare_process.sh
    
    mkdir -p datadir_anchored
    
    # Create basic Kaldi datadir structure
    echo "audio audio A" > datadir_anchored/reco2file_and_channel
    echo "audio audio.wav" > datadir_anchored/wav.scp
    
    # Create smart segments using anchor information
    smart_segment_for_alignment.py !{anchored_segments} datadir_anchored
    '''
}

process align_with_anchors {
    memory '4GB'
    cpus 1
    label 'with_gpu'
    
    input:
    path datadir_anchored
    path audio
    path anchor_mapping

    output:
    path 'alignments.json', emit: alignments_json
    path 'alignments.ctm', emit: alignments_ctm
    path 'alignment_confidence.json', emit: confidence

    shell:
    '''
    . !{projectDir}/bin/prepare_process.sh

    utils/copy_data_dir.sh !{datadir_anchored} datadir_anchored_hires
    
    steps/make_mfcc.sh --nj 1 \
        --mfcc-config !{params.rootdir}/kaldi-data/!{params.acoustic_model}/conf/mfcc.conf \
        datadir_anchored_hires || echo "Warning: steps/make_mfcc.sh resulted in error"
    steps/compute_cmvn_stats.sh datadir_anchored_hires || exit 1
    utils/fix_data_dir.sh datadir_anchored_hires

    steps/online/nnet2/extract_ivectors_online.sh  --nj 1 \
      datadir_anchored_hires !{params.rootdir}/kaldi-data/!{params.acoustic_model}/ivector_extractor ivectors || exit 1;

    mkdir dict
    cp -r !{params.rootdir}/kaldi-data/dict .
    rm -f dict/lexicon*.txt
    echo "<unk>   UNK" > dict/lexicon.txt
    echo "<sil>   SIL" >> dict/lexicon.txt

    cat datadir_anchored_hires/text | perl -npe 's/\\S+\\s//; s/\\s+/\\n/g;' | grep -v "^\\s*$" | sort | uniq | ${ET_G2P_ROOT}/run.sh |  perl -npe 's/\\(\\d\\)(\\s)/\\1/' | \
      perl -npe 's/\\b(\\w+) \\1\\b/\\1\\1 /g; s/(\\s)jj\\b/\\1j/g; s/(\\s)tttt\\b/\\1tt/g; s/(\\s)pppp\\b/\\1pp/g;  ' | uniq >> dict/lexicon.txt

    utils/prepare_lang.sh --phone-symbol-table !{params.rootdir}/kaldi-data/!{params.acoustic_model}/phones.txt --sil-prob 0.01 dict '<unk>' dict/tmp lang
    cp -r !{params.rootdir}/kaldi-data/!{params.acoustic_model}/conf .      

    steps/nnet3/align.sh --retry-beam 300 --use-gpu true --nj 1 --online-ivector-dir ivectors \
      --scale-opts '--transition-scale=1.0 --self-loop-scale=1.0 --acoustic-scale=1.0' \
      datadir_anchored_hires lang !{params.rootdir}/kaldi-data/!{params.acoustic_model} ali
      
    steps/get_train_ctm.sh --frame-shift 0.03 --use-segments false \
      datadir_anchored_hires lang ali .

    ctm2json.py ctm datadir_anchored_hires/segments !{datadir_anchored}/text > raw_alignments.json
    utils/convert_ctm.pl datadir_anchored_hires/segments <(echo audio audio 1) ctm > raw_alignments.ctm
    
    # Calculate alignment confidence scores
    python3 -c "
import json
with open('!{anchor_mapping}', 'r') as f:
    mapping = json.load(f)
confidence = {'high_confidence_words': len([w for w in mapping.get('anchors', []) if w.get('confidence', 0) > !{params.confidence_threshold}]),
              'total_anchors': len(mapping.get('anchors', [])),
              'edit_distance': mapping.get('edit_distance', 0)}
with open('alignment_confidence.json', 'w') as f:
    json.dump(confidence, f, indent=2)
"
    
    cp raw_alignments.json alignments.json
    cp raw_alignments.ctm alignments.ctm
    '''
}

process merge_with_anchors {
    memory '2GB'
    cpus 1
    
    input:
    path raw_alignments_ctm
    path raw_alignments_json
    path anchor_mapping
    path reference_json
    
    output:
    path 'merged_alignments.ctm', emit: merged_ctm
    
    shell:
    '''
    export LANG=C.UTF-8
    export LC_ALL=C.UTF-8
    
    merge_alignments_with_anchors.py \
        --new-alignments-ctm !{raw_alignments_ctm} \
        --new-alignments-json !{raw_alignments_json} \
        --reference !{reference_json} \
        --anchor-mapping !{anchor_mapping} \
        --confidence-threshold !{params.confidence_threshold} \
        --output-ctm merged_alignments.ctm \
        --output-json /dev/null \
        --output-report /dev/null
    '''
}

process output_final_ctm {
    memory '500MB'
    
    publishDir "${params.out_dir}", mode: 'copy', overwrite: true

    input:
    val basename
    path merged_alignments_ctm

    output:
    path "result.ctm"

    shell:
    '''
    export LANG=C.UTF-8
    export LC_ALL=C.UTF-8
    
    # Clean CTM by removing punctuation marks at end of words
    cat !{merged_alignments_ctm} | perl -npe 's/[.,!?;:]$//' > result.ctm
    '''
}

workflow {
    audio_file = Channel.fromPath(params.in_audio)
    edited_json_file = Channel.fromPath(params.edited_json)
    
    to_wav(audio_file)
    
    // Handle different reference input types
    reference_json = Channel.empty()
    
    if (params.reference_json != "") {
        reference_json = Channel.fromPath(params.reference_json)
    } else if (params.reference_trs != "") {
        trs_file = Channel.fromPath(params.reference_trs)
        convert_reference_trs(trs_file)
        reference_json = convert_reference_trs.out.reference_json
    } else if (params.reference_ctm != "") {
        ctm_file = Channel.fromPath(params.reference_ctm)
        convert_reference_ctm(ctm_file) 
        reference_json = convert_reference_ctm.out.reference_json
    }
    
    // Main alignment workflow
    if (params.use_anchoring) {
        compare_and_anchor(edited_json_file, reference_json)
        create_smart_segments(compare_and_anchor.out.anchored_segments, to_wav.out.basename)
        align_with_anchors(create_smart_segments.out.datadir_anchored, to_wav.out.audio, compare_and_anchor.out.anchor_mapping)
        // Skip merge_with_anchors - use direct alignment output for now
        output_final_ctm(
            to_wav.out.basename,
            align_with_anchors.out.alignments_ctm
        )
    } else {
        // Fallback to standard alignment without anchoring
        create_smart_segments(edited_json_file, to_wav.out.basename)
        align_with_anchors(create_smart_segments.out.datadir_anchored, to_wav.out.audio, Channel.empty())
        
        output_final_ctm(
            to_wav.out.basename,
            align_with_anchors.out.alignments_ctm
        )
    }
}