#!/usr/bin/env nextflow

params.in = "/home/aivo_olevi/tmp/speechfiles/intervjuu2018080910.mp3"
params.out = "result.json"
params.out_format = "json"
params.do_music_detection = "no" // yes or no 
params.do_speaker_id = "no" // yes or no
params.speaker_id_server_url = ""
params.file_ext = "mp3"
params.srcdir = "/opt/kaldi-offline-transcriber"
params.acoustic_model = "tdnn_7d_online"
params.njobs = 2
params.nthreads = 4
params.decode_cmd = "run.pl"
params.kaldi_root = "/opt/kaldi"
params.et_g2p_fst = "/opt/et-g2p-fst"
params.lm_scale = 10
audio_file = file(params.in)


process to_wav {
    input:
    path audio_file

    output:
    file 'audio.wav' into audio
    
    script:
    if( params.file_ext == 'wav' )
        """
        sox $params.in -c 1 audio.wav rate -v 16k
        """

    else if( params.file_ext == 'mp3' || params.file_ext == 'mpga' || params.file_ext == 'm4a' || params.file_ext == 'mp4' )
        """
        ffmpeg -i $params.in -f sox - | sox -t sox - -c 1 -b 16 -t wav audio.wav rate -v 16k	
        """

    else if( params.file_ext == 'ogg' || params.file_ext == 'mp2' || params.file_ext == 'flac' )
        """
        sox $params.in -c 1 audio.wav rate -v 16k
        """

    else
        error "Invalid alignment params.file_ext: $params.file_ext"

}

process diarization {
    input:
    file audio

    output: 
    file 'show.seg' into show_seg

    script:
        diarization_opts = params.do_music_detection == 'yes' ? '-m' : '' 
        """
        echo "audio 1 0 1000000000 U U U 1" > show.uem.seg
        WORK_DIR=\$PWD
        cd $params.srcdir
        ./scripts/diarization.sh $diarization_opts \${WORK_DIR}/$audio \${WORK_DIR}/show.uem.seg
        """
}

process segments {
    input:
    file show_seg

    output:
    file 'segments' into segments

    script:
    """
    echo audio audio A > reco2file_and_channel
    cat $show_seg | cut -f "3,4,8" -d " " | \
	while read LINE ; do \
		start=`echo \$LINE | cut -f 1,2 -d " " | perl -ne '@t=split(); \$start=\$t[0]/100.0; printf("%08.3f", \$start);'`; \
		end=`echo \$LINE   | cut -f 1,2 -d " " | perl -ne '@t=split(); \$start=\$t[0]/100.0; \$len=\$t[1]/100.0; \$end=\$start+\$len; printf("%08.3f", \$end);'`; \
		sp_id=`echo \$LINE | cut -f 3 -d " "`; \
		echo audio-\${sp_id}---\${start}-\${end} audio \$start \$end; \
	done > segments
	if [ ! -s segments } ]; then \
	  echo "audio-dummy---0.000-0.110 audio 0.0 0.110" > segments; \
	fi
	cat segments | perl -npe 's/\\s+.*//; s/((.*)---.*)/\\1 \\2/' > utt2spk
    ${params.srcdir}/utils/utt2spk_to_spk2utt.pl utt2spk > spk2utt
    """
}

// MFCC calculation
process mfcc {
    input:
    file segments
    file audio

    output:
    file 'feats.scp' into feats
    file 'ivectors/*' into ivectors
    file 'utt2spk' into utt2spk

    script:
    """
    ln -s ${params.kaldi_root}/egs/wsj/s5/utils
    ln -s ${params.kaldi_root}/egs/wsj/s5/steps
    ln -s ${params.srcdir}/build

    cat $segments | perl -npe 's/\\s+.*//; s/((.*)---.*)/\\1 \\2/' > utt2spk
    ${params.srcdir}/utils/utt2spk_to_spk2utt.pl utt2spk > spk2utt
    echo \"audio audio.wav\" > wav.scp
    make_mfcc.sh \
        --mfcc-config ${params.srcdir}/build/fst/${params.acoustic_model}/conf/mfcc.conf \
        --cmd ${params.decode_cmd} \
        --nj ${params.njobs} \
        \$PWD || exit 1
	steps/compute_cmvn_stats.sh \$PWD || exit 1
	utils/fix_data_dir.sh \$PWD

    steps/online/nnet2/extract_ivectors_online.sh --cmd \"${params.decode_cmd}\" --nj ${params.njobs} \
		. build/fst/${params.acoustic_model}/ivector_extractor ivectors || exit 1;
    """
}

// Do 1-pass decoding using chain online models
process one_pass_decoding {
    // Do 1-pass decoding using chain online models
    input:
    file feats
    file 'ivectors/*' from ivectors
    file utt2spk

    output:
    path "${params.acoustic_model}_pruned_unk*" into pruned_unk

    script:
    """
    ln -s ${params.kaldi_root}/egs/wsj/s5/steps
    ln -s ${params.kaldi_root}/egs/wsj/s5/utils
    ln -s ${params.srcdir}/build
    mkdir -p ${params.acoustic_model}_pruned_unk
    (cd ${params.acoustic_model}_pruned_unk; for f in ${params.srcdir}/build/fst/${params.acoustic_model}/*; do ln -s \$f; done)

    steps/nnet3/decode.sh --num-threads $params.nthreads --acwt 1.0  --post-decode-acwt 10.0 \
	    --skip-scoring true --cmd \"${params.decode_cmd}\" --nj ${params.njobs} \
	    --online-ivector-dir ivectors \
	    --skip-diagnostics true \
      ${params.srcdir}/build/fst/${params.acoustic_model}/graph_prunedlm_unk . ${params.acoustic_model}_pruned_unk/decode || exit 1;
    (cd ${params.acoustic_model}_pruned_unk; ln -s ${params.srcdir}/build/fst/${params.acoustic_model}/graph_prunedlm_unk graph)
    """
}

// Rescore lattices with a larger language model
process rescore_lattices {
    // Rescore lattices with a larger language model
    input: 
    path pruned_unk from pruned_unk

    output:
    path "${params.acoustic_model}_pruned_rescored_main_unk*" into pruned_rescored_unk

    script:
    """
    ln -s ${params.kaldi_root}/egs/wsj/s5/steps
    ln -s ${params.kaldi_root}/egs/wsj/s5/utils
    ln -s ${params.kaldi_root}/egs/wsj/s5/local
    ln -s ${params.srcdir}/build
    mkdir -p ${params.acoustic_model}_pruned_rescored_main_unk
	(cd ${params.acoustic_model}_pruned_rescored_main_unk; for f in ${params.srcdir}/build/fst/${params.acoustic_model}/*; do ln -s \$f; done)

	steps/lmrescore_const_arpa.sh \
	  build/fst/data/prunedlm_unk build/fst/data/largelm_unk \
	  . \
	  ${params.acoustic_model}_pruned_unk/decode ${params.acoustic_model}_pruned_rescored_main_unk/decode || exit 1;
	cp -r --preserve=links ${params.acoustic_model}_pruned_unk/graph ${params.acoustic_model}_pruned_rescored_main_unk	
    """
}

process main_rnnlm {
    input:
    path pruned_unk from pruned_unk
    path pruned_rescored_unk from pruned_rescored_unk

    output:
    path "${params.acoustic_model}_pruned_rescored_main_rnnlm_unk*" into main_rnnlm

    script:
    """
    ln -s ${params.kaldi_root}/egs/wsj/s5/rnnlm
    ln -s ${params.kaldi_root}/egs/wsj/s5/steps
    ln -s ${params.kaldi_root}/egs/wsj/s5/utils
    ln -s ${params.kaldi_root}/egs/wsj/s5/local
    ln -s ${params.srcdir}/build
    
    mkdir -p ${params.acoustic_model}_pruned_rescored_main_rnnlm_unk
	(cd ${params.acoustic_model}_pruned_rescored_main_rnnlm_unk; for f in ${params.srcdir}/build/fst/${params.acoustic_model}/*; do ln -s \$f; done)
	rnnlm/lmrescore_pruned.sh \
	    --skip-scoring true \
	    --max-ngram-order 4 \
      build/fst/data/largelm_unk \
      build/fst/data/rnnlm_unk \
      . \
	  ${params.acoustic_model}_pruned_rescored_main_unk/decode \
      ${params.acoustic_model}_pruned_rescored_main_rnnlm_unk/decode
	cp -r --preserve=links ${params.acoustic_model}_pruned_unk/graph ${params.acoustic_model}_pruned_rescored_main_rnnlm_unk/
    """
}

process decode {
    input:
    path main_rnnlm from main_rnnlm

    output:
    file 'segmented.ctm' into segmented_ctm

    script:
    """
    ln -s ${params.kaldi_root}/egs/wsj/s5/rnnlm
    ln -s ${params.kaldi_root}/egs/wsj/s5/steps
    ln -s ${params.kaldi_root}/egs/wsj/s5/utils
    ln -s ${params.kaldi_root}/egs/wsj/s5/local
    ln -s ${params.srcdir}/build
    ln -s ${params.srcdir}/scripts
    
    frame_shift_opt=""; \
	if [ -f ${main_rnnlm}/frame_subsampling_factor ]; then \
	  factor=`cat ${main_rnnlm}/frame_subsampling_factor`; \
	  frame_shift_opt=\"--frame-shift 0.0\$factor\"; \
	fi; \
	steps/get_ctm.sh \$frame_shift_opt . ${main_rnnlm}/graph ${main_rnnlm}/decode
    
	frame_shift_opt=""; \
	if [ -f  ${main_rnnlm}/frame_subsampling_factor ]; then \
	  factor=`cat ${main_rnnlm}/frame_subsampling_factor`; \
	  frame_shift_opt=\"--frame-shift 0.0\$factor\"; \
	fi; \
	${params.srcdir}/local/get_ctm_unk.sh --use_segments false \$frame_shift_opt \
	  --unk-p2g-cmd \"python3 ${params.srcdir}/local/unk_p2g.py --p2g-cmd \'python3 ${params.et_g2p_fst}/g2p.py --inverse --fst  ${params.et_g2p_fst}/data/chars.fst --nbest 1\'\" \
	  --unk-word \'<unk>\' \
	  --min-lmwt ${params.lm_scale} \
	  --max-lmwt ${params.lm_scale} \
	  . ${main_rnnlm}/graph ${main_rnnlm}/decode

    cat ${main_rnnlm}/decode/score_${params.lm_scale}/..ctm  | perl -npe \'s/(.*)-(S\\d+)---(\\S+)/\\1_\\3_\\2/\' > segmented.splitw2.ctm

    python3 scripts/compound-ctm.py \
		\"python3 scripts/compounder.py build/fst/data/compounderlm/G.fst build/fst/data/compounderlm/words.txt\" \
		< segmented.splitw2.ctm > with-compounds.ctm
    
    cat with-compounds.ctm | grep -v \"++\" |  grep -v \"\\[sil\\]\" | grep -v -e \" \$\$\" | perl -npe \'s/\\+//g\' | sort -k1,1 -k 3,3g > segmented.ctm
    """
}

process json {
    input:
    file segmented_ctm

    storeDir 'results'

    output:
    file params.out into results

    script:
    if( params.do_speaker_id == 'yes' && params.do_music_detection == 'yes' )
        """
        python3 ${params.srcdir}/local/segmented_ctm2json.py --speaker-names sid-result.json --pms-seg show.pms.seg $segmented_ctm > ${params.out}
        """
    else if( params.do_speaker_id == 'yes' && params.do_music_detection == 'yes' )
        """
        """
    else if( params.do_speaker_id == 'no' && params.do_music_detection == 'yes' )
        """
        """
    else
        """
        python3 ${params.srcdir}/local/segmented_ctm2json.py $segmented_ctm > ${params.out}
        """
}