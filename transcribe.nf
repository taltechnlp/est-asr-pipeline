#!/usr/bin/env nextflow

params.in = "/home/aivo_olevi/tmp/speechfiles/intervjuu2018080910.mp3"
params.do_music_detection = "no" // yes or no 
params.fileext = "mp3"
params.srcdir = "/opt/kaldi-offline-transcriber"
params.acoustic_model = "tdnn_7d_online"
params.njobs = 1
params.nthreads = 1
params.decode_cmd = "run.pl"
params.kaldi_root = "/opt/kaldi"
audio_file = file(params.in)


process transcribe {
    input:
    path audio_file

    output:
    file 'audio.wav' into audio
    
    script:
    if( params.fileext == 'wav' )
        """
        sox $params.in -c 1 audio.wav rate -v 16k
        """

    else if( params.fileext == 'mp3' || params.fileext == 'm4a' || params.fileext == 'mp4' )
        """
        ffmpeg -i $params.in -f sox - | sox -t sox - -c 1 -b 16 -t wav audio.wav rate -v 16k	
        """

    else if( params.fileext == 'ogg' || params.fileext == 'mp2' || params.fileext == 'flac' )
        """
        sox $params.in -c 1 audio.wav rate -v 16k
        """

    else
        error "Invalid alignment params.fileext: $params.fileext"

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
        cd \$WORK_DIR
        sleep 2
        echo \$PWD
        ls -l
        cat show.seg > show1.seg
        """
}

process scp {
    input:
    file audio

    output:
    file 'wav.scp' into scp

    script:
    "echo \"audio audio.wav\" > wav.scp"
}

process reco2file_and_channel {
    output:
    file "reco2file_and_channel" into reco2file_and_channel

    script:
    "echo audio audio A > reco2file_and_channel"
}

process segments {
    input:
    file show_seg
    file reco2file_and_channel

    output:
    file 'segments' into segments

    script:
    """
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
	
    """
}

process utt2spk {
    input:
    file segments

    output:
    file 'utt2spk' into utt2spk

    script:
    """
    cat $segments | perl -npe 's/\\s+.*//; s/((.*)---.*)/\\1 \\2/' > utt2spk
    """

}

process spk2utt {
    input:
    file utt2spk

    output:
    file 'spk2utt' into spk2utt

    script:
    """
    ${params.srcdir}/utils/utt2spk_to_spk2utt.pl $utt2spk > spk2utt
    """
}

process mfcc {
    input:
    file spk2utt
    file utt2spk
    file scp
    file segments
    file audio

    output:
    file 'feats.scp' into feats

    script:
    """
    ln -s ${params.kaldi_root}/egs/wsj/s5/utils
    ln -s ${params.kaldi_root}/egs/wsj/s5/steps
    make_mfcc.sh \
        --mfcc-config ${params.srcdir}/build/fst/${params.acoustic_model}/conf/mfcc.conf \
        --cmd ${params.decode_cmd} \
        --nj ${params.njobs} \
        \$PWD || exit 1
	steps/compute_cmvn_stats.sh \$PWD || exit 1
	utils/fix_data_dir.sh \$PWD
    """
}

process ivectors {
    input:
    file feats
    file utt2spk

    output:
    file 'ivectors/*' into ivectors

    script:
    """
    ln -s ${params.kaldi_root}/egs/wsj/s5/steps
    ln -s ${params.kaldi_root}/egs/wsj/s5/utils
    ln -s ${params.srcdir}/build

    steps/online/nnet2/extract_ivectors_online.sh --cmd \"${params.decode_cmd}\" --nj ${params.njobs} \
		. build/fst/${params.acoustic_model}/ivector_extractor ivectors || exit 1;
    """
}

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

process restore_lattices {
    // Rescore lattices with a larger language model
    input: 
    path pruned from pruned_unk

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