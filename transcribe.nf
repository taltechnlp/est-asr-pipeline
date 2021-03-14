#!/usr/bin/env nextflow

params.in = "$HOME/src-audio/test.mp3"
params.do_music_detection = "no" // yes or no 
params.fileext = "mp3"
params.srcdir = "/home/aivo_olevi/dev/kaldi-offline-transcriber"
params.acoustic_model = ""
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
    "echo audio $audio > wav.scp"
}

process reco2file_and_channel {
    output:
    file "reco2file_and_channel" into reco2file_and_channel

    script:
    "echo audio audio A > reco2file_and_channel"
}

process segments {
    input:
    file show_seg from show_seg
    file reco2file_and_channel from reco2file_and_channel

    output:
    file 'segments' into segments

    script:
    """
    cat $show_seg | cut -f 3,4,8 -d " " | \
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
    cat $segments | perl -npe 's/\s+.*//; s/((.*)---.*)/\1 \2/' > utt2spk
    """

}

process spk2utt {
    input:
    file utt2spk

    output:
    file 'spk2utt' into spk2utt

    script:
    """
    WORK_DIR=\$PWD
    cd $params.srcdir
    utils/utt2spk_to_spk2utt.pl \${WORK_DIR}/$utt2spk > spk2utt
    """
}

process mfcc {
    input:
    file spk2utt

    script:
    """
    steps/make_mfcc.sh --mfcc-config build/fst/$(ACOUSTIC_MODEL)/conf/mfcc.conf --cmd "$$decode_cmd" --nj $(njobs) \
		build/trans/$* build/trans/$*/exp/make_mfcc $@ || exit 1
	steps/compute_cmvn_stats.sh build/trans/$* build/trans/$*/exp/make_mfcc $@ || exit 1
	utils/fix_data_dir.sh build/trans/$*
    """
}

// seg .subscribe { println "File: ${it.name} => ${it.text}" }
