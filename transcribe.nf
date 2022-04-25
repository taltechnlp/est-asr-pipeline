#!/usr/bin/env nextflow

def env = System.getenv()
params.in = "/home/aivo_olevi/tmp/ao_1.wav"
params.do_music_detection = true
params.do_speaker_id = true
params.do_punctuation = true
params.do_language_id = true
audio_file = file(params.in)


process to_wav {
    memory '500MB'

    input:
    path audio_file

    output:
    file 'audio.wav' into audio
    
    shell:
        """
        ffmpeg -i !{audio_file} -f sox - | sox  -t sox - -c 1 -b 16 -t wav audio.wav rate -v 16k 
        """

}

process diarization {
    memory '5GB'
    
    input:
    file audio

    output: 
    file 'show.seg' into show_seg
    file 'show.pms.seg' into show_pms_seg 

    script:
        diarization_opts = params.do_music_detection ? '-m' : '' 
        """
        echo "audio 1 0 1000000000 U U U 1" > show.uem.seg
        diarization.sh $diarization_opts $audio show.uem.seg
        """
}

process prepare_initial_data_dir {
    memory '1GB'

    input:
      file show_seg
      file audio

    output:
      path 'init_datadir' into init_datadir optional true

    shell:
    '''
    . !{projectDir}/bin/prepare_process.sh
    
    if [ -s !{show_seg} ]; then
      mkdir init_datadir
      echo audio audio A > init_datadir/reco2file_and_channel
      cat !{show_seg} | cut -f "3,4,8" -d " " | \
      while read LINE ; do \
        start=`echo $LINE | cut -f 1,2 -d " " | perl -ne '@t=split(); $start=$t[0]/100.0; printf("%08.3f", $start);'`; \
        end=`echo $LINE   | cut -f 1,2 -d " " | perl -ne '@t=split(); $start=$t[0]/100.0; $len=$t[1]/100.0; $end=$start+$len; printf("%08.3f", $end);'`; \
        sp_id=`echo $LINE | cut -f 3 -d " "`; \
        echo audio-${sp_id}---${start}-${end} audio $start $end; \
      done > init_datadir/segments
      cat init_datadir/segments | perl -npe 's/\\s+.*//; s/((.*)---.*)/\\1 \\2/' > init_datadir/utt2spk
      utils/utt2spk_to_spk2utt.pl init_datadir/utt2spk > init_datadir/spk2utt
      echo "audio !{audio}" > init_datadir/wav.scp
    fi
    '''
}

process language_id {
    memory '4GB'
    
    input:
      path init_datadir
      file audio    
    
    output:
      path 'datadir' into datadir optional true
    
    shell:
    if ( params.do_language_id )
      '''
      . !{projectDir}/bin/prepare_process.sh
      
      extract_lid_features_kaldi.py !{init_datadir} .
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
    


process mfcc {
    memory '1GB'
    
    input:
    path datadir
    file audio    

    output:
    path 'datadir_hires' into datadir_hires
    path 'ivectors' into ivectors

    shell:
    '''
    . !{projectDir}/bin/prepare_process.sh
	
    utils/copy_data_dir.sh !{datadir} datadir_hires
    steps/make_mfcc.sh --nj 1 \
        --mfcc-config !{params.rootdir}/build/fst/!{params.acoustic_model}/conf/mfcc.conf \
        datadir_hires || exit 1
    steps/compute_cmvn_stats.sh datadir_hires || exit 1
    utils/fix_data_dir.sh datadir_hires

    steps/online/nnet2/extract_ivectors_online.sh  --nj 1 \
      datadir_hires !{params.rootdir}/build/fst/!{params.acoustic_model}/ivector_extractor ivectors || exit 1;
    '''
}





process speaker_id {
    memory '5GB'
    
    cpus "${params.nthreads}"
    
    input:
      path datadir
      file audio    
      path conf from "$projectDir/conf"

    output:
      file 'sid-result.json' into sid_result

    shell:
        if (params.do_speaker_id)
          '''
          . !{projectDir}/bin/prepare_process.sh
          ln -v -s !{params.rootdir}/kaldi-data
          ln -v -s !{params.rootdir}/build

          utils/copy_data_dir.sh !{datadir} datadir_sid
          
          # MFCC for Speaker ID, since the features for MFCC are different from speech recognition
          steps/make_mfcc.sh --nj 1 \
              --mfcc-config kaldi-data/sid/mfcc_sid.conf \
              datadir_sid || exit 1
          steps/compute_cmvn_stats.sh datadir_sid || exit 1
          sid/compute_vad_decision.sh --nj 1 datadir_sid || exit 1

          # i-vectors for each speaker in our audio file
          sid/extract_ivectors.sh  --nj 1 --num-threads !{params.nthreads} \
              kaldi-data/sid/extractor_2048 datadir_sid .

          # cross-product between trained speakers and diarized speakers
          join -j 2 \
              <(cut -d " " -f 1 kaldi-data/sid/name_ivector.scp | sort ) \
              <(cut -d " " -f 1 spk_ivector.scp | sort ) > trials

          ivector-plda-scoring --normalize-length=true \
              "ivector-copy-plda --smoothing=0.3 kaldi-data/sid/lda_plda - |" \
              "ark:ivector-subtract-global-mean scp:kaldi-data/sid//name_ivector.scp ark:- | transform-vec kaldi-data/sid/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
              "ark:ivector-subtract-global-mean kaldi-data/sid/mean.vec scp:spk_ivector.scp ark:- | transform-vec kaldi-data/sid/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
              trials lda_plda_scores
          
	        cat lda_plda_scores | sort -k2,2 -k3,3nr | awk \'{print $3, $1, $2}\' | uniq -f2 | awk \'{if ($1 > !{params.sid_similarity_threshold}) {print $3, $2}}\' | \
	        perl -npe \'s/^\\S+-(S\\d+)/\\1/; s/_/ /g;\' | python -c \'import json, sys; spks={s.split()[0]:{"name" : " ".join(s.split()[1:])} for s in sys.stdin}; json.dump(spks, sys.stdout);\' > sid-result.json
          '''
        else
          '''
          echo "{}" >  sid-result.json
          '''
}



// Do 1-pass decoding using chain online models
process one_pass_decoding {
    memory '5GB'  
    cpus "${params.nthreads}"
    
    // Do 1-pass decoding using chain online models
    input:
      path datadir_hires
      path ivectors
      //path build from params.build_dir 
      path conf from "$projectDir/conf"
	
    output:
      path "${params.acoustic_model}_pruned_unk" into pruned_unk
   
    
    shell:
      '''
      . !{projectDir}/bin/prepare_process.sh
      
      mkdir -p !{params.acoustic_model}_pruned_unk
      for f in !{params.rootdir}/build/fst/!{params.acoustic_model}/?*; do ln -s $f !{params.acoustic_model}_pruned_unk/; done

      steps/nnet3/decode.sh  --nj 1 --num-threads !{params.nthreads} --acwt 1.0  --post-decode-acwt 10.0 \
        --skip-scoring true \
        --online-ivector-dir ivectors \
        --skip-diagnostics true \
        !{params.rootdir}/build/fst/!{params.acoustic_model}/graph_prunedlm_unk datadir_hires !{params.acoustic_model}_pruned_unk/decode || exit 1;
      ln -s !{params.rootdir}/build/fst/!{params.acoustic_model}/graph_prunedlm_unk !{params.acoustic_model}_pruned_unk/graph
        
      '''
}


process rnnlm_rescoring {
    memory '5GB'
    
    input:
    
      path pruned_unk from pruned_unk
      path datadir_hires

    output:
      path "${params.acoustic_model}_pruned_rnnlm_unk" into pruned_rnnlm_unk

    shell:
      '''
      . !{projectDir}/bin/prepare_process.sh
      
      mkdir -p !{params.acoustic_model}_pruned_rnnlm_unk	
      for f in !{params.rootdir}/build/fst/!{params.acoustic_model}/*; do ln -s $f !{params.acoustic_model}_pruned_rnnlm_unk/; done
        rnnlm/lmrescore_pruned.sh \
        --skip-scoring true \
        --max-ngram-order 4 \
        !{params.rootdir}/build/fst/data/prunedlm_unk \
        !{params.rootdir}/build/fst/data/rnnlm_unk \
        datadir_hires \
      !{pruned_unk}/decode \
        !{params.acoustic_model}_pruned_rnnlm_unk/decode
      cp -r --preserve=links !{params.acoustic_model}_pruned_unk/graph !{params.acoustic_model}_pruned_rnnlm_unk/
      '''
}



process lattice2ctm {
    memory '4GB'
    
    input:
      path pruned_rnnlm_unk from pruned_rnnlm_unk
      path datadir_hires

    output:
      file 'segmented.ctm' into segmented_ctm
      file 'with-compounds.ctm' into with_compounds_ctm

    shell:
      '''
      . !{projectDir}/bin/prepare_process.sh
      
      frame_shift_opt=""
      if [ -f !{pruned_rnnlm_unk}/frame_subsampling_factor ]; then
        factor=`cat !{pruned_rnnlm_unk}/frame_subsampling_factor`
        frame_shift_opt="--frame-shift 0.0$factor"; 
      fi; 
      
      get_ctm_unk.sh --use_segments false $frame_shift_opt \
        --unk-p2g-cmd "python3 !{projectDir}/bin/unk_p2g.py --p2g-cmd \'python3 /opt/et-g2p-fst/g2p.py --inverse --fst /opt/et-g2p-fst/data/chars.fst --nbest 1\'" \
        --unk-word "<unk>" \
        --min-lmwt !{params.lm_scale} \
        --max-lmwt !{params.lm_scale} \
        datadir_hires !{pruned_rnnlm_unk}/graph !{pruned_rnnlm_unk}/decode

      cat !{pruned_rnnlm_unk}/decode/score_!{params.lm_scale}/datadir_hires.ctm  | perl -npe "s/(.*)-(S\\d+)---(\\S+)/\\1_\\3_\\2/" > segmented.splitw2.ctm

      python3 !{projectDir}/bin/compound-ctm.py \
        "python3 !{projectDir}/bin/compounder.py !{params.rootdir}/build/fst/data/compounderlm/G.fst !{params.rootdir}/build/fst/data/compounderlm/words.txt" \
        < segmented.splitw2.ctm > with-compounds.ctm
      
      cat with-compounds.ctm | grep -v "++" |  grep -v "\\[sil\\]" | grep -v -e " \$" | perl -npe "s/\\+//g" | sort -k1,1 -k 3,3g > segmented.ctm    
      '''
}



process to_json {    
    memory '500MB'
    
    input:
      file segmented_ctm
      file with_compounds_ctm
      file sid_result
      file show_pms_seg

    output:
      file "unpunctuated.json" into unpunctuated_json

    shell:
      if (params.do_speaker_id && params.do_music_detection)
          """
          python3 !{projectDir}/bin/segmented_ctm2json.py --speaker-names !{sid_result} --pms-seg !{show_pms_seg} !{segmented_ctm} > unpunctuated.json
          """
      else if (params.do_speaker_id && !params.do_music_detection )
          """
          python3 !{projectDir}/bin/segmented_ctm2json.py --speaker-names !{sid_result} !{segmented_ctm} > unpunctuated.json
          """
      else if (!params.do_speaker_id  && params.do_music_detection)
          """
          python3 !{projectDir}/bin/segmented_ctm2json.py --pms-seg !{show_pms_seg} !{segmented_ctm} > unpunctuated.json
          """
      else
          """
          python3 !{projectDir}/bin/segmented_ctm2json.py !{segmented_ctm} > unpunctuated.json
          """
}


process punctuation {
    memory '4GB'
    
    input:
      file unpunctuated_json

    output:
      file "punctuated.json" into punctuated_json

    shell:
      if (params.do_punctuation)
        '''
        WORK_DIR=$PWD
        cd !{params.rootdir}/punctuator-data/est_punct2
        TEMP_FILE1=$(mktemp); 
        TEMP_FILE2=$(mktemp);
        cat $WORK_DIR/!{unpunctuated_json} > TEMP_FILE1 
        python2 punctuator_pad_emb_json.py Model_stage2p_final_563750_h256_lr0.02.pcl TEMP_FILE1 TEMP_FILE2  
        cat TEMP_FILE2 > $WORK_DIR/punctuated.json
        rm TEMP_FILE1 TEMP_FILE2 
        cd $WORK_DIR
        '''
      else
        '''
        cp !{unpunctuated_json} punctuated.json
        '''
}




process output {
    memory '500MB'
    
    publishDir "results/${audio_file.baseName}", mode: 'copy', overwrite: true

    input:
      file with_compounds_ctm
      file punctuated_json

    output:
      file "result.json" into result_json
      file "result.srt" into result_srt
      file "result.trs" into result_trs
      file "result.ctm" into result_ctm
      file "result.with-compounds.ctm" into result_with_compounds_ctm

    script:
      json = punctuated_json
      """
      normalize_json.py words2numbers.py $json > result.json

      cat $with_compounds_ctm | unsegment-ctm.py | LC_ALL=C sort -k 1,1 -k 3,3n -k 4,4n > with-compounds.synced.ctm
      cat with-compounds.synced.ctm | grep -v \"<\" > result.ctm
      cat result.ctm | ctm2with-sil-ctm.py > result.with-compounds.ctm
      json2trs.py --fid trs $json > result.trs
      json2srt.py result.json > result.srt
      """
}

process empty_output {

    publishDir "results/${audio_file.baseName}", mode: 'copy', overwrite: true

    input:
      val a from datadir.ifEmpty{ 'EMPTY' }
      val b from init_datadir.ifEmpty{ 'EMPTY' }
    when:
      a == 'EMPTY' || b == 'EMPTY' 

    output:
      file "result.json" into empty_result_json
      file "result.srt" into empty_result_srt
      file "result.trs" into empty_result_trs
      file "result.ctm" into empty_result_ctm
      file "result.with-compounds.ctm" into empty_result_with_compounds_ctm

    script:
      json = file("assets/empty.json")
      with_compounds_ctm = file("assets/empty.ctm")
      """
      cp $json result.json
      json2trs.py $json > result.trs      
      touch result.srt result.ctm result.with-compounds.ctm
      """
}

