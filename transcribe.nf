#!/usr/bin/env nextflow

def env = System.getenv()
params.in = "/home/aivo_olevi/tmp/ao_1.wav"
params.out_dir = "results/"
params.do_speaker_id = true
params.do_punctuation = true
params.do_language_id = true
audio_file = file(params.in)

out_dir = ""
if (params.out_dir[-1] != "/") {
  out_dir = params.out_dir + "/"
}
else out_dir = params.out_dir


process to_wav {
    memory '500MB'
    cpus 2

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
    file 'show.uem.seg' into show_uem_seg 

    script:        
        """
        find_speech_segments.py $audio show.uem.seg
        diarization.sh $audio show.uem.seg
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
        len_in_frames=`echo $LINE | cut -f 2 -d " "`; \
        start=`echo $LINE | cut -f 1,2 -d " " | perl -ne '@t=split(); $start=$t[0]/100.0; printf("%08.3f", $start);'`; \
        end=`echo $LINE   | cut -f 1,2 -d " " | perl -ne '@t=split(); $start=$t[0]/100.0; $len=$t[1]/100.0; $end=$start+$len; printf("%08.3f", $end);'`; \
        sp_id=`echo $LINE | cut -f 3 -d " "`; \
        if  [ ${len_in_frames} -gt 30 ]; then \
          echo audio-${sp_id}---${start}-${end} audio $start $end; \
        fi
      done > init_datadir/segments
      cat init_datadir/segments | perl -npe 's/\\s+.*//; s/((.*)---.*)/\\1 \\2/' > init_datadir/utt2spk
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
      file audio    
    
    output:
      path 'datadir' into datadir optional true
    
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
    memory '5GB'
    
    cpus params.nthreads
    
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



process decode {
    memory '16GB'
    cpus 1
    label 'with_gpu'

    input:
      path datadir
      file audio    
      
    output:
      file 'trans.hyp' into trans_hyp
      
    shell:
      '''
      . !{projectDir}/bin/prepare_process.sh
      
      extract_segments_to_wavs.sh --nj 1 !{datadir} data_segmented data_segmented/wav
      
      echo "." > test.tsv
      
      awk \'{print($2)}\' data_segmented/wav.scp > wav_locations.tmp
      
      awk \'{print($2)}\' data_segmented/wav.scp | xargs soxi -s | paste wav_locations.tmp - >> test.tsv
      
      awk \'{print("")}\' data_segmented/wav.scp > test.ltr
      
      python ${FAIRSEQ_ROOT}/examples/speech_recognition/infer.py \
        . \
        --batch-size 1 \
        --task audio_finetuning  \
        --path !{params.rootdir}/models/fairseq/xls-r-960M-1h/checkpoint_best.pt \
        --results-path . \
        --w2l-decoder kenlm --lm-weight 1.0 --word-score 0.0 --sil-weight 0 --lm-model  !{params.rootdir}/models/fairseq/xls-r-960M-1h/interpolated.4g.arpa.bin \
        --criterion ctc --labels ltr \
        --max-tokens 4000000 --post-process sentencepiece \
        --gen-subset test \
        --lexicon !{params.rootdir}/models/fairseq/xls-r-960M-1h/lexicon \
        --beam 20 --normalize 	      
      
      cut -f 1 -d " " data_segmented/wav.scp > uttids
      python ${FAIRSEQ_ROOT}/scripts/spm_decode.py --model !{params.rootdir}/models/fairseq/xls-r-960M-1h/spm.model --input hypo.units-checkpoint_best.pt-test.txt | \
        perl -npe \'s/\\|/ /g;\' | \
        perl -npe \'s/(.*)(\\(\\S+\\))$/\\2 \\1/\' | perl -npe \'s/\\(None-(\\d+)\\)/\\1/\' | sort -n | cut -f 2- -d \' \' | \
        paste uttids - > trans.hyp

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
        --mfcc-config !{params.rootdir}/kaldi-data/!{params.acoustic_model}/conf/mfcc.conf \
        datadir_hires || exit 1
    steps/compute_cmvn_stats.sh datadir_hires || exit 1
    utils/fix_data_dir.sh datadir_hires

    steps/online/nnet2/extract_ivectors_online.sh  --nj 1 \
      datadir_hires !{params.rootdir}/kaldi-data/!{params.acoustic_model}/ivector_extractor ivectors || exit 1;
    '''
}

process hyp2ctm {
    memory '4GB'
    cpus 1
    label 'with_gpu'
    
    input:
      file trans_hyp
      path datadir_hires
      path ivectors

    output:
      file 'segmented.ctm' into segmented_ctm

    shell:
      '''
      . !{projectDir}/bin/prepare_process.sh
      mkdir dict
    	cp -r !{params.rootdir}/kaldi-data/dict .
      rm -f dict/lexicon*.txt
      echo "<unk>   UNK" > dict/lexicon.txt
      echo "<sil>   SIL" >> dict/lexicon.txt
      cat !{trans_hyp}  | perl -npe 's/\\S+\\s//; s/\\s+/\\n/g;' | grep -v "^\\s*$" | sort | uniq | ${ET_G2P_ROOT}/run.sh |  perl -npe 's/\\(\\d\\)(\\s)/\\1/' | \
        perl -npe 's/\\b(\\w+) \\1\\b/\\1\\1 /g; s/(\\s)jj\\b/\\1j/g; s/(\\s)tttt\\b/\\1tt/g;' | uniq >> dict/lexicon.txt
      
      utils/prepare_lang.sh --phone-symbol-table !{params.rootdir}/kaldi-data/!{params.acoustic_model}/phones.txt --sil-prob 0.01 dict '<unk>' dict/tmp lang
      cp -r !{params.rootdir}/kaldi-data/!{params.acoustic_model}/conf .
      utils/copy_data_dir.sh !{datadir_hires} datadir_hires_hyp
      
      cp !{trans_hyp} datadir_hires_hyp/text
      steps/nnet3/align.sh --use-gpu true --nj 1 --online-ivector-dir !{ivectors} \
        --scale-opts '--transition-scale=1.0 --self-loop-scale=1.0 --acoustic-scale=1.0' \
        datadir_hires_hyp lang !{params.rootdir}/kaldi-data/!{params.acoustic_model} ali
        
      steps/get_train_ctm.sh --frame-shift 0.03 --use-segments false \
        datadir_hires_hyp lang ali .
      cat ctm | perl -npe "s/(.*)-(S\\d+)---(\\S+)/\\1_\\3_\\2/" | sort -k1,1 -k 3,3g >  segmented.ctm
        
      
      '''
}



process to_json {    
    memory '500MB'
    
    input:
      file segmented_ctm
      file sid_result
      file show_uem_seg

    output:
      file "unpunctuated.json" into unpunctuated_json

    shell:
      if (params.do_speaker_id)
          """
          python3 !{projectDir}/bin/segmented_ctm2json.py --speaker-names !{sid_result} --pms-seg !{show_uem_seg} !{segmented_ctm} > unpunctuated.json
          """
      else 
          """
          python3 !{projectDir}/bin/segmented_ctm2json.py --pms-seg !{show_uem_seg} !{segmented_ctm} > unpunctuated.json
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
        TEMP_FILE1=$WORK_DIR/tmp1
        TEMP_FILE2=$WORK_DIR/tmp2
        cat $WORK_DIR/!{unpunctuated_json} > $TEMP_FILE1 
        python2 punctuator_pad_emb_json.py Model_stage2p_final_563750_h256_lr0.02.pcl $TEMP_FILE1 $TEMP_FILE2  
        cat $TEMP_FILE2 > $WORK_DIR/punctuated.json
        #rm TEMP_FILE1 TEMP_FILE2 
        cd $WORK_DIR
        '''
      else
        '''
        cp !{unpunctuated_json} punctuated.json
        '''
}




process output {
    memory '500MB'
    
    publishDir "${out_dir}${audio_file.baseName}", mode: 'copy', overwrite: true

    input:
      file punctuated_json
      file segmented_ctm

    output:
      file "result.json" into result_json
      file "result.srt" into result_srt
      file "result.trs" into result_trs
      file "result.ctm" into result_ctm

    script:
      json = punctuated_json
      """
      normalize_json.py words2numbers.py $json > result.json

      cat $segmented_ctm | unsegment-ctm.py | LC_ALL=C sort -k 1,1 -k 3,3n -k 4,4n > result.ctm
      json2trs.py --fid trs $json > result.trs
      json2srt.py result.json > result.srt
      """
}

process empty_output {

    publishDir "${out_dir}/${audio_file.baseName}", mode: 'copy', overwrite: true

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

    script:
      json = file("assets/empty.json")
      with_compounds_ctm = file("assets/empty.ctm")
      """
      cp $json result.json
      json2trs.py $json > result.trs      
      touch result.srt result.ctm 
      """
}

