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
    file 'audio.wav' glob false into audio
    
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
    file 'show.seg' into show_seg optional true
    file 'show.uem.seg' into show_uem_seg 

    script:        
        """
        find_speech_segments.py $audio show.uem.seg
        if [ -s show.uem.seg ]; then
          diarization.sh $audio show.uem.seg
        fi
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
        start=`echo $LINE | cut -f 1,2 -d " " | perl -ne '@t=split(); $start=$t[0]/100.0; printf("%09.3f", $start);'`; \
        end=`echo $LINE   | cut -f 1,2 -d " " | perl -ne '@t=split(); $start=$t[0]/100.0; $len=$t[1]/100.0; $end=$start+$len; printf("%09.3f", $end);'`; \
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
    memory '2GB'
    label 'with_gpu'
    
   
    input:
      path datadir
      file audio    

    output:
      file 'sid-result.json' into sid_result

    shell:
        if (params.do_speaker_id)
          '''
          spk_id.sh --device 1 --accelerator gpu --load-checkpoint !{params.rootdir}/models/spk_id/version_295816/checkpoints/last.ckpt \
            --predict-datadir datadir --predictions-file spk_preds.txt --segments-per-spk 10
          
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
      path 'datadir_25sec' into datadir_25sec optional true
      
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
      path datadir_25sec
      file audio    
      
    output:
      file 'trans.hyp' into trans_hyp
      
    shell:
      '''
      whisper-transcribe.py --beam 1 --is_multilingual --language et !{params.rootdir}/models/whisper-large-et-v3-ct2 !{audio} datadir_25sec/segments | tee trans.hyp
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
      path datadir_25sec
      file trans_hyp    
      file audio    

    output:
      file 'alignments.json' into alignments_json
      file 'alignments.ctm' into alignments_ctm

    shell:
      '''
      . !{projectDir}/bin/prepare_process.sh

      utils/copy_data_dir.sh !{datadir_25sec} datadir_25sec_hires
      cat !{trans_hyp} | perl -npe 's/ [[:punct:]]/ /g;' >  datadir_25sec_hires/text      
      steps/make_mfcc.sh --nj 1 \
          --mfcc-config !{params.rootdir}/kaldi-data/!{params.acoustic_model}/conf/mfcc.conf \
          datadir_25sec_hires || exit 1
      steps/compute_cmvn_stats.sh datadir_25sec_hires || exit 1
      utils/fix_data_dir.sh datadir_25sec_hires

      steps/online/nnet2/extract_ivectors_online.sh  --nj 1 \
        datadir_25sec_hires !{params.rootdir}/kaldi-data/!{params.acoustic_model}/ivector_extractor ivectors || exit 1;

      mkdir dict
      cp -r !{params.rootdir}/kaldi-data/dict .
      rm -f dict/lexicon*.txt
      echo "<unk>   UNK" > dict/lexicon.txt
      echo "<sil>   SIL" >> dict/lexicon.txt

      cat datadir_25sec_hires/text | perl -npe 's/\\S+\\s//; s/\\s+/\\n/g;' | grep -v "^\\s*$" | sort | uniq | ${ET_G2P_ROOT}/run.sh |  perl -npe 's/\\(\\d\\)(\\s)/\\1/' | \
        perl -npe 's/\\b(\\w+) \\1\\b/\\1\\1 /g; s/(\\s)jj\\b/\\1j/g; s/(\\s)tttt\\b/\\1tt/g; s/(\\s)pppp\\b/\\1pp/g;  ' | uniq >> dict/lexicon.txt

      utils/prepare_lang.sh --phone-symbol-table !{params.rootdir}/kaldi-data/!{params.acoustic_model}/phones.txt --sil-prob 0.01 dict '<unk>' dict/tmp lang
      cp -r !{params.rootdir}/kaldi-data/!{params.acoustic_model}/conf .      

      steps/nnet3/align.sh --retry-beam 100 --use-gpu true --nj 1 --online-ivector-dir ivectors \
        --scale-opts '--transition-scale=1.0 --self-loop-scale=1.0 --acoustic-scale=1.0' \
        datadir_25sec_hires lang !{params.rootdir}/kaldi-data/!{params.acoustic_model} ali
        
      steps/get_train_ctm.sh --frame-shift 0.03 --use-segments false \
        datadir_25sec_hires lang ali .

      ctm2json.py ctm datadir_25sec_hires/segments !{trans_hyp} > alignments.json
      utils/convert_ctm.pl datadir_25sec_hires/segments <(echo audio audio 1) ctm > alignments.ctm
              
      '''
}




process to_json {    
    memory '500MB'
    
    input:
      path datadir
      file show_uem_seg
      file sid_result
      file alignments_json

    output:
      file "punctuated.json" into punctuated_json

    shell:
      if (params.do_speaker_id)
          """
          # foo
          python3 !{projectDir}/bin/resegment_json.py --new-turn-sil-length 1.0 --speaker-names !{sid_result} --pms-seg !{show_uem_seg} !{datadir} !{alignments_json} > punctuated.json
          """
      else 
          """
          python3 !{projectDir}/bin/resegment_json.py --new-turn-sil-length 1.0 --pms-seg !{show_uem_seg} !{datadir} !{alignments_json} > punctuated.json
          """

}

/*

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
}*/




process output {
    memory '500MB'
    
    publishDir "${out_dir}${audio_file.baseName}", mode: 'copy', overwrite: true

    input:
      file punctuated_json
      file alignments_ctm

    output:
      file "result.json" into result_json
      file "result.srt" into result_srt
      file "result.trs" into result_trs
      file "result.ctm" into result_ctm

    script:
      json = punctuated_json
      """
      normalize_json.py words2numbers.py $punctuated_json > result.json

      json2trs.py --fid trs $punctuated_json > result.trs
      json2srt.py result.json > result.srt
      cp -L $alignments_ctm result.ctm
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


