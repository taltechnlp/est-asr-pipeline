#!/bin/bash

# Begin configuration section.
stage=1

echo "$0 $@"  # Print the command line for logging

if [ -z "$KALDI_ROOT" ]; then
  echo "Please set KALDI_ROOT"
  exit 1;
fi

rm -f steps utils sid rnnlm
ln -s $KALDI_ROOT/egs/wsj/s5/steps
ln -s $KALDI_ROOT/egs/wsj/s5/utils
ln -s $KALDI_ROOT/egs/sre08/v1/sid
ln -s $KALDI_ROOT/scripts/rnnlm


am=cnn_tdnn_1d_online
lexicon=language_model/lexicon.txt
pruned_lm=language_model/interpolated.pruned9.4g.arpa.gz
rnnlm_dir=language_model/rnnlm
compounder_lm=language_model/compounder.pruned9.4g.arpa.gz

. ./utils/parse_options.sh

export PATH=${KALDI_ROOT}/tools/openfst/bin:utils:${KALDI_ROOT}/src/bin:${KALDI_ROOT}/tools/openfst/bin:${KALDI_ROOT}/src/fstbin/:${KALDI_ROOT}/src/lmbin/:${KALDI_ROOT}/src/rnnlmbin/:$PATH

set -e
set -u
set -o pipefail

if [ $stage -le 1 ]; then
    rm -rf build/fst/${am}
    mkdir -p build/fst/${am}
    cp -r kaldi-data/${am}/* build/fst/${am}
    perl -i -npe 's#=.*online/#=build/fst/${am}/#' build/fst/${am}/conf/*.conf
    if [ ! -e build/fst/${am}/cmvn_opts ]; then \
        echo "--norm-means=false --norm-vars=false" > build/fst/${am}/cmvn_opts; \
    fi
fi


if [ $stage -le 2 ]; then
  # Make a lexicon from the user-provided lexicon
    rm -rf build/fst/data/dict
    mkdir -p build/fst/data/dict
    cp -r kaldi-data/dict/* build/fst/data/dict
    rm -f build/fst/data/dict/lexicon.txt build/fst/data/dict/lexiconp.txt
    cat kaldi-data/dict/lexicon.txt  | egrep "^<" > build/fst/data/dict/lexicon.txt  
    cat $lexicon | perl -npe 's/\(\d\)(\s)/\1/' >> build/fst/data/dict/lexicon.txt  
fi


if [ $stage -le 3 ]; then
    echo "Constructing UNK word LM"
    rm -rf build/fst/data/unk_lang_model
    utils/lang/make_unk_lm.sh --cmd utils/run.pl build/fst/data/dict build/fst/data/unk_lang_model
fi

if [ $stage -le 4 ]; then
    echo "Making build/fst/data/prunedlm/G.fst from the provided ARPA LM"
    rm -rf build/fst/data/prunedlm
    mkdir -p build/fst/data/prunedlm
    utils/prepare_lang.sh --phone-symbol-table build/fst/$am/phones.txt build/fst/data/dict '<unk>' build/fst/data/dict/tmp build/fst/data/prunedlm
    gunzip -c $pruned_lm | arpa2fst --disambig-symbol=#0 \
        --read-symbol-table=build/fst/data/prunedlm/words.txt - build/fst/data/prunedlm/G.fst
    echo "Checking how stochastic G is (the first of these numbers should be small):"
    fstisstochastic build/fst/data/prunedlm/G.fst || echo "not stochastic (probably OK)"    
    utils/validate_lang.pl build/fst/data/prunedlm || exit 1
fi


if [ $stage -le 5 ]; then
    echo "Making build/fst/data/prunedlm_unk from build/fst/data/prunedlm"
    rm -rf build/fst/data/prunedlm_unk
    utils/prepare_lang.sh --unk-fst build/fst/data/unk_lang_model/unk_fst.txt build/fst/data/dict "<unk>" build/fst/data/prunedlm build/fst/data/prunedlm_unk
    cp build/fst/data/prunedlm/G.fst build/fst/data/prunedlm_unk
fi


if [ $stage -le 6 ]; then
    echo "Compiling decoding graph"
    rm -rf build/fst/build/fst/${am}/graph_prunedlm_unk
    self_loop_scale_arg=""
    if [ -f build/fst/${am}/frame_subsampling_factor ]; then
      factor=`cat build/fst/${am}/frame_subsampling_factor`
      if [ $factor -eq "3" ]; then
        self_loop_scale_arg="--self-loop-scale 1.0 "
      fi
    fi
    utils/mkgraph.sh $self_loop_scale_arg build/fst/data/prunedlm_unk build/fst/${am} build/fst/${am}/graph_prunedlm_unk
    rm -rf build/fst/data/prunedlm_unk/tmp
fi


if [ $stage -le 7 ]; then
    echo "Preparing RNNLM"
    rm -rf build/fst/data/rnnlm_unk
    cp $rnnlm_dir/config/unigram_probs.txt $rnnlm_dir/unigram_probs.txt
    rnnlm/change_vocab.sh build/fst/data/prunedlm/words.txt \
        $rnnlm_dir build/fst/data/rnnlm_unk
fi    
    

if [ $stage -le 8 ]; then
    echo "Converting compounder LM to FST"
    rm -rf build/fst/data/compounderlm
    mkdir -p build/fst/data/compounderlm
    cat $lexicon | perl -npe 's/(\(\d\))?\s.+//' | uniq | ./bin/make-compounder-symbols.py > build/fst/data/compounderlm/words.txt
    zcat $compounder_lm | \
        grep -v '<s> <s>' | \
        grep -v '</s> <s>' | \
        grep -v '</s> </s>' | \
        arpa2fst --disambig-symbol='#0' --read-symbol-table=build/fst/data/compounderlm/words.txt - > arpa2fst_result.fst
    cat arpa2fst_result.fst  | fstproject --project_type=output | fstarcsort --sort_type=ilabel > build/fst/data/compounderlm/G.fst 
    rm arpa2fst_result.fst
fi
