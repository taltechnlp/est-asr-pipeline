#!/bin/bash

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

# Function to validate FST files
validate_fst() {
    local fst_file=$1
    if [ ! -f "$fst_file" ]; then
        echo "Error: FST file $fst_file not found."
        exit 1
    fi

    if ! fstinfo "$fst_file" > /dev/null; then
        echo "Error: FST file $fst_file is corrupt or has an invalid format."
        exit 1
    fi

    echo "FST file $fst_file is valid."
}

echo "Compiling decoding graph"
rm -rf build/fst/build/fst/${am}/graph_prunedlm_unk
self_loop_scale_arg=""

if [ -f build/fst/${am}/frame_subsampling_factor ]; then
  factor=`cat build/fst/${am}/frame_subsampling_factor`
  if [ $factor -eq "3" ]; then
    self_loop_scale_arg="--self-loop-scale 1.0 "
  fi
fi
compile_models/mkgraph.sh $self_loop_scale_arg build/fst/data/prunedlm_unk build/fst/${am} build/fst/${am}/graph_prunedlm_unk
rm -rf build/fst/data/prunedlm_unk/tmp
exit 0