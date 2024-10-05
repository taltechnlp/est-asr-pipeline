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

echo "Making build/fst/data/prunedlm/G.fst from the provided ARPA LM"
rm -rf build/fst/data/prunedlm
mkdir -p build/fst/data/prunedlm
utils/prepare_lang.sh --phone-symbol-table build/fst/$am/phones.txt build/fst/data/dict '<unk>' build/fst/data/dict/tmp build/fst/data/prunedlm
gunzip -c $pruned_lm | arpa2fst --disambig-symbol=#0 \
    --read-symbol-table=build/fst/data/prunedlm/words.txt - build/fst/data/prunedlm/G.fst
echo "Checking how stochastic G is (the first of these numbers should be small):"
fstisstochastic build/fst/data/prunedlm/G.fst || echo "not stochastic (probably OK)"    
utils/validate_lang.pl build/fst/data/prunedlm || exit 1
exit 0