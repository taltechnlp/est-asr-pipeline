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

echo "Constructing UNK word LM"
rm -rf build/fst/data/unk_lang_model
utils/lang/make_unk_lm.sh --cmd utils/run.pl build/fst/data/dict build/fst/data/unk_lang_model
exit 0