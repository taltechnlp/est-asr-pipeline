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

echo "Converting compounder LM to FST"
rm -rf build/fst/data/compounderlm
mkdir -p build/fst/data/compounderlm
cat $lexicon | perl -npe 's/(\(\d\))?\s.+//' | uniq | ./bin/make-compounder-symbols.py > build/fst/data/compounderlm/words.txt
zcat $compounder_lm | \
    grep -v '<s> <s>' | \
    grep -v '</s> <s>' | \
    grep -v '</s> </s>' | \
    arpa2fst --disambig-symbol='#0' --read-symbol-table=build/fst/data/compounderlm/words.txt - > arpa2fst_result.fst
validate_fst arpa2fst_result.fst
cat arpa2fst_result.fst | fstproject --project_type=output | fstarcsort --sort_type=ilabel > build/fst/data/compounderlm/G.fst 
rm arpa2fst_result.fst
exit 0