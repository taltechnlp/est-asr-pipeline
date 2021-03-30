FROM alumae/kaldi-offline-transcriber-et

LABEL maintainer="Aivo Olev"

COPY scripts/diarization.sh /opt/kaldi-offline-transcriber/scripts/diarization.sh
COPY .gitignore /opt/kaldi-offline-transcriber/.gitignore
COPY transcribe.nf /opt/kaldi-offline-transcriber/

RUN cd /opt/kaldi-offline-transcriber && wget -qO- https://get.nextflow.io | bash

WORKDIR /opt/kaldi-offline-transcriber

ARG KALDI_ROOT="/home/speech/tools/kaldi-trunk"
ENV PATH="utils:${KALDI_ROOT}/src/bin:${KALDI_ROOT}/tools/openfst/bin:${KALDI_ROOT}/src/fstbin/:${KALDI_ROOT}/src/gmmbin/:${KALDI_ROOT}/src/featbin/:${KALDI_ROOT}/src/lm/:${KALDI_ROOT}/src/sgmmbin/:${KALDI_ROOT}/src/sgmm2bin/:${KALDI_ROOT}/src/fgmmbin/:${KALDI_ROOT}/src/latbin/:${KALDI_ROOT}/src/nnet2bin/:${KALDI_ROOT}/src/online2bin/:${KALDI_ROOT}/src/kwsbin:${KALDI_ROOT}/src/lmbin:$(PATH):${KALDI_ROOT}/src/ivectorbin:${KALDI_ROOT}/src/nnet3bin:${KALDI_ROOT}/src/rnnlmbin:${PATH}"

CMD ["/bin/bash"]   