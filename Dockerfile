FROM debian:12 as base 

RUN apt-get update && apt-get install -y --no-install-recommends  \
    g++ \
    gfortran \
    make \
    automake \
    autoconf \
    bzip2 \
    unzip \
    wget \
    sox \
    libtool \
    git \
    zlib1g-dev \
    ca-certificates \
    patch \
    python-is-python3 \
    gfortran \
    libatlas3-base \
    libtool-bin \
    python3 \
    python3-pip \
    python3-dev \
    sox \
    ffmpeg \
    subversion \
    wget \
    libarchive-dev \
    zlib1g-dev \
    locales \
    procps \
    default-jre-headless && \
    apt-get clean autoclean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
ENV LD_LIBRARY_PATH=/opt/kaldi/tools/openfst/lib:$LD_LIBRARY_PATH
ENV CPLUS_INCLUDE_PATH=/opt/kaldi/tools/openfst/include:$CPLUS_INCLUDE_PATH
ENV LIBRARY_PATH=/opt/kaldi/tools/openfst/lib:$LIBRARY_PATH

RUN git clone --depth 1 --branch aarch64-compatible https://github.com/andrikaro/kaldi.git /opt/kaldi && \
    cd /opt/kaldi/tools && \
    ./extras/install_openblas.sh  && \
    make -j $(nproc) && \
    cd /opt/kaldi/src && \
    ./configure --shared --mathlib=OPENBLAS && \
    make depend -j $(nproc) && \
    make -j $(nproc) && \ 
    ldconfig

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-aarch64.sh -b \
    && rm -f Miniconda3-latest-Linux-aarch64.sh 

RUN pip install pynini && pip cache purge

RUN conda install -c conda-forge scipy simplejson pytest numpy  cmake ninja libmamba libarchive -y && \
    conda install -c conda-forge pytorch -y && \
    conda clean -y --all 

# Install torchaudio from source
RUN git clone --branch release/2.4 https://github.com/pytorch/audio.git
RUN cd audio && \
    BUILD_SOX=1 python setup.py install && \
    cd .. && \
    rm -rf audio

RUN conda install -c pytorch cpuonly -y && \
    conda clean -y --all
RUN conda install ruamel.yaml && \
    conda clean -y --all && \
    pip install aesara kaldiio speechbrain && \
    pip cache purge

WORKDIR /opt
  
RUN git clone https://github.com/alumae/et-g2p-fst.git    

RUN git clone -b 'v4.0' --single-branch --depth 1 https://github.com/snakers4/silero-vad.git


RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8
    

RUN cd /opt/kaldi/tools && \
    extras/install_pocolm.sh

ENV HOME /opt
ENV LD_LIBRARY_PATH /usr/local/lib

RUN ln -s -f /usr/bin/python3 /usr/bin/python

# Set up punctuator    
COPY est_punct2-aesara.tar.gz /opt/est-asr-pipeline/est_punct2-aesara.tar.gz

RUN mkdir -p /opt/est-asr-pipeline && \
    cd /opt/est-asr-pipeline && \
    cat est_punct2-aesara.tar.gz | tar xvz && \
    rm est_punct2-aesara.tar.gz

COPY kaldi-offline-transcriber-data-2021-06-11.tgz /opt/est-asr-pipeline/kaldi-offline-transcriber-data-2021-06-11.tgz


RUN cd /opt/est-asr-pipeline && \
    cat kaldi-offline-transcriber-data-2021-06-11.tgz | tar xvz && \
    rm kaldi-offline-transcriber-data-2021-06-11.tgz
   
COPY bin /opt/est-asr-pipeline/bin

ENV KALDI_ROOT /opt/kaldi

FROM base AS stage_1_to_5

COPY compile_models/stage_1.sh compile_models/stage_2.sh compile_models/stage_3.sh compile_models/stage_4.sh compile_models/stage_5.sh /opt/est-asr-pipeline/compile_models/

RUN cd /opt/est-asr-pipeline && \
    touch -m path.sh && \
    ./compile_models/stage_1.sh && \
    ./compile_models/stage_2.sh && \
    ./compile_models/stage_3.sh && \
    ./compile_models/stage_4.sh && \
    ./compile_models/stage_5.sh


FROM stage_1_to_5 AS stage_6
COPY compile_models/stage_6.sh compile_models/mkgraph.sh /opt/est-asr-pipeline/compile_models/
RUN cd /opt/est-asr-pipeline && \
    ./compile_models/stage_6.sh


FROM stage_6 AS stage_7
COPY compile_models/stage_7.sh /opt/est-asr-pipeline/compile_models/
RUN cd /opt/est-asr-pipeline && \
    ./compile_models/stage_7.sh

FROM stage_7 AS stage_8
COPY compile_models/stage_8.sh /opt/est-asr-pipeline/compile_models/
RUN cd /opt/est-asr-pipeline && \
    ./compile_models/stage_8.sh

# This can be removed once the base data pack has been fixed
RUN echo '--sample-frequency=16000' >  /opt/est-asr-pipeline/kaldi-data/sid/mfcc_sid.conf && \
    echo '--frame-length=25' >>  /opt/est-asr-pipeline/kaldi-data/sid/mfcc_sid.conf && \
    echo '--low-freq=20' >>  /opt/est-asr-pipeline/kaldi-data/sid/mfcc_sid.conf && \
    echo '--high-freq=7600' >>  /opt/est-asr-pipeline/kaldi-data/sid/mfcc_sid.conf && \
    echo '--num-mel-bins=30' >>  /opt/est-asr-pipeline/kaldi-data/sid/mfcc_sid.conf && \
    echo '--num-ceps=24' >>  /opt/est-asr-pipeline/kaldi-data/sid/mfcc_sid.conf && \
    echo '--snip-edges=false' >>  /opt/est-asr-pipeline/kaldi-data/sid/mfcc_sid.conf

# run punctuation once on dummy data, just to get the model compiled and cached
RUN cd /opt/est-asr-pipeline/punctuator-data/est_punct2 && \
    echo {} > tmp1 && \
    python3 punctuator_pad_emb_json.py Model_stage2p_final_563750_h256_lr0.02.pcl tmp1 tmp2 && \
    rm tmp1 tmp2 || echo "OK";

# cache model for language ID
RUN cd /opt/est-asr-pipeline/bin && \
    ./extract_lid_features_kaldi.py foo fii  || echo "OK";

# cache model for speech activity detection
RUN cd /opt/est-asr-pipeline/bin && \
    ./find_speech_segments.py foo fii  || echo "OK";

RUN cd /opt/kaldi/src/ivectorbin && \
    make -j $(nproc) || echo "OK" && \
    ldconfig

SHELL ["/bin/bash", "-c"]
RUN cd /opt/est-asr-pipeline/punctuator-data/est_punct2 && \
    conda create -n aesara-env python=3.10.15 -y && \
    conda init && \
    source ~/.bashrc && \
    conda activate aesara-env && \
    conda install -c conda-forge aesara numpy pytorch -y && \
    conda clean -a -y

COPY extracted_punct/punctuator-data/est_punct2/*.py /opt/est-asr-pipeline/punctuator-data/est_punct2/

ENV PATH="/opt/kaldi/src/ivectorbin:${PATH}"
ENV LD_LIBRARY_PATH=/opt/kaldi/tools/openfst/lib:$LD_LIBRARY_PATH

CMD ["/bin/bash"]    
