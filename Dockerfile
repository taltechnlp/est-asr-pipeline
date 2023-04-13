FROM kaldiasr/kaldi:latest 
MAINTAINER Tanel Alumae <alumae@gmail.com>

RUN apt-get update && apt-get install -y  \
    autoconf \
    automake \
    bzip2 \
    g++ \
    gfortran \
    git \
    libatlas3-base \
    libtool-bin \
    make \
    python2.7 \
    python-pip \
    python-dev \
    sox \
    ffmpeg \
    subversion \
    wget \
    zlib1g-dev && \
    apt-get clean autoclean && \
    apt-get autoremove -y


ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN conda --version

RUN conda install -c conda-forge pynini=2.1.3

RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch

RUN conda install ruamel.yaml && \
    pip install kaldiio && \
    pip install simplejson && \
    pip install pytest

RUN pip install speechbrain

WORKDIR /opt
  
RUN git clone https://github.com/alumae/et-g2p-fst.git    

RUN git clone -b 'v4.0' --single-branch --depth 1 https://github.com/snakers4/silero-vad.git

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y locales

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8
    
RUN apt-get install -y openjdk-8-jre-headless

RUN cd /opt/kaldi/tools && \
    extras/install_pocolm.sh

ENV HOME /opt
ENV LD_LIBRARY_PATH /usr/local/lib

RUN ln -s -f /usr/bin/python2 /usr/bin/python && \
    apt-get install -y python-numpy python-scipy python3-simplejson python3-pytest && \
    pip2 install theano --no-deps

# Set up punctuator    
RUN mkdir -p /opt/est-asr-pipeline && \
    cd /opt/est-asr-pipeline && \
    wget -q -O - http://bark.phon.ioc.ee/tanel/est_punct2.tar.gz | tar xvz

RUN cd /opt/est-asr-pipeline && \
    wget -q -O - http://bark.phon.ioc.ee/tanel/kaldi-offline-transcriber-data-2021-06-11.tgz | tar xvz
   

COPY bin /opt/est-asr-pipeline/bin

ENV KALDI_ROOT /opt/kaldi

RUN cd /opt/est-asr-pipeline && \
    touch -m path.sh && \
    ./bin/compile_models.sh

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
    python2 punctuator_pad_emb_json.py Model_stage2p_final_563750_h256_lr0.02.pcl tmp1 tmp2 && \
    rm tmp1 tmp2 || echo "OK";

# cache model for language ID
RUN cd /opt/est-asr-pipeline/bin && \
    ./extract_lid_features_kaldi.py foo fii  || echo "OK";

# cache model for speech activity detection
RUN cd /opt/est-asr-pipeline/bin && \
    ./find_speech_segments.py foo fii  || echo "OK";


RUN apt-get install -y procps



CMD ["/bin/bash"]    
