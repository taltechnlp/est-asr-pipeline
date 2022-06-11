FROM kaldiasr/kaldi:gpu-latest 
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
    zlib1g-dev \
    openjdk-8-jdk-headless \
    ant && \
    apt-get clean autoclean && \
    apt-get autoremove -y

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y locales

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8


ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN conda --version

RUN conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

RUN conda install ruamel.yaml && \
    pip install kaldiio && \
    pip install simplejson && \
    pip install pytest && \
    pip install pydub  && \
    pip install editdistance && \
    pip install soundfile 

RUN pip install speechbrain

WORKDIR /opt
  
RUN git clone https://github.com/alumae/et-g2p.git && \
    cd /opt/et-g2p && \
    ant jar
    

RUN git clone https://github.com/pytorch/fairseq && \
    cd fairseq && \
    git checkout cf8ff8c3c5242e6e71e8feb40de45dd699f3cc08 && \
    pip install --editable ./
    
    
RUN apt-get -y install -y build-essential libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev libatlas-dev libfftw3-dev


RUN apt-get update && \
    apt-get install -y software-properties-common lsb-release && \
    apt-get clean all && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
    apt-get update  && \
    apt-get install -y kitware-archive-keyring  && \
    rm /etc/apt/trusted.gpg.d/kitware.gpg && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6AF7F09730B3F0A4 && \
    apt-get update  && \
    apt-get -y install cmake

    
RUN git clone https://github.com/kpu/kenlm.git && \
    cd kenlm && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make -j 4    
    
RUN git clone https://github.com/flashlight/flashlight.git && \
    cd flashlight && \
    git checkout 0031ca105ea71f9cc163e6f60e6c6dc5b504883b && \
    cd bindings/python && \
    USE_MKL=0 KENLM_ROOT=/opt/kenlm python3 setup.py install


ENV HOME /opt
ENV LD_LIBRARY_PATH /usr/local/lib

RUN ln -s -f /usr/bin/python2 /usr/bin/python && \
    apt-get install -y python-numpy python-scipy python3-simplejson python3-pytest && \
    pip2 install theano --no-deps && \
    pip2 install six

# Set up punctuator    
RUN mkdir -p /opt/est-asr-pipeline && \
    cd /opt/est-asr-pipeline && \
    wget -q -O - http://bark.phon.ioc.ee/tanel/est_punct2.tar.gz | tar xvz

RUN cd /opt/est-asr-pipeline && \
    wget -q -O - http://bark.phon.ioc.ee/tanel/kaldi-offline-transcriber-data-2021-06-11.tgz | tar xvz
   

COPY bin /opt/est-asr-pipeline/bin

ENV KALDI_ROOT /opt/kaldi

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


COPY models /opt/est-asr-pipeline/models

RUN apt-get install -y procps

RUN conda install -c conda-forge pynini=2.1.3

CMD ["/bin/bash"]    
