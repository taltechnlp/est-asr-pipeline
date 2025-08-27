FROM nvcr.io/nvidia/kaldi:22.12-py3
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

RUN conda init bash && \
    conda config --set channel_priority flexible && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda tos accept --override-channels --channel conda-forge || echo "conda-forge TOS already accepted or not required"

RUN conda install python=3.9

RUN conda install -c conda-forge pynini=2.1.3

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


RUN conda install ruamel.yaml && \
    pip install kaldiio && \
    pip install simplejson && \
    pip install pytest && \
    pip install soundfile pandas && \
    pip install lxml

RUN pip install speechbrain pytorch-lightning==1.9.0 


RUN pip install whisper-ctranslate2 faster-whisper 'ctranslate2<4.0'

RUN pip install pyannote.audio==3.1

WORKDIR /opt
  
RUN git clone https://github.com/alumae/et-g2p-fst.git    

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y locales

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8
    
RUN apt-get install -y openjdk-8-jdk-headless ant


ENV HOME /opt
ENV LD_LIBRARY_PATH /usr/local/lib

    


RUN mkdir -p /opt/est-asr-pipeline && \
    cd /opt/est-asr-pipeline && \
    wget -q -O - https://cs.taltech.ee/staff/tanel.alumae/data/est-asr-pipeline-whisper-models.2024-10-11.tgz | tar xvz
   

COPY bin /opt/est-asr-pipeline/bin
COPY assets /opt/est-asr-pipeline/assets

ENV KALDI_ROOT /opt/kaldi



# cache model for language ID
RUN cd /opt/est-asr-pipeline/bin && \
    ./extract_lid_features_kaldi.py foo fii  || echo "OK";

ARG HF_TOKEN

# cache model for speaker diarization
RUN cd /opt/est-asr-pipeline && \
    ./bin/diarize.py foo.wav foo.rttm || echo "Should download model and error out, it's OK"

# cache SpkID feature model
RUN pip install huggingface_hub && \
    huggingface-cli download speechbrain/spkrec-ecapa-voxceleb
    

RUN apt-get install -y procps

RUN git clone https://github.com/alumae/et-g2p.git && \
    cd /opt/et-g2p && \
    ant dist

CMD ["/bin/bash"]    
