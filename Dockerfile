FROM alumae/kaldi-offline-transcriber-et

LABEL maintainer="Aivo Olev"

COPY scripts/diarization.sh /opt/kaldi-offline-transcriber/scripts/diarization.sh
COPY .gitignore /opt/kaldi-offline-transcriber/.gitignore
COPY transcribe.nf /opt/kaldi-offline-transcriber/

RUN apt-get update && apt-get install -y procps     

CMD ["/bin/bash"]   