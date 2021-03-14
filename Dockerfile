FROM alumae/kaldi-offline-transcriber-et

COPY scripts/diarization.sh /opt/kaldi-offline-transcriber/scripts/diarization.sh
COPY .gitignore /opt/kaldi-offline-transcriber/.gitignore
COPY transcribe.nf /opt/kaldi-offline-transcriber/

RUN cd /opt/kaldi-offline-transcriber
   
CMD ["/bin/bash"]   