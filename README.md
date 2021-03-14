# kaldi-offline-transcriber-nextflow
Nextflow based speech processing

## Usage
First build the container:
    docker build . -t nextflow:latest

Start a container (name is "nextflow") and put it into background (`-d`). Also, mount a local
directory `~/tmp/speechfiles` as the container directory `/opt/speechfiles`.
  
    mkdir -p ~/tmp/speechfiles
    docker run --name nextflow -v ~/tmp/speechfiles:/opt/speechfiles --rm -d -t nextflow:latest
