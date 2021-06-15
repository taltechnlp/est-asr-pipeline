# kaldi-offline-transcriber-nextflow
Nextflow based speech processing

## Usage
First build the container:

    docker build . -t nextflow:latest

Start a container (name is "nextflow") and put it into background (`-d`). Also, mount a local
directory `~/tmp/speechfiles` as the container directory `/opt/speechfiles`.
  
    mkdir -p ~/tmp/speechfiles
    docker run --name nextflow -v ~/tmp/speechfiles:/opt/speechfiles --rm -d -t nextflow:latest

Install Nextflow locally:

    wget -qO- https://get.nextflow.io | bash

To transcribe:

    nextflow run transcribe.nf -with-docker nextflow -with-report report.html -with-trace -with-dag flowchart.png -with-weblog 'http://localhost:7700/process/'

    nextflow run transcribe.nf -with-docker nextflow -with-weblog 'http://localhost:7700/process/'