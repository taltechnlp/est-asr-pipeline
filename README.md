# ASR pipeline for Estonian speech recognition

This project uses the Nextflow workflow engine to transcribe Estonian speech recordings to text.

Nextflow offers great support for container technologies like Docker and different cluster and cloud environments as well as Kubernetes.

*This pipeline requires a server with a fairly modern NViudia GPU!*

## Installation

The project uses Nextflow which depends mainly on Java. Both should be installed locally.
The pre-built models, scripts and Kaldi tookit is consumed via Docker and so Docker also needs to be installed.

This configuration has only been used on Linux. Because of Docker use other OS-s could be possible but because of configuration tricks used in the nextflow.config files, Dockerizing this project might be easiest.

Install Nextflow locally (depends on Java 8, refer to official documentation in case of troubles):

    wget -qO- https://get.nextflow.io | bash

Pull the required Docker image, containing models and libraries (recommended):


    docker pull europe-north1-docker.pkg.dev/speech2text-218910/repo/est-asr-pipeline:1.0b

Also, install [NVIDIA Container Runtime](https://developer.nvidia.com/nvidia-container-runtime)


## Usage

### Using a prebuilt Docker image

Run:

    nextflow run transcribe.nf -profile docker --in /path/to/some_audiofile.mp3 

If you didn't pull the Docker image before, then the first invocation might take some time, because the required Docker image
containing all the needed models and libraries is automatically pulled from the remote registry.

A successful invocation will result with something like this:

    $ nextflow run transcribe.nf -profile docker --in /path/to/some_audiofile.mp3
    N E X T F L O W  ~  version 21.10.6
    Launching `transcribe.nf` [festering_crick] - revision: ee9a3dc173
    executor >  local (11)
    [5a/bad7f2] process > to_wav                   [100%] 1 of 1 ✔
    [c2/4733fb] process > diarization              [100%] 1 of 1 ✔
    [08/d271ef] process > prepare_initial_data_dir [100%] 1 of 1 ✔
    [23/9c6450] process > language_id              [100%] 1 of 1 ✔
    [95/5dec5e] process > speaker_id               [100%] 1 of 1 ✔
    [ad/15e65d] process > decode                   [100%] 1 of 1 ✔
    [c9/f8d8f5] process > mfcc                     [100%] 1 of 1 ✔
    [e6/668fbf] process > hyp2ctm                  [100%] 1 of 1 ✔
    [e2/2b4452] process > to_json                  [100%] 1 of 1 ✔
    [2a/32162e] process > punctuation              [100%] 1 of 1 ✔
    [2d/698ef9] process > output                   [100%] 1 of 1 ✔
    [-        ] process > empty_output             -
    Completed at: 03-apr-2023 15:53:01
    Duration    : 4m 44s
    CPU hours   : 0.1
    Succeeded   : 11



The transcription result in different formats is put to the directory `results/some_audiofile`
(where `some_audiofile` corresponds to the "basename" of your input file):

    $ ls results/some_audiofile/
    result.ctm  result.json  result.srt  result.trs  result.with-compounds.ctm   result.txt

### Running on cluster

#### SGE

Run:

    nextflow run transcribe.nf -profile docker,sge --in /path/to/some_audiofile.mp3 
    

#### SLURM

Run:

    nextflow run transcribe.nf -profile docker,slurm --in /path/to/some_audiofile.mp3 

In both cases, some things in nextflow.config might need to be modified.

### Running without Docker

TODO

### Command line parameters

Firstly, the main script (transcribe.nf) already has default values for input parameters. All of these can be provided via the command line when executing the script using the nextflow executable. To use a parameter, ignore the 'params.' part and prepend two hyphens '--'. So 'params.in' becomes '--in'. The following parameter should always be provided (unless the default value is satisfactory):

-   --in <filename> - The name and location of the audio recording in you local system that needs to be transcribed.
-   --out_dir <path> - The path to the directory where results will be stored. By default a relative directory "results/".
-   `--do_speaker_id true|false` - Include speaker diarization and identification. The result will include speaker names. Some Estonian celebrities and radio personalities will be identified by their name, others will be give ID-s. By default `true`.
-   `--do_punctuation true|false` - Whether to attempt punctuation recovery and add punctuation to the transcribed text. By default `true`.
-   `--do_language_id true|false` - Whether to apply a language ID model to discard speech segements that are not in Estonian. By default `true`.


### Command line options

Nextflow allows additional command line options:

-   with-report - Generates a human-readable HTML execution report by default into the current folder. Optional, useful to dig into resource consumption details.
-   with-trace - Generates a machine-readable CSV execution report of all the steps in the script. Places it into the current folder by default. Useful to gather general execution statistics.
-   with-dag "filename.png"- Generates a visual directed execution graph. Shows dependecies between script processes. Not very useful.
-   with-weblog "your-api-endpoint" - Sends real-time statistics to the provided API endpoint. This is used by our https://github.com/taltechnlp/est-asr-backend backend server to gather real-time progress information and estimate queue length.
