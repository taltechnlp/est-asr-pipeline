# ASR pipeline for Estonian speech recognition

This project uses the Nextflow workflow engine to transcribe Estonian speech recordings to text.

Nextflow offers great support for container technologies like Docker and different cluster and cloud environments as well as Kubernetes.

## Installation

The project uses Nextflow which depends mainly on Java. Both should be installed locally.
The pre-built model, scripts and Kaldi tookit is consumed via Docker and so Docker also needs to be installed.

This configuration has only been used on Linux. Because of Docker use other OS-s could be possible but because of configuration tricks used in the nextflow.config files, Dockerizing this project might be easiest.

Install Nextflow locally (depends on Java 8, refer to official documentation in case of troubles):

    wget -qO- https://get.nextflow.io | bash

## Usage

First build the container:

    docker build . -t nextflow

Start the container (named "nextflow:latest") and put it into background (`-d`). Also, mount a local directory `~/tmp/speechfiles` to the following directory in the container: `/opt/speechfiles`.

    mkdir -p ~/tmp/speechfiles
    docker run --name nextflow -v ~/tmp/speechfiles:/opt/speechfiles --rm -d -t nextflow:latest

To transcribe a speech recording provide at least input file, output file and file format as parameters. Also the -with-docker command line option should refer to the locally built and running Docker container:

    nextflow run transcribe.nf -with-docker nextflow --in ~/audio.mp3 --out result.json --file_ext mp3

There is a project which enables to set up an API server and a simple user interface to upload files and retrieve results from this workflow. It can be useful for hosting this workflow: https://github.com/taltechnlp/est-asr-backend

## Configuration

Nextflow enables various configuration options.

### Command line parameters

Firstly, the main script (transcribe.nf) already has default values for input parameters. All of these can be provided via the command line when executing the script using the nextflow executable. To use a parameter, ignore the 'params.' part and prepend two hyphens '--'. So 'params.in' becomes '--in'. The following parameter should always be provided (unless the default value is satisfactory):

-   --in - The name and location of the audio recording in you local system that needs to be transcribed.
-   --file_ext - the file extention of the input file. This does not have to exactly match the actual file name but is important to determine how to turn the file into ´wav´ format. Supported options: ´wav´, ´ḿp3´, ´mpga´, ´m4a´, ´mp4´, ´oga´, ´ogg´, ´mp2´, ´flac´.
-   --out - The output file name. Cannot be a location in the local system. Should be a unique file name. By default will be saved into the /results folder of this project.
-   --out_format - Output format of the transcription. By default `json`. Supported options: ´json´, ´trs´, ´with-compounds´, ´txt´, ´srt´.
-   --do_speaker_id - By default 'yes'. Include speaker diarization and identification. The result will include speaker names. Some Estonian celebrities and radio personalities will be identified by their name, others will be give ID-s.
-   --do-punctuation - Whether to attempt punctuation recovery and add punctuation to the transcribed text.

### Configuration file

Additional configuration is currently provided via the nextflow.config file. The following parameters should be changed if you need advanced execution optimizations:

-   nthreads - By default the script will use 4 system threads. Should be changed in case you are executing the script in parallel multiple times or want to use a different nubmer of threads per execution.

The rest of the parameters are there because the transcribe.nf file needs these. Those should never be changed unless you are deliberately changing the script (replacing the acoustic model or optimizing other parameters).

Nextflow offers great support for cluster and cloud environments and Kubernetes. Please consult Nextflow documentation in order to configure these.

### Command line options

Nextflow allows additional command line options:

-   -with-docker - Allows the script to use Docker. This should always be used with this script and must refer to the container that needed to be built locally.
-   -with-report - Generates a human-readable HTML execution report by default into the current folder. Optional, useful to dig into resource consumption details.
-   with-trace - Generates a machine-readable CSV execution report of all the steps in the script. Places it into the current folder by default. Useful to gather general execution statistics.
-   with-dag "filename.png"- Generates a visual directed execution graph. Shows dependecies between script processes. Not very useful.
-   with-weblog "your-api-endpoint" - Sends real-time statistics to the provided API endpoint. This is used by our https://github.com/taltechnlp/est-asr-backend backend server to gather real-time progress information and estimate queue length.
