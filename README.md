# ASR pipeline for Estonian speech recognition

This project uses the Nextflow workflow engine to transcribe Estonian speech recordings to text.

Nextflow offers great support for container technologies like Docker and different cluster and cloud environments as well as Kubernetes.

## Installation

The project uses Nextflow which depends mainly on Java. Both should be installed locally. Nextflow 22.10.x or earlier has to be used, newer versions will not work at the moment!

The pre-built model, scripts and Kaldi tookit is consumed via Docker and so Docker also needs to be installed.

This configuration has only been used on Linux. Because of Docker use other OS-s could be possible but because of configuration tricks used in the nextflow.config files, Dockerizing this project might be easiest.

Install Nextflow locally (depends on Java 8, refer to official documentation in case of troubles):

    wget -qO- https://get.nextflow.io | bash

Pull the required Docker image, containing models and libraries (recommended):

    docker pull alumae/est-asr-pipeline:0.1.4

## Usage

### Using a prebuilt Docker image

Run:

    NXF_VER=22.10.0 nextflow run transcribe.nf --in /path/to/some_audiofile.mp3

*NXF_VER=22.10.0 sets the Nextflow version. Later version don't support the syntax in transcribe.nf*

If you didn't pull the Docker image before, then the first invocation might take some time, because the required Docker image
containing all the needed models and libraries is automatically pulled from the remote registry.

A successful invocation will result with something like this:

    N E X T F L O W  ~  version 21.10.6
    Launching `transcribe.nf` [backstabbing_baekeland] - revision: 7ee707faa8
    executor >  local (12)
    [ae/4b81cd] process > to_wav                   [100%] 1 of 1 ✔
    [27/370fbd] process > diarization              [100%] 1 of 1 ✔
    [ec/c58cee] process > prepare_initial_data_dir [100%] 1 of 1 ✔
    [c6/299141] process > language_id              [100%] 1 of 1 ✔
    [d8/9fcbc7] process > mfcc                     [100%] 1 of 1 ✔
    [85/a6589b] process > speaker_id               [100%] 1 of 1 ✔
    [09/f7d366] process > one_pass_decoding        [100%] 1 of 1 ✔
    [0e/fd2533] process > rnnlm_rescoring          [100%] 1 of 1 ✔
    [c0/461429] process > lattice2ctm              [100%] 1 of 1 ✔
    [b4/04eeee] process > to_json                  [100%] 1 of 1 ✔
    [37/3d7f31] process > punctuation              [100%] 1 of 1 ✔
    [86/90a123] process > output                   [100%] 1 of 1 ✔
    [d3/23d672] process > empty_output             -
    Completed at: 14-apr-2022 10:55:48
    Duration    : 3m 31s
    CPU hours   : 0.1
    Succeeded   : 12

The transcription result in different formats is put to the directory `results/some_audiofile`
(where `some_audiofile` corresponds to the "basename" of your input file):

    $ ls results/some_audiofile/
    result.ctm  result.json  result.srt  result.trs  result.with-compounds.ctm

### Running on cluster

TODO

### Running without Docker

TODO

### Command line parameters

Firstly, the main script (transcribe.nf) already has default values for input parameters. All of these can be provided via the command line when executing the script using the nextflow executable. To use a parameter, ignore the 'params.' part and prepend two hyphens '--'. So 'params.in' becomes '--in'. The following parameter should always be provided (unless the default value is satisfactory):

-   --in <filename> - The name and location of the audio recording in you local system that needs to be transcribed.
-   --out_dir <path> - The path to the directory where results will be stored. By default a relative directory "results/".
-   `--do_speaker_id true|false` - Include speaker diarization and identification. The result will include speaker names. Some Estonian celebrities and radio personalities will be identified by their name, others will be give ID-s. By default `true`.
-   `--do_punctuation true|false` - Whether to attempt punctuation recovery and add punctuation to the transcribed text. By default `true`.
-   `--do_language_id true|false` - Whether to apply a language ID model to discard speech segements that are not in Estonian. By default `true`.

### Configuration file

Additional configuration is currently provided via the nextflow.config file. The following parameters should be changed if you need advanced execution optimizations:

-   nthreads - By default the script will use 2 system threads for more CPU-intensive parts of the transcription pipeline. Should be changed in case you are executing the script in parallel multiple times or want to use a different nubmer of threads per execution.

Nextflow offers great support for cluster and cloud environments and Kubernetes. Please consult Nextflow documentation in order to configure these.

### Command line options

Nextflow allows additional command line options:

-   with-report - Generates a human-readable HTML execution report by default into the current folder. Optional, useful to dig into resource consumption details.
-   with-trace - Generates a machine-readable CSV execution report of all the steps in the script. Places it into the current folder by default. Useful to gather general execution statistics.
-   with-dag "filename.png"- Generates a visual directed execution graph. Shows dependecies between script processes. Not very useful.
-   with-weblog "your-api-endpoint" - Sends real-time statistics to the provided API endpoint. This is used by our https://github.com/taltechnlp/est-asr-backend backend server to gather real-time progress information and estimate queue length.
