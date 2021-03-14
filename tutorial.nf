#!/usr/bin/env nextflow

params.str = 'Hello world!'

process splitLetters {

    output:
    file 'chunk_*' into letters

    """
    printf '${params.str}' | split -b 6 - chunk_
    """
}


process convertToUpper {

    input:
    file x from letters.flatten()

    output:
    stdout result

    """
    rev $x
    """
}

    cat $seq[show.seq] | cut -f 3,4,8 -d " " | \
	while read LINE ; do \
		start=`echo \$LINE | cut -f 1,2 -d " " | perl -ne '@t=split(); \$start=\$t[0]/100.0; printf("%08.3f", \$start);'`; \
		end=`echo \$LINE   | cut -f 1,2 -d " " | perl -ne '@t=split(); \$start=\$t[0]/100.0; \$len=\$t[1]/100.0; \$end=\$start+\$len; printf("%08.3f", \$end);'`; \
		sp_id=`echo \$LINE | cut -f 3 -d " "`; \
		echo audio-\${sp_id}---\${start}-\${end} audio \$start \$end; \
	done > segments
	if [ ! -s segments } ]; then \
	  echo "audio-dummy---0.000-0.110 audio 0.0 0.110" > segments; \
	fi


process utt2spk {
    input:
    file segments

    output:
    file "utt2spk" into utt2spk

    script:
    """
    cat $segments | perl -npe 's/\s+.*//; s/((.*)---.*)/\1 \2/' > utt2spk
    """
}

process spk2utt {
    input:
    file utt2spk

    output:
    file "spk2utt" into spk2utt

    script:
    """
    WORK_DIR=\$PWD
    cd $params.srcdir
    ./utils/utt2spk_to_spk2utt.pl $utt2spk > \${WORK_DIR}/spk2utt
    """
}

result.view { it.trim() }