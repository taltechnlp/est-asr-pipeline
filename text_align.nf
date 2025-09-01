#!/usr/bin/env nextflow

/*
 * Text-based alignment pipeline using edit distance
 * Aligns reference text with ASR output and adjusts word timings
 */

nextflow.enable.dsl = 2

params.asr_json = ""
params.reference_json = ""
params.out_dir = "results_text_align"
params.help = false

if (params.help) {
    log.info """
    Text-based alignment pipeline
    
    Usage:
        nextflow run text_align.nf --asr_json ASR_OUTPUT.json --reference_json REFERENCE.json --out_dir OUTPUT_DIR
    
    Parameters:
        --asr_json         Path to ASR output JSON (transcribe.nf format)
        --reference_json   Path to reference JSON (simple segments format)
        --out_dir          Output directory (default: results_text_align)
    """
    exit 0
}

if (!params.asr_json || !params.reference_json) {
    log.error "Both --asr_json and --reference_json parameters are required"
    exit 1
}

/*
 * Process to perform text-based alignment using edit distance
 */
process perform_text_alignment {
    memory '2GB'
    cpus 1
    
    input:
    path asr_json, name: 'asr_input.json'
    path reference_json, name: 'reference_input.json'
    
    output:
    path 'alignment_result.json', emit: alignment_result
    path 'alignment_stats.json', emit: alignment_stats
    path 'debug_log.txt', emit: debug_log
    
    script:
    """
    text_based_alignment.py \\
        --asr-json asr_input.json \\
        --reference-json reference_input.json \\
        --output alignment_result.json \\
        --stats alignment_stats.json \\
        --debug debug_log.txt
    """
}

/*
 * Process to generate CTM file from alignment results
 */
process generate_ctm {
    memory '1GB'
    cpus 1
    publishDir params.out_dir, mode: 'copy'
    
    input:
    path alignment_result
    
    output:
    path 'result.ctm', emit: ctm
    
    script:
    """
    json_to_ctm_aligned.py ${alignment_result} > result.ctm
    """
}

/*
 * Process to generate alignment statistics report
 */
process generate_statistics {
    memory '1GB'  
    cpus 1
    publishDir params.out_dir, mode: 'copy'
    
    input:
    path alignment_stats
    
    output:
    path 'alignment_report.txt', emit: report
    path 'alignment_stats.json', emit: stats_json
    
    script:
    """
    alignment_stats.py ${alignment_stats} > alignment_report.txt
    cp ${alignment_stats} alignment_stats_copy.json
    mv alignment_stats_copy.json alignment_stats.json
    """
}

/*
 * Main workflow
 */
workflow {
    asr_json_ch = Channel.fromPath(params.asr_json, checkIfExists: true)
    reference_json_ch = Channel.fromPath(params.reference_json, checkIfExists: true)
    
    // Perform text-based alignment
    alignment_results = perform_text_alignment(asr_json_ch, reference_json_ch)
    
    // Generate CTM output
    ctm_result = generate_ctm(alignment_results.alignment_result)
    
    // Generate statistics report
    stats_result = generate_statistics(alignment_results.alignment_stats)
    
    // Log completion
    ctm_result.ctm.view { "CTM file generated: $it" }
    stats_result.report.view { "Statistics report generated: $it" }
}

workflow.onComplete {
    log.info "Text alignment pipeline completed successfully"
    log.info "Results available in: ${params.out_dir}"
}