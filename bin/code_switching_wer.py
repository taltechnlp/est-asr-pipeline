import argparse
import re
import string
import xml.etree.ElementTree as ET
from pathlib import Path
from subprocess import CalledProcessError, run
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm
from estnltk import Text

# Constants
EMBEDDED_LANG = "E"
MATRIX_LANG = "M"
TRS_SUFFIX = ".trs"
STM_SUFFIX = ".stm"
CTM_SUFFIX = ".ctm"
Triplet = Tuple[str, Optional[int], Optional[int]]  # (op, ref_idx, hyp_idx)


def preprocess_estonian_annotations(text: str) -> str:
    r"""
    Preprocesses Estonian annotation protocols for SCLITE compatibility.
    
    Estonian TRS annotations:
    - (word) : unrecognized/uncertain word suggestions
    - prefix()suffix : partial word recognition
    - (maybe) : uncertainty markers
    
    These are converted to SCLITE-compatible optionally deletable words using
    parentheses, which work with the -D flag.

    Args:
        text (str): Text with Estonian annotations

    Returns:
        str: Text with SCLITE-compatible optionally deletable word markers

    Examples
    --------
    >>> preprocess_estonian_annotations("Ma (lähen) poodi ja ta (sõit)s autoga.")
    'Ma (lähen) poodi ja ta (sõits) autoga.'
    """
    if not text.strip():
        return text
    
    # Handle partial recognition patterns with proper Estonian semantics:
    # The confident part must be present, uncertain part is optional
    
    # 1. prefix()suffix -> prefix (suffix) - prefix required, suffix optional
    text = re.sub(r'(\w+)\(\)(\w+)', r'\1 (\2)', text)
    
    # 2. word(suffix) -> word (suffix) - word required, suffix optional  
    text = re.sub(r'(\w+)\((\w+)\)', r'\1 (\2)', text)
    
    # 3. prefix() -> (prefix) - prefix uncertain, can be optional
    text = re.sub(r'(\w+)\(\)', r'(\1)', text)
    
    # 4. ()suffix -> (prefix) suffix - prefix optional, suffix required
    text = re.sub(r'\(\)(\w+)', r'(*) \1', text)
    
    # 5. Handle empty parentheses - convert to generic optionally deletable marker
    # This prevents insertion errors when hypothesis has words at empty () positions
    text = re.sub(r'\(\)\s*', '(*) ', text)
    
    # The (word) and (maybe) patterns are already in correct SCLITE format
    # They will be handled by SCLITE's -D flag as optionally deletable words
    
    return text

def postprocess_text(text: str) -> str:
    r"""
    Normalizes and cleans input text for consistent tokenization and comparison.
    
    This function first handles Estonian annotation protocols, then performs
    Estonian-specific text normalization using EstNLTK, including morphological 
    lemmatization, followed by basic cleaning operations while preserving 
    optionally deletable word markers for SCLITE.

    Args:
        text (str): The input string to be normalized.

    Returns:
        str: The normalized and cleaned string, with Estonian-specific normalization,
        SCLITE-compatible annotations, punctuation removed, extra spaces collapsed, 
        and all characters in lowercase.

    Examples
    --------
    >>> postprocess_text("Ma (lähen) poodi ja ta (sõit)s autoga.")
    'mina (minema) pood ja tema (sõitma) auto'
    """
    if not text.strip():
        return ""
    
    # Step 1: Handle Estonian annotation protocols first
    text = preprocess_estonian_annotations(text)
    
    try:
        # Step 2: Apply Estonian normalization while preserving optionally deletable markers
        # Split text into segments, preserving parentheses structure
        segments = re.split(r'(\([^)]*\))', text)
        normalized_segments = []
        
        for segment in segments:
            if not segment.strip():
                continue
                
            if segment.startswith('(') and segment.endswith(')'):
                # This is an optionally deletable word - normalize content but keep parentheses
                inner_text = segment[1:-1].strip()
                if inner_text == '*':
                    # Special marker for empty parentheses - preserve as-is
                    normalized_segments.append('(*)')
                elif inner_text:
                    normalized_inner = _normalize_text_segment(inner_text)
                    if normalized_inner:
                        normalized_segments.append(f'({normalized_inner})')
            else:
                # Regular text segment
                normalized_segment = _normalize_text_segment(segment)
                if normalized_segment:
                    normalized_segments.append(normalized_segment)
        
        text = ' '.join(normalized_segments)
        
    except Exception as e:
        # Fallback to basic normalization if EstNLTK fails
        print(f"Warning: Estonian normalization failed, using basic normalization: {e}")
        # Still preserve parentheses for SCLITE
        text = text.lower()
        # Remove punctuation except parentheses
        text = re.sub(r'[^\w\s()]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    return text.strip()

def _normalize_text_segment(text: str) -> str:
    """
    Normalizes a text segment using EstNLTK (internal helper function).
    
    Args:
        text (str): Text segment to normalize
        
    Returns:
        str: Normalized text segment
    """
    if not text.strip():
        return ""
        
    try:
        # Estonian-specific normalization using EstNLTK
        est_text = Text(text)
        est_text.tag_layer('tokens')
        est_text.tag_layer('compound_tokens')  
        est_text.tag_layer('words')
        est_text.tag_layer('morph_analysis')
        
        # Extract normalized words using lemmatization
        normalized_words = []
        for i, word in enumerate(est_text['words']):
            word_text = word.text
            
            # Skip punctuation tokens (except parentheses which are handled separately)
            if word_text in string.punctuation and word_text not in '()':
                continue
                
            # Try to get lemma from morphological analysis
            if i < len(est_text['morph_analysis']):
                morph_span = est_text['morph_analysis'][i]
                if morph_span.annotations:
                    # Get first annotation's lemma
                    annotation = morph_span.annotations[0]
                    if 'lemma' in annotation:
                        word_text = annotation['lemma']
            
            # Additional Estonian number and time normalization
            # Convert time format (e.g., "15:30" -> "15 30")
            word_text = re.sub(r'(\d+):(\d+)', r'\1 \2', word_text)
            
            # Handle decimal numbers (e.g., "25." -> "25")
            word_text = re.sub(r'^(\d+)\.$', r'\1', word_text)
            
            # Split compound numbers/tokens
            tokens = word_text.split()
            normalized_words.extend(tokens)
        
        result = ' '.join(normalized_words)
        
    except Exception:
        # Basic fallback normalization
        result = text.lower()
        result = re.sub(r'[^\w\s]', '', result)
        result = re.sub(r'\s+', ' ', result)
    
    return result.strip()

def tokenize(text: str) -> List[str]:
    """Exactly the same tokenization used everywhere."""
    txt = postprocess_text(text)
    return txt.split() if txt else []


def trs_to_stm(trs_file: Union[str, Path], out_path_stm: Union[str, Path]) -> None:
    """
    Converts a Transcriber XML (.trs) file into an STM (Segment Time Marked) file.

    Args:
        trs_file (str): Path to the input .trs XML file.
        out_path_stm (str): Path to output the generated .stm file.

    Notes:
        STM formatting conventions for SCLITE compatibility.
    """
    filename = Path(trs_file).stem
    root = ET.parse(trs_file).getroot()

    # Step 1: Build speaker ID → real name map
    # NOTE maps have to be build during each function call as they differ from file to file
    speaker_map = {}
    for speaker_el in root.findall(".//Speakers/Speaker"):
        speaker_id = speaker_el.attrib.get("id")
        speaker_name = speaker_el.attrib.get("name")
        if speaker_id and speaker_name:
            speaker_map[speaker_id] = speaker_name.replace(
                " ", "_"
            )  # sanitize name for STM

    # Step 2: Process each turn
    turns = root.findall(".//Turn")
    stm_lines = []

    for turn in turns:
        speaker_id = turn.attrib.get("speaker", "").strip()
        if not speaker_id or speaker_id not in speaker_map:
            continue

        speaker_name = speaker_map[speaker_id]
        turn_start = float(turn.attrib["startTime"])
        turn_end = float(turn.attrib["endTime"])

        current_start = turn_start
        current_text = []
        
        # Process initial Turn text content first
        if turn.text:
            current_text.append(turn.text.strip())

        for el in turn:
            if el.tag == "Sync":
                sync_time = float(el.attrib["time"])
                if current_text:
                    text = postprocess_text(" ".join(current_text))
                    if text:
                        stm_lines.append(
                            f"{filename} 1 {speaker_name} {current_start:.2f} {sync_time:.2f} {text}"
                        )
                current_start = sync_time
                current_text = []

                if el.tail:
                    current_text.append(el.tail.strip())
            else:
                if el.text:
                    current_text.append(el.text.strip())
                if el.tail:
                    current_text.append(el.tail.strip())

        if current_text:
            final_text = postprocess_text(" ".join(current_text))
            if final_text:
                stm_lines.append(
                    f"{filename} 1 {speaker_name} {current_start:.2f} {turn_end:.2f} {final_text}"
                )

    with open(out_path_stm, "w", encoding="utf-8") as f:
        f.write("\n".join(stm_lines) + "\n")

def normalize_ctm(input_ctm: Union[str, Path], output_ctm: Union[str, Path]) -> None:
    """
    Normalizes a CTM file by applying Estonian text processing to word column.
    
    Args:
        input_ctm (str): Path to the input .ctm file.
        output_ctm (str): Path to output the normalized .ctm file.
    """
    normalized_lines = []
    
    with open(input_ctm, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) < 5:
                # Invalid CTM line format, skip
                continue
                
            # Extract components: file channel start_time duration word [confidence]
            file_id = parts[0]
            channel = parts[1] 
            start_time = parts[2]
            duration = parts[3]
            word = parts[4]
            confidence = parts[5] if len(parts) > 5 else None
            
            # Apply Estonian normalization to the word
            normalized_word = postprocess_text(word)
            if not normalized_word:
                # If normalization results in empty string, skip this word
                continue
            
            # Reconstruct the line
            if confidence:
                normalized_line = f"{file_id} {channel} {start_time} {duration} {normalized_word} {confidence}"
            else:
                normalized_line = f"{file_id} {channel} {start_time} {duration} {normalized_word}"
            
            normalized_lines.append(normalized_line)
    
    with open(output_ctm, "w", encoding="utf-8") as f:
        f.write("\n".join(normalized_lines) + "\n")

def trs_wordId2lang(
    trs_file: Union[str, Path],
) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    trs_file = Path(trs_file)
    root = ET.parse(trs_file).getroot()

    word_id2lang: Dict[str, str] = {}
    word2lang: List[Tuple[str, str]] = []

    in_en_block = False
    word_idx = 0

    for turn in root.findall(".//Turn"):
        # process .text at the start of the <Turn>
        if turn.text:
            for tok in tokenize(turn.text):
                lang = "E" if in_en_block else "M"
                word_id2lang[str(word_idx)] = lang
                word2lang.append((tok, lang))
                word_idx += 1

        for elem in turn:
            if elem.tag == "Event":
                desc = (elem.attrib.get("desc") or "").lower()
                extent = (elem.attrib.get("extent") or "").lower()

                if desc.startswith("en"):
                    if extent == "begin":
                        in_en_block = True
                    elif extent == "end":
                        in_en_block = False
                    elif extent == "previous" and word_idx > 0:
                        # mark the last word as English
                        word_id2lang[str(word_idx - 1)] = "E"
                        prev_word, _ = word2lang[-1]
                        word2lang[-1] = (prev_word, "E")

            # text that comes *after* this tag (very important!)
            if elem.tail:
                for tok in tokenize(elem.tail):
                    lang = "E" if in_en_block else "M"
                    word_id2lang[str(word_idx)] = lang
                    word2lang.append((tok, lang))
                    word_idx += 1

    return word_id2lang, word2lang


def _token_ranges(text: str) -> List[Tuple[int, int, int]]:
    r"""
    Computes the character span for each whitespace-separated token in the input text.

    For input string :math:`s`, this function returns a list of tuples
    :math:`(i, \text{start}, \text{end})`, where each tuple gives the token index and
    its start and end column offset in the string.

    Args:
        text (str): The input string to tokenize.

    Returns:
        List[Tuple[int, int, int]]: List of (token_index, start_column, end_column)
        for each token.

    Example
    -------
    >>> _token_ranges("foo bar")
    [(0, 0, 3), (1, 4, 7)]
    """

    ranges = []
    next_idx = 0
    for match in re.finditer(r"\S+", text):
        start, end = match.span()
        token = match.group()
        if re.fullmatch(r"\*+", token):
            ranges.append((None, start, end))
        else:
            ranges.append((next_idx, start, end))
            next_idx += 1
    return ranges, next_idx


def _idx_from_char(ranges: List[Tuple[int, int, int]], col: int) -> Optional[int]:
    r"""
    Finds the token index whose printed span covers a given character column.

    Args:
        ranges (List[Tuple[int, int, int]]): Output from `_token_ranges`, describing the
            character ranges of tokens.
        col (int): The character column index to test.

    Returns:
        Optional[int]: The index of the token covering column `col`, or None if none.

    Example
    -------
    >>> _idx_from_char([(0,0,3),(1,4,7)], 5)
    1
    """
    for idx, start, end in ranges:
        if start <= col < end:
            return idx  # may be None for *** fillers
    return None


def _idx_from_next(ranges: List[Tuple[int, int, int]], col: int) -> Optional[int]:
    r"""
    Finds the first token index that starts at or after a given character column.

    Args:
        ranges (List[Tuple[int, int, int]]): Output from `_token_ranges`.
        col (int): The character column index.

    Returns:
        Optional[int]: Index of the first token starting at or after `col`, or None.

    Example
    -------
    >>> _idx_from_next([(0,0,3),(1,4,7)], 3)
    1
    """

    after = [
        (idx, start) for idx, start, _ in ranges if idx is not None and start >= col
    ]
    return min(after, key=lambda p: p[1])[0] if after else None


def _parse_block(
    block: str, ref_offset: int, hyp_offset: int
) -> Tuple[List[Triplet], int, int]:
    r"""
    Parses a SCLITE output alignment block to extract operation triplets and token counts.

    Each block contains reference (REF), hypothesis (HYP), and evaluation (Eval) lines
    with alignment markers ('S', 'I', 'D'). This function aligns the markers to tokens,
    mapping local indices to global indices using provided offsets.

    Args:
        block (str): SCLITE block string for one segment.
        ref_offset (int): Global offset to add to reference token indices.
        hyp_offset (int): Global offset to add to hypothesis token indices.

    Returns:
        Tuple[
            List[Tuple[str, Optional[int], Optional[int]]],
            int,
            int
        ]:
            - List of (op, global_ref_idx, global_hyp_idx) triplets, where op is in {'S','I','D'}
            - Number of reference tokens in block
            - Number of hypothesis tokens in block

    Notes
    -----
    - 'S' = substitution, 'I' = insertion, 'D' = deletion.
    - Token indices are made global by adding the supplied offsets.

    Example
    -------
    >>> _parse_block(block_string, 10, 20)
    ([('S', 10, 20), ('D', 11, None)], 2, 1)
    """

    lines = block.rstrip("\n").splitlines()
    ref_line = next((l for l in lines if l.startswith("REF:")), None)  # noqa: E741
    hyp_line = next((l for l in lines if l.startswith("HYP:")), None)  # noqa: E741
    eval_line = next((l for l in lines if l.startswith("Eval:")), None)  # noqa: E741
    if not (ref_line and hyp_line and eval_line):
        return [], 0, 0

    ref_txt = ref_line[ref_line.find(":") + 1 :]
    hyp_txt = hyp_line[hyp_line.find(":") + 1 :]
    eval_txt = eval_line[eval_line.find(":") + 1 :]

    ref_ranges, n_ref_real = _token_ranges(ref_txt)
    hyp_ranges, n_hyp_real = _token_ranges(hyp_txt)

    trips: List[Triplet] = []

    for col, mark in enumerate(eval_txt):
        if mark not in "SID":
            continue

        ref_idx = _idx_from_char(ref_ranges, col)
        hyp_idx = _idx_from_char(hyp_ranges, col)

        # fall‑back if marker sits in white‑space or in a *** token
        if mark == "S":
            if ref_idx is None:
                ref_idx = _idx_from_next(ref_ranges, col)
            if hyp_idx is None:
                hyp_idx = _idx_from_next(hyp_ranges, col)
        elif mark == "D":
            if ref_idx is None:
                ref_idx = _idx_from_next(ref_ranges, col)
        elif mark == "I":
            if hyp_idx is None:
                hyp_idx = _idx_from_next(hyp_ranges, col)

        g_ref = ref_idx + ref_offset if ref_idx is not None else None
        g_hyp = hyp_idx + hyp_offset if hyp_idx is not None else None
        trips.append((mark, g_ref, g_hyp))

    return trips, n_ref_real, n_hyp_real


def parse_sclite(sclite_out: str) -> List[Triplet]:
    r"""
    Parses the full output of SCLITE to extract global triplets of alignment operations.

    The function splits the output into blocks, parses each one, and returns a
    concatenated list of (op, global_ref_idx, global_hyp_idx) triplets.

    Args:
        sclite_out (str): Multi-block SCLITE output as a string.

    Returns:
        List[Tuple[str, Optional[int], Optional[int]]]: List of
        (operation, global reference index, global hypothesis index) triplets.

    Example
    -------
    >>> parse_sclite(sclite_output)
    [('S', 0, 0), ('I', None, 1), ...]
    """
    blocks = re.split(r"\n(?=id: \()", sclite_out.strip())
    ref_offset = hyp_offset = 0
    all_trips: List[Triplet] = []

    for blk in blocks:
        trips, n_ref_real, n_hyp_real = _parse_block(blk, ref_offset, hyp_offset)
        all_trips.extend(trips)
        ref_offset += n_ref_real
        hyp_offset += n_hyp_real

    return all_trips


def debug_embedded_lang_errors(
    stm: Union[Path, str],
    ctm: Union[Path, str],
    triplets: List[Triplet],
    trs_wordId2lang: Dict[str, str],
    stm_dir: Union[Path, str],
    filename_stem: Union[Path, str],
) -> None:
    with open(stm, "r", encoding="utf-8") as f:
        stm_text = f.read()
    ref_words = []
    for line in stm_text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 6:
            ref_words.extend(parts[5:])

    with open(ctm, "r", encoding="utf-8") as f:
        ctm_text = f.read()
    hyp_words = []
    for line in ctm_text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            hyp_words.append(parts[4])

    debug_lines = []
    for op, i_ref, i_hyp in triplets:
        if i_ref is not None and str(i_ref) in trs_wordId2lang and trs_wordId2lang[str(i_ref)] == EMBEDDED_LANG:
            if op == "S":
                ref_word = ref_words[i_ref] if i_ref < len(ref_words) else "[OOR]"
                hyp_word = (
                    hyp_words[i_hyp]
                    if i_hyp is not None and i_hyp < len(hyp_words)
                    else "[OOR]"
                )
                debug_lines.append(f"S\t{ref_word}\t{hyp_word}")
            elif op == "D":
                ref_word = ref_words[i_ref] if i_ref < len(ref_words) else "[OOR]"
                debug_lines.append(f"D\t{ref_word}\t[DELETED]")
            elif op == "I":
                hyp_word = (
                    hyp_words[i_hyp]
                    if i_hyp is not None and i_hyp < len(hyp_words)
                    else "[OOR]"
                )
                debug_lines.append(f"I\t[INSERTED]\t{hyp_word}")
    debug_path = stm_dir / f"{filename_stem}_embedded_debug.txt"
    with open(debug_path, "w", encoding="utf-8") as dbg:
        dbg.write("OP\tREF_WORD\tHYP_WORD\n")
        for line in debug_lines:
            dbg.write(line + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute Word Error Rate (WER) for Embedded Language segments in code-switching ASR transcripts using SCLITE.\n\n"
            "This script processes reference transcriptions in .trs (Transcriber XML) format, "
            "where embedded language spans are explicitly marked, and hypothesis transcriptions in .ctm format. "
            "It converts reference .trs files to .stm format (required by SCLITE), runs SCLITE for alignment, "
            "and reports overall and embedded-language-specific WER metrics.\n\n"
            "Arguments:\n"
            "  --trs-ref-dir-path : Directory containing .trs reference files (with embedded language tags)\n"
            "  --ctm-hyp-dir-path : Directory containing .ctm hypothesis files\n"
            "  --stm-ref-dir-path : Directory to output .stm reference files for SCLITE\n"
            "\n"
            "Example usage:\n"
            "  python code_switching_wer.py \\\n"
            "    --trs-ref-dir-path ./data/refs \\\n"
            "    --ctm-hyp-dir-path ./data/hyps \\\n"
            "    --stm-ref-dir-path ./data/stm"
        )
    )
    parser.add_argument(
        "--trs-ref-dir-path",
        type=str,
        required=True,
        help="Directory containing .trs reference (REF) files with embedded language markup.",
    )

    parser.add_argument(
        "--ctm-hyp-dir-path",
        type=str,
        required=True,
        help="Directory containing .ctm hypothesis (HYP) files.",
    )

    parser.add_argument(
        "--stm-ref-dir-path",
        type=str,
        required=True,
        help="Directory for outputting .stm reference (REF) files (required by SCLITE).",
    )

    parser.add_argument(
        "--debug-embedded-lang-errors",
        type=bool,
        required=False,
        default=False,
        help="Whether to print out errors made in the embedded language words",
    )

    return parser.parse_args()


def main():
    r"""
    Main routine for code-switching WER evaluation with SCLITE.
    """

    args = parse_args()

    trs_dir = Path(args.trs_ref_dir_path)
    ctm_dir = Path(args.ctm_hyp_dir_path)
    stm_dir = Path(args.stm_ref_dir_path)

    try:
        stm_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Failed to create STM directory '{stm_dir}': {e}")
        return

    try:
        trs_files = list(trs_dir.glob("*.trs"))
    except Exception as e:
        print(f"Failed to list .trs files in '{trs_dir}': {e}")
        return

    if len(trs_files) == 0:
        print(f"Zero .trs files were found in {trs_dir}. Exiting.")
        return

    N_ref_total = 0
    S_total = 0
    I_total = 0
    D_total = 0

    N_ref_embedded = 0
    S_embedded = 0
    I_embedded = 0
    D_embedded = 0

    for trs in tqdm(trs_files, desc="Computing WER", dynamic_ncols=True):
        try:
            filename_stem = trs.stem
            out_stm = stm_dir / f"{filename_stem}{STM_SUFFIX}"
            hyp_ctm = ctm_dir / f"{filename_stem}{CTM_SUFFIX}"
            trs_to_stm(trs_file=trs, out_path_stm=out_stm)
            word_id2lang_map, word2lang = trs_wordId2lang(trs_file=trs)

            if not hyp_ctm.exists():
                print(
                    f"Warning: Hypothesis CTM file does not exist: {hyp_ctm}. Skipping."
                )
                continue

            # Create normalized CTM file for consistent Estonian processing
            normalized_ctm = stm_dir / f"{filename_stem}_normalized{CTM_SUFFIX}"
            normalize_ctm(input_ctm=hyp_ctm, output_ctm=normalized_ctm)

            cmd = [
                "sclite",
                "-r",
                str(out_stm),
                "stm",
                "-h",
                str(normalized_ctm),
                "ctm",
                "-i",
                "rm",
                "-c",
                "NOASCII",
                "-o",
                "all",
                "stdout",
                "-F",
                "-D",
            ]

            try:
                sclite_out = run(cmd, capture_output=True, check=True, text=True).stdout
            except CalledProcessError as e:
                print(
                    f"Failed to run SCLITE on {out_stm} and {normalized_ctm}.\nError: {e.stderr}"
                )
                continue
            triplets = parse_sclite(sclite_out=sclite_out)
            N_ref_total += len(word_id2lang_map)
            N_ref_embedded += sum(
                1 for v in word_id2lang_map.values() if v == EMBEDDED_LANG
            )

            for op, i_ref, _ in triplets:
                if op == "S":
                    S_total += 1
                elif op == "I":
                    I_total += 1
                elif op == "D":
                    D_total += 1

                if i_ref is not None and str(i_ref) in word_id2lang_map and word_id2lang_map[str(i_ref)] == EMBEDDED_LANG:
                    if op == "S":
                        S_embedded += 1
                    elif op == "I":
                        I_embedded += 1
                    elif op == "D":
                        D_embedded += 1
            if args.debug_embedded_lang_errors:
                debug_embedded_lang_errors(
                    stm=out_stm,
                    ctm=normalized_ctm,
                    triplets=triplets,
                    trs_wordId2lang=word_id2lang_map,
                    stm_dir=stm_dir,
                    filename_stem=filename_stem,
                )

        except Exception as e:
            print(f"Error processing file '{trs}': {e}")
            continue

    if N_ref_total == 0:
        print("No reference words found. Exiting.")
        return

    WER = (D_total + I_total + S_total) / N_ref_total * 100
    
    print(f"SCLITE General WER: {WER:.2f}%")
    
    if N_ref_embedded > 0:
        CSW_WER = (D_embedded + I_embedded + S_embedded) / N_ref_embedded * 100
        MATRIX_WER = ((D_total + I_total + S_total) - (D_embedded + I_embedded + S_embedded)) / (N_ref_total - N_ref_embedded) * 100
        print(f"SCLITE Embedded Language WER: {CSW_WER:.2f}%")
        print(f"SCLITE Matrix Language WER: {MATRIX_WER:.2f}%")
    else:
        print("No embedded language words found - showing general WER only.")


if __name__ == "__main__":
    main()
