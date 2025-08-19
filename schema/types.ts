export interface Word {
  word_with_punctuation: string;
  start: number;
  end: number;
  word: string;
  punctuation: string;
}

export interface Turn {
  speaker: string;
  start: number;
  end: number;
  transcript: string;
  words: Word[];
  unnormalized_transcript: string;
}

export interface Section {
  type: "speech";
  start: number;
  end: number;
  turns?: Turn[];
}

export interface Speaker {
  [speakerId: string]: Record<string, unknown>;
}

export interface BestHypothesis {
  speakers: Speaker;
  sections: Section[];
}

export interface Alternative {
  rank: number;
  text: string;
  avg_logprob: number;
}

export interface Segment {
  start: number;
  end: number;
  alternatives: Alternative[];
}

export interface AlternativesData {
  language: string;
  segments: Segment[];
}

export interface Metadata {
  n_best: number;
  has_word_alignments: boolean;
  basename: string;
}

export interface TranscriptionResult {
  best_hypothesis: BestHypothesis;
  alternatives: AlternativesData;
  metadata: Metadata;
}