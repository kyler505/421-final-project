from __future__ import annotations

import argparse
from pathlib import Path

from .mimic import iter_sentence_candidates
from .unlabeled_cache import UnlabeledCandidate, filter_candidates, sentence_hash, write_cache


def build_parser():
    p = argparse.ArgumentParser(description='Extract unlabeled candidate sentences from MIMIC notes')
    p.add_argument('--notes', required=True, help='Path to NOTEEVENTS.csv.gz')
    p.add_argument('--output', required=True, help='Output CSV path')
    p.add_argument('--max-words', type=int, default=128)
    p.add_argument('--max-sentences-per-note', type=int, default=25, help='Optional cap on emitted sentences per note (0 = no cap)')
    p.add_argument('--limit', type=int, default=0, help='Optional cap on emitted rows (0 = no cap)')
    p.add_argument('--candidate-cap', type=int, default=0, help='Optional cap on kept candidates after filtering')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    rows: list[UnlabeledCandidate] = []
    for note_id, sent in iter_sentence_candidates(
        args.notes,
        max_words=args.max_words,
        max_sentences_per_note=args.max_sentences_per_note,
    ):
        row_id = f'{note_id}:{emitted}'
        rows.append(
            UnlabeledCandidate(
                row_id=row_id,
                note_id=str(note_id),
                sentence=sent,
                sentence_hash=sentence_hash(sent),
                word_count=len(sent.split()),
            )
        )
        emitted += 1
        if args.limit and emitted >= args.limit:
            break
    kept, dropped = filter_candidates(rows, per_note_cap=args.max_sentences_per_note, candidate_cap=args.candidate_cap)
    write_cache(kept, out_path)
    print(f'wrote {len(kept)} kept candidates to {out_path}')
    if dropped:
        audit_path = out_path.with_suffix('.audit.csv')
        write_cache(dropped[:1000], audit_path)
        print(f'wrote {len(dropped)} dropped-candidate samples to {audit_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
