from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .mimic import iter_sentence_candidates


def build_parser():
    p = argparse.ArgumentParser(description='Extract unlabeled candidate sentences from MIMIC notes')
    p.add_argument('--notes', required=True, help='Path to NOTEEVENTS.csv.gz')
    p.add_argument('--output', required=True, help='Output CSV path')
    p.add_argument('--max-words', type=int, default=128)
    p.add_argument('--max-sentences-per-note', type=int, default=25, help='Optional cap on emitted sentences per note (0 = no cap)')
    p.add_argument('--limit', type=int, default=0, help='Optional cap on emitted rows (0 = no cap)')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['row_id', 'text'])
        for note_id, sent in iter_sentence_candidates(
            args.notes,
            max_words=args.max_words,
            max_sentences_per_note=args.max_sentences_per_note,
        ):
            writer.writerow([f'{note_id}:{emitted}', sent])
            emitted += 1
            if args.limit and emitted >= args.limit:
                break
    print(f'wrote {emitted} unlabeled candidates to {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
