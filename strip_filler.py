#!/usr/bin/env python3
"""Remove meta/filler sentences from PACKaGE paragraphs.

Drops sentences such as "No further details are provided.", "The context is unclear.",
"These are separate statements." from paragraph_plain / paragraph_role / paragraph_rr
in lockstep. Sentences with a causal verb (gave rise to, resulted in, ...) are never
removed. Node tags inside a removed sentence are dropped as a pair; a tag that opens
or closes outside the sentence is kept, so the remaining tags stay balanced.

  python3 strip_filler.py --input "data/package_dataset 4_uncleaned.csv" --output "data/package_dataset 4.csv"
  (omit --output for a dry run that only prints counts)
"""
import argparse, csv, re, collections

csv.field_size_limit(10**7)
COLS = ['paragraph_plain', 'paragraph_role', 'paragraph_rr']
TAGRE = r'</?(?:source|intermediate|leaf)[^>]*>'
TAG = re.compile(TAGRE)

KEY = re.compile(r'\b(details?|context|contexts|information|specifics?|circumstances)\b', re.I)
NEG = re.compile(r'\b(no|not|never|unclear|unspecified|unavailable|unknown|undisclosed|limited|lacks?|lacking|remains?|absent|missing)\b', re.I)
ST  = re.compile(r'\bstatements?\b', re.I)
# real causal content must never be treated as filler
CAUSAL = re.compile(r'\b(gave rise to|gives rise to|give rise to|resulted in|results in|result in|led to|leads to|brought about|brings about|triggered|triggers|caused|enabled|hindered|contributed to|induced)\b', re.I)
FAM = [re.compile(p, re.I) for p in [
    r'^(these|they|those|both)\s+(are|were)\s+(two\s+)?(presented\s+as\s+|reported\s+as\s+)?(separate|distinct|brief|general|high-level|independent)\s+(observations?|points?|notes?|summaries|descriptions?)\.$',
    r'^(these|those|both)\s+(observations?|points?)\s+(are|were)\s+(presented|noted|stated)\s+(separately|independently|succinctly|for\s+clarity)\.$',
    r'^they\s+(are|were)\s+presented\s+as\s+separate\s+(points?|observations?)\.$',
    r'^each\s+(observation|point|note)\s+(stands\s+on\s+its\s+own|stands\s+alone|is\s+presented\s+independently)\.$',
    r'^(this|it)\s+is\s+a\s+(brief|concise|general|short)\s+(summary|note|observation|description)\.$',
    r'^these\s+appear\s+to\s+be\s+separate\s+observations\.$',
    r'^both\s+points\s+are\s+stated\s+succinctly\.$',
]]
SPLIT = re.compile(r'(?<=[.!?])\s+|(?<=[.!?]["”])\s+')


def is_filler(s):
    if not s.endswith('.') or len(s.split()) > 14 or CAUSAL.search(s):
        return False
    if KEY.search(s) and NEG.search(s): return 'detail'
    if ST.search(s):                    return 'statement'
    if any(p.match(s) for p in FAM):    return 'family'
    return False


def sent_regex(s):
    """char-level tag-tolerant matcher: tags may sit anywhere, even mid-word"""
    parts, opt = [], r'(?:\s*(?:' + TAGRE + r')\s*)*'
    for ch in s:
        parts.append(r'(?:\s|' + TAGRE + r')+' if ch.isspace() else re.escape(ch) + opt)
    parts[-1] = re.escape(s[-1])   # no trailing tag-slurp past the final '.'
    return re.compile(''.join(parts))


def valid_matches(text, rx):
    out = []
    lead = re.compile(r'(?:\s|' + TAGRE + r')+$')
    for m in rx.finditer(text):
        start, pre = m.start(), text[:m.start()]
        lm = lead.search(pre)          # pull preceding tags/space into the span
        if lm: start, pre = lm.start(), pre[:lm.start()]
        if pre and pre[-1] not in '.!?"”’': continue
        if m.end() < len(text) and not text[m.end()].isspace(): continue
        out.append((start, m.end()))
    return out


def balance(inner):
    """tags to keep when deleting `inner`: orphan closers (their opener is before the
    span) stay, orphan openers (their closer is after) stay; fully-inside pairs go."""
    stack, orphan_close = [], []
    for m in TAG.finditer(inner):
        t = m.group(0)
        if t.startswith('</'):
            if stack: stack.pop()
            else: orphan_close.append(t)
        else:
            stack.append(t)
    kept = orphan_close + stack
    dropped_pairs = (len(TAG.findall(inner)) - len(kept)) // 2
    return (' ' + ' '.join(kept) + ' ' if kept else ''), dropped_pairs


def cut(text, spans):
    dropped = 0
    for a, b in sorted(spans, reverse=True):
        rep, d = balance(text[a:b])
        dropped += d
        text = text[:a] + rep + text[b:]
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', default=None)
    args = ap.parse_args()

    with open(args.input, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    removed = collections.Counter()
    pairs_dropped = 0
    mismatch = 0
    for r in rows:
        cands = {}
        for s in SPLIT.split(r['paragraph_plain']):
            s = s.strip()
            k = is_filler(s)
            if k: cands[s] = k
        if not cands: continue
        spans = {c: [] for c in COLS}
        for s, kind in cands.items():
            rx = sent_regex(s)
            ms = {c: valid_matches(r[c], rx) for c in COLS}
            n = len(ms['paragraph_plain'])
            if n == 0: continue
            if any(len(ms[c]) != n for c in COLS): mismatch += 1
            for c in COLS: spans[c].extend(ms[c])
            removed[(kind, s)] += n
        for c in COLS:
            if spans[c]:
                r[c], d = cut(r[c], spans[c])
                pairs_dropped += d

    if args.output:
        with open(args.output, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows)
        print('WROTE', args.output)

    per = collections.Counter()
    for (k, s), n in removed.items(): per[k] += n
    print('removed per column:', sum(removed.values()), dict(per), '| unique:', len(removed))
    print('tag pairs dropped (across all 3 cols):', pairs_dropped, '| column count mismatches:', mismatch)


if __name__ == '__main__':
    main()
