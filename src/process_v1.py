'''

Run with:

% python src/process_v1.py data/mkw.tsv data/tsv/GO_NSAF/GO_NSAF.tsv

'''

import sys

import os
import os.path as op
import errno as er

import numpy as np
import pandas as pd

import re
import decimal as de

# ---------------------------------------------------------------------------

SEP = '~'

PPFX = 'S'
REFPFX = PPFX + '0'
REFSFX = 'C0'
CTRLSFXS = set(('C0', 'C1'))

PPFX = 'Set'
REFPFX = PPFX + '1'
REFSFX = 'cq_126_sn_sum'
CTRLSFXS = set(('cq_126_sn_sum', 'cq_131_sn_sum'))

AVGSFX = 'AV'
PRINT_SUMS = False

# ---------------------------------------------------------------------------

def parse_hdr(h, sep=SEP):
    return tuple(h.split(sep) if sep in h else [h, ''])

def build_hdr(pfx, sfx):
    return SEP.join([pfx, sfx])

def get_headers_for_prefix(df, pfx, omit_avcol=False):
    keep = lambda x: x[0] == pfx
    if omit_avcol:
        k0 = keep
        keep = lambda x: k0(x) and x[1] != AVGSFX

    return [build_hdr(*ps) for ps in
            filter(keep,
                   [parse_hdr(c) for c in df.columns])]

def get_pfxs(df):
    return set([p for p, s in [parse_hdr(c) for c in df.columns]
                if s in CTRLSFXS])

def get_first_control_col_pfxs(df):
    return [p for p, s in [parse_hdr(c) for c in df.columns] if s == REFSFX]

# ---------------------------------------------------------------------------

def split_first_col(df):
    tmp = np.vstack(df.iloc[:, 0].map(lambda x: np.array(re.split(r'\|', x))).values)
    df0 = df.iloc[:, [0]]
    df1 = pd.DataFrame(tmp[:, [1, 0]], columns=('Uniprot', 'Sp/Tr'), index=df.index)
    df2 = df.iloc[:, 1:]
    return pd.concat((df0, df1, df2), axis=1)

def dropcols(df, todrop):
    if not callable(todrop):
        if isinstance(todrop, basestring):
            todrop = (todrop,)
        s = set(todrop)
        assert s.issubset(set(df.columns))
        todrop = lambda x: x in s
    return df.loc[:, filter(lambda x: not todrop(x), df.columns)]

# ---------------------------------------------------------------------------

def refsum(df, pfx):
    return float(df.loc[:, build_hdr(pfx, AVGSFX)].sum())

def diagframe(v, labels=None):
    if labels is None:
        labels = v.index
    return pd.DataFrame(np.diag(v), columns=labels, index=labels)

def colscale(df, tgt, s):
    assert len(s) == len(tgt)
    rawvals = df.loc[:, tgt]
    ret = df.copy()
    ret.loc[:, tgt] = rawvals.dot(diagframe(s))
    return ret

def rowscale(df, s):
    return diagframe(s).dot(df)

# ---------------------------------------------------------------------------

def mkdirp(path):
    try:
        os.makedirs(path)
    except OSError, e:
        if e.errno != er.EEXIST: raise

def output_dir(inputfile, _basename=lambda p: op.splitext(op.basename(p))[0]):
    return op.join('data', _basename(__file__), _basename(inputfile))

def output_path(output_dir, basename):
    return op.join(output_dir, re.sub(r'(\.\w+)?$', '.tsv', basename))

def to_tsv(df):
    return df.to_csv(index=False, sep='\t').rstrip('\n')

to_hex = np.vectorize(float.hex)
to_dec = np.vectorize(lambda f: 'inf' if f == np.inf else str(de.Decimal(f)))

def sortcols(df):

    cols = df.columns
    peptcolq = lambda c: c.endswith(' Peptides')
    minpeptcol = min(filter(lambda i: peptcolq(cols[i]), range(len(cols))))

    def key((i, hdr)):

        isdatacol = SEP in hdr
        ispeptcol = peptcolq(hdr)

        assert not (isdatacol and ispeptcol)

        if not (isdatacol or ispeptcol):
            return (0, i)

        pfx, sfx = parse_hdr(hdr, sep=SEP if isdatacol else ' ')

        idx = re.sub('^' + PPFX, '', pfx)
        try:
            n, s = int(idx), ''
        except ValueError, e:
            if not e.message.startswith('invalid literal for int() with base 10'):
                raise
            n, s = sys.maxint, idx

        if ispeptcol:
            return (0, minpeptcol, n, s, sfx)

        if sfx == AVGSFX:
            m = -3
        elif sfx == REFSFX:
            m = -2
        elif sfx in CTRLSFXS:
            m = -1
        else:
            m = i

        return (1, n, s, m, sfx)

    return df.loc[:, [c for _, c in sorted(enumerate(cols), key=key)]]

def write_tbl(df, output_dir, basename):
    df = df.copy()
    normalized_cols = [c for c in df.columns if SEP in c]

    # sums = '\t'.join(to_hex(df.loc[:, normalized_cols].sum(axis=0)))
    sums = '\t'.join(to_dec(df.loc[:, normalized_cols].sum(axis=0)))

    # df.loc[:, normalized_cols] = to_hex(df.loc[:, normalized_cols])
    df.loc[:, normalized_cols] = to_dec(df.loc[:, normalized_cols])

    out = to_tsv(df)

    out = re.sub(r'(\.\d*[^0\D])0+(?=[\t\n]|$)', r'\1', out, re.M)
    out = re.sub(r'\.0*(?=[\t\n]|$)', '', out, re.M)
    outpath = output_path(output_dir, basename)
    with open(outpath, 'w') as fh:
        print >> fh, out

        if PRINT_SUMS:
            w = df.shape[1]
            for _ in 0, 1:
                print >> fh, '\t' * (w-1)

            print >> fh, '%s\t%s' % ('\t'.join(['' for c in df.columns
                                                if not c in normalized_cols]), sums)

    print >> sys.stderr, outpath

# ---------------------------------------------------------------------------

def add_av_cols(df):
    df0 = df.copy()
    for pfx in get_pfxs(df0):
        avcolhdr = build_hdr(pfx, AVGSFX)
        ctrlhdrs = [build_hdr(pfx, sfx) for sfx in CTRLSFXS]
        df0.loc[:, avcolhdr] = df0.loc[:, ctrlhdrs].mean(axis=1)
    return df0

def normalization_a(df):
    for pfx in get_pfxs(df):
        n = refsum(df, pfx)
        tgt = get_headers_for_prefix(df, pfx, pfx == REFPFX)
        d = df.loc[:, tgt].sum()
        df = colscale(df, tgt, n/d)
    return df

def normalization_b(df):
    df0 = df.copy()
    num = refsum(df, REFPFX)
    for pfx in get_first_control_col_pfxs(df0):
        s = num/refsum(df, pfx)
        tgt = get_headers_for_prefix(df0, pfx, pfx == REFPFX)
        df0.loc[:, tgt] = s * df0.loc[:, tgt].values
    return df0

def normalization_ab(df):
    tgt = sum([get_headers_for_prefix(df, pfx, pfx == REFPFX) for pfx in
               get_first_control_col_pfxs(df)], [])
    refcol = build_hdr(REFPFX, AVGSFX)
    s = float(df.loc[:, refcol].sum())/df.loc[:, tgt].sum()
    return colscale(df, tgt, s)

def normalization_c(df):

    df0 = df.copy()

    refsfx = AVGSFX

    refcol = build_hdr(REFPFX, refsfx)
    ref = df0.loc[:, refcol].astype(np.float64)
    idx = df0.index

    for pfx in get_pfxs(df0):
        controlcol = build_hdr(pfx, refsfx)

        hh = (df0.loc[:, controlcol] != 0)
        ii = idx[hh]

        s = ref[ii]/(df0.loc[ii, controlcol].astype(np.float64))
        jj = get_headers_for_prefix(df, pfx, pfx == REFPFX)

        df0.loc[ii, jj] = rowscale(df0.loc[ii, jj], s)

        if np.any(~hh):
            ii0 = idx[~hh]
            df0.loc[ii0, jj] = np.inf

    return df0

# ---------------------------------------------------------------------------

def main(inputfile, go_nsaf_file, refpfx=None):
    global REFPFX

    if refpfx is None:
        refpfx = REFPFX
    else:
        REFPFX = refpfx

    global PPFX
    PPFX = re.search(r'^(.*?)\d+$', REFPFX).group(1)

    df = pd.io.parsers.read_table(inputfile, sep='\t')

    outdir = output_dir(inputfile)
    mkdirp(outdir)

    # 1. remove 'reverse hits'
    # 2. remove contaminant proteins
    df = df[df['Protein Id'].str.contains(r'^(?!##).*(?<!_contaminant)$')]
    
    # 3. separate Uniprot identifiers
    df = split_first_col(df)
    df.index = df.iloc[:, 0]

    # 5. remove data not relevant to main dataset
    # (minus the relabeling of data columns: the proposed relabeling
    # scheme(s) are not bijective)
    df = dropcols(df,
                  lambda x: re.match(r'Set(?:(?:12|AtoD|EtoH)\b|5~cq_1(?:28|30)[ab]_sn_sum$)',
                                     x))

    # --------------------------------------------------------------------------

    godf = dropcols(pd.io.parsers.read_table(go_nsaf_file, sep='\t'), 'Gene')

    godf.columns = [re.sub(r'(?<=^Uniprot)-l$', '', s) for s in godf.columns]

    assert set.issuperset(set(godf.loc[:, 'Uniprot']),
                          set(df.loc[:, 'Uniprot']))

    # 6. add GO terms
    df = df.merge(godf.loc[:, ['Uniprot', 'GOCC']],
                  on='Uniprot', how='inner')

    # 7. add spectral counts data
    # OMITTED

    # --------------------------------------------------------------------------

    df = add_av_cols(df)

    # 4. reorder the data columns
    df = sortcols(df)
    write_tbl(df, outdir, 'prenorm')

    # (just for qc)
    dfab = normalization_ab(df)

    # 8.
    df = normalization_a(df)
    write_tbl(df, outdir, 'norm_A')

    # 9.
    df = normalization_b(df)
    write_tbl(df, outdir, 'norm_B')

    def chk(x, y):
        k = sum([get_headers_for_prefix(x, pfx) for pfx in
                 get_first_control_col_pfxs(x)], [])
        assert np.max(np.abs((x.loc[:, k] - y.loc[:, k]).values.ravel())) < 1e-9
    chk(dfab, df)
    del dfab

    # 10.
    df = normalization_c(df)
    write_tbl(df, outdir, 'norm_C')

    # 11. remove all control data apart from Set1 controls
    # OMITTED
    # 12. sort data such that proteins with >= 1 peptides quantified in all samples are analyzed first
    # OMITTED
    # 13. normalize data for each protein to a maximum of 1
    # OMITTED

if __name__ == "__main__":
    # PRINT_SUMS = True

    main(*sys.argv[1:])
