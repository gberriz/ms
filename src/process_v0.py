import numpy as np
import pandas as pd
import re

# ---------------------------------------------------------------------------

import os.path as op
import os
import errno as er

def mkdirp(path):
    try:
        os.makedirs(path)
    except OSError, e:
        if e.errno != er.EEXIST: raise

def _basename(path):
    return op.splitext(op.basename(path))[0]

def output_dir(inputfile):
    return op.join('data', _basename(__file__), _basename(inputfile))

def output_path(output_dir, basename):
    return op.join(output_dir, re.sub(r'(\.\w+)?$', '.tsv', basename))

def write_tbl(df, output_dir, basename):
    # out = re.sub(r'(\t\d+)\.(?:0+|0{5,}\d+)?(?:\t|\n|\z)', '\1',
    #              df.to_csv(index=False, sep='\t')).rstrip('\n')
    out = df.to_csv(index=False, sep='\t').rstrip('\n')
    out = re.sub(r'0{5,}\d*', '', out)
    out = re.sub(r'\.0*(?=[\t\n]|$)', '', out)
    outpath = output_path(output_dir, basename)
    #df.to_csv(outpath, index=False, float_format='%0.1f', sep='\t')
    with open(outpath, 'w') as fh:
        print >> fh, out

        w = df.shape[1]
        for _ in 0, 1:
            print >> fh, '\t' * (w-1)

        sums = '\t'.join(['%02.f' % x
                          for x in df.loc[:, [c for c in df.columns if SEP in c]].sum(axis=0)])

        print >> fh, '%s\t%s' % ('\t'.join(['' for c in df.columns if not SEP in c]), sums)

    import sys
    print >> sys.stderr, 'output to %s' % outpath

def split_first_col(df):
    tmp = np.vstack(df.iloc[:, 0].map(lambda x: np.array(re.split(r'\|', x))).values)
    df0 = pd.DataFrame(tmp[:, [1, 0]], columns=('Uniprot-l', 'db'), index=df.index)
    df1 = df.iloc[:, 1:]
    return pd.concat((df0, df1), axis=1)

def dropcols(df, todrop):
    if not callable(todrop):
        if isinstance(todrop, basestring):
            todrop = (todrop,)
        s = set(todrop)
        assert s.issubset(set(df.columns))
        todrop = lambda x: x in s
    return df.loc[:, filter(lambda x: not todrop(x), df.columns)]

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

def get_headers_for_prefix(df, pfx):
    return [build_hdr(*ps) for ps in
            filter(lambda x: x[0] == pfx,
                   [parse_hdr(c) for c in df.columns])]

def get_countscols(df):
    return sum([get_headers_for_prefix(df, pfx) for pfx in
                get_first_control_col_pfxs(df)], [])

def rowscale(df, s):
    return diagframe(s).dot(df)

def almost_equal(a, b, reltol=1e-6):
    d = np.max((np.max(np.abs(a)), np.max(np.abs(b))))
    if d == 0:
        return True
    n = np.max(np.abs(a - b))
    return n/d < reltol

# ---------------------------------------------------------------------------

#REFSFX = 'cq_126'
SEP = '~'
REFSET = 'S0'
REFSFX = 'C0'

# ---------------------------------------------------------------------------

def parse_hdr(h):
    return tuple(h.split(SEP) if SEP in h else [h, ''])

def build_hdr(pfx, sfx):
    return SEP.join([pfx, sfx])

# ---------------------------------------------------------------------------


def get_first_control_col_pfxs(df):
    return [p for p, s in [parse_hdr(c) for c in df.columns] if s == REFSFX]

def normalize_ab(df):
    tgt = get_countscols(df)
    refcol = build_hdr(REFSET, REFSFX)
    s = float(df.loc[:, refcol].sum())/df.loc[:, tgt].sum()
    return colscale(df, tgt, s)

def normalize_a(df):
    for pfx in get_first_control_col_pfxs(df):
        refcol = build_hdr(pfx, REFSFX)
        tgt = get_headers_for_prefix(df, pfx)
        s = float(df.loc[:, refcol].sum())/df.loc[:, tgt].sum()
        df = colscale(df, tgt, s)
    return df

def normalize_b(df):
    df0 = df.copy()
    refcol = build_hdr(REFSET, REFSFX)
    num = float(df0.loc[:, refcol].sum())
    for pfx in get_first_control_col_pfxs(df0):
        ctrlcol = build_hdr(pfx, REFSFX)
        s = num/float(df0.loc[:, ctrlcol].sum())
        tgt = get_headers_for_prefix(df0, pfx)
        df0.loc[:, tgt] = s * df0.loc[:, tgt].values
    return df0

def normalize_c(df):
    refcol = build_hdr(REFSET, REFSFX)
    df0 = df.copy()
    ref = df0.loc[:, refcol].astype(np.float64)
    idx = df.index
    for pfx in get_first_control_col_pfxs(df):
        controlcol = build_hdr(pfx, REFSFX)

        hh = (df0.loc[:, controlcol] != 0)
        ii = idx[hh]

        s = ref[ii]/df0.loc[ii, controlcol]
        jj = get_headers_for_prefix(df, pfx)

        df0.loc[ii, jj] = rowscale(df0.loc[ii, jj], s)

        if np.any(~hh):
            ii0 = idx[~hh]
            jj0 = [j for j in jj if j != controlcol]
            df0.loc[ii0, jj0] = np.inf

    return df0

def main(inputfile):
    df0 = pd.io.parsers.read_table(inputfile, sep='\t')
    df0.index = df.iloc[:, 0]
    ret = [df0]
    ret.append(normalize_a(df0, refsfx='a'))
    ret.append(normalize_b(ret[-1], refset='S0', refsfx='a'))
    ret.append(normalize_ab(df0, refset='S0', refsfx='a'))

    assert almost_equal(ret[-2].iloc[:, 2:].values,
                        ret[-1].iloc[:, 2:].values)

    ret.append(normalize_c(ret[-1], refset='S0', refsfx='a'))

    return ret

def main(inputfile):
    df = pd.io.parsers.read_table(inputfile, sep='\t')

    # 1. & 2.
    #df = df[df['Protein Id'].str.contains('^(?!##).*(?<!_contaminant)$')]

    # 3.
    #df = split_first_col(df)
    #df.index = df.iloc[:, 0]

    # 5.
    #df = dropcols(df, lambda x: re.match(r'^Set(?:12|AtoD|EtoH|5~cq_1(?:28|30)[ab]_sn_sum)\b', x))

    outdir = output_dir(inputfile)
    mkdirp(outdir)
    # 8.
    df = normalize_a(df)
    write_tbl(df, outdir, 'norm_A')

    # 9.
    df = normalize_b(df)
    write_tbl(df, outdir, 'norm_B')

    # 8. & 9.
    #df = normalize_ab(df)

    # 10.
    df = normalize_c(df)
    write_tbl(df, outdir, 'norm_C')

    return

    # godf_tsv = 'data/tsv/GO_NSAF/GO_NSAF.tsv'
    # godf = dropcols(pd.io.parsers.read_table(godf_tsv, sep='\t'), 'Gene')

    # assert set.issuperset(set(godf.loc[:, 'Uniprot-l']),
    #                       set(df.loc[:, 'Uniprot-l']))

    # # df = df.merge(godf.loc[:, ['Uniprot-l', 'GOCC']],
    # #               on='Uniprot-l', how='inner')

    # df = df.merge(godf, on='Uniprot-l', how='inner')

    # # cn = pd.io.parsers.read_table('data/tsv/convert_names.tsv', sep='\t')
    # # lkp = dict(zip(cn.iloc[:, 0], cn.iloc[:, 1]))

    # # df.columns = [lkp.get(c, c) for c in df.columns]

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
