import sys

import numpy as np
import pandas as pd
import re
import os.path as op
import os
import errno as er

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

def parse_hdr(h):
    return tuple(h.split(SEP) if SEP in h else [h, ''])

def build_hdr(pfx, sfx):
    return SEP.join([pfx, sfx])

def get_headers_for_prefix(df, pfx):
    return [build_hdr(*ps) for ps in
            filter(lambda x: x[0] == pfx,
                   [parse_hdr(c) for c in df.columns])]

def get_ctrl_headers_for_prefix(df, pfx):
    return [build_hdr(*ps) for ps in
            filter(lambda x: x[0] == pfx and x[1] in CTRLSFXS,
                   [parse_hdr(c) for c in df.columns])]

def get_pfxs(df):
    return set([p for p, s in [parse_hdr(c) for c in df.columns]
                if s in CTRLSFXS])

def get_first_control_col_pfxs(df):
    return [p for p, s in [parse_hdr(c) for c in df.columns] if s == REFSFX]

def get_countscols(df):
    return sum([get_headers_for_prefix(df, pfx) for pfx in
                get_first_control_col_pfxs(df)], [])

def refsum(df, pfx):
    return float(df.loc[:, build_hdr(pfx, AVGSFX)].sum())

def add_av_cols(df):
    df0 = df.copy()
    for pfx in get_pfxs(df0):
        avcolhdr = build_hdr(pfx, AVGSFX)
        ctrlhdrs = [build_hdr(pfx, sfx) for sfx in CTRLSFXS]
        df0.loc[:, avcolhdr] = df0.loc[:, ctrlhdrs].mean(axis=1)
    return df0

def normalize_a(df):
    for pfx in get_pfxs(df):
        tgt = get_headers_for_prefix(df, pfx)
        n = refsum(df, pfx)
        d = df.loc[:, tgt].sum()
        df = colscale(df, tgt, n/d)
    return df

def normalize_b(df):
    df0 = df.copy()
    num = refsum(df, REFPFX)
    for pfx in get_first_control_col_pfxs(df0):
        s = num/refsum(df, pfx)
        tgt = get_headers_for_prefix(df0, pfx)
        # tgt = [j for j in get_headers_for_prefix(df0, pfx)
        #        if not parse_hdr(j)[1] in CTRLSFXS.union((AVGSFX,))]
        df0.loc[:, tgt] = s * df0.loc[:, tgt].values
    return df0


def rowscale(df, s):
    return diagframe(s).dot(df)

def normalize_c(df):

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
        jj = get_headers_for_prefix(df, pfx)

        # jj = [j for j in get_headers_for_prefix(df0, pfx)
        #       if not parse_hdr(j)[1] in CTRLSFXS.union((AVGSFX,))]

        df0.loc[ii, jj] = rowscale(df0.loc[ii, jj], s)

        if np.any(~hh):
            ii0 = idx[~hh]

            if pfx == REFPFX:
                # leave zeros in the AV column of the reference set intact
                jj = filter(lambda p: parse_hdr(p)[1] != AVGSFX, jj)

            # jj0 = [j for j in jj
            #        if not parse_hdr(j)[1] in CTRLSFXS.union((AVGSFX,))]
            # df0.loc[ii0, jj0] = np.inf

            df0.loc[ii0, jj] = np.inf

    return df0

def almost_equal(a, b, reltol=1e-6):
    d = np.max((np.max(np.abs(a)), np.max(np.abs(b))))
    if d == 0:
        return True
    n = np.max(np.abs(a - b))
    return n/d < reltol

def main(inputfile):
    df0 = pd.io.parsers.read_table(inputfile, sep='\t')
    df0.index = df.iloc[:, 0]
    ret = [df0]
    ret.append(normalize_a(df0, refsfx='a'))
    ret.append(normalize_b(ret[-1], refpfx='S0', refsfx='a'))
    ret.append(normalize_ab(df0, refpfx='S0', refsfx='a'))

    assert almost_equal(ret[-2].iloc[:, 2:].values,
                        ret[-1].iloc[:, 2:].values)

    ret.append(normalize_c(ret[-1], refpfx='S0', refsfx='a'))

    return ret

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

def to_tsv(df):
    return df.to_csv(index=False, sep='\t', float_format='%0.2f').rstrip('\n')

def write_tbl(df, output_dir, basename):

    def key((i, hdr)):

        if not SEP in hdr:
            return (0, i)

        pfx, sfx = parse_hdr(hdr)

        idx = re.sub('^' + PPFX, '', pfx)
        try:
            n, s = int(idx), ''
        except ValueError, e:
            if not e.message.startswith('invalid literal for int() with base 10'):
                raise
            n, s = sys.maxint, idx

        if sfx == AVGSFX:
            m = -3
        elif sfx == REFSFX:
            m = -2
        elif sfx in CTRLSFXS:
            m = -1
        else:
            m = i

        return (1, n, s, m, sfx)

    sorted_cols = [c for _, c in sorted(enumerate(df.columns), key=key)]

    df0 = df.loc[:, sorted_cols]

    out = to_tsv(df0)

    out = re.sub(r'(\.\d*[^0\D])0+(?=[\t\n]|$)', r'\1', out)
    out = re.sub(r'\.0*(?=[\t\n]|$)', '', out)
    outpath = output_path(output_dir, basename)
    with open(outpath, 'w') as fh:
        print >> fh, out
        w = df0.shape[1]
        for _ in 0, 1:
            print >> fh, '\t' * (w-1)

        sums = '\t'.join(['%02.f' % x
                          for x in df0.loc[:, [c for c in sorted_cols if SEP in c]].sum(axis=0)])
        
        print >> fh, '%s\t%s' % ('\t'.join(['' for c in sorted_cols if not SEP in c]), sums)

    print >> sys.stderr, outpath

def DISABLED__main__DISABLED(inputfile):
    df = pd.io.parsers.read_table(inputfile, sep='\t')

    outdir = output_dir(inputfile)
    mkdirp(outdir)


    # --------------------------------------------------------------------------

    # 1. & 2.
    df = df[df['Protein Id'].str.contains('^(?!##).*(?<!_contaminant)$')]

    # 3.
    df = split_first_col(df)
    df.index = df.iloc[:, 0]

    # 5.
    df = dropcols(df, lambda x: re.match(r'^Set(?:12|AtoD|EtoH|5~cq_1(?:28|30)[ab]_sn_sum)\b', x))

    # --------------------------------------------------------------------------

    godf_tsv = 'data/tsv/GO_NSAF/GO_NSAF.tsv'
    godf = dropcols(pd.io.parsers.read_table(godf_tsv, sep='\t'), 'Gene')

    godf.columns = [re.sub(r'(?<=^Uniprot)-l$', '', s) for s in godf.columns]

    assert set.issuperset(set(godf.loc[:, 'Uniprot']),
                          set(df.loc[:, 'Uniprot']))

    df = df.merge(godf.loc[:, ['Uniprot', 'GOCC']],
                  on='Uniprot', how='inner')

    # --------------------------------------------------------------------------

    df = add_av_cols(df)
    write_tbl(df, outdir, 'prenorm')

    # 8.
    df = normalize_a(df)
    write_tbl(df, outdir, 'norm_a')

    # 9.
    #df = add_av_cols(df)
    df = normalize_b(df)
    write_tbl(df, outdir, 'norm_b')

    # 8. & 9.
    #df = normalize_ab(df)

    # 10.
    #df = add_av_cols(df)
    df = normalize_c(df)
    write_tbl(df, outdir, 'norm_c')

    return

    df = df.merge(godf, on='Uniprot-l', how='inner')

    # cn = pd.io.parsers.read_table('data/tsv/convert_names.tsv', sep='\t')
    # lkp = dict(zip(cn.iloc[:, 0], cn.iloc[:, 1]))

    # df.columns = [lkp.get(c, c) for c in df.columns]


def main(inputfile, refpfx=None):
    global REFPFX

    if refpfx is None:
        refpfx = REFPFX
    else:
        REFPFX = refpfx

    global PPFX
    PPFX = re.search(r'^(.*?)\d+$', REFPFX).group(1)

    df = pd.io.parsers.read_table(inputfile, sep='\t')

    #outdir = op.join(output_dir(inputfile), refpfx)
    outdir = output_dir(inputfile)
    #mkdirp(op.join(outdir, refpfx))
    mkdirp(outdir)

    # # 1. & 2.
    # df = df[df['Protein Id'].str.contains('^(?!##).*(?<!_contaminant)$')]

    # # 3.
    # df = split_first_col(df)
    # df.index = df.iloc[:, 0]

    # # 5.
    # df = dropcols(df, lambda x: re.match(r'^Set(?:12|AtoD|EtoH|5~cq_1(?:28|30)[ab]_sn_sum)\b', x))

    # --------------------------------------------------------------------------

    # godf_tsv = 'data/tsv/GO_NSAF/GO_NSAF.tsv'
    # godf = dropcols(pd.io.parsers.read_table(godf_tsv, sep='\t'), 'Gene')

    # godf.columns = [re.sub(r'(?<=^Uniprot)-l$', '', s) for s in godf.columns]

    # assert set.issuperset(set(godf.loc[:, 'Uniprot']),
    #                       set(df.loc[:, 'Uniprot']))

    # df = df.merge(godf.loc[:, ['Uniprot', 'GOCC']],
    #               on='Uniprot', how='inner')
    # --------------------------------------------------------------------------

    df = add_av_cols(df)
    write_tbl(df, outdir, 'prenorm')

    # 8.
    df = normalize_a(df)
    write_tbl(df, outdir, 'norm_A')

    # 9.
    # df = add_av_cols(df)
    df = normalize_b(df)
    write_tbl(df, outdir, 'norm_B')

    # 8. & 9.
    #df = normalize_ab(df)

    # 10.
    # if (REFPFX == 'Set4'):
    #     import pdb
    #     pdb.set_trace()
    #     pass
    # df = add_av_cols(df)
    df = normalize_c(df)
    write_tbl(df, outdir, 'norm_C')


if __name__ == "__main__":

    main(*sys.argv[1:])
