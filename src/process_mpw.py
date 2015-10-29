import numpy as np
import pandas as pd
import re
import os.path as op
import os
import errno as er

# ---------------------------------------------------------------------------

SEP = '~'
REFPFX = 'Set1'
REFSFX = 'cq_126_sn_sum'
AVGSFX = 'AVERAGE'
CTRLSFXS = set(('cq_126_sn_sum', 'cq_131_sn_sum'))

# ---------------------------------------------------------------------------

# def split_first_col(df):
#     tmp = np.vstack(df.iloc[:, 0].map(lambda x: np.array(re.split(r'\|', x))).values)
#     df0 = pd.DataFrame(tmp[:, [1, 0]], columns=('Uniprot', 'Sp/Tr'), index=df.index)
#     df1 = df.iloc[:, 1:]
#     return pd.concat((df0, df1), axis=1)

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

def normalize_ab(df):
    tgt = get_countscols(df)
    refcol = build_hdr(REFPFX, REFSFX)
    s = float(df.loc[:, refcol].sum())/df.loc[:, tgt].sum()
    return colscale(df, tgt, s)

# not this...
# def normalize_a(df):
#     for pfx in get_pfxs(df):
#         refcol = build_hdr(pfx, REFSFX)
#         tgt = get_headers_for_prefix(df, pfx)
#         s = float(df.loc[:, refcol].sum())/df.loc[:, tgt].sum()
#         df = colscale(df, tgt, s)
#     return df

# def refsum(df, pfx):
#     ctrlhdrs = [build_hdr(pfx, sfx) for sfx in CTRLSFXS]
#     return float(df.loc[:, ctrlhdrs].values.ravel().sum()/len(ctrlhdrs))

def refsum(df, pfx):
    return float(df.loc[:, build_hdr(pfx, AVGSFX)].sum())

def add_av_cols(df):
    df0 = df.copy()
    for pfx in get_pfxs(df0):
        avcolhdr = build_hdr(pfx, AVGSFX)
        ctrlhdrs = [build_hdr(pfx, sfx) for sfx in CTRLSFXS]
        df0.loc[:, avcolhdr] = df0.loc[:, ctrlhdrs].mean(axis=1)
    return df0

# nor this...
# def normalize_a(df):
#     for pfx in get_pfxs(df):
#         tgt = get_headers_for_prefix(df, pfx)
#         ctrlhdrs = [build_hdr(pfx, sfx) for sfx in CTRLSFXS]
#         n = refsum(df, pfx)
#         d = df.loc[:, tgt].sum()
#         df = colscale(df, tgt, n/d)
#     return df


def normalize_a(df):
    for pfx in get_pfxs(df):
        tgt = get_headers_for_prefix(df, pfx)
        n = refsum(df, pfx)
        d = df.loc[:, tgt].sum()
        df = colscale(df, tgt, n/d)
    return df



# def normalize_b(df):
#     df0 = df.copy()
#     refcol = build_hdr(REFPFX, REFSFX)
#     num = float(df0.loc[:, refcol].sum())
#     for pfx in get_first_control_col_pfxs(df0):
#         ctrlcol = build_hdr(pfx, REFSFX)
#         s = num/float(df0.loc[:, ctrlcol].sum())
#         tgt = get_headers_for_prefix(df0, pfx)
#         df0.loc[:, tgt] = s * df0.loc[:, tgt].values
#     return df0

def normalize_b(df):
    df0 = df.copy()
    refcol = build_hdr(REFPFX, REFSFX)
    num = refsum(df, REFPFX)
    for pfx in get_first_control_col_pfxs(df0):
        s = num/refsum(df, pfx)
        #tgt = get_headers_for_prefix(df0, pfx)
        tgt = [j for j in get_headers_for_prefix(df0, pfx)
               if not parse_hdr(j)[1] in CTRLSFXS.union((AVGSFX,))]
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
        jj = [j for j in get_headers_for_prefix(df0, pfx)
              if not parse_hdr(j)[1] in CTRLSFXS.union((AVGSFX,))]

        if pfx == 'Set3':
            import pdb
            pdb.set_trace()
            pass

        df0.loc[ii, jj] = rowscale(df0.loc[ii, jj], s)

        if np.any(~hh):
            ii0 = idx[~hh]
            # jj0 = [j for j in jj
            #        if not parse_hdr(j)[1] in CTRLSFXS.union((AVGSFX,))]
            # df0.loc[ii0, jj0] = np.inf
            df0.loc[ii0, jj] = np.inf

    return df0


# def normalize_c(df):

#     df0 = df.copy()

#     refsfx = REFSFX

#     refcol = build_hdr(REFPFX, refsfx)
#     ref = df0.loc[:, refcol].astype(np.float64)
#     idx = df.index

#     for pfx in get_pfxs(df0):
#         controlcol = build_hdr(pfx, refsfx)

#         hh = (df0.loc[:, controlcol] != 0)
#         ii = idx[hh]

#         s = ref[ii]/(df0.loc[ii, controlcol].astype(np.float64))
#         jj = [j for j in get_headers_for_prefix(df0, pfx)
#               if not parse_hdr(j)[1] in CTRLSFXS.union((AVGSFX,))]

#         df0.loc[ii, jj] = rowscale(df0.loc[ii, jj], s)

#         if np.any(~hh):
#             ii0 = idx[~hh]
#             # jj0 = [j for j in jj
#             #        if not parse_hdr(j)[1] in CTRLSFXS.union((AVGSFX,))]
#             # df0.loc[ii0, jj0] = np.inf
#             df0.loc[ii0, jj] = np.inf

#     return df0


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
    return op.join('data', 'processed', _basename(inputfile))

def output_path(output_dir, basename):
    return op.join(output_dir, re.sub(r'(\.\w+)?$', '.tsv', basename))

def write_tbl(df, output_dir, basename):
    outpath = output_path(output_dir, basename)

    def key((i, hdr)):
        if not SEP in hdr:
            return (0, i)
        pfx, sfx = parse_hdr(hdr)
        n = int(pfx[3:])
        if sfx == REFSFX:
            m = -3
        elif sfx in CTRLSFXS:
            m = -2
        elif sfx == AVGSFX:
            m = -1
        else:
            m = i
        return (1, -n, m)

    sorted_cols = [c for _, c in sorted(enumerate(df.columns), key=key)
                   if c == 'Uniprot' or SEP in c]

    df0 = df.loc[:, sorted_cols]

    #cn = pd.io.parsers.read_table('data/orig/Convert names_MH.tsv', sep='\t')
    #cn = pd.io.parsers.read_table('data/tsv/Convert names2/Sheet1.tsv', sep='\t')
    cn = pd.io.parsers.read_table('data/tsv/convert_names.tsv', sep='\t')
    lkp = dict(zip(cn.iloc[:, 0], cn.iloc[:, 1]))

    df0.columns = [lkp.get(c, c) for c in df0.columns]

    df0.to_csv(outpath, index=False, sep='\t')
    import sys
    print >> sys.stderr, outpath

def main(inputfile):
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


def main(inputfile, refpfx):

    global REFPFX
    REFPFX = refpfx

    df = pd.io.parsers.read_table(inputfile, sep='\t')

    outdir = op.join(output_dir(inputfile), refpfx)
    mkdirp(op.join(outdir, refpfx))

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
    write_tbl(df, outdir, 'norm_A')

    # 9.
    df = add_av_cols(df)
    df = normalize_b(df)

    import pdb
    pdb.set_trace()
    pass

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
    import sys

    inputfile = sys.argv[1]

    #for i in '1234':
    for i in '4':
        main(inputfile, 'Set%s' % i)
