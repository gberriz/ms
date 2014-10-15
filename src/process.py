import numpy as np
import pandas as pd
import re

# ---------------------------------------------------------------------------

SEP = '~'
REFSET = 'Set1'
REFSFX = 'cq_126_sn_sum'

# ---------------------------------------------------------------------------

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

def parse_hdr(h):
    return tuple(h.split(SEP) if SEP in h else [h, ''])

def build_hdr(pfx, sfx):
    return SEP.join([pfx, sfx])

def get_headers_for_prefix(df, pfx):
    return [build_hdr(*ps) for ps in
            filter(lambda x: x[0] == pfx,
                   [parse_hdr(c) for c in df.columns])]

def get_first_control_col_pfxs(df):
    return [p for p, s in [parse_hdr(c) for c in df.columns] if s == REFSFX]

def get_countscols(df):
    return sum([get_headers_for_prefix(df, pfx) for pfx in
                get_first_control_col_pfxs(df)], [])

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

def rowscale(df, s):
    return diagframe(s).dot(df)

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
    ret.append(normalize_b(ret[-1], refset='S0', refsfx='a'))
    ret.append(normalize_ab(df0, refset='S0', refsfx='a'))

    assert almost_equal(ret[-2].iloc[:, 2:].values,
                        ret[-1].iloc[:, 2:].values)

    ret.append(normalize_c(ret[-1], refset='S0', refsfx='a'))

    return ret

def main(inputfile):
    df = pd.io.parsers.read_table(inputfile, sep='\t')

    # 1. & 2.
    df = df[df['Protein Id'].str.contains('^(?!##).*(?<!_contaminant)$')]

    # 3.
    df = split_first_col(df)
    df.index = df.iloc[:, 0]

    # 5.
    df = dropcols(df, lambda x: re.match(r'^Set(?:12|AtoD|EtoH|5~cq_1(?:28|30)[ab]_sn_sum)\b', x))

    # 8. & 9.
    df = normalize_ab(df)

    # 10.
    df = normalize_c(df)


if __name__ == "__main__":
    main('data/orig/BrCa_Sets1_12_AtoD_EtoH_AtoD2_EtoH2_DrugsA_B_Tryps_140806_Quant.tsv')
