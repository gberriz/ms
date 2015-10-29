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
        #return (1, -n, m)
        return (1, n, m)

    sorted_cols = [c for _, c in sorted(enumerate(df.columns), key=key)
                   if c == 'Uniprot' or SEP in c]

    df0 = df.loc[:, sorted_cols]

    cn = pd.io.parsers.read_table('data/tsv/convert_names.tsv', sep='\t')
    lkp = dict(zip(cn.iloc[:, 0], cn.iloc[:, 1]))

    df0.columns = [lkp.get(c, c) for c in df0.columns]

    df0.to_csv(outpath, index=False, sep='\t')
    import sys
    print >> sys.stderr, outpath


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


