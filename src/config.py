class Config:
    """Note, current pickle uses pre_2003_list and set_2003_list together."""

    pre_2003_list = [
        'ons', 'wc02', 'pcy', 'jud', 'tor', 'pal02', 'pr2', 'g02', 'f02', 'dkm', 'ody', 'wc01', 'apc',
        '7ed', 'pls', 'pal01', 'g01', 'f01', 'mpr', 'inv', 'btd', 'wc00', 's00', 'nem', 'pelp', 'pal00',
        'fnm', 'g00', 'psus', 'brb', 'mmq', 'pwos', 'pwor', 'wc99', 'pgru', 'ptk', 'uds', '6ed',
        'ulg', 'pal99', 'g99', 'ath', 'palp', 'wc98', 'tugl', 'ugl', 'sth', 'jgp', 'tmp', 'ppre',
        'wc97', 'wth', 'ppod', 'pvan', 'por', 'pmic', 'past', '5ed', 'vis', 'itp', 'mgb', 'mir',
        'pcel', 'parl', 'rqs', 'all', 'ptc', 'hml', 'chr', 'ice', '4ed', 'scg', 'plgm', 'pmei', 'fem',
        'phpr', 'drk', 'pdrc', 'sum', 'leg', '3ed', 'atq', 'arn', 'cei', 'ced', '2ed', 'leb', 'lea',
        # removed sets: 'fbb', 'prm', 's99', 'usg', 'p02', 'exo', 'pred', 'rin', 'ren', '4bb'
    ]

    set_2003_list = [
        'mrd', 'dst', '5dn', 'chk', 'bok', 'sok', 'rav', 'gpt', 'dis', 'csp', 'tsp', 'plc', 'fut', '10e',
        'mor', 'shm', 'eve', 'ala', 'con', 'arb', 'm10', 'zen', 'wwk', 'roe', 'm11', 'som', 'mbs',
        'nph', 'm12', 'isd', 'dka', 'avr', 'm13', 'rtr', 'gtc', 'dgm', 'm14', 'ths', 'bng', 'jou'
    ]

    """ UNUSED *** => if you want to use any of these sets from 2015 or past, you till need to build your own pickle"""
    # Core & expansion sets with 2015 frame
    set_2015_list = [
        'm15', 'ktk', 'frf', 'dtk', 'bfz', 'ogw', 'soi', 'emn', 'kld', 'aer', 'akh', 'hou', 'xln', 'rix', 'dom'
        # removed sets: 'lrw'
    ]

    # Box sets
    set_box_list = [
        'evg', 'drb', 'dd2', 'ddc', 'td0', 'v09', 'ddd', 'h09', 'dde', 'dpa', 'v10', 'ddf', 'td0', 'pd2', 'ddg',
        'cmd', 'v11', 'ddh', 'pd3', 'ddi', 'v12', 'ddj', 'cm1', 'td2', 'ddk', 'v13', 'ddl', 'c13', 'ddm', 'md1',
        'v14', 'ddn', 'c14', 'ddo', 'v15', 'ddp', 'c15', 'ddq', 'v16', 'ddr', 'c16', 'pca', 'dds', 'cma', 'c17',
        'ddt', 'v17', 'ddu', 'cm2', 'ss1', 'gs1', 'c18'
    ]
    # Supplemental sets
    set_sup_list = ['hop', 'arc', 'pc2', 'cns', 'cn2', 'e01', 'e02', 'bbd']

    masters = ['ema', 'mm2', 'vma', 'mma', 'me4', 'me3', 'me2', 'me1']
    """"""

    ALL_SETS = pre_2003_list + set_2003_list + set_2015_list + set_box_list + set_sup_list + masters
