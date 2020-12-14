import numpy as np
import pandas as pd
import tensorflow as tf

def get_stack(ncf, names):
    return np.stack([
        np.array(ncf.variables.get(name))
    for name in names], axis=-1)


# === Sigrid3 Codes ===

concentration_codes = {
     0: 'Ice Free',
     1: 'Less than 1/10 (open water)',
     2: 'Bergy water',
    10: '1/10',
    20: '2/10',
    30: '3/10',
    40: '4/10',
    50: '5/10',
    60: '6/10',
    70: '7/10',
    80: '8/10',
    90: '9/10',
    91: '9+/10',
    92: '10/10'
}
_c_c2i = {k: i for i, (k, v) in enumerate(concentration_codes.items())}
_c_i2c = {i: k for i, (k, v) in enumerate(concentration_codes.items())}
_c_i2n = {i: v for i, (k, v) in enumerate(concentration_codes.items())}

stage_codes = {
     0: 'Ice Free',
    80: 'No stage of development',
    81: 'New Ice',
    82: 'Nilas, Ice Rind',
    83: 'Young Ice',
    84: 'Grey Ice',
    85: 'Grey - White Ice',
    86: 'First Year Ice',
    87: 'Thin First Year Ice',
    88: 'Thin First Year Ice Stage 1',
    89: 'Thin First Year Ice Stage 2',
    91: 'Medium First Year Ice',
    93: 'Thick First Year Ice',
    95: 'Old Ice',
    96: 'Second Year Ice',
    97: 'Multi-Year Ice',
    98: 'Glacier Ice'
}
_s_c2i = {k: i for i, (k, v) in enumerate(stage_codes.items())}
_s_i2c = {i: k for i, (k, v) in enumerate(stage_codes.items())}
_s_i2n = {i: v for i, (k, v) in enumerate(stage_codes.items())}

form_codes = {
     0: 'Pancake Ice',
     1: 'Shuga / Small Ice Cake, Brash Ice',
     2: 'Ice Cake',
     3: 'Small Floe',
     4: 'Medium Floe',
     5: 'Big Floe',
     6: 'Vast Floe',
     7: 'Giant Floe',
     8: 'Fast Ice',
     9: 'Growlers, Floebergs or Floebits',
    10: 'Icebergs',
    11: 'Strips and Patches, Concentrations 1/10',
    12: 'Strips and Patches, Concentrations 2/10',
    13: 'Strips and Patches, Concentrations 3/10',
    14: 'Strips and Patches, Concentrations 4/10',
    15: 'Strips and Patches, Concentrations 5/10',
    16: 'Strips and Patches, Concentrations 6/10',
    17: 'Strips and Patches, Concentrations 7/10',
    18: 'Strips and Patches, Concentratons 8/10',
    19: 'Strips and Patches, Concentrations 9/10',
    20: 'Strips and Patches, Concentrations 10/10',
    21: 'Level Ice',
}
_f_c2i = {k: i for i, (k, v) in enumerate(form_codes.items())}
_f_i2c = {i: k for i, (k, v) in enumerate(form_codes.items())}
_f_i2n = {i: v for i, (k, v) in enumerate(form_codes.items())}

type_codes = {
    'W': 'Water - sea ice free',
    'I': 'Ice - of any concentration',
}
_t_c2i = {k: i for i, (k, v) in enumerate(type_codes.items())}
_t_i2c = {i: k for i, (k, v) in enumerate(type_codes.items())}
_t_i2n = {i: v for i, (k, v) in enumerate(type_codes.items())}


def encode_column(column):
    d = {}
    if column.name.startswith('C'):
        d = _c_c2i
    elif column.name.startswith('S'):
        d = _s_c2i
    elif column.name.startswith('F'):
        d = _f_c2i
    elif column.name == 'POLY_TYPE':
        d = _t_c2i
    return column.apply(lambda x: d.get(x, -1))


def get_labels(ncf):
    poly_index = get_stack(ncf, ['polygon_icechart'])[...,0]
    poly_data = get_stack(ncf, ['polygon_codes'])[:,0]
    info = list(map(lambda x: x.split(';'), poly_data))
    info.append(['0'] + ['-1'] * 14)
    info = pd.DataFrame(columns=info[0], data=info[1:])
    info = info.apply(lambda x: x if x.name == 'POLY_TYPE' else x.astype(int))
    info = info.set_index('id')
    info = info.apply(encode_column)

    id_2_idx = {v: i for i, v in enumerate(info.index)}
    poly_index = np.vectorize(id_2_idx.get)(poly_index)

    # poly_type = info['POLY_TYPE'].values[poly_index]
    # concentration = info[['CT', 'CA', 'CB', 'CC']].values[poly_index]
    concentration = info['CT'].values[poly_index]
    # stage = info[['SA', 'SB', 'SC']].values[poly_index]
    # form = info[['FA', 'FB', 'FC']].values[poly_index]

    # return poly_type, concentration, stage, form
    return concentration
