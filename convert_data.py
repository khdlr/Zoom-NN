from multiprocessing import Pool
from tqdm import tqdm
from pathlib import Path
import h5netcdf
from itertools import product
import data_utils
import numpy as np


def load_tile(tile_path):
    with h5netcdf.File(tile_path, 'r') as ncf:
        freqs = [6.9, 7.3, 10.7, 18.7, 23.8, 36.5, 89.0]
        polarizations = ['v', 'h']
        btemp_names = [f'btemp_{freq}{pol}' for pol, freq in product(polarizations, freqs)]

        sar = data_utils.get_stack(ncf, ['sar_primary', 'sar_secondary']).astype(np.float32)
        nersc_sar = data_utils.get_stack(ncf,
                ['nersc_sar_primary', 'nersc_sar_secondary']).astype(np.float32)
        btemp  = data_utils.get_stack(ncf, btemp_names).astype(np.float32) / 200
        poly_type, concentration, stage, form = data_utils.get_labels(ncf)


    # Invalid pixels masking
    valid = (poly_type != -1) | np.isfinite(sar).all(axis=-1)
    poly_type[~valid] = -1
    concentration[~valid] = -1
    stage[~valid] = -1
    form[~valid] = -1

    H, W, _ = sar.shape
    valid_rows = valid.any(axis=1)
    first_valid_row = np.argmax(valid_rows)
    last_valid_row = H - np.argmax(valid_rows[::-1])

    valid_cols = valid.any(axis=0)
    first_valid_col = np.argmax(valid_cols)
    last_valid_col = W - np.argmax(valid_cols[::-1])

    previous_pixels = np.prod(sar.shape)
    sar = sar[first_valid_row:last_valid_row+1,
              first_valid_col:last_valid_col+1]
    sar = np.nan_to_num(sar)

    nersc_sar = nersc_sar[first_valid_row:last_valid_row+1,
                          first_valid_col:last_valid_col+1]
    nersc_sar = np.nan_to_num(nersc_sar)

    btemp = btemp[first_valid_row//50:last_valid_row//50+1,
                  first_valid_col//50:last_valid_col//50+1]
    btemp = np.nan_to_num(btemp)

    poly_type = poly_type[first_valid_row:last_valid_row+1,
                          first_valid_col:last_valid_col+1]

    concentration = concentration[first_valid_row:last_valid_row+1,
                                  first_valid_col:last_valid_col+1]

    stage = stage[first_valid_row:last_valid_row+1,
                  first_valid_col:last_valid_col+1]

    form = form[first_valid_row:last_valid_row+1,
                first_valid_col:last_valid_col+1]

    assert sar.shape[:2] == nersc_sar.shape[:2]
    assert sar.shape[:2] == poly_type.shape[:2]
    assert sar.shape[:2] == concentration.shape[:2]
    assert sar.shape[:2] == stage.shape[:2]
    assert sar.shape[:2] == form.shape[:2]

    post_pixels = np.prod(sar.shape)
    print(f'Tossed out {100 * (1 - post_pixels / previous_pixels)}% of data (invalid rows/columns)')

    return sar, nersc_sar, btemp, poly_type, concentration, stage, form


def convert_tile(tilepath, write=True):
    print('converting', tilepath)
    out_npz = tilepath.with_suffix('.npz')
    sar, nersc_sar, btemp, poly_type, concentration, stage, form = load_tile(tilepath)
    if write:
        np.savez_compressed(f'data_np/{out_npz.name}',
            sar=sar,
            nersc_sar=nersc_sar,
            btemp=btemp,
            poly_type=poly_type,
            concentration=concentration,
            stage=stage,
            form=form
        )


if __name__ == '__main__':
    all_ncs = list(Path('data').glob('*.nc'))
    with Pool(40) as pool:
        r = list(tqdm(pool.imap(convert_tile, all_ncs), total=len(all_ncs)))
