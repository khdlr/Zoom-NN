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
        labels = data_utils.get_labels(ncf)

    sar = np.nan_to_num(sar)
    nersc_sar = np.nan_to_num(sar)
    btemp = np.nan_to_num(btemp)

    return sar, nersc_sar, btemp, labels


def convert_tile(tilepath):
    out_npz = tilepath.with_suffix('.npz')
    sar, nersc_sar, btemp, labels = load_tile(tilepath)
    poly_type, concentration, stage, form = labels
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
    with Pool(32) as pool:
        r = list(tqdm(pool.imap(convert_tile, all_ncs), total=len(all_ncs)))
