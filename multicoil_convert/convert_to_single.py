import pathlib
import h5py

from utils.fastmri.data.transforms import tensor_to_complex_np
from utils.fastmri.data import transforms
from tqdm import tqdm
from utils import fastmri
from argparse import ArgumentParser
from pathlib import Path

def convert(in_path, out_path):
    print('BEGINNING CONVERSION')
    new_files_attrs = {}
    new_files_data = {}
    for fname in tqdm(list(in_path.glob("*.h5"))):
        with h5py.File(fname, "r") as hf:
            kspace = transforms.to_tensor(hf['kspace'][()])
            image = fastmri.ifft2c(kspace)
            image = fastmri.rss(image, dim=1)
            new_kspace = fastmri.fft2c(image)
            new_kspace = tensor_to_complex_np(new_kspace)

            new_files_attrs[fname.name] = {
                'acquisition': hf.attrs['acquisition'],
                'max': hf.attrs['max'],
                'norm': hf.attrs['norm'],
                'patient_id': hf.attrs['patient_id']
            }

            new_files_data[fname.name] = {
                'ismrmrd_header': hf['ismrmrd_header'][()],
                'kspace': new_kspace,
                'reconstruction_rss': hf['reconstruction_rss'][()]
            }

    print('SAVING CONVERTED FILES')
    out_path.mkdir(exist_ok=True, parents=True)
    for fname, data in tqdm(new_files_data.items()):
        with h5py.File(out_path / fname, "w") as hf:
            hf.create_dataset('ismrmrd_header', data=new_files_data[fname]['ismrmrd_header'])
            hf.create_dataset('kspace', data=new_files_data[fname]['kspace'])
            hf.create_dataset('reconstruction_rss', data=new_files_data[fname]['reconstruction_rss'])
            hf.attrs.create('acquisition', data=new_files_attrs[fname]['acquisition'])
            hf.attrs.create('max', data=new_files_attrs[fname]['max'])
            hf.attrs.create('norm', data=new_files_attrs[fname]['norm'])
            hf.attrs.create('patient_id', data=new_files_attrs[fname]['patient_id'])

    print('CONVERSION COMPLETE')


parser = ArgumentParser(add_help=False)

parser.add_argument(
    "--data_type",
    type=str,
    required=True,
    help="Path to the data",
)

args = parser.parse_args()
data_path = Path(f'/storage/fastMRI_brain/data/multicoil_{args.data_type}')
out_path = Path(f'/storage/fastMRI_brain/data/singlecoil_{args.data_type}')
# data_path = Path(f'/Users/mattbendel/Desktop/Professional/PhD/ComparisonStudy/test_dir/multicoil_{args.data_type}')
# out_path = Path(f'/Users/mattbendel/Desktop/Professional/PhD/ComparisonStudy/test_dir/singlecoil_{args.data_type}')

convert(data_path, out_path)
