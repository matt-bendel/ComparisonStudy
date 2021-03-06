import h5py

from utils.fastmri.data.transforms import tensor_to_complex_np
from utils.fastmri.data import transforms
from tqdm import tqdm
from utils import fastmri
from argparse import ArgumentParser
from pathlib import Path
from torch.nn import functional as F

def preprocess(in_path, out_path):
    print('BEGINNING PROCESS')
    new_files_attrs = {}
    new_files_data = {}
    for fname in tqdm(list(in_path.glob("*.h5"))):
        with h5py.File(fname, "r") as hf:
            kspace = transforms.to_tensor(hf['kspace'][()])
            image = fastmri.ifft2c(kspace)

            if image.shape[-2] < 320:
                diff = (320 - image.shape[-2]) // 2 + 1
                padding = (0, 0, diff, diff, diff, diff, 0, 0)
                image = F.pad(image, padding, "constant", 0)

            image = transforms.complex_center_crop(image, (320, 320))
            
            half_num_slices = image.shape[0] // 2
            num_to_trim = half_num_slices // 2
            
            image = image[0+num_to_trim:image.shape[0]-num_to_trim]

            new_kspace = fastmri.fft2c(image)
            new_kspace = tensor_to_complex_np(new_kspace)
            
            if hf.attrs['acquisition'] == 'AXT2':
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

    print('SAVING PROCESSED FILES')
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

    print('PROCESS COMPLETE')


parser = ArgumentParser(add_help=False)

parser.add_argument(
    "--data_type",
    type=str,
    required=True,
    help="Path to the data",
)

args = parser.parse_args()
data_path = Path(f'/storage/fastMRI_brain/data/singlecoil_{args.data_type}')
out_path = Path(f'/storage/fastMRI_brain/data/Matt_preprocessed_data/T2/singlecoil_{args.data_type}')
# data_path = Path(f'/Users/mattbendel/Desktop/Professional/PhD/ComparisonStudy/test_dir/singlecoil_{args.data_type}')
# out_path = Path(f'/Users/mattbendel/Desktop/Professional/PhD/ComparisonStudy/test_dir/preprocessed/singlecoil_{args.data_type}')

preprocess(data_path, out_path)
