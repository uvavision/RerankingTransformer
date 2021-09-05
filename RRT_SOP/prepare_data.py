import os
import zipfile
import os.path as osp
from glob import glob

from sacred import Experiment
from torchvision.datasets.utils import download_url


ex = Experiment('Prepare SOP')


@ex.config
def config():
    sop_dir = osp.join('data', 'Stanford_Online_Products')
    sop_url = 'ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip'
    train_file = 'train.txt'
    test_file = 'test.txt'


@ex.capture
def download_extract_sop(sop_dir, sop_url):
    download_url(sop_url, root=osp.dirname(sop_dir))
    filename = osp.join(osp.dirname(sop_dir), osp.basename(sop_url))
    with zipfile.ZipFile(filename) as zipf:
        zipf.extractall(path=osp.dirname(sop_dir))


@ex.capture
def generate_sop_train_test(sop_dir, train_file, test_file):
    original_train_file = osp.join(sop_dir, 'Ebay_train.txt')
    original_test_file = osp.join(sop_dir, 'Ebay_test.txt')
    train_file = osp.join(sop_dir, train_file)
    test_file = osp.join(sop_dir, test_file)

    with open(original_train_file) as f_images:
        train_lines = f_images.read().splitlines()[1:]
    with open(original_test_file) as f_images:
        test_lines = f_images.read().splitlines()[1:]

    train = [','.join((l.split()[-1], str(int(l.split()[1]) - 1))) for l in train_lines]
    test = [','.join((l.split()[-1], str(int(l.split()[1]) - 1))) for l in test_lines]

    with open(train_file, 'w') as f:
        f.write('\n'.join(train))
    with open(test_file, 'w') as f:
        f.write('\n'.join(test))


@ex.main
def prepare_sop():
    download_extract_sop()
    generate_sop_train_test()


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    ex.run()
