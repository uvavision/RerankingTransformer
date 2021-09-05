import _init_paths
import os
import os.path as osp
from glob import glob
from PIL import Image
from copy import deepcopy
from sacred import Experiment
from utils import pickle_load, pickle_save, json_save, ReadSolution


def extract_resolution(data_dir, records, gnd=None):
    outs = []
    for i in range(len(records)):
        entry = records[i]
        name, label = entry.split(',')
        path = osp.join(data_dir, name)
        if gnd is not None:
            bbx = gnd['gnd'][i]['bbx']
            width  = int(bbx[2] - bbx[0] + 1)
            height = int(bbx[3] - bbx[1] + 1)
        else:
            try:
                img = Image.open(path)
            except Warning:
                print('corrupted image:', i, name)
            width, height = img.size
        line = ','.join([name, label, str(width), str(height)])
        outs.append(line)
        if i % 1000 == 0:
            print(i)
    return outs


###########################################################################
## GLD V2

ex1 = Experiment('Prepare GLD V2')


def load_gldv2_train_csv(data_dir, csv_path):
    jpgs = glob(osp.join(data_dir, '**/*.jpg'), recursive=True)
    jpgs = ['/'.join(x.split('/')[2:]) for x in jpgs]
    name_to_jpg = {osp.splitext(osp.basename(x))[0]: x for x in jpgs}
    entry_list = []
    with open(csv_path) as records:
        lines = records.read().splitlines()
    for i in range(1, len(lines)):
        entry = lines[i].split(',')
        name, landmark_id = entry[0], entry[-1]
        entry_list.append((name.strip('"'), landmark_id))
    outs = [','.join([name_to_jpg[x[0]], str(x[1])]) for x in entry_list]
    return outs


def load_gldv2_solution_csv(data_dir):
    query_jpgs   = sorted(glob(osp.join(data_dir, 'test', '**/*.jpg'),  recursive=True))
    gallery_jpgs = sorted(glob(osp.join(data_dir, 'index', '**/*.jpg'), recursive=True))
    query_jpgs   = ['/'.join(x.split('/')[2:]) for x in query_jpgs]
    gallery_jpgs = ['/'.join(x.split('/')[2:]) for x in gallery_jpgs]

    public_solution, private_solution, _ = ReadSolution(
        osp.join(data_dir, 'meta', 'retrieval_solution_v2.1.csv'), 
        'retrieval'
    )
    solution = deepcopy(public_solution)
    solution.update(private_solution)
    valid_jpgs = [x for x in query_jpgs if osp.splitext(osp.basename(x))[0] in solution]
    query_jpgs = sorted(valid_jpgs)

    query_name_to_path = {osp.splitext(osp.basename(x))[0]: x for x in query_jpgs}
    gallery_name_to_path = {osp.splitext(osp.basename(x))[0]: x for x in gallery_jpgs}
    query_name_to_label, gallery_name_to_label = {}, {}
    with open(osp.join(data_dir, 'meta', 'index_image_to_landmark.csv')) as records:
        lines = records.read().splitlines()
    for i in range(1, len(lines)):
        entry = lines[i].split(',')
        name = entry[0].strip('"')
        landmark_id = int(entry[-1])
        gallery_name_to_label[name] = landmark_id
    for query_name in query_name_to_path.keys():
        index_names = solution[query_name]
        label = gallery_name_to_label[index_names[0]]
        query_name_to_label[query_name] = label
    gnd = { 'public': public_solution, 'private': private_solution }
    query_outs   = [','.join([query_name_to_path[x],   str(query_name_to_label[x])]) for x in query_name_to_path.keys()]
    gallery_outs = [','.join([gallery_name_to_path[x], str(gallery_name_to_label[x])]) for x in gallery_name_to_path.keys()]
    return query_outs, gallery_outs, gnd


@ex1.config 
def config():
    data_dir = osp.join('data', 'gldv2')
    train_file = 'train_clean.txt'
    test_file  = 'test_query.txt'
    index_file = 'test_gallery.txt'
    gnd_file   = 'gnd_gld.pkl'
    require_resolution = True


@ex1.main
def generate_gldv2_train_test(data_dir, train_file, test_file, index_file, gnd_file, require_resolution):
    train_file = osp.join(data_dir, train_file)
    test_file  = osp.join(data_dir, test_file)
    index_file = osp.join(data_dir, index_file) 
    gnd_file   = osp.join(data_dir, gnd_file)
    train = load_gldv2_train_csv(osp.join(data_dir, 'train'), osp.join(data_dir, 'meta', 'train.csv'))
    test, index, solution = load_gldv2_solution_csv(data_dir)
    if require_resolution:
        train = extract_resolution(data_dir, train)
        test  = extract_resolution(data_dir, test)
        index = extract_resolution(data_dir, index)
    with open(train_file, 'w') as f:
        f.write('\n'.join(train))
    with open(test_file, 'w') as f:
        f.write('\n'.join(test))
    with open(index_file, 'w') as f:
        f.write('\n'.join(index))
    pickle_save(gnd_file, solution)


###########################################################################
## Revisited Oxford/Paris
def load_revisited_dataset(data_dir, gnd_file):
    # prefix = osp.join(data_dir.split('/')[-1], 'jpg')
    prefix = 'jpg'
    gnd = pickle_load(gnd_file)
    query_names   = gnd['qimlist']
    gallery_names = gnd['imlist']


    categories = []
    for x in query_names:
        cat = '_'.join(x.split('_')[:-1])
        categories.append(cat)
    for x in gallery_names:
        cat = '_'.join(x.split('_')[:-1])
        categories.append(cat)
    categories = sorted(list(set(categories)))
    cat_to_label = dict(zip(categories, range(len(categories))))

    query_outs   = [','.join([ osp.join(prefix, x+'.jpg'), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in query_names]
    gallery_outs = [','.join([ osp.join(prefix, x+'.jpg'), str(cat_to_label['_'.join(x.split('_')[:-1])]) ]) for x in gallery_names]
    return query_outs, gallery_outs


ex2 = Experiment('Prepare Revisited Oxford5K')


@ex2.config
def config():
    data_dir   = osp.join('data', 'oxford5k')
    test_file  = 'test_query.txt'
    index_file = 'test_gallery.txt'
    gnd_file   = 'gnd_roxford5k.pkl'
    require_resolution = True


@ex2.main
def generate_oxford(data_dir, test_file, index_file, gnd_file, require_resolution):
    test_file  = osp.join(data_dir, test_file)
    index_file = osp.join(data_dir, index_file) 
    gnd_file   = osp.join(data_dir, gnd_file)
    test, index = load_revisited_dataset(data_dir, gnd_file)
    gnd = pickle_load(gnd_file)

    if require_resolution:
        test  = extract_resolution(data_dir, test, gnd)
        index = extract_resolution(data_dir, index)

    with open(test_file, 'w') as f:
        f.write('\n'.join(test))
    with open(index_file, 'w') as f:
        f.write('\n'.join(index))


ex3 = Experiment('Prepare Revisited Paris6K')


@ex3.config
def config():
    data_dir   = osp.join('data', 'paris6k')
    test_file  = 'test_query.txt'
    index_file = 'test_gallery.txt'
    gnd_file   = 'gnd_rparis6k.pkl'
    require_resolution = True


@ex3.main
def generate_paris(data_dir, test_file, index_file, gnd_file, require_resolution):
    test_file  = osp.join(data_dir, test_file)
    index_file = osp.join(data_dir, index_file) 
    gnd_file   = osp.join(data_dir, gnd_file)
    test, index = load_revisited_dataset(data_dir, gnd_file)

    if require_resolution:
        test  = extract_resolution(data_dir, test)
        index = extract_resolution(data_dir, index)

    with open(test_file, 'w') as f:
        f.write('\n'.join(test))
    with open(index_file, 'w') as f:
        f.write('\n'.join(index))


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    # ex1.run()
    ex2.run()
    ex3.run()
