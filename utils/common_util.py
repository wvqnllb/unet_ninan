from os import listdir
from os.path import splitext

dir_img = 'data/imgs/'
dir_mask = 'data/masks/'
mask_suffix='_a'

n_test = 29


ids = [splitext(file)[0] for file in listdir(dir_img)
            if not file.startswith('.')]


