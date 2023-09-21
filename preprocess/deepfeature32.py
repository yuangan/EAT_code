from glob import glob
import os
import numpy as np

out='./deepfeature32/'
os.makedirs(out, exist_ok=True)
allfeas = glob('./video_fps25/*.npy')

for fea in allfeas:
    feature = np.load(fea).astype(np.float32)
    name = os.path.basename(fea)
    np.save(os.path.join(out, name), feature)

