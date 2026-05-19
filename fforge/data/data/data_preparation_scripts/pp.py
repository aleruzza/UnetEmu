import pandas as pd
import numpy as np
import os
from tqdm import tqdm
coord = []
for i in tqdm(range(0,1000)):
    if os.path.exists(f'indacorun/outputs/out_{i:05}/summary30.dat'):
        blob = open(f'indacorun/outputs/out_{i:05}/summary30.dat', 'rb').read()
        lines = blob.decode('us-ascii')
        split = lines[lines.find('Planet 0 out of 1'):].split('\n')[1].split('\t')
        x, y = float(split[0]), float(split[1])
    #print(ppos)
        coord.append([x, y])
    else:
        continue
cc = np.array(coord)
np.save('pcoord_test.npy', cc)
