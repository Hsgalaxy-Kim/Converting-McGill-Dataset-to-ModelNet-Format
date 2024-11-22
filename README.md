# Converting-McGill-Dataset-to-ModelNet-Format

This repository provides code for transforming the McGill dataset into the ModelNet format to enable training on the McGill dataset using PointNet.
(This repository provides code to create a McGill version of `modelnet40_normal_resampled`.)

PointNet: https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git

# Install Dataset

First, download the dataset(.ply) from the [McGill 3D Shape Benchmark](https://www.cim.mcgill.ca/~shape/benchMark/).

Second, extract each file(.gz) and organize the folders as follows.
```
project-root/
├── data/
│   ├── airplane/
│   │   ├── b1.ply
│   │   ├── b2.ply
│   │   ├── ...
│   ├── ant/
│   │   ├── 1.ply
│   │   ├── 2.ply
│   │   ├── ...
│   ├── ...
│   ├── teddy-bears/
│   │   ├── b1.ply
│   │   ├── b2.ply
│   │   ├── ...
```

# Convert Dataset

You should download python library.

`numpy`, `trimesh`

```
pip install numpy
pip install trimesh
```

Edit base_path in `McGill_convert.py`. This is the folder for converted McGill dataset.

Run the `McGill_convert.py`.

```
python McGill_convert.py 
```

Finally, you can use the dataloader for modelnet40_normal_resampled directly with McGill.
