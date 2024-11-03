# Readme

## Required package

numpy, open3d, scipy, tqdm, sklearn, opencv

## Usage  

for bev.py

```bash
python3 bev.py --front ${FRONT IMAGE PATH} --bev ${BEV IMAGE PATH}
```  

for reconstruct.py

GLOBAL SET = 1 ; Use global registration (RANSAC).
GLOBAL SET = 0 ; Not use global registration (RANSAC).

ICP VERSION can be "my_icp" or "open3d".

```bash
python3 reconstruct.py -f ${FLOOR NUMBER} --global_reg ${GLOBAL SET} -v ${ICP VERSION}  
```
