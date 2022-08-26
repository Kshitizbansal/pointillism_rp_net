#!/bin/sh
cd pointillism/lib/utils/iou3d/
python3 setup.py install

cd ../roipool3d/
python3 setup.py install

cd /pointillism
$@
