1、Install and Run YoDe-Segmentation modle on image
# YoDe-Segmentation

## Usage
### Step 1: Environ building
```bash
conda create -n YoDe_Seg python==3.10.11
conda activate YoDe_Seg
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
cd "your dir to `YODE-Segmentation`" # change directory
pip install -r requirements.txt
pip install YoDe-Segmentation-v2==1.0.1
```

### Step 2: Weight configuration


### Step 3: 
Put the images you want to predict in the [test_img](https://github.com/OneChorm/YoDe-Segmentation/blob/master/test_img) folder

### Step 4: Predict
```bash
yode_seg 
```

