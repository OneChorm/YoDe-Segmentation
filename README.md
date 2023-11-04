# YoDe-Segmentation

## Usage
### Step 1: Environ building
```bash
conda create -n YoDe_Seg python==3.10.11
conda activate YoDe_Seg
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
cd "your dir to `YoDe-Segmentation`" # change directory
pip install -r requirements.txt
pip install YoDe-Segmentation-v2==1.0.1
```

### Step 2: Weight configuration
Put [the weight of Deeplabv3 model](https://drive.google.com/file/d/1ipJNPU5tmCcYDZIbo7_veMu5idQjdbiQ/view?usp=sharing) and [the weight of YOLOV5 model](https://drive.google.com/file/d/1tXX_-RE2sL2U7lRvFfOBUBTIIIN_MhnN/view?usp=sharing) in the [weights](https://github.com/OneChorm/YoDe-Segmentation/tree/master/weights) folder

### Step 3: Where should you put the images you want to predict
Put the files or images you want to predict in the [test_img](https://github.com/OneChorm/YoDe-Segmentation/blob/master/test_img) folder

### Step 4: Predict
```bash
yode_seg 
```
## Datasets
You can download the Datasets at [YoDe-Segmentation_data](https://figshare.com/articles/journal_contribution/YoDe-Segmentation_DATA_zip/24456277)
