# Automatic LV segmentation for cardiac DTI (cDTI)

One command: give it the **average DWI** and **mean diffusivity (MD)** map of each slice, get back the
**LV myocardium mask + RV insertion points in the original image space**, ready for CarDpy
(MD / FA / helix angle).

```
DWI ─► YOLO heart crop ─► square crop + resize 256 ─► nnUNet LV (5-fold ensemble) ─► YOLO insertion points ─► un-crop ─► mask
```

---

## 1. Install (once)

Python 3.10–3.12. A CUDA GPU is recommended; CPU works (slower).

```bash
git clone https://github.com/sash9966/nnUNet_cDTI.git
cd nnUNet_cDTI/deploy          # everything you need is in deploy/ -- run all commands from here
python -m venv .venv && source .venv/bin/activate      # optional but recommended
pip install -r requirements.txt
```

> GPU: if `python -c "import torch; print(torch.cuda.is_available())"` prints `False` on a GPU machine,
> install the CUDA build of PyTorch first (see pytorch.org), then rerun `pip install -r requirements.txt`.

## 2. Get the weights (once)

The weights (~4 GB) are on Hugging Face: **`sash99666/cdti-lv-baseline`** (private — ask for access).
Because the repo is private you log in once, whichever way you download:

1. Make a Hugging Face account and send the username to the authors to be added.
2. Create a **read** token at huggingface.co/settings/tokens, then in a terminal: `hf auth login`

Then pick one:

- **A — automatic:** do nothing. The first run downloads the weights into `deploy/deployable_weights/`.
- **B — your own folder** (shared drive, offline machine, …):
  ```bash
  hf download sash99666/cdti-lv-baseline --local-dir /path/to/cdti_weights
  ```
  and add `--weights /path/to/cdti_weights` when you run. Keep the folder layout as downloaded.

## 3. Prepare your input

One 2-D NIfTI per slice and contrast, all in one folder:

```
input/
  Patient01_slice01_0000.nii.gz    # average DWI           (required)
  Patient01_slice01_0001.nii.gz    # mean diffusivity (MD) (required)
  Patient01_slice02_0000.nii.gz
  Patient01_slice02_0001.nii.gz
  ...
```

Rules:
- `_0000` = average DWI, `_0001` = MD. Same base name for both files of a slice.
- Give the maps **as reconstructed — do not normalize them**; the pipeline normalizes
  (DWI min–max, MD ÷ 4) exactly as in training.
- MD in **µm²/ms** (= 10⁻³ mm²/s; myocardium ≈ 1.5–1.7).
- DWI and MD of a slice must have the same shape (same image grid).

From CarDpy / numpy arrays:

```python
import os, nibabel as nib, numpy as np
os.makedirs('input', exist_ok=True)
for s in range(dwi.shape[2]):                    # dwi, md: (H, W, slices)
    for ch, img in ((0, dwi), (1, md)):
        nib.save(nib.Nifti1Image(img[:, :, s].astype(np.float32), np.eye(4)),
                 f'input/Patient01_slice{s+1:02d}_{ch:04d}.nii.gz')
```

## 4. Run

```bash
python deployable.py --input input --output output                                  # option A
python deployable.py --input input --output output --weights /path/to/cdti_weights  # option B
```

or from Python / a notebook (start Python or Jupyter **inside `deploy/`** — from the repository root,
Python would import this repo's own `nnunetv2/` folder instead of the installed package):

```python
from deployable import CardiacPipeline
CardiacPipeline(weights_dir='/path/to/cdti_weights').predict_folder('input', 'output')
```

## 5. Output

Two files per slice, same size and grid as your input DWI:

| file | contents |
|---|---|
| `output/<case>_lv.nii.gz` | **LV myocardium only** (0/1) — use this for MD / FA / helix angle |
| `output/<case>.nii.gz` | 1 = LV myocardium, 2 = anterior and 3 = inferior RV insertion point |

In `<case>.nii.gz` the insertion points are small disks painted over the myocardium, so its label 1 has
small holes there — that's why the separate `_lv` mask exists.
Intermediate files (crop box, 256 crops, raw LV) are kept in `output/_work/`.

---

## How the model was built

| | |
|---|---|
| Data | Stanford cDTI, manually annotated (Hannum): SmartHealth + Directions-vs-Averages cohorts, protocol DiVO_15_04 |
| Test set | 39 slices from 6 volunteers (46, 47, 49, 50, 51, 52), excluded from every training fold |
| Crop model | YOLO box detector on the average DWI (`yolo/crop.pt`) |
| LV model | nnUNet v2, 2-D, ResEnc-M plans, 5-fold ensemble, 100 epochs, DA5 augmentation, inputs DWI + MD (`noNorm`; normalization done in preprocessing) |
| Insertion-point model | single-class YOLO on the average DWI, top-2 points (`yolo/rvip.pt`) |

### Results on the held-out test set (39 slices, 6 volunteers, never used in training)

| training data / inputs | LV Dice | precision | recall | helix angle pitch, automatic − manual (median abs.) |
|---|---|---|---|---|
| SmartHealth only, 4 contrasts (DWI, MD, eigenvector, FA) | 0.902 | 0.884 | 0.923 | 3.9 °/wall |
| SmartHealth + Dir-vs-Avg, 4 contrasts | 0.909 | 0.886 | 0.936 | 3.7 °/wall |
| **SmartHealth + Dir-vs-Avg, DWI + MD — this model** | **0.909** | 0.889 | 0.932 | 4.2 °/wall |

More data helped a little; dropping to DWI + MD cost nothing, so the shipped model needs only those two maps.
MD / FA / DWI inside the predicted mask agree with the manual mask to ~1% / ~2% / ~3% (median).
Helix angle (CarDpy, computed from the raw diffusion data inside each mask): pitch −61.0 °/wall with the
manual mask vs −61.7 with this model (mean; bias −0.6, r = 0.87 per slice).
Insertion-point error: median ~2.3 mm, but a few slices miss by 30–55 mm.

| | |
|---|---|
| ![LV Dice](results/dice_violin.png) | ![Helix angle pitch](results/gtpred_HAP.png) |
| ![MD](results/gtpred_MD.png) | ![FA](results/gtpred_FA.png) |
| ![DWI](results/gtpred_DWI.png) | |

![Insertion points](results/ip_hausdorff_violin.png)

Split violins: grey = value inside the manual mask, colour = inside the automatic mask; lines pair the same slice.
All numbers (mean / median / std per run): [`results/summary.csv`](results/summary.csv).

Training and evaluation code (ongoing development): branch `crop-alignment-and-hap-fixes`
(`PaperRun_cDTI_Pipeline.ipynb`, `MODEL_CARD.md`).

## Limitations

- Single center and protocol; other scanners/protocols are not validated — check the masks.
- Requires reconstructed DWI and MD maps (no tensor fitting here).
- Treat insertion points as an initialization and review them.

## Files

| file | what |
|---|---|
| `deploy/deployable.py` | the whole pipeline (config at the top, CLI at the bottom) |
| `deploy/yolo_pipeline.py` | YOLO crop / insertion-point helpers |
| `deploy/requirements.txt` | dependencies (installs the standard `nnunetv2` from PyPI — no need to install this repo) |
| `results/` | evaluation plots + summary numbers shown above (no images or data) |

Everything outside `deploy/` is our nnU-Net fork and research code used to train and evaluate the model.

---

# Based on nnU-Net

This repository is a fork of nnU-Net (MIC-DKFZ). Original documentation below.

# Welcome to the new nnU-Net!

Click [here](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) if you were looking for the old one instead.

Coming from V1? Check out the [TLDR Migration Guide](documentation/tldr_migration_guide_from_v1.md). Reading the rest of the documentation is still strongly recommended ;-)

## **2024-04-18 UPDATE: New residual encoder UNet presets available!**
Residual encoder UNet presets substantially improve segmentation performance.
They ship for a variety of GPU memory targets. It's all awesome stuff, promised! 
Read more :point_right: [here](documentation/resenc_presets.md) :point_left:

Also check out our [new paper](https://arxiv.org/pdf/2404.09556.pdf) on systematically benchmarking recent developments in medical image segmentation. You might be surprised!

# What is nnU-Net?
Image datasets are enormously diverse: image dimensionality (2D, 3D), modalities/input channels (RGB image, CT, MRI, microscopy, ...), 
image sizes, voxel sizes, class ratio, target structure properties and more change substantially between datasets. 
Traditionally, given a new problem, a tailored solution needs to be manually designed and optimized  - a process that 
is prone to errors, not scalable and where success is overwhelmingly determined by the skill of the experimenter. Even 
for experts, this process is anything but simple: there are not only many design choices and data properties that need to 
be considered, but they are also tightly interconnected, rendering reliable manual pipeline optimization all but impossible! 

![nnU-Net overview](documentation/assets/nnU-Net_overview.png)

**nnU-Net is a semantic segmentation method that automatically adapts to a given dataset. It will analyze the provided 
training cases and automatically configure a matching U-Net-based segmentation pipeline. No expertise required on your 
end! You can simply train the models and use them for your application**.

Upon release, nnU-Net was evaluated on 23 datasets belonging to competitions from the biomedical domain. Despite competing 
with handcrafted solutions for each respective dataset, nnU-Net's fully automated pipeline scored several first places on 
open leaderboards! Since then nnU-Net has stood the test of time: it continues to be used as a baseline and method 
development framework ([9 out of 10 challenge winners at MICCAI 2020](https://arxiv.org/abs/2101.00232) and 5 out of 7 
in MICCAI 2021 built their methods on top of nnU-Net, 
 [we won AMOS2022 with nnU-Net](https://amos22.grand-challenge.org/final-ranking/))!

Please cite the [following paper](https://www.google.com/url?q=https://www.nature.com/articles/s41592-020-01008-z&sa=D&source=docs&ust=1677235958581755&usg=AOvVaw3dWL0SrITLhCJUBiNIHCQO) when using nnU-Net:

    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
    method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.


## What can nnU-Net do for you?
If you are a **domain scientist** (biologist, radiologist, ...) looking to analyze your own images, nnU-Net provides 
an out-of-the-box solution that is all but guaranteed to provide excellent results on your individual dataset. Simply 
convert your dataset into the nnU-Net format and enjoy the power of AI - no expertise required!

If you are an **AI researcher** developing segmentation methods, nnU-Net:
- offers a fantastic out-of-the-box applicable baseline algorithm to compete against
- can act as a method development framework to test your contribution on a large number of datasets without having to 
tune individual pipelines (for example evaluating a new loss function)
- provides a strong starting point for further dataset-specific optimizations. This is particularly used when competing 
in segmentation challenges
- provides a new perspective on the design of segmentation methods: maybe you can find better connections between 
dataset properties and best-fitting segmentation pipelines?

## What is the scope of nnU-Net?
nnU-Net is built for semantic segmentation. It can handle 2D and 3D images with arbitrary 
input modalities/channels. It can understand voxel spacings, anisotropies and is robust even when classes are highly
imbalanced.

nnU-Net relies on supervised learning, which means that you need to provide training cases for your application. The number of 
required training cases varies heavily depending on the complexity of the segmentation problem. No 
one-fits-all number can be provided here! nnU-Net does not require more training cases than other solutions - maybe 
even less due to our extensive use of data augmentation. 

nnU-Net expects to be able to process entire images at once during preprocessing and postprocessing, so it cannot 
handle enormous images. As a reference: we tested images from 40x40x40 pixels all the way up to 1500x1500x1500 in 3D 
and 40x40 up to ~30000x30000 in 2D! If your RAM allows it, larger is always possible.

## How does nnU-Net work?
Given a new dataset, nnU-Net will systematically analyze the provided training cases and create a 'dataset fingerprint'. 
nnU-Net then creates several U-Net configurations for each dataset: 
- `2d`: a 2D U-Net (for 2D and 3D datasets)
- `3d_fullres`: a 3D U-Net that operates on a high image resolution (for 3D datasets only)
- `3d_lowres` → `3d_cascade_fullres`: a 3D U-Net cascade where first a 3D U-Net operates on low resolution images and 
then a second high-resolution 3D U-Net refined the predictions of the former (for 3D datasets with large image sizes only)

**Note that not all U-Net configurations are created for all datasets. In datasets with small image sizes, the 
U-Net cascade (and with it the 3d_lowres configuration) is omitted because the patch size of the full 
resolution U-Net already covers a large part of the input images.**

nnU-Net configures its segmentation pipelines based on a three-step recipe:
- **Fixed parameters** are not adapted. During development of nnU-Net we identified a robust configuration (that is, certain architecture and training properties) that can 
simply be used all the time. This includes, for example, nnU-Net's loss function, (most of the) data augmentation strategy and learning rate.
- **Rule-based parameters** use the dataset fingerprint to adapt certain segmentation pipeline properties by following 
hard-coded heuristic rules. For example, the network topology (pooling behavior and depth of the network architecture) 
are adapted to the patch size; the patch size, network topology and batch size are optimized jointly given some GPU 
memory constraint. 
- **Empirical parameters** are essentially trial-and-error. For example the selection of the best U-net configuration 
for the given dataset (2D, 3D full resolution, 3D low resolution, 3D cascade) and the optimization of the postprocessing strategy.

## How to get started?
Read these:
- [Installation instructions](documentation/installation_instructions.md)
- [Dataset conversion](documentation/dataset_format.md)
- [Usage instructions](documentation/how_to_use_nnunet.md)

Additional information:
- [Learning from sparse annotations (scribbles, slices)](documentation/ignore_label.md)
- [Region-based training](documentation/region_based_training.md)
- [Manual data splits](documentation/manual_data_splits.md)
- [Pretraining and finetuning](documentation/pretraining_and_finetuning.md)
- [Intensity Normalization in nnU-Net](documentation/explanation_normalization.md)
- [Manually editing nnU-Net configurations](documentation/explanation_plans_files.md)
- [Extending nnU-Net](documentation/extending_nnunet.md)
- [What is different in V2?](documentation/changelog.md)

Competitions:
- [AutoPET II](documentation/competitions/AutoPETII.md)

[//]: # (- [Ignore label]&#40;documentation/ignore_label.md&#41;)

## Where does nnU-Net perform well and where does it not perform?
nnU-Net excels in segmentation problems that need to be solved by training from scratch, 
for example: research applications that feature non-standard image modalities and input channels,
challenge datasets from the biomedical domain, majority of 3D segmentation problems, etc . We have yet to find a 
dataset for which nnU-Net's working principle fails!

Note: On standard segmentation 
problems, such as 2D RGB images in ADE20k and Cityscapes, fine-tuning a foundation model (that was pretrained on a large corpus of 
similar images, e.g. Imagenet 22k, JFT-300M) will provide better performance than nnU-Net! That is simply because these 
models allow much better initialization. Foundation models are not supported by nnU-Net as 
they 1) are not useful for segmentation problems that deviate from the standard setting (see above mentioned 
datasets), 2) would typically only support 2D architectures and 3) conflict with our core design principle of carefully adapting 
the network topology for each dataset (if the topology is changed one can no longer transfer pretrained weights!) 

## What happened to the old nnU-Net?
The core of the old nnU-Net was hacked together in a short time period while participating in the Medical Segmentation 
Decathlon challenge in 2018. Consequently, code structure and quality were not the best. Many features 
were added later on and didn't quite fit into the nnU-Net design principles. Overall quite messy, really. And annoying to work with.

nnU-Net V2 is a complete overhaul. The "delete everything and start again" kind. So everything is better 
(in the author's opinion haha). While the segmentation performance [remains the same](https://docs.google.com/spreadsheets/d/13gqjIKEMPFPyMMMwA1EML57IyoBjfC3-QCTn4zRN_Mg/edit?usp=sharing), a lot of cool stuff has been added. 
It is now also much easier to use it as a development framework and to manually fine-tune its configuration to new 
datasets. A big driver for the reimplementation was also the emergence of [Helmholtz Imaging](http://helmholtz-imaging.de), 
prompting us to extend nnU-Net to more image formats and domains. Take a look [here](documentation/changelog.md) for some highlights.

# Acknowledgements
<img src="documentation/assets/HI_Logo.png" height="100px" />

<img src="documentation/assets/dkfz_logo.png" height="100px" />

nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) 
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the 
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).
