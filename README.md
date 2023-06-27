# RS-SUM
RS-SUM: ADOPTING SELF-SUPERVISED LEARNING INTO UNSUPERVISED VIDEO SUMMARIZATION THROUGH RESTORATIVE SCORE
This repository contains the code implementation of RS-SUM, a method described in the paper "Adopting Self-Supervised Learning into Unsupervised Video Summarization through Restorative Score." The paper has been accepted to IEEE ICIP 2023.

Introduction
RS-SUM is a novel approach to unsupervised video summarization that leverages self-supervised learning and restorative score. It aims to automatically generate concise summaries from unannotated videos without the need for manual supervision. The method incorporates self-supervised learning techniques to learn meaningful representations from video data, which are then used to calculate a restorative score for each frame. Frames with high restorative scores are selected to construct the final video summary.

Citation
If you use RS-SUM or find it helpful in your research, please consider citing our paper:

bibtex
Copy code
@inproceedings{mehryar2023icip,
  title={ADOPTING SELF-SUPERVISED LEARNING INTO UNSUPERVISED VIDEO SUMMARIZATION THROUGH RESTORATIVE SCORE},
  author={Abbasi, Mehryar and Saeedi, Parvaneh},
  booktitle={Proceedings of IEEE International Conference on Image Processing (ICIP)},
  year={2023},
  organization={IEEE}
}
Usage
To run the code, follow the instructions below:

Set the --video_type parameter to either "SumMe" or "TVSum" based on the video dataset you want to use.

Set the --mask_ratio parameter to define the masking ratio percentage during training.

Set the --window_s parameter to specify the masking window size during training.

Set the --mask_ratio_fs parameter to determine the masking ratio for frame score generation.

Set the --window_s_fs parameter to set the masking window size for frame score generation.

Set the --n_epochs parameter to define the number of epochs for training.

Set the --losstype parameter to "CB" for combined Cos for cosine embedding loss or "L1" for L1 loss.

Run main.py to train the RS-SUM model.

bash
Copy code
python main.py
Run compute_fscore.py to calculate the F-score.
bash
Copy code
python compute_fscore.py
