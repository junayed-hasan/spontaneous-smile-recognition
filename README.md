# DeepMarkerNet: Leveraging Supervision from the Duchenne Marker for Spontaneous Smile Recognition 

(Paper Accepted for Publication in **Pattern Recognition Letters** on 26 September, 2024)



[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11.0-red.svg)](https://pytorch.org/)

## Abstract

Distinguishing between spontaneous and posed smiles from videos poses a significant challenge in pattern classification literature. This project introduces DeepMarkerNet, a novel deep learning solution that leverages Duchenne Marker (D-Marker) features to improve spontaneous smile recognition performance. Our multi-task learning framework integrates a transformer network with the utilization of facial D-Markers for accurate smile classification.

Unlike past methods, our approach simultaneously predicts the class of the smile and associated facial D-Markers using two different feed-forward neural networks, creating a symbiotic relationship that enriches the learning process. The novelty lies in incorporating supervisory signals from pre-calculated D-Markers, harmonizing the loss functions through a weighted average. This allows our training to utilize the benefits of D-Markers without requiring their computation during inference.

## Key Features

- Multi-task learning framework combining smile classification and D-Marker prediction
- Transformer network integration for improved feature extraction
- Utilization of D-Marker features as supervisory signals
- State-of-the-art performance on multiple smile datasets

## Dependencies

- Python 3.8
- numpy 1.21.5
- Pillow 9.0.1
- dlib 19.24.0
- opencv-python 4.6.0.66
- torch 1.11.0
- torchvision 0.12.0
- vidaug 1.5
- einops 0.6.0
- tqdm 4.64.1
- colorama 0.4.6

## Datasets

To reproduce our results, obtain the following datasets:

1. [UVA-NEMO database](https://www.uva-nemo.org)
2. [MMI database](https://mmifacedb.eu)
3. [BBC database](https://www.bbc.co.uk/science/humanbody/mind/surveys/smiles/)
4. [SPOS database](https://www.oulu.fi/cmvs/node/41317)

Note: Only landmarks data from UVA-NEMO are provided in this repository. For other datasets, you can extract landmarks using the code from [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html).

## Usage

### Training DeepMarkerNet

To train the model on the UVA-NEMO dataset:

```bash
python smile_point.py --fold 0
```

This command trains the model using fold 0 of the UVA-NEMO dataset. The trained model weights will be saved in the `uva/labels` folder.

## Results

Our model achieves state-of-the-art results on four well-known smile datasets: UvA-NEMO, BBC, MMI facial expression, and SPOS datasets. Detailed results and comparisons can be found in our paper (link to be added upon publication).

## Citation
```bash
@article{HASAN2024,
title = {DeepMarkerNet: Leveraging supervision from the Duchenne Marker for spontaneous smile recognition},
journal = {Pattern Recognition Letters},
year = {2024},
issn = {0167-8655},
doi = {https://doi.org/10.1016/j.patrec.2024.09.015},
url = {https://www.sciencedirect.com/science/article/pii/S0167865524002770},
author = {Mohammad Junayed Hasan and Kazi Rafat and Fuad Rahman and Nabeel Mohammed and Shafin Rahman}
}
```

## Contributing

We welcome contributions to improve DeepMarkerNet. Please feel free to submit issues and pull requests.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any queries regarding the code or paper, please open an issue or contact Mohammad Junayed Hasan (junayedhasan100@gmail.com).

---

Copyright © 2024 Mohammad Junayed Hasan. All rights reserved.
