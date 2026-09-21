# HiPPO

This recipe provides a reproducible evaluation setup for the original [HiPPO (IJCNLP-AACL 2025)](https://aclanthology.org/2025.ijcnlp-long.45/) method on speechocean762.

## Setup
Install the Python dependencies with:

```bash
conda create -n hippo python=3.9
conda activate hippo
python -m pip install -r requirements.txt
```

## Usage

```bash


# Closed response (ground-truth)
bash run.sh --response-mode closed --gpu 0

# Open response (medium.en followed by large-v3)
bash run.sh --response-mode open --gpu 0

# Others
bash run.sh --help

```

## Citation

If you find this repository useful, please cite the following paper:

```bibtex
@inproceedings{yan-etal-2025-hippo,
    title = "{H}i{PPO}: Exploring A Novel Hierarchical Pronunciation Assessment Approach for Spoken Languages",
    author = "Yan, Bi-Cheng  and Wang, Hsin Wei  and Chao, Fu-An  and Lo, Tien-Hong  and Hsu, Yung-Chang  and Chen, Berlin",
    editor = "Inui, Kentaro and Sakti, Sakriani  and Wang, Haofen  and Wong, Derek F.  and Bhattacharyya, Pushpak  and Banerjee, Biplab  and Ekbal, Asif  and Chakraborty, Tanmoy  and Singh, Dhirendra Pratap",
    booktitle = "Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "The Asian Federation of Natural Language Processing and The Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.ijcnlp-long.45/",
    doi = "10.18653/v1/2025.ijcnlp-long.45",
    pages = "810--823",
    ISBN = "979-8-89176-298-5"
}
```
