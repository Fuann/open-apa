# MultiPA paper reproduction

This recipe provides a reproducible evaluation setup for the original [MultiPA (Interspeech 2024)](https://www.isca-archive.org/interspeech_2024/chen24c_interspeech.html/) method on speechocean762 and MultiPA.


### Setup
Install the Python dependencies with:

```bash
conda create -n multipa python=3.9
conda activate multipa
python -m pip install --upgrade "pip<24.1"
python -m pip install -r requirements.txt
```

### Usage

```bash
# MultiPA
bash run.sh --test-data multipa

# speechocean762 (closed + open)
bash run.sh --test-data speechocean762
```

## Citation

If you find this repository useful, please cite the following paper:

```bibtex
@inproceedings{chen24c_interspeech,
  title     = {{MultiPA: A Multi-task Speech Pronunciation Assessment Model for Open Response Scenarios}},
  author    = {Yu-Wen Chen and Zhou Yu and Julia Hirschberg},
  year      = {2024},
  booktitle = {{Interspeech 2024}},
  pages     = {297--301},
  doi       = {10.21437/Interspeech.2024-123},
  issn      = {2958-1796},
}
```