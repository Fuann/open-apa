<p align="center">
  <img src="assets/openapa-banner.png" alt="OpenAPA — Open-response pronunciation assessment" width="100%">
</p>

#

**OpenAPA** is an open-source benchmark and evaluation toolkit for pronunciation assessment in open-response scenarios. It is designed to grow with new datasets, methods, and evaluation protocols.

Currently supported:

- Recipes: [MultiPA](egs/multipa) and [HiPPO](egs/hippo)
- Evaluation levels: word and utterance

## Quick start

Clone the repository and run the MultiPA recipe:

```bash
git clone https://github.com/Fuann/open-apa.git
cd open-apa/egs/multipa

conda create -n multipa python=3.9
conda activate multipa
python -m pip install --upgrade "pip<24.1"
python -m pip install -r requirements.txt

# MultiPA open-response evaluation
bash run.sh --test-data multipa

# SpeechOcean762 closed- and open-response evaluation
bash run.sh --test-data speechocean762
```

## License

OpenAPA is released under the [BSD 3-Clause License](LICENSE).
