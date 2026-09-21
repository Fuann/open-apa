# Third-party notices

OpenAPA includes evaluation recipes, adapted source code, and references to
third-party datasets and model artifacts. The OpenAPA BSD 3-Clause License
applies only to contributions owned by the OpenAPA contributors. Third-party
materials remain subject to their respective licenses and terms.

## MultiPA

The `egs/multipa` recipe contains code adapted from the original
[MultiPA](https://github.com/yuwchen/MultiPA) implementation.

License: MIT

Copyright (c) 2023 yuwchen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The released MultiPA model and pilot data are obtained separately from the
[authors' Hugging Face repository](https://huggingface.co/yuwchen/multipa),
which identifies them as Apache-2.0. Users should review the repository's
current model card and terms before use.

## Charsiu

Parts of the alignment implementation in `egs/multipa` are adapted from
[Charsiu](https://github.com/lingjzhu/charsiu).

License: MIT

Copyright (c) 2021 jzhu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## HiPPO and CTC-based GOP

The `egs/hippo` recipe reproduces the
[HiPPO](https://github.com/bicheng1225/HIPPO) method and downloads its released
checkpoint at runtime. It also downloads model artifacts from
[CTC-based-GOP](https://github.com/frank613/CTC-based-GOP). These upstream
repositories did not include an explicit license file when this notice was
prepared. Their inclusion here is attribution, not a grant of rights. Users
should consult the respective authors or repositories for applicable terms
before redistributing their code or model artifacts.

## SpeechOcean762

The fixed ASR reference transcripts under
`references/speechocean762/` are derived from the
[SpeechOcean762](https://huggingface.co/datasets/mispeech/speechocean762)
dataset, which is distributed under the Apache License 2.0. OpenAPA does not
redistribute the source audio.

## Other dependencies

The recipes install additional third-party Python packages and pretrained
models. Those components are not relicensed by OpenAPA and remain governed by
their own licenses. See each recipe's `requirements.txt`, download URLs, and
upstream documentation for the complete dependency set and current terms.
