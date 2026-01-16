# PlaNet

A PyTorch implementation of **PlaNet: Learning Latent Dynamics for Planning from Pixels** [[1]](https://www.google.com/search?q=%23references), specifically adapted for the **Safety Gymnasium** environment.

This repository builds upon the [Kaixhin/PlaNet](https://github.com/Kaixhin/PlaNet) codebase. 


## Usage

To train an agent on a specific Safety Gymnasium task, run:

```bash
python main.py
```


## Requirements

The following dependencies are required:

* [PyTorch](https://pytorch.org/get-started/locally)
* [Safety Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium)
* [OpenCV Python](https://pypi.org/project/opencv-python/)

You can install all dependencies at once using:

```bash
pip install -r requirements.txt
```


## Links

* [PlaNet Paper (arXiv)](https://arxiv.org/abs/1811.04551)
* [PlaNet Project Page](https://danijar.com/project/planet)


## Acknowledgements

* [@Kaixhin](https://www.google.com/search?q=https://github.com/Kaixhin) for the [PlaNet PyTorch implementation](https://github.com/Kaixhin/PlaNet) that served as the foundation for this project.
* [@danijar](https://github.com/danijar) for the original [Google Research PlaNet](https://github.com/google-research/planet) implementation.
* [@PKU-Alignment](https://www.google.com/search?q=https://github.com/PKU-Alignment) for the [Safety Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) environment.


## References

[1] [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)