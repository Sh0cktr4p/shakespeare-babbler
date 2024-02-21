# Shakespeare Babbler: An Infinite Text Generating Transformer

This project is based on the excellent tutorial by Andrej Carpathy (https://www.youtube.com/watch?v=kCc8FmEb1nY).

More on transformers: https://arxiv.org/abs/1706.03762

Trains a character-level decoder-only transformer model on text inputs.
Once trained, models can be used to auto-generate text based on the training dataset.
The text generator use a double buffer method to print characters and generate new ones in parallel.


## Installation

This project requires python >= 3.10.

To install, run

```
pip install .
```

## Training

For training, execute

```
python train_language_model.py dataset.path=path/to/dataset.txt
```

If you want/need to run the training on the cpu:

```
python train_language_model.py dataset.path=path/to/dataset.txt training.device=cpu
```

By default, models are stored in the `models/` folder, log `.csv` files are stored in the `logs/` folder. To change these values, take a look at the configuration in `config/`. This project uses hydra (https://hydra.cc/docs/intro/), allowing for changing the configuration either in the config file or from the command line. 

## Text Generation

To generate text using a trained model, execute:

```
python text_generator.py dataset.path=path/to/dataset.txt
```

The dataset path is required to obtain the charset for encoding/decoding.


## Dockerization

To build the docker image, run

```
./build_docker.sh
```

To run, execute

```
./run_docker.sh
```