# Kanji rnn

Generate Kanji using RNN

## Setup

To run this code you need the following:

- a machine with multiple GPUs
- Python3
- Numpy, TensorFlow

## Training the model

Use the `main_entry.py` script to train the model. To train the default model on CIFAR-10 simply use:

```
python3 main_entry.py
```

You might want to at least change the `--data_dir` and `--save_dir` which point to paths on your system to download the data to (if not available), and where to save the checkpoints.

**I want to train on fewer GPUs**. To train on fewer GPUs we recommend using `CUDA_VISIBLE_DEVICES` to narrow the visibility of GPUs to only a few and then run the script. Don't forget to modulate the flag `--nr_gpu` accordingly.

**I want to train on my own dataset**. Have a look at the `DataLoader` classes in the `data/` folder. You have to write an analogous data iterator object for your own dataset and the code should work well from there.

## Pretrained model checkpoint

## Citation

### Data

The data should be placed in the `data_store` folder, and the structure is like:

```
--data_store
```
