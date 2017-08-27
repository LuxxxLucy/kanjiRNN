# Kanji RNN

Using RNN to generate Kanji Characters (vector image).

Example Training Sketches (20 randomly chosen out of 11000 [KanjiVG](http://kanjivg.tagaini.net/) dataset):

![Example Training Sketches](https://cdn.rawgit.com/hardmaru/sketch-rnn/master/example/training.svg)

Generated Sketches (Temperature = 0.1):

![Generated Sketches](https://cdn.rawgit.com/hardmaru/sketch-rnn/master/example/output.svg)

# requirement

```
svgwrite
xml.etree.ElementTree
argparse
cPickle
svg.path
```

# usages

run `python train,py --data_set kanji`

You can use `utils.py` to play out some random training data after the svg files have been copied in:

```
%run -i utils.py
loader = SketchLoader(data_filename = 'tuberlin')
draw_stroke_color(random.choice(loader.raw_data))
```

The default values are in `train.py`

```
--rnn_size RNN_SIZE             size of RNN hidden state (256)
--num_layers NUM_LAYERS         number of layers in the RNN (2)
--model MODEL                   rnn, gru, or lstm (lstm)
--batch_size BATCH_SIZE         minibatch size (100)
--seq_length SEQ_LENGTH         RNN sequence length (300)
--num_epochs NUM_EPOCHS         number of epochs (500)
--save_every SAVE_EVERY         save frequency (250)
--grad_clip GRAD_CLIP           clip gradients at this value (5.0)
--learning_rate LEARNING_RATE   learning rate (0.005)
--decay_rate DECAY_RATE         decay rate after each epoch (adam is used) (0.99)
--num_mixture NUM_MIXTURE       number of gaussian mixtures (24)
--data_scale DATA_SCALE         factor to scale raw data down by (15.0)
--keep_prob KEEP_PROB           dropout keep probability (0.8)
--stroke_importance_factor F    gradient boosting of sketch-finish event (200.0)
--dataset_name DATASET_NAME     name of directory containing training data (kanji)
```

## Sampling a Sketch

I've included a pretrained model in `/save` so it should work out of the box.

Running `python sample.py --filename output --num_picture 10 --dataset_name kanji`

## More useful links, pointers, datasets

- Alex Graves' [paper](http://arxiv.org/abs/1308.0850) on text sequence and handwriting generation.

- [KanjiVG](http://kanjivg.tagaini.net/). Fantastic Database of Kanji Stroke Order.

# License

MIT
