# Kanji RNN

still under development!!!!cannot run without dataset!

Using RNN to generate Kanji Characters (vector image).

Example Training Sketches (20 randomly chosen out of 11000 [KanjiVG](http://kanjivg.tagaini.net/) dataset):

![Example Training Sketches](https://cdn.rawgit.com/hardmaru/sketch-rnn/master/example/training.svg)

Generated Sketches (Temperature = 0.1):

![Generated Sketches](https://cdn.rawgit.com/hardmaru/sketch-rnn/master/example/output.svg)

# requirement

```
python 3
tqdm
tensorflow
keras
svgwrite
xml.etree.ElementTree
argparse
cPickle
svg.path
```

# usages

run `python main_entry.py --data_set kanji --mode train` to start training. Or you can load a pre-trained model (which is not uploaded yet) and only want it to generate kanji using `python main_entry.py --mode sample` (if you do not specify mode argument, the default setting is 'sample')

The default values are in `main_entry.py`

## Sampling a Sketch

Running `python main_entry.py --filename output --num_picture 10 --dataset_name kanji`

## More useful links, pointers, datasets

- Alex Graves' [paper](http://arxiv.org/abs/1308.0850) on text sequence and handwriting generation.

- [KanjiVG](http://kanjivg.tagaini.net/). Fantastic Database of Kanji Stroke Order.

# License

MIT
