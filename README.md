# Text To Image Synthesis 


This is an experimental tensorflow implementation of synthesizing images. The images are synthesized using the GAN-CLS Algorithm from the paper [Generative Adversarial Text-to-Image Synthesis][2]. This implementation is built on top of the excellent [DCGAN in Tensorflow][3]. The following is the model architecture.

![Model architecture](http://i.imgur.com/dNl2HkZ.jpg)

Image Source : [Generative Adversarial Text-to-Image Synthesis][2] Paper

## Requirements
- [TensorLayer](https://github.com/zsdonghao/tensorlayer) 1.3.1 (or later)
- [Tensorflow][4] 0.10 (or later)
- [NLTK][8] : for tokenizer

## Datasets
- The model is currently trained on the [flowers dataset][9]. Download the images from [here][9] and save them in ```102flowers/102flowers/*.jpg```. Also download the captions from [this link][10]. Extract the archive, copy the ```text_c10``` folder and paste it in ```102flowers/text_c10/class_*```.

## Codes
- `train_dcgan.py` use DCGAN to generate new images.
- `train_txt2im.py` use GAN-CLS to generate new images conditioned on text.
- `utils.py` helper functions.


## References
- [Generative Adversarial Text-to-Image Synthesis][2] Paper
- [Generative Adversarial Text-to-Image Synthesis][11] Torch Code
- [Skip Thought Vectors][1] Paper
- [Skip Thought Vectors][12] Code
- [Generative Adversarial Text-to-Image Synthesis with Skip Thought Vectors](https://github.com/paarthneekhara/text-to-image) TensorFlow code
- [DCGAN in Tensorflow][3]

## Credits


## License
Apache 2.0


[1]:http://arxiv.org/abs/1506.06726
[2]:http://arxiv.org/abs/1605.05396
[3]:https://github.com/carpedm20/DCGAN-tensorflow
[4]:https://github.com/tensorflow/tensorflow
[5]:http://www.h5py.org/
[6]:https://github.com/Theano/Theano
[7]:http://scikit-learn.org/stable/index.html
[8]:http://www.nltk.org/
[9]:http://www.robots.ox.ac.uk/~vgg/data/flowers/102/
[10]:https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/view
[11]:https://github.com/reedscot/icml2016
[12]:https://github.com/ryankiros/skip-thoughts
[13]:https://github.com/ryankiros/skip-thoughts#getting-started
[14]:https://bitbucket.org/paarth_neekhara/texttomimagemodel/raw/74a4bbaeee26fe31e148a54c4f495694680e2c31/latest_model_flowers_temp.ckpt