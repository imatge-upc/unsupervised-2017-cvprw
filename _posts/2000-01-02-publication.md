---
title: "publication"
bg: blue
color: white
fa-icon: book
---

*This paper instroduces an unsupervised framework to extract semantically rich features for video representation. Inspired by how the human visual system groups objects based on motion cues, we propose a deep convolutional neural network that disentangles motion, foreground and background information. The proposed architecture consists of a 3D convolutional feature encoder for blocks of 16 frames, which is trained for reconstruction tasks over the first and last frames of the sequence. The model is trained with a fraction of videos from the UCF-101 dataset taking as ground truth the bounding boxes around the activity regions. Qualitative results indicate that the network can successfully update the foreground appearance based on pure-motion features. The benefits of these learned features are shown in a discriminative classification task when compared with a random initialization of the network weights, providing a gain of accuracy above the 10%.*

Find our paper on [arXiv](https://arxiv.org/abs/1707.04092)  or download the [PDF file](https://github.com/imatge-upc/unsupervised-2017-cvprw/raw/gh-pages/lin-2017-cvprw.pdf). An [extended abstract](https://openreview.net/forum?id=HkJLyTwgZ&noteId=HkJLyTwgZ) of this publication was accepted as poster in the [CVPR 2017 Workshop on Brave new ideas for motion representations in videos II](http://bravenewmotion.github.io/).

If you find this work useful, please consider citing:

```
Lin, Xunyu, Victor Campos, Xavier Giro-i-Nieto, Jordi Torres, and Cristian Canton Ferrer. "Disentangling Motion, Foreground and Background Features in Videos." CVPR 2017 Workshop: Brave new ideas for motion representations in videos II (2017).
}```

<pre>
@InProceedings{lin2017disentangling,
  title={Disentangling Motion, Foreground and Background Features in Videos},
  author={Lin, Xunyu and Campos, Victor and Giro-i-Nieto, Xavier and Torres, Jordi and Ferrer, Cristian Canton},
  booktitle = {CVPR 2017 Workshop: Brave new ideas for motion representations in videos II},
  month = {July},
  year={2017}
  }
</pre>
