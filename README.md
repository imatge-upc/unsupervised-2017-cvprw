# Disentangling Foreground, Background and Motion Features in Videos
This repo contains the source codes for our work as in title. Please refer to our [project webpage](https://imatge-upc.github.io/unsupervised-2017-cvprw/) or [original paper](https://arxiv.org/pdf/1707.04092.pdf) for more details.

## Dataset

This project requires [UCF-101 dataset](http://crcv.ucf.edu/data/UCF101.php) and its [localization annotations](http://www.thumos.info/download.html) (bonding box for action region). Please note that the annotations only contain bounding boxes for 24 classes out of 101. We only use these 24 classes for further experiments.

### Download link
```
UCF-101: http://crcv.ucf.edu/data/UCF101/UCF101.rar
Annotations (version 1): http://crcv.ucf.edu/ICCV13-Action-Workshop/index.files/UCF101_24Action_Detection_Annotations.zip 
```

### Dataset split

We split our dataset into training set, validation set and test set. Split lists of each set can be found under **`dataset`** folder.

### Generate TF-Records

As we are dealing with videos, using TF-records in TensorFlow can help to reduce I/O overheads. (Please refer to the [official documentation](https://www.tensorflow.org/api_guides/python/reading_data) if you're not familiar with TF-records). Each [**`SequenceExample`**](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/core/example/example.proto) in our TF-records includes 32 video frames, corresponding masks and so on.

A brief description of how we generate TF-records from videos and annotations: for each video, we split it into multiple chunks consisting of 32 frames and save each chunk as an example. The corresponding masks are generated through manipulations on localization annotations.

In order to generate TF-records used by this project, you need to modify certain paths in **`tools/generate_tfrecords.py`**. Including, **`videos_directory`**, **`annotation_directory`** and so on. As we use FFMPEG to decode videos, you may want to install it with the command below if you are using Anaconda:



```
conda install ffmpeg
```

After installation of FFMPEG, you need to specify the path to executable FFMPEG binary file in **`tools/ffmpeg_reader.py`**. (Usually it's just **`~/anaconda/bin/ffmpeg`** if you are using Anaconda). After specifying path to FFMPEG, you are good to go! Run the script as below to generate those TF-records:

```
python tools/generate_tfrecords.py
```

## Training & Testing

Our codes for training and testing are organized in the following fashion: we have scripts under **`models/`** to construct TensorFlow graph for each model. And under top path, we have scripts named as **`***_[train|val|test].py`**. These are the scripts accomplish external call and training/validation/test of each model.


