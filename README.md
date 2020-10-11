# 3i4K
Intonation-aided intention identification for Korean

## Requirements
fastText, Keras (TensorFlow), Numpy, Librosa<br/>
**Currently available for python 3.5 and upper version is in implementation**

## Word Vector
[Pretrained 100dim fastText vector](https://drive.google.com/open?id=1jHbjOcnaLourFzNuP47yGQVhBTq6Wgor)
* Download this and unzip THE .BIN FILE in the NEW FOLDER named 'vectors'
* This can be replaced with whatever model the user employs, but it requires an additional training.

## Dataset
### (20.10.13) We updated our LICENSE to [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/) and rearranged [our guideline](https://docs.google.com/document/d/1-dPL5MfsxLbWs7vfwczTKgBq_1DX9u1wxOgOPn1tOss/edit#) (in Korean) so as to be used officially. The international version is to be prepared with the publication.
### (19.06.06) We provide train & validation, and test set separately [here](https://github.com/warnikchow/3i4k/tree/master/data/train_val_test), for an easier Keras-based implementation.
The (train+validation) : test ratio is 9 : 1, and train : validation ratio is also 9 : 1 (thus, in total, 0.81 : 0.09 : 0.1).
### (19.02.28) [**A Renewed version of the final corpus**](https://github.com/warnikchow/3i4k/blob/master/data/fci.txt)
The renewed version of the corpus is uploaded along with the models. May not be changed unless severe defect is observed.
### (18.12.28) **Dataset under modification!!** 
We've found a few misclassified utterances and undergoing modification, thus the true-final version will be disclosed before Fabrary. Pilot implemenation of the system (e.g., as [tutorial](https://github.com/warnikchow/dlk2nlp)) is less involved with this problem, but do not cite this dataset as a benchmark until Fabrary. The notice will be available as soon as possible.
### (18.11.22) A final version of dataset and the new model is uploaded! 
The next version will incoporate much more utterances and will be treated as a separate dataset.
### FCI: A seven-class text corpus for the classification of conversation-style and non-canonical Korean utterances
* F: Fragments (nouns, noun phrases, incomplete sentences etc.) (FRs)
* C: Clear-cut cases (statements, questions, commands, rhetorical questions, rhetorical commands) (CCs)
* I: Intonation-dependent utterances (IUs)
### Corpus composition<br/>
<image src="https://github.com/warnikchow/3i4k/blob/master/images/portion.PNG" width="400"><br/>
* IAA: *kappa* = 0.85 for Corpus 1
* Data for FCI module is labeled in 0-6, split in train:test with ratio 9:1. 
* Available in *data* folder.

  
### Block diagram<br/>
<image src="https://github.com/warnikchow/3i4k/blob/master/images/fig1.png" width="500"><br/>

## System Description
* Easy start: Demonstration.exe
<pre><code> python3 3i4k_demo.py </code></pre>

### Intention Identification
- Given only a text input, the system classifies the input into one the aforementioned 7 categories. Available in demo.
- Text classification is also available in demo;  a corpus (input: filename without '.txt') can be categorized into 7 classes.
- Available by importing module
<pre><code> from classify import pred_only_text('sentence_you_choose') </code></pre>
<pre><code> from classify import classify_document('filename_you_choose') </code></pre>

### Speech Intention Understanding
- Available by importing module
<pre><code> from classify import pred_audio_text('speechfilename_you_choose', 'sentence_you_choose') </code></pre>

## Annotation Guideline and Acknowledgement
The [annotation guideline (in Korean)](https://docs.google.com/document/d/1-dPL5MfsxLbWs7vfwczTKgBq_1DX9u1wxOgOPn1tOss/edit#) (previous version is [here](https://drive.google.com/open?id=1AvxzEHr7wccMw7LYh0J3Xbx5GLFfcvMW)) was elaborately constructed by Won Ik Cho, with the great help of Ha Eun Park and Dae Ho Kook. Also, the authors appreciate Jong In Kim, Kyu Hwan Lee, and Jio Chung from SNU Spoken Language Processing laboratory (SNU SLP) for providing the useful corpus for the analysis. We note that this work was supported by the Technology Innovation Program (10076583, Development of free-running speech recognition technologies for embedded robot system) funded By the Ministry of Trade, Industry & Energy (MOTIE, Korea).

## Citation
### For the utilization of the [word vector dictionary](https://drive.google.com/open?id=1jHbjOcnaLourFzNuP47yGQVhBTq6Wgor), cite the following:
```
@article{cho2018real,
	title={Real-time Automatic Word Segmentation for User-generated Text},
	author={Cho, Won Ik and Cheon, Sung Jun and Kang, Woo Hyun and Kim, Ji Won and Kim, Nam Soo},
	journal={arXiv preprint arXiv:1810.13113},
	year={2018}
}
```
### For the utilization of the [annotation guideline](https://docs.google.com/document/d/1-dPL5MfsxLbWs7vfwczTKgBq_1DX9u1wxOgOPn1tOss/edit#) or the [dataset](https://github.com/warnikchow/3i4k/blob/master/data/fci.txt), cite the following:
```
@article{cho2018speech,
	title={Speech Intention Understanding in a Head-final Language: A Disambiguation Utilizing Intonation-dependency},
	author={Cho, Won Ik and Lee, Hyeon Seung and Yoon, Ji Won and Kim, Seok Min and Kim, Nam Soo},
	journal={arXiv preprint arXiv:1811.04231},
	year={2018}
}
```

## YouTube demo (non-audio-input version; for past submission)
https://youtu.be/OlvLlH8JgmM
