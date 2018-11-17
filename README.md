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
### FCI: A seven-class text corpus for the classification of conversation-style and non-canonical Korean utterances
* F: Fragments (nouns, noun phrases, incomplete sentences etc.) (FRs)
* C: Clear-cut cases (statements, questions, commands, rhetorical questions, rhetorical commands) (CCs)
* I: Intonation-dependent utterances (IUs)
### Corpus composition<br/>
<image src="https://github.com/warnikchow/3i4k/blob/master/images/portion.PNG" width="400"><br/>
* IAA: *kappa* = 0.85 for Corpus 1
* Data for FCI module is labeled in 0-6, split in train:test with ratio 9:1. 
* Available in *data* folder.
#### For the data, an advanced version (more examples on fragments and IUs, rearrangement for some utterances) is being prepared. Might be released before 2019.	
  
### Block diagram<br/>
<image src="https://github.com/warnikchow/3i4k/blob/master/images/fig1.png" width="500"><br/>

## System Description
* Easy start: Demonstration.exe
<pre><code> python3 3i4k_demo.py </code></pre>

### Intention Identification
- Given only a text input, the system classifies the input into one the aforementioned 7 categories. Available in demo.
- Text classification is also available in demo;  a corpus (input: filename without '.txt') can be categorized into 7 classes.
- Available by importing module
<pre><code> from classify import pred_only_text </code></pre>
<pre><code> from classify import classify_document </code></pre>

### Speech Intention Understanding
- Available by importing module
<pre><code> from classify import pred_audio_text </code></pre>

## Annotation Guideline (in Korean)
https://drive.google.com/open?id=1AvxzEHr7wccMw7LYh0J3Xbx5GLFfcvMW

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
### For the utilization of the [annotation guideline](https://drive.google.com/open?id=1AvxzEHr7wccMw7LYh0J3Xbx5GLFfcvMW) or the [dataset](https://github.com/warnikchow/3i4k/blob/master/data/fci.txt), cite the following:
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
