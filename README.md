# 3i4K
Intonation-aided identification of intention for Korean

## Requirements
fastText, Keras (TensorFlow), Numpy, Librosa

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
<image src="https://github.com/warnikchow/3i4k/blob/master/images/portion.PNG" width="500"><br/>
### Block diagram<br/>
<image src="https://github.com/warnikchow/3i4k/blob/master/images/fig1.png" width="700"><br/>

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

## YouTube demo (non-audio-input version; for past submission)
https://youtu.be/OlvLlH8JgmM
