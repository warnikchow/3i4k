# 3i4K
Intonation-based identification of intention for Korean

## Requirements
fastText, Keras (TensorFlow), Numpy

## Word Vector (model.bin)
https://drive.google.com/file/d/1QtkE0hiBT97c5AMNln-F3Sx7F3xZL0xc/view?usp=sharing
* Download this and unzip THE .BIN FILE in the same folder with 'onlychar_fast_execute.py'.
* This can be replaced with whatever model the user employs, but it requires an additional training.

## System Description
* The system was trained with 'classify.py' (line by line!)
* Easy start: Python3 execute file
<pre><code> python3 onlychar_fast_execute.py </code></pre>

### Intention Identification
 - First, this system detects the fragments.
 - Next, if the input sentence clearly shows its intention among statement, question, command, rhetorical question and rhetorical command (idiomatic expression), this directly decides the intention.
 - If the intonation information is indispensable, this requires an auxiliary input and makes an intonation-aided decision.
 
### Text Classification
 - This system classifies a corpus (input: filename without '.txt') into 7 categories: fragments, intonation-dependent utterances, and previously mentioned 5 clear-cut cases.

## Annotation Guideline (in Korean)
https://drive.google.com/file/d/1JvZpCQEa4FkFgDAKO3VKJueFYOpJDUNQ/view?usp=sharing

## YouTube demo
https://youtu.be/OlvLlH8JgmM
