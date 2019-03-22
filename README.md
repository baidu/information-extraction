# Information Extraction Baseline System—InfoExtractor
## Abstract
InfoExtractor is an information extraction baseline system based on the Schema constrained Knowledge Extraction dataset(SKED). InfoExtractor adopt a pipeline architecture with a p-classification model and a so-labeling model which are both implemented with PaddlePaddle. The p-classification model is a multi-label classification which employs a stacked Bi-LSTM with max-pooling network, to identify the predicate involved in the given sentence. Then a deep Bi-LSTM-CRF network is adopted with BIEO tagging scheme in the so-labeling model to label the element of subject and object mention, given the predicate which is distinguished in the p-classification model. The F1 value of InfoExtractor on the development set is 0.668.

## Getting Started
### Environment Requirements
Paddlepaddle v1.2.0 </br>
Numpy </br>
Memory requirement 10G for training and 6G for infering

### Step 1: Install paddlepaddle
For now we’ve only tested on PaddlePaddle Fluid v1.2.0, please install PaddlePaddle firstly and see more details about PaddlePaddle in [PaddlePaddle Homepage](http://www.paddlepaddle.org/).

### Step 2: Download the training data, dev data and schema files
Please download the training data, development data and schema files from [the competition website](http://lic2019.ccf.org.cn/kg), then unzip files and put them in ```./data/``` folder.
```
cd data
unzip train_data.json.zip 
unzip dev_data.json.zip
cd -
```
### Step 3: Get the vocabulary file
Obtain high frequency words from the field ‘postag’ of training and dev data, then compose these high frequency words into a vocabulary list.
```
python lib/get_vocab.py ./data/train_data.json ./data/dev_data.json > ./dict/word_idx
```
### Step 4: Train p-classification model
First, the classification model is trained to identify predicates in sentences. Note that if you need to change the default hyper-parameters, e.g. hidden layer size or whether to use GPU for training (By default, CPU training is used), etc. Please modify the specific argument in ```./conf/IE_extraction.conf```, then run the following command:
```
python bin/p_classification/p_train.py --conf_path=./conf/IE_extraction.conf
```
The trained p-classification model will be saved in the folder ```./model/p_model```.
### Step 5: Train so-labeling model
After getting the predicates that exist in the sentence, a sequence labeling model is trained to identify the s-o pairs corresponding to the relation that appear in the sentence. </br>
Before training the so-labeling model, you need to prepare the training data that meets the training model format to train a so-labeling model.
```
python lib/get_spo_train.py  ./data/train_data.json > ./data/train_data.p
python lib/get_spo_train.py  ./data/dev_data.json > ./data/dev_data.p
```
To train a so labeling model, you can run:
```
python bin/so_labeling/spo_train.py --conf_path=./conf/IE_extraction.conf
```
The trained so-labeling model will be saved in the folder ```./model/spo_model```.

### Step 6: Infer with two trained models
After the training is completed, you can choose a trained model for prediction. The following command is used to predict with the last model. You can also use the development set to select the optimal model for prediction. To do inference by using two trained models with the demo test data (under ```./data/test_demo.json```), please execute the command in two steps:
```
python bin/p_classification/p_infer.py --conf_path=./conf/IE_extraction.conf --model_path=./model/p_model/final/ --predict_file=./data/test_demo.json > ./data/test_demo.p
python bin/so_labeling/spo_infer.py --conf_path=./conf/IE_extraction.conf --model_path=./model/spo_model/final/ --predict_file=./data/test_demo.p > ./data/test_demo.res
```
The predicted SPO triples will be saved in the folder ```./data/test_demo.res```.

## Discussion
If you have any questions, you can submit an issue in github and we will respond periodically. </br>


## Copyright and License
Copyright 2019 Baidu.com, Inc. All Rights Reserved </br>
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may otain a copy of the License at </br>
```http://www.apache.org/licenses/LICENSE-2.0``` </br>
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# APPENDIX
In the released dataset, the field ‘postag’ of sentences represents the segmentation and part-of-speech tagging information. The abbreviations of part-of-speech tagging (PosTag) and their corresponding part of speech meanings are shown in the following table. </br>
In addition, the given segmentation and part-of-speech tagging of the dataset are only references and can be replaced with other segmentation results.</br>

|POS| Meaning |
|:---|:---|
| n |common nouns|
| f | localizer |
| s | space |
| t | time|
| nr | noun of people|
| ns | noun of space|
| nt | noun of time|
| nw | noun of work|
| nz | other proper noun|
| v | verbs |
| vd | verb of adverbs|
| vn |verb of noun|
| a | adjective |
| ad | adjective of adverb|
| an | adnoun |
| d | adverbs |
| m | numeral |
| q | quantity|
| r | pronoun |
| p | prepositions |
| c | conjunction |
| u | auxiliary |
| xc | other function word |
| w | punctuations |
