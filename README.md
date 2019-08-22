## 1. INTRODUCTION

*   In this project, I tried to train a Sentiment Analyzer/Text Classifier for Beyazperde movie critics.

There are lots of resources for English Sentiment Analysis but in Turkish, we have limited resources 

for Sentiment Analyzing. In order to increase resources about Turkish Sentiment Analyzing, I started 

to this project. 


*   Sentiment Analyzing is a branch of Natural Language Processing. In this field’s projects usually 

there are some unlabeled data and, you try to predict which class they belong to. In order to implement 

this process, there are some Sentiment Analyzing steps.


<br>

## 2. SENTIMENT ANALYZING STEPS

<br>

### 2.1. DATA FETCHING

*   The first rule is to get adequate dataset to train your model efficiently. Here, I have 4995 movie 

critics from Beyazperde. You can find it from this [link](http://www.beyazperde.com/filmler/elestiriler-beyazperde/).


<br>

### 2.2. DATA PREPROCESSING

*   This step is the crucial step for any kind of Machine Learning model training. Real life data is 

not always clean. So, you must process your dataset as possible as. In Machine Learning, there 

is a ratio that is, data preprocessing/cleaning is 80% and modelling is 20% of overall work. So, I 

also splitted data preprocessing into sub steps. 


<br>

#### 2.2.1. LOAD DATASET
		
* [Dataset](https://github.com/slmttndrk/job1/blob/master/sample_beyazperde_dataset.csv) is in the form of csv file

<br>

#### 2.2.2. ELIMINATE NAN VALUES
		
*   Nan values is not useful for training model

<br>

#### 2.2.3. ARRANGE DATASET TO AVOID OVERFITTING

*   If the sizes are unbalanced the model overfits while prediction

<br>

#### 2.2.4. ELIMINATE TURKISH STOPWORDS AND PUNCTUATIONS

*   [Stopwords](https://github.com/slmttndrk/job1/blob/master/stopwords.txt) and punctuations are unnecessary for training model

<br>

#### 2.2.5. NORMALIZATION

*   This corrects the miswritten words and throws meaningless words away

<br>

#### 2.2.6. STEMMING/LEMMATIZATION

*   This removes the suffixes and gives us the root of each word


<br>

### 2.3. DATA CLASSIFICATION
	
*   In this step, you choose a Machine Learning algorithm for Sentiment Analyzing/Text Classification. 

All algorithms can be used, but I chose the [Multinomial Naive Bayes algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html). Since, it gives 

better accuracy scores on Sentiment Analyzing/Text Classification. This algorithm assumes that 

the presence of a particular feature in a class is unreletad to the presence of any other feature. 

I also, splitted data classification into some sub steps.

<br>
	
#### 2.3.1. SPLITTING [TRAIN AND TEST DATA](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html):

*   Usually, we partition the dataset into 80% as training and 20% as testing data

<br>
    
#### 2.3.2. VECTORIZATION

*   There are some methods such as Bag Of Words, Count Vectorizer and [Tfidf Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html). I chose Tfidf Vectorizer.

<br>

#### 2.3.3. [GRIDSEARCHCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html):

*   This method enables us to find the best hyperparameter for the model

<br>

#### 2.3.4. FIT AND PREDICT

*   The model learns by fitting and analyzes the sentiment by predicting

<br>
    
#### 2.3.5. OBSERVING [ACCURACY](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html), [F1, PRECISION AND RECALL](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) SCORES:

*   This scores are useful for comparing model’s success

<br>
    
#### 2.3.6. OBSERVING [CONFUSION MATRIX](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) AND PREDICTION PROBABILITIES

*   This gives us an intuition of how confidently the model makes the predictions

<br>
    
#### 2.3.7. [TEN-FOLD CROSS VALIDATION](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html):

*   This enables us to train our model with different samples of the same dataset so that, we can check if it 
    
learned correctly or not


<br>

### 2.4. MODEL PIPELINING AND PICKLING

*   In this step, I create a pipeline for the model. [Pipelining](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) prevents us from repeating all steps again 

and again. With the help of pipelining, when I give any raw unlabeled data, at first, the model preprocess

it and then, makes prediction. So, it makes our model reusable.


*   [Pickling](https://scikit-learn.org/stable/modules/model_persistence.html) a model means transforming it into binary form. It makes our model portable. When you want to

use the model in different projects, by just loading this pickled file, you can use the model and get

predictions wherever you want. 


<br>

## 3. CONCLUSION

*   In this project, I learned the concept of Text Classification/Sentiment Analyzing. It also provided 

me knowledge base for Natural Language Processing. Since, getting and preprocessing the dataset is 

the crucial part of any Machine Learning model training. 


<br>

## 4. IMPROVEMENTS

*   The model score can be improved by increasing the number of "Turkish Stopwords" or "Beyazperde Dataset". 

In both cases, model will be trained more efficiently. 

<br>

## 5. RESOURCES/THANKS

*   I completed this project in cooperation with [Verius Technology Company](https://verius.com.tr/).The training dataset (Beyazperde) 

and data preprocessing tools (normalization, stemming) are provided me by them. I also used python libraries 

such as: Sklearn, Pandas, Numpy, Nltk.


<br>

