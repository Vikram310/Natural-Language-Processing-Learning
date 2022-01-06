# Natural-Language-Processing-Learning

[**Day1**](https://www.linkedin.com/posts/vikram--krishna_datawithvikram-datascience-careers-activity-6884374937344335874-bziD)

**💡 Word2Vec**: 
- A two layer neural network to generate word embeddings given a text corpus. Word Embeddings is mapping of words in a vector space. It Preserves relationship between words. It Deals with addition of new words in the vocabulary. Also, it gives better results in lots of deep learning applications. 
- The word2vec objective function causes the word that occur in similar contexts to have similar embeddings. The Idea is to design a model whose parameters are the word vectors. Then, we train the model on a certain objective. At every iteration, we run our model, evaluate the error, and follow an update rule that has some notion of penalizing the model parameters that caused the error. It is done using 2 algorithms and 2 training methods.

**Algorithms**
  - Continuous Bag of Words:  It aims to predict a center word from the surrounding context in terms of word vectors. 
  - Skip-gram : It does the opposite of CBOW, and predicts the distribution of context words from a center word.

**Training Methods**
  - Negative Sampling - It defines an objective by sampling negative samples
  - Hierarchical Softmax: It defines an objective using an efficient tree structure to compute the probabilities for all the vocabulary. 
 
- Reference:
  - [Paper on Word2Vec](http://arxiv.org/pdf/1301.3781.pdf)
  - [Negative sampling paper](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

[**Day2**](https://www.linkedin.com/posts/vikram--krishna_datawithvikram-datascience-careers-activity-6884737641330413568-QZNn)

**💡 Skip-gram**: 
- The training objective of the Skip-gram model is to find word representations that are useful for predicting the surrounding words in a sentence or a document. The basic Skip-gram formulation defines the probability using the SoftMax function. we use SoftMax function because

      Max -> amplifies the probability to largest word-vector
      soft -> still assigns some probability to smallest word-vector
- To train the model, we gradually adjust the parameters to minimize the loss by finding the gradient of the objective function. 

**Negative sampling**
 - Loss functions of CBOW and skip-gram are quite expensive to compute because of the SoftMax normalization where we sum over size of the vocabulary (|V|). For every training step, instead of looping over entire vocabulary, we just sample out several negative samples. 
 - We sample from the noise distribution whose probabilities match the ordering of the frequency of the vocabulary.The “negative samples” are selected using a “unigram distribution”, where more frequent words are more likely to be selected as negative samples.
- To augment our formulation of the problem of incorporating negative sampling, we need to update the objective function, gradients and update-rules. Skip-gram with negative sampling will change the task of predicting neighboring words to answer the question if two words are neighbors or not? This will change the task of Multi classification to Binary Classification and we replace the expensive SoftMax function with less-expensive sigmoid function.
 
- Reference:
  - [Negative sampling paper](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

