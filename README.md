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

