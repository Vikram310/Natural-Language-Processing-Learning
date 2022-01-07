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

[**Day3**](https://www.linkedin.com/posts/vikram--krishna_datawithvikram-datascience-careers-activity-6885096211460833280-bowt)

**💡 Hierarchical SoftMax**: 

- It is much more efficient efficient alternative to the normal SoftMax. In practice, hierarchical SoftMax tends to be better for infrequent words, while negative sampling works better for frequent words and lower dimensional vectors. Hierarchical SoftMax uses a binary tree to represent all words in the vocabulary. Each leaf of the tree is a word, and there is a unique path from root to leaf. 
- The main advantage is that instead of evaluating W output nodes in the neural network to obtain the probability distribution, it is needed to evaluate only about log2 (W) nodes.In this model, there is no output representation for words. Instead, each node of the graph (except the root and the leaves) is associated to a vector that the model is going to learn.
- In this model, the probability of a word w given a vector w(i) ,    P(w|wi), is equal to the probability of a random walk starting in the root and ending in the leaf node corresponding to w.The main advantage in computing the probability this way is that the cost is only O(log(|V|)), corresponding to the length of the path. 
- To train the model, our goal is still to minimize the negative log likelihood [− log P(w|wi)]. But instead of updating output vectors per word, we update the vectors of the nodes in the binary tree that are in the path from root to leaf node.
- The structure of the tree used by the hierarchical SoftMax has a considerable effect on the performance. After many experiments they suggested to use a binary Huffman tree, as it assigns short codes to the frequent words which results in fast training. It has been observed before that grouping words together by their frequency works well as a very simple speedup technique for the neural network based language models
 
- Reference:
  - [Negative sampling paper](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

