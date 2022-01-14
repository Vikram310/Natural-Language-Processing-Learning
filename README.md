# Natural-Language-Processing-Learning

[**Day 1**](https://www.linkedin.com/posts/vikram--krishna_datawithvikram-datascience-careers-activity-6884374937344335874-bziD)

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

[**Day 2**](https://www.linkedin.com/posts/vikram--krishna_datawithvikram-datascience-careers-activity-6884737641330413568-QZNn)

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

[**Day 3**](https://www.linkedin.com/posts/vikram--krishna_datawithvikram-datascience-careers-activity-6885096211460833280-bowt)

**💡 Hierarchical SoftMax**: 

- It is much more efficient efficient alternative to the normal SoftMax. In practice, hierarchical SoftMax tends to be better for infrequent words, while negative sampling works better for frequent words and lower dimensional vectors. Hierarchical SoftMax uses a binary tree to represent all words in the vocabulary. Each leaf of the tree is a word, and there is a unique path from root to leaf. 
- The main advantage is that instead of evaluating W output nodes in the neural network to obtain the probability distribution, it is needed to evaluate only about log2 (W) nodes.In this model, there is no output representation for words. Instead, each node of the graph (except the root and the leaves) is associated to a vector that the model is going to learn.
- In this model, the probability of a word w given a vector w(i) ,    P(w|wi), is equal to the probability of a random walk starting in the root and ending in the leaf node corresponding to w.The main advantage in computing the probability this way is that the cost is only O(log(|V|)), corresponding to the length of the path. 
- To train the model, our goal is still to minimize the negative log likelihood [− log P(w|wi)]. But instead of updating output vectors per word, we update the vectors of the nodes in the binary tree that are in the path from root to leaf node.
- The structure of the tree used by the hierarchical SoftMax has a considerable effect on the performance. After many experiments they suggested to use a binary Huffman tree, as it assigns short codes to the frequent words which results in fast training. It has been observed before that grouping words together by their frequency works well as a very simple speedup technique for the neural network based language models
 
- Reference:
  - [Negative sampling paper](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

[**Day 4**](https://www.linkedin.com/posts/vikram--krishna_datawithvikram-datascience-careers-activity-6886183414148538368-JZfB)

**💡 Word Embeddings**: 

- Word Embeddings are the texts converted into numbers. There may be different numerical representations of the same text.

**💡 Word Embeddings**: 

- Similar words tend to occur together and will have a similar context. For a given corpus, the co-occurrence of a pair of words say w1 and w2 is the number of times they have appeared together in a Context Window. How to form Co-occurrence matrix:

      1. The matrix A stores co-occurrences of words.
      2. In this method, we count the number of times each word appears inside a window of a particular size around the word of interest.
      3. Calculate this count for all the words in the corpus.
- For a huge corpus, this co-occurrence matrix could become really complex (high-dimension). To avoid this problem, we use Singular value decomposition(SVD) and principal component analysis(PCA) are two eigenvalue methods used to reduce a high-dimensional dataset into fewer dimensions while retaining important information.
- Both of these methods give us word vectors that are more than sufficient to encode semantic and syntactic (part of speech) information but are associated with many other   problem:
  
      1. The dimensions of the matrix change very often (new words are added very frequently and corpus changes in size).
      2. The matrix is extremely sparse since most words do not cooccur.
      3. The matrix is very high dimensional in general ( 106 *106 )
      4. Quadratic cost to train (i.e. to perform SVD)
      5. Requires the incorporation of some hacks on X to account for the drastic imbalance in word frequency

[**Day 5**](https://www.linkedin.com/posts/vikram--krishna_datawithvikram-datascience-careers-activity-6886547204912230400-r5n-)

**💡 Glove for word representations**: 

- Semantic vector space models of language represent each word with a real-valued vector. These vectors can be used as features in a variety of applications, such as information retrieval, document classification, question answering, named entity recognition, and parsing. There were two popular methods for word embeddings, which are Matrix Factorization methods and Shallow window based methods. 
- Matrix Factorization methods (such as LSA and HAL) are count-based, effectively leveraging global statistical information. They are primarily used to capture word similarities and do poorly on tasks such as word analogy, indicating a sub-optimal vector space structure. 
- Shallow-window based methods (such as skip-gram and CBOW). learn word embeddings by making predictions in local context windows. These models demonstrate the capacity to capture complex linguistic patterns beyond word similarity, but fail to make use of the global co-occurrence statistics.
- Glove was introduced to overcome the problems in previous models. GloVe, is a new global log-bilinear regression model for the unsupervised learning of word representations that outperforms other models on word analogy, word similarity, and named entity recognition tasks. 
- GloVe (Global Vectors) consists of a weighted least squares model that trains on global word-word co-occurrence counts and thus makes efficient use of statistics. The model produces a word vector space with meaningful sub-structure. It shows state-of-the-art performance on the word analogy task, and outperforms other current methods on several word similarity tasks.

Reference:
- [Paper on Glove](http://nlp.stanford.edu/pubs/glove.pdf)

[**Day 6**](https://www.linkedin.com/posts/vikram--krishna_datawithvikram-datascience-careers-activity-6887277701821030401-bokF)

**💡 Intrinsic evaluation of word vectors**: 

- Intrinsic evaluation of word vectors is the evaluation of a set of word vectors generated by an embedding technique (such as Word2Vec or Glove) on specific intermediate subtasks (such as analogy completion). These subtasks are typically simple and fast to compute and thereby allow us to help understand the system used to generate the word vectors.  An intrinsic evaluation should typically return to us a number that indicates the performance of those word vectors on the evaluation subtask.
- Fast to compute performance and helps understand sub-system. The only problem is it needs positive correlation with real task to determine usefulness. Correlation evaluation and Word vector analogies are few examples of Intrinsic evaluation. Word vector analogies evaluate word vectors by how well their cosine distance after addition captures intuitive semantic and syntactic analogy questions

**💡 Extrinsic evaluation of word vectors**: 

- Extrinsic evaluation of word vectors is the evaluation of a set of word vectors generated by an embedding technique on the real task at hand. These tasks are typically elaborate and slow to compute. Typically, optimizing over an underperforming extrinsic evaluation system does not allow us to determine which specific subsystem is at fault and this motivates the need for intrinsic evaluation. It is slow to compute performance and unclear if subsystem is the problem, other subsystems, or internal interactions. 

Reference:
- [Paper on evaluation methods](http://www.aclweb.org/anthology/D15-1036)

[**Day 7**](https://www.linkedin.com/posts/vikram--krishna_datawithvikram-datascience-careers-activity-6887638675119841280-HCfS)

**💡 Tokenization**: 

- Tokenization is a way of separating a piece of text into smaller units called tokens. Here, tokens can be either word, characters, or subwords. Hence, tokenization can be broadly classified into 3 types – word, character, and subword (n-gram characters) tokenization. As tokens are the building blocks of Natural Language, the most common way of processing the raw text happens at the token level. As tokens are the building blocks of Natural Language, the most common way of processing the raw text happens at the token level.

**Word Tokenization**: 

- Word Tokenization is the most commonly used tokenization algorithm. It splits a piece of text into individual words based on a certain delimiter. Depending upon delimiters, different word-level tokens are formed. Pretrained Word Embeddings such as Word2Vec and GloVe come under word tokenization. One of the major issues with word tokens is dealing with Out Of Vocabulary (OOV) words. OOV words refer to the new words which are encountered at testing. These new words do not exist in the vocabulary. Hence, these methods fail in handling OOV words.

**Character Tokenization**:

- Character Tokenization splits apiece of text into a set of characters. It overcomes the drawbacks we saw above about Word Tokenization. It handles OOV words coherently by preserving the information of the word. It breaks down the OOV word into characters and represents the word in terms of these characters. Character tokens solve the OOV problem but the length of the input and output sentences increases rapidly as we are representing a sentence as a sequence of characters. As a result, it becomes challenging to learn the relationship between the characters to form meaningful words.

**Sub-word Tokenization**:

- Subword Tokenization splits the piece of text into subwords (or n-gram characters). For example, words like lower can be segmented as low-er, smartest as smart-est, and so on. Transformer-based models – the SOTA in NLP relies on Subword Tokenization algorithms for preparing vocabulary. Byte-pair encoding (BPE) is a simple form of data compression algorithm in which the most common pair of consecutive bytes of data is replaced with a byte that does not occur in that data.

[**Implementation**](https://github.com/Vikram310/Natural-Language-Processing-Learning/blob/main/Code%20files/Tokenization.ipynb) 

Reference:
- Google
