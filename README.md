# NW453-ZeroWaste-WebCrawl
Assignment for Northwestern University MSDS Course 453

Abstract

With climate change and sustainability top of mind these days, understanding what we can do as individual citizens of this planet to reduce our personal impact on the environment has become a hot topic. The goal of my first assignment was to develop a corpus comprised of blog posts and articles about the zero waste movement. Zero waste attempts to have "no trash to be sent to landfills, incinerators, or the ocean" (Zero waste 2020). My second assignment took the next step in the natural language processing lifecycle and applied vectorization techniques to the collected documents. The third assignment uncovered more about the collected corpus through cluster analysis and multidimensional scaling. This assignment is aimed at leveraging neural networks to generate content related to the zero waste movement through the use of long short term memory models.

Introduction

Throughout our assignments my goal has been to better understand the leading factors in the zero waste movement, as well as how different influencers speak about the movement. If you frequent social media sites you may come across influencers discussing their favorite products or encouraging the average person to make more sustainable choices. These influencers create content that range from video, images, and text. These social media platforms are outreach tools and typically look to attract followers to their website. A website has more in-depth content in the form of blog posts. This web of networks has one thing in common – content. And in a world that doesn’t sleep, an influencer may be expected to generate new content daily.

My goal for this assignment was to build a neural network to learn from the corpus collected in previous assignments and fit a model that can generate new content related to the zero waste movement. The sections of this paper are: Literature Review where similar work done by others will be reviewed, Methods discusses the various neural network modeling techniques used, Results lays out the best model and related hyperparameters, and Conclusion summarizes the findings and discusses next steps.

Literature Review

Text generation has been done before and most applications were built using the long short-term memory (LSTM) method. Prior to the LSTM method was introduced, recurrent neural networks (RNN) were the primary neural network used. RNNs introduced loops, unlike convolutional neural networks (CNN), which allowed them to “remember” previous events or examples (Olah 2015). LSTMs built upon this capability by “remembering” information farther back in history. Memory cells open and close to allow access to each memory cell’s constant error carrousel (Hochreiter and Schmidhuber 1997).

Many fellow data scientists have built text generation tools using a combination of TensorFlow, Keras, and Python. Much of the code necessary for text generation applications prepares the data, such as creating rolling windows of words and the corresponding next word, which is then used to train a neural network (Campion 2018). Creating the sequences of words from existing sentences in the corpus, then shuffling and splitting the data, is also paramount to avoid overfitting (A. 2018). Finally, when generating text, the seed text needs to be tokenized before being used to predict the following word (Bansal 2018).
    
Methods

As with past assignments, a web crawler scrapes the text of a specific set of webpages, then follows all links on those webpages and scrapes the text from the second layer of webpages too. I manually cleaned the text using the NLTK package in Python (Brownlee 2019) and calculated word vectors for each document (Lane 2019). In a less manual approach, I leveraged the CountVectorizer and TfidfVectorizer from scikit-learn as well as the Tokenizer from Keras (Brownlee 2019). Finally, pre-trained document embeddings from Doc2Vec were used to take advantage of neural networks built on millions of documents (Lane 2019). Some of this preparation is not used in the fitting of neural networks but is an important exploratory data analysis step.

After exploring the collected data, it is time to further prepare the data for use in a neural network. This means converting the data from webpage documents into sequences of words, and then converting those sequences into numbers. This last step was more difficult than expected as neural networks require the data in specific formats. Nevertheless, the first step was to take the corpus and break each document into sentences. Then each sentence was split into words by whitespace. Punctuation was left in so that the neural network would not produce run-on sentences. Once these sentence lists were prepared, they were vectorized and broken into smaller sequences of words. These word sequences were semi-redundant in that the first and second sequence would share some of the same words. The next word in the sequence was also stored in a separate list as the dependent variable.

Finally, the data was shuffled and split into a training and test set. As previously stated, the data format was difficult to understand. I tried different types of arrays and reshaping, running out of memory many times. In addition to data format struggles, it seemed the dense layer in my neural networks were expecting a different sized array than what was provided. But these dense layers were not the first layer in my network so I would have expected them to conform to the size of the output of the previous layer.

After all the data gathering, exploration, and preparation, we are brought to the moment of truth. The random seed was set to allow reproducible results, the batch size was set to 32, and the dropout was set to 0.2. An initial Binary LSTM neural network was compiled and fit on a training data set of 331,851 sentences. I would think that the final dense layer would output predictions for all the words in the corpus and the word with the largest value would be the word predicted by the model. Unfortunately, I was not able to get a categorical neural network to work with this data despite my many attempts.

In additional to the Binary LSTM model, a Binary Bidirectional LSTM model was fitted. A bidirectional LSTM model looks at the data both forwards and backwards, which may sound like cheating if you were working with time series data, as there is no way you can see into the future, but for text generation this is not an issue.

Results

In the results I was able to achieve, accuracy is poor and my loss values are negative, which I do not understand. Accuracy improves in the first two epochs but then remains steady. Perhaps the network gets caught in a local minima so making changes to the learning rate would be helpful.

Other potential improvements to the networks would be to add additional layers, try different activation methods, or run for longer epochs. On the data preparation aspect, the current setup create words sequences of 5 words, but I wonder what kind of impact it would have on the model if this were increased.

Conclusions

In this project as a whole I learned a great deal. The scrapy package seems powerful, but difficult to master. I most likely used it when a most simple approach would have sufficed for my needs. Data normalization and vectorization was fun and open to interpretation. Minor changes made at this level affect everything else, so the choices being made must be thoughtful. Clustering, too, is subjective and is only as good as the data collected. In this last assignment, the power of neural networks was promising, though getting the inputs correctly formatted continued to be an obstacle I could not overcome. Given the complexity and power of LSTM networks, they take a good deal of time to fit. This left little time to tweak the hyperparameters. Overall, this assignment got me thinking out of the box of what I could accomplish simply from text I scraped off the internet.

References

A., E. (June 4, 2018). Word-level lstm text generator: Creating automatic song lyrics with neural networks. Medium. https://medium.com/coinmonks/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb

Campion, D. (January 11, 2018). Text generation using bidirectional lstm and doc2vec models 1/3. Medium. https://medium.com/@david.campion/text-generation-using-bidirectional-lstm-and-doc2vec-models-1-3-8979eb65cb3a

Bansal, S. (March 26, 2018). Language modelling and text generation using lstms — deep learning for nlp. Medium. https://medium.com/@shivambansal36/language-modelling-text-generation-using-lstms-deep-learning-for-nlp-ed36b224b275

Brownlee, J. (2019). Deep learning for natural language processing: Develop deep learning models for natural language in python. Machine Learning Mastery.

Hochreiter, S. and Schmidhuber, J. (1997). Long short-term memory. Neural Computation 9(8): 1735-1780.

Lane, H., Howard, C., and Hapke, H. M. (2019). Natural language processing in action: Understanding, analyzing, and generating text with python. Manning.

Olah, C. (August 27, 2015). Understanding lstm networks. http://colah.github.io/posts/2015-08-Understanding-LSTMs/

Zero waste. (April 23, 2020). In Wikipedia. https://en.wikipedia.org/w/index.php?title=Zero_waste&oldid=952695900
