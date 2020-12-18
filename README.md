# NY-Times-articles-Topic-modelling-NLP

Topic modelling using python
(A natural language based approach)



Can you explain briefly about your project?
 
Basically, I am going to consider a dataset with text data, and I am going to create topics out of the data and predict topic for each sentence. For instance, if there is a dataset which explains the sale of amazon products with their details which include product name, price, MRP, color etc., and here from the above data I am going to divide the data into topics such as brands.

What is topic modelling?

It is basically a statistical way to discover the abstract topics from a corpus of data, for analyzing vast quantities of textual data and to understand the hidden semantic structures behind that.
We have huge amount of text data in the form of corpus (collections of documents), each document consists of several set of continuous set of words and we are interested in finding how often these words comes up in the document and this helps to get topics from the corpus. This easily helps to analyze the information.

Explain the process involved in this project?

1. I need to consider a dataset and from the reference of Kaggle (open source for datasets) I am considering “The New York times Dataset”. 
2. Loading the dataset into particular dataset that can be induced in python using pandas.
3. Pre-processing the data which includes tokenizing, lemmatization, n-grams, removing stop words etc. This is one of the most important step that we need to consider in this project, I am working on it and I will finalize ones I have clear understanding of the dataset.
4. Extracting features from the corpus, here we are going to use eighter “TF-IDF” or “Count-Vectorization” for converting the text document tokens into numerical format (document matrix with numerical value for each word).
 5.  Now our data is ready for modelling, so we are going to categorize the data into models using “Latent Dirichlet Allocation” also called as LDA. There are two LDA’s in ML one is used for text data and other is used for numerical data for reducing dimensions of the data, which is known as linear discriminant analysis. Also, we can use NMF which is also used for topic modelling for text data.
6. Visualizing the topics that are generated using few python libraries.


Are you going to consider any other data for this project?

Yes, actually I am working right now on the dataset that was described above, but I might add few other data to test my model performance. Because, there are lot of datasets available similar to the one that I have chosen.

Can you explain about the dataset and why did you consider this dataset and what is your motivation?

My main motivation behind this project is to categorize the topics from New York times dataset, here in this dataset each article represents a single document, and there are vast number of articles present with different themes and topics. By using topic modelling technique, I am going to predict the topics and categorize each article into different topics and present them in different data frames. The main reason behind choosing this dataset is because of its diversity in topics. 
Finally, I am going to visualize the articles with respective theme/topic.
 
What modelling technique are you going to implement and why?

In today’s world there are four different ways to perform topic modelling: LSA, LDA, NMF, and recently developed deep learning techniques.
So, for this dataset I have considered to use LDA (latent Dirichlet Allocation), particularly it uses Dirichlet priors for document topic and word-topic distribution, leading itself to a better generalization than other technique. Thus, in cases where the size of dataset is large, and the topic probability will not be remained fixed for document LDA is most helpful. Furthermore, we can control sparsity by using hyperparameter tuning in LDA.
As our dataset is large and there might be many topics that are interlinked and interrelated, I am thinking to implement LDA method for a better modelling.

What are the pre-processing techniques that you are going to do with your data?

I am not sure about that because, I am trying to analyze the dataset now and within few days I will figure out my preprocessing techniques, as of now I can guarantee that I am going to use few basic preprocessing such as removing stop words, lemmatization, tokenization, lowercasing, normalization, and noise removal.
