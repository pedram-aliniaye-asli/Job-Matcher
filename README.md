**Job Description and Resume Similarity Calculator
**
Hey there!

This project is my initial foray into building a tool that compares job descriptions (JDs) and resumes to assess how well a candidate's qualifications match a specific job opening. It's a work in progress, and I'm excited to explore different approaches and see how far I can take it!

In this first version, I'm using Word2Vec to create word embeddings, which are numerical representations that capture semantic relationships between words. This helps compare keywords extracted from both the JD and resume. But that's just the tip of the iceberg! I plan to experiment with other word embedding libraries like GloVe to see if they yield even better results.

Furthermore, this code compares keywords, words, and sentences for similarity. However, I envision expanding the scope to include named entity recognition (NER). This would allow the program to identify specific skills and qualifications mentioned in the JD and pinpoint their presence in the resume.

Overall, this is just the beginning. I'm eager to delve deeper into different NLP techniques and models to refine this project and make it a more robust JD-resume similarity assessment tool.


A more detailed of what is happening in the code:
   ** Text Preprocessing:**
        Removes HTML tags, URLs, and punctuation from both the JD and resume.
        Tokenizes the text into words.
        Converts all words to lowercase.
        Removes stop words (common words like "the", "a", and "an").
        Applies stemming/lemmatization to reduce words to their base form (e.g., "running" becomes "run").
        Removes words consisting only of numbers.
        Eliminates extra whitespaces.

   ** Keyword Extraction (TF-IDF):**
        Identifies the most relevant keywords in each paragraph (section) of the JD using TF-IDF (Term Frequency-Inverse Document Frequency). TF-IDF considers both the frequency of a word within a document and its rarity across documents in the corpus.
        Creates a shortlist by selecting the top 35% of words ranked by TF-IDF score.
        Generates word embeddings for these keywords using Word2Vec, a popular technique for representing words as numerical vectors that capture semantic relationships between words.

    **Sentence Embedding (Transformer Model):**
        Splits the JD and resume text into sentences.
        Employs a pre-trained sentence transformer model (all-MiniLM-L6-v2) to generate dense vector representations for each sentence. Sentence transformers learn to encode sentences that convey similar meanings into similar vector spaces.

   ** Similarity Calculation:**
        Calculates the cosine similarity between the sentence and word embeddings of the JD and resume to quantify how similar they are. Cosine similarity is a metric used to measure the similarity between two vectors by finding the cosine of the angle between them.

  **  Overall Similarity Score:**
        Averages the sentence and word embedding similarity scores to get a final similarity score between the JD and resume. This score indicates how well a candidate's skills and experience, as described in their resume, align with the requirements mentioned in the job description.
