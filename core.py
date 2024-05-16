import re
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import download as nltkdown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltkdown('stopwords')
nltkdown('wordnet')

def prepare_text(in_text,text_type):
    #text_type --> jd: Job Descritiopn, rc: Resume/CV
    text = {'type':'','content':''}
    if text_type == 'jd':
        pars = in_text.split('\n\n')
        for i, par in enumerate(pars):
            pars[i] = preprocess_text(par)
        text['content'] = pars
        text['type'] = 'jd'
    elif text_type == 'rc':
        text['content'] = [preprocess_text(in_text)]
        text['type'] = 'rc'
    return text

def preprocess_text(text):
    # Removing HTML tags, URLs, punctuation 
    text = re.sub(r'<[^>]+>', '', text)   
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenization
    words = word_tokenize(text)
    # Lowercasing
    words = [word.lower() for word in words]
    # Removing Stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Removing items with all numbers
    words = [word for word in words if not word.isdigit()]
    # Removing Extra Whitespaces
    text = ' '.join(words)
    
    return text

def get_keywords(in_text,text_type):
    pars = {'type':'','content':{},'keywords_list':{}}
    if text_type == 'jd':
        for i, par in enumerate(in_text):
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(sent_tokenize(par))
            # Get keywords 
            keywords = vectorizer.get_feature_names_out()
            # Get TF-IDF scores
            tfidf_scores = tfidf_matrix.toarray()[0]
            # Create a dictionary of words and their TF-IDF scores
            term_tfidf_scores = {term: score for term, score in zip(keywords, tfidf_scores)}
            sorted_terms = sorted(term_tfidf_scores.items(), key=lambda x: x[1], reverse=True)
            #getting top 35% as the keywords
            top_n = int(len(sorted_terms) * 0.35)
            keywords = [term for term, _ in sorted_terms[:top_n]]

            #keywords embeddings
            sentences = sent_tokenize(par)
            data = []
            for sentence in sentences:
                temp = []
                for word in word_tokenize(sentence):
                    temp.append(word)
                data.append(temp)
            words_model = Word2Vec(data, vector_size=100, window=5, min_count=1, sg=0)
            words_embeddings = words_model.wv
            keywords_embeddings = [words_embeddings[keyword] for keyword in keywords if keyword in words_embeddings]
            pars['type'] = text_type
            pars['content']['par'+str(i)] = [keywords_embeddings]
            pars['keywords_list']['par'+str(i)] = keywords
    elif text_type == 'rc':
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sent_tokenize(in_text[0]))
        # Get keywords 
        keywords = vectorizer.get_feature_names_out()
        # Get TF-IDF scores
        tfidf_scores = tfidf_matrix.toarray()[0]
        # Create a dictionary of words and their TF-IDF scores
        term_tfidf_scores = {term: score for term, score in zip(keywords, tfidf_scores)}
        sorted_terms = sorted(term_tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        #getting top 35% as the keywords
        top_n = int(len(sorted_terms) * 0.35)
        keywords = [term for term, _ in sorted_terms[:top_n]]

        #keywords embeddings
        sentences = sent_tokenize(in_text[0])
        data = []
        for sentence in sentences:
            temp = []
            for word in word_tokenize(sentence):
                temp.append(word)
            data.append(temp)
        words_model = Word2Vec(data, vector_size=100, window=5, min_count=1, sg=0)
        words_embeddings = words_model.wv
        keywords_embeddings = [words_embeddings[keyword] for keyword in keywords if keyword in words_embeddings]
        pars['type'] = text_type
        pars['content']['only_par'] = [keywords_embeddings]
        pars['keywords_list']['only_par'] = keywords

    return pars




def embed_text(in_text,text_type):
    pars = {'type':'','content':{}}
    if text_type == 'jd':
        for i, par in enumerate(in_text):
            # Sentence Tokenization
            sentences = sent_tokenize(par)
            # Transformer-based Sentence Embedding
            sent_model = SentenceTransformer("all-MiniLM-L6-v2")
            sentence_embeddings = sent_model.encode(sentences)
            data = []
            # Word2Vec Word Embedding
            for sentence in sentences:
                temp = []
                for word in word_tokenize(sentence):
                    temp.append(word)
                data.append(temp)
            word_model = Word2Vec(data, vector_size=100, window=5, min_count=1, sg=0)
            word_embeddings = word_model.wv
            word_embeddings = [word_embeddings[word] for tokens in data for word in tokens if word in word_embeddings]
            pars['type'] = text_type
            pars['content']['par'+str(i)]= [sentence_embeddings, word_embeddings]
    elif text_type == 'rc':
        # Sentence Tokenization
        sentences = sent_tokenize(in_text[0])
        # Transformer-based Sentence Embedding
        sent_model = SentenceTransformer("all-MiniLM-L6-v2")
        sentence_embeddings = sent_model.encode(sentences)
        data = []
        # Word2Vec Word Embedding
        for sentence in sentences:
            temp = []
            for word in word_tokenize(sentence):
                temp.append(word)
            data.append(temp)
        word_model = Word2Vec(data, vector_size=100, window=5, min_count=1, sg=0)
        word_embeddings = word_model.wv
        word_embeddings = [word_embeddings[word] for tokens in data for word in tokens if word in word_embeddings]
        pars['type'] = text_type
        pars['content']['only_par'] = [sentence_embeddings, word_embeddings]

    return pars


def cal_similarity(array1,array2):
    #reshaping to 2D array
    array1 = np.array(array1).reshape(1,-1)
    array2 = np.array(array2).reshape(1,-1)
    # Ensure both arrays have the same number of features (columns)
    max_cols = max(array1.shape[1], array2.shape[1])
    array1 = np.pad(array1, ((0, 0), (0, max_cols - array1.shape[1])), mode='constant')
    array2 = np.pad(array2, ((0, 0), (0, max_cols - array2.shape[1])), mode='constant')
    similarity = cosine_similarity(array1, array2)
    #return similarity[0][0]
    return np.mean(similarity[0][0])

def start(job_txt,resume_txt):
    #prepare the texts for analysis
    prepared_jd = prepare_text(job_txt,'jd')
    prepared_resume = prepare_text(resume_txt,'rc')

    #get word and sentence embeddings
    jd_embeddings = embed_text(prepared_jd['content'],prepared_jd['type'])
    rc_embeddings = embed_text(prepared_resume['content'],prepared_resume['type'])

    #get keyword embeddings
    jd_keywords = get_keywords(prepared_jd['content'],prepared_jd['type'])
    rc_keywords = get_keywords(prepared_resume['content'],prepared_resume['type'])

    word_embed = []
    sent_embed = []
    for i, par in enumerate(jd_embeddings['content']):
        word_embed.append(cal_similarity(jd_embeddings['content'][f'par{i}'][1],rc_embeddings['content']['only_par'][1]))
        sent_embed.append(cal_similarity(jd_embeddings['content'][f'par{i}'][0],rc_embeddings['content']['only_par'][0]))
    keyword_embed = []
    for i, par in enumerate(jd_keywords['content']):
        keyword_embed.append(cal_similarity(jd_keywords['content'][f'par{i}'][0],rc_keywords['content']['only_par'][0]))
    print(np.mean(word_embed))
    print(word_embed)

    print(jd_keywords['keywords_list'])
    print(rc_keywords['keywords_list'])
           
    print(np.mean(sent_embed))
    print(np.mean(keyword_embed))
    print(np.mean(np.array([np.mean(word_embed),np.mean(sent_embed),np.mean(keyword_embed)])))
def test(a,b):
    a = input('Enter the Job Description: ')
    b = input('Enter the Resume/CV: ')
    start(a,b)
