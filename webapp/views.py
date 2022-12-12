from django.shortcuts import render

# Create your views here.
def index(request):
    context = {}
    if request.method == 'POST':
        mdata = []
        for doc in request.FILES.getlist('docs'):
            mdata.append(((doc.read()).decode('utf-8')).replace('\n', ' '))

        context['results'], context['coherence'], context['imgs'], context['sal_img'] = lda_tagme(mdata, int(request.POST['topics-nb']), float(request.POST['tagme-th']))

    return render(request, 'index.html', context)



def text_preprocess(texts):
    from nltk.corpus import stopwords
    import spacy
    import gensim
    from gensim.utils import simple_preprocess

    import nltk
    nltk.download('stopwords')
    
    # NLTK Stop words
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
    data_words = list(sent_to_words(texts))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]
    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]
    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)
    # Initialize spacy 'en_core_web_sm' model, keeping only tagger component (for efficiency)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    new_data = data_lemmatized
    return new_data



def lda_tagme(mdata, lda_nb_topics, tagme_threshold):
    import math
    # TagMe
    import tagme
    tagme.GCUBE_TOKEN = "3f5df154-2197-4930-ac6b-d9f5f3992adf-843339462"
    # Gensim
    import gensim
    import gensim.corpora as corpora
    from gensim.models import CoherenceModel
    import warnings
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    # Plotting tools
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import io
    import urllib, base64
    from scipy.special import softmax


    # Data Preprocessing
    data = text_preprocess(mdata)
    data_attached = []
    for d in data:
        data_attached.append(' '.join(d))

    
    # Create Dictionary
    id2word = corpora.Dictionary(data)
    # Create Corpus
    texts = data
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=lda_nb_topics,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )


    # Recuperer la liste des topics
    topics = lda_model.show_topics(num_topics=0, num_words=len(id2word.items()), formatted=False)
    # Recuperer chaque mot et frequence de chaque topic
    topic_words = []
    topic_freqs = []
    for topic in topics:
        topic_word = []
        topic_freq = []
        for word in topic[1]:
            topic_word.append(word[0])
            topic_freq.append(word[1])
        topic_words.append(topic_word)
        topic_freqs.append(topic_freq)

    # Recuperer la liste des mots
    words = []
    for word in id2word.items():
        words.append(word[1])

    # Calculer la frequence de chaque mot
    words_freq = [0] * len(words)
    for topic in topics:
        for word in topic[1]:
            words_freq[words.index(word[0])] += word[1]
    # Calculer la probabilité qu'un topic genere un mot
    topic_proba = 1 / len(id2word.items())
    # Calculer le terme saliency de chaque mot
    words_saliency = []
    for i in range(len(words)):
        sum_t = 0
        for ti in range(len(topic_words)):
            word_freq = topic_freqs[ti][topic_words[ti].index(words[i])]
            sum_t += word_freq * math.log10(word_freq / topic_proba)
        words_saliency.append(words_freq[i] * sum_t)

    # Trier les mots par "term saliency"
    salient_words = []
    words_copy = words.copy()
    words_saliency_copy = words_saliency.copy()
    for i in range(len(words)):
        index = words_saliency_copy.index(max(words_saliency_copy))
        salient_words.append((words_copy[index], words_saliency_copy[index]))
        del words_copy[index]
        del words_saliency_copy[index]


    salient_words_list = []
    saliency_list = []
    for t in salient_words:
        salient_words_list.append(t[0])
        saliency_list.append(t[1])

    # Word Cloud : most salient words
    word_score_dict = dict(zip(salient_words_list[:30], softmax(saliency_list[:30])))
    plt.figure(figsize=(16, 4), dpi=200)
    plt.axis("off")
    plt.imshow(
        WordCloud(width=1600,
                height=400,
                background_color='white'
        ).generate_from_frequencies(word_score_dict)
    )
    fig = plt.gcf()
    imgdata = io.BytesIO()
    fig.savefig(imgdata, format='png')
    imgdata.seek(0)
    string = base64.b64encode(imgdata.read())
    html_sal_img = 'data:image/png;base64,' + urllib.parse.quote(string)

    # Word Cloud : topic words
    img_count = 0
    html_imgs = []
    for i in range(len(topic_words)):
        img_count += 1
        word_score_dict = dict(zip(topic_words[i][:30], softmax(topic_freqs[i][:30])))
        plt.figure(figsize=(16, 4), dpi=200)
        plt.axis("off")
        plt.imshow(
            WordCloud(width=1600,
                    height=400,
                    background_color='white'
            ).generate_from_frequencies(word_score_dict)
        )
        fig = plt.gcf()
        imgdata = io.BytesIO()
        fig.savefig(imgdata, format='png')
        imgdata.seek(0)
        string = base64.b64encode(imgdata.read())
        html_imgs.append(('data:image/png;base64,' + urllib.parse.quote(string), img_count))
    

    # Generation des annotations TagMe
    tagme_mentions = []
    tagme_entities = []
    tagme_scores   = []
    for datum in data_attached:
        tagme_mentions_datum = []
        tagme_entities_datum = []
        tagme_scores_datum   = []
        lunch_annotations = tagme.annotate(datum)
        for ann in lunch_annotations.get_annotations(tagme_threshold):
            tagme_mentions_datum.append(ann.mention)
            tagme_entities_datum.append(ann.entity_title)
            tagme_scores_datum.append(ann.score)
        tagme_mentions.append(tagme_mentions_datum)
        tagme_entities.append(tagme_entities_datum)
        tagme_scores.append(tagme_scores_datum)


    # Calculer l'intersection entre les resultats du LDA et de TagMe
    used_entities = []
    used_saliency = []
    for i in range(len(tagme_mentions)):
        for j in range(len(tagme_mentions[i])):
            tagme_mention = tagme_mentions[i][j]
            tagme_entity  = tagme_entities[i][j]
            tagme_score   = tagme_scores[i][j]
            
            if tagme_mention in words:
                if tagme_entity not in used_entities:
                    used_entities.append(tagme_entity)
                    used_saliency.append(words_saliency[words.index(tagme_mention)] * tagme_score)
                else:
                    used_saliency[used_entities.index(tagme_entity)] += words_saliency[words.index(tagme_mention)] * tagme_score


    # Trier et Afficher les Resultats
    results = []
    used_entities_copy = used_entities.copy()
    used_saliency_copy = used_saliency.copy()
    max_proba = max(used_saliency)

    this_rank = 0
    for i in range(len(used_entities)):
        index = used_saliency_copy.index(max(used_saliency_copy))

        this_rank       +=  1
        this_entity     =   used_entities_copy[index]
        this_count      =   0
        this_doc_count  =   0
        this_proba      =   used_saliency_copy[index] / max_proba
        
        if this_proba < 0.01:
            break

        for tagme_entities_datum in tagme_entities:
            if this_entity in tagme_entities_datum:
                this_doc_count += 1
            this_count += tagme_entities_datum.count(this_entity)
        
        results.append((this_rank, this_entity, this_count, this_doc_count, int(this_proba * 100)))
        del used_entities_copy[index]
        del used_saliency_copy[index]

    coherence_model_lda = CoherenceModel(model=lda_model, texts=data, dictionary=id2word, coherence='c_v')
    coherence_lda = float(int(coherence_model_lda.get_coherence() * 10000)) / 10000

    return results, coherence_lda, html_imgs, html_sal_img
    