from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# 1. 准备语料库
corpus = ["This is an example sentence.", "Another example sentence."]

# 2. 文本预处理
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# 3. 构建训练数据
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, sg=1, min_count=1)

# 4. 训练模型
model.train(tokenized_corpus, total_examples=len(tokenized_corpus), epochs=10)

# 5. 保存模型
model.save("word2vec_model")