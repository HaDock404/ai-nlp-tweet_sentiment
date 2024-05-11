import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import ast
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

data = pd.read_csv('./data/tweets.csv', encoding='latin-1', header=None)
data = data.rename(columns={data.columns[0]: 'target'})
data = data.rename(columns={data.columns[1]: 'id'})
data = data.rename(columns={data.columns[2]: 'date'})
data = data.rename(columns={data.columns[3]: 'flag'})
data = data.rename(columns={data.columns[4]: 'user'})
data = data.rename(columns={data.columns[5]: 'text'})
data['text'] = data['text'].apply(lambda x: re.sub(r'\S*@\S*\s?', '', x))
data['text'] = data['text'].apply(lambda x: re.sub(r'http\S+', '', x))
data['text'] = data['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
data['text'] = data['text'].apply(lambda x: re.sub(r'\b\w\b', '', x))
data['text'] = data['text'].apply(lambda x: re.sub(r'\d', '', x))
data['text'] = data['text'].apply(lambda x: re.sub(r'\s+', ' ', x))
data['text'] = data['text'].apply(lambda x: x.lower())

data['text_token'] = 0

for i in range(len(data['text'])):
    data['text_token'][i] = nltk.word_tokenize(data['text'][i])


def remove_stopwords(word_list):
    stop_words = set(stopwords.words('english'))
    filtered_words = []
    for word in word_list:
        if word not in stop_words:
            filtered_words.append(word)
    return filtered_words


for i in range(len(data['text_token'])):
    data['text_token'][i] = remove_stopwords(data['text_token'][i])


def lemmatize_words(word_list):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word, get_pos(word))
                        for word in word_list]
    return lemmatized_words


def get_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV,
                "J": wordnet.ADJ}
    return tag_dict.get(tag, wordnet.NOUN)


for i in range(len(data['text_token'])):
    data['text_token'][i] = lemmatize_words(data['text_token'][i])

word_dataset = './data/normalized_dataset.csv'
data.to_csv(word_dataset, index=False)

df = pd.read_csv("./data/normalized_dataset.csv")

array_list = df['text_token'].values
data_list = []
for item in array_list:
    data_list.append(ast.literal_eval(item))

df_list = pd.DataFrame({'text_token': data_list})
df = df.drop(columns=['text_token'])
df['text_token'] = df_list['text_token']
df['words'] = df['text_token'].apply(lambda x: ' '.join(x))
df = df.drop(columns=['text_token'])

word_dataset = './data/cleaned_dataset.csv'
df.to_csv(word_dataset, index=False)

df_neg = df[df['target'] == 0].sample(500000)
df_pos = df[df['target'] == 4].sample(500000)
df_pos['target'] = 1
liste_concat = [df_neg, df_pos]
df_sample = pd.concat([df_neg, df_pos], ignore_index=True)
df_sample = df_sample.sample(frac=1).reset_index(drop=True)

sample_df = './data/sample_dataset.csv'
df_sample.to_csv(sample_df, index=False)

df = pd.read_csv("./data/sample_dataset.csv")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

saving_path = './data/train_df.csv'
train_df.to_csv(saving_path, index=False)

saving_path = './data/test_df.csv'
test_df.to_csv(saving_path, index=False)
