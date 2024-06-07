import pandas as pd
import time
from datetime import datetime, timedelta
import numpy as np
import csv
from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
from torch.utils.data import DataLoader, TensorDataset
import torch

# Load pre-trained model tokenizer (vocabulary)


date_start = '2019-01-01'
date_end = '2021-02-15'
reddit_file_path = 'r_wallstreetbets_posts.csv'
processed_fp = 'processed_reddits.csv'


def create_processed_reddit(date_start, date_end, search_strings):
    df = pd.read_csv(reddit_file_path)
    columns_to_keep = ['title', 'created_utc'] 
    df = df[columns_to_keep]
    df['created_utc'] = pd.to_datetime(df['created_utc'],unit='s')
    df['date'] = df['created_utc'].dt.date
    df['time'] = df['created_utc'].dt.time
    df = df.drop(columns=['created_utc'])
    mask = df['title'].str.contains('|'.join(search_strings), case=False, na=False)
    tesla_df = df[mask]
    start_date = pd.to_datetime(date_start)
    end_date = pd.to_datetime(date_end)
    tesla_df = tesla_df[(tesla_df['date'] >= start_date) & (tesla_df['date'] <= end_date)]
    tesla_df = text_preprocessing(tesla_df,'title')
    tesla_df = tesla_df.drop(columns='title')
    output_file_path = processed_fp
    tesla_df.to_csv(output_file_path, index=False)
    print(f"DataFrame has been saved to {output_file_path}")

def text_preprocessing(df,col_name):
    #remove URL
    df['processed'] = df[col_name].str.replace(r'http(\S)+', r'')
    df['processed'] = df['processed'].str.replace(r'http ...', r'')
    df['processed'] = df['processed'].str.replace(r'http', r'')
    df[df['processed'].str.contains(r'http')]
   # remove RT, @
    df['processed'] = df['processed'].str.replace(r'(RT|rt)[ ]*@[ ]*[\S]+',r'')
    df[df['processed'].str.contains(r'RT[ ]?@')]
    df['processed'] = df['processed'].str.replace(r'@[\S]+',r'')
    #remove non-ascii words and characters
    df['processed'] = [''.join([i if ord(i) < 128 else '' for i in text]) for text in df['processed']]
    df['processed'] = df['processed'].str.replace(r'_[\S]?',r'')
    #remove &, < and >
    df['processed'] = df['processed'].str.replace(r'&amp;?',r'and')
    df['processed'] = df['processed'].str.replace(r'&lt;',r'<')
    df['processed'] = df['processed'].str.replace(r'&gt;',r'>')
    # remove extra space
    df['processed'] = df['processed'].str.replace(r'[ ]{2, }',r' ')
    # insert space between punctuation marks
    df['processed'] = df['processed'].str.replace(r'([\w\d]+)([^\w\d ]+)', r'\1 \2')
    df['processed'] = df['processed'].str.replace(r'([^\w\d ]+)([\w\d]+)', r'\1 \2')
    # lower case and strip white spaces at both ends
    df['processed'] = df['processed'].str.lower()
    df['processed'] = df['processed'].str.strip()

    df['word_count'] = [len(text.split(' ')) for text in df['processed']]
    df['word_count'].value_counts()
    df = df[df['word_count']>3]
    df = df.drop_duplicates(subset=['processed'])
    return df

embeddings_list = [] 
def get_embeddings(dataloader, model, device):
    model.eval()  # Set model to evaluation mode
    all_embeddings = []
    with torch.no_grad():  # Disable gradient calculation
        for batch in dataloader:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state

            # Optionally, take the [CLS] token embedding
            cls_embeddings = last_hidden_states[:, 0, :].cpu().numpy()
            all_embeddings.extend(cls_embeddings)

    return all_embeddings
    

def create_bert_embeddings():
    file_path = processed_fp
    df = pd.read_csv(file_path)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    encoded_inputs = tokenizer(df['processed'].tolist(), padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=16)
    embeddings = get_embeddings(dataloader, model, device)
    df['embeddings'] = embeddings
    start_date = df['date'].min()
    end_date = df['date'].max()
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    delta = timedelta(days=1)
    embedding_dict = dict()
    current_date = start_date
    df['date'] = pd.to_datetime(df['date'])
    while current_date <= end_date:
        current_df = df[df['date'] == current_date]
        embedding_dict[current_date.strftime('%Y-%m-%d')] = np.mean(current_df['embeddings'], axis=0)
        current_date += delta
    return embedding_dict

def process_stock(stockname):
    filepath = 'input/Data/Stocks/' + stockname + '.csv'
    data = pd.read_csv(filepath)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    data['increase'] = (data['Open'].shift(-1) < data['Open'].shift(-2)).astype(int)
    data = data.drop(columns=['High','Low','Close','Adj Close','Volume'])
    start_date = pd.to_datetime(date_start)
    end_date = pd.to_datetime(date_end)
    df = data.loc[start_date:end_date]
    delta = timedelta(days=1)
    embedding_dict = create_bert_embeddings()
    current_date = start_date
    embeddings = []
    increases = []
    while current_date <= end_date:
        if(current_date.strftime('%Y-%m-%d') in embedding_dict and current_date in df.index):
            row = df.loc[current_date]
            increases.append(row['increase'])
            embeddings.append(embedding_dict[current_date.strftime('%Y-%m-%d')])
        current_date += delta
        full_df = pd.DataFrame()
    full_df['embedding'] = embeddings
    full_df['increase'] = increases
    full_df = full_df.dropna()
    return full_df
    

def main():
    stock_strings = [['tesla', 'TSLA', 'Tesla'],['microsoft','Microsoft','MSFT'],['apple','Apple','AAPL'],['NVIDIA','nvdia','NVDA', 'Nvdia'],['amd','AMD','Amd'],['META','meta','facebook','Meta'],['intel','Intel','INTC']]
    stocks = ['TSLA','MSFT','AAPL','NVDA','AMD','META','INTC']
    i = 0
    df = pd.DataFrame(columns=['embedding', 'increase'])
    for search in stock_strings:
        stock = stocks[i]
        create_processed_reddit(date_start, date_end, search)
        df = pd.concat([df, process_stock(stock)], ignore_index=True)
        i+=1
    df.to_pickle('stocks_dataframe.pkl')

if __name__ == "__main__":
    main()