import pandas as pd

def preprocess_data(df, xColumnName, yColumnName, removableWords, label2id):
  df = df[[xColumnName,yColumnName]]
  df.loc[:,yColumnName] = [(float(label2id[label])) for label in df.loc[:,yColumnName]]
  remove_words = lambda x: ' '.join([word.lower() for word in x.split() if word not in removableWords])
  df.loc[:,xColumnName] = df[xColumnName].apply(remove_words)
  x, y = df[xColumnName].values.tolist(), df[yColumnName].values.tolist()
  return (x,y)