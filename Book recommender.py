#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


books = pd.read_csv('books.csv') 
users = pd.read_csv('users.csv')
ratings = pd.read_csv('ratings.csv')


# In[3]:


books.head()


# In[4]:


users.head()


# In[5]:


ratings.head()


# # Popularity Based RS

# In[6]:


ratings_with_name = ratings.merge(books,on='ISBN')


# In[7]:


ratings_with_name


# In[8]:


num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':'num_ratings'},inplace=True)
num_rating_df


# In[9]:


avg_rating_df = ratings_with_name.groupby('Book-Title').mean()['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating':'avg_rating'},inplace=True)
avg_rating_df


# In[10]:


popular_df = num_rating_df.merge(avg_rating_df,on='Book-Title')
popular_df


# In[11]:


popular_df = popular_df[popular_df['num_ratings']>=250].sort_values('avg_rating',ascending=False).head(50)
popular_df


# In[12]:


popular_df = popular_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_rating']]


# In[13]:


popular_df


# In[14]:


import pickle
pickle.dump(popular_df,open('popular.pkl','wb'))


# # Collaborative filtering RS

# In[15]:


ratings_with_name


# In[16]:


x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
selected_users = x[x].index


# In[17]:


filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(selected_users)]
filtered_rating


# In[18]:


y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
y


# In[19]:


famous_books = y[y].index


# In[20]:


final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
final_ratings


# In[21]:


pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
pt


# In[22]:


pt.fillna(0,inplace=True)


# In[23]:


pt


# In[24]:


from sklearn.metrics.pairwise import cosine_similarity


# In[25]:


ss = cosine_similarity(pt)
ss


# In[26]:


ss.shape


# In[27]:


def recommend(book_name):
    # index fetch
    index = np.where(pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(ss[index])),key=lambda x:x[1],reverse=True)[1:5]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(item)
    
    return data


# In[28]:


recommend('1984')


# In[30]:


pickle.dump(pt,open('pt.pkl','wb'))


# In[31]:


pickle.dump(books,open('books.pkl','wb'))


# In[32]:


pickle.dump(ss,open('similarity_scores.pkl','wb'))


# In[ ]:




