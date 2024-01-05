import streamlit as st
import pickle
import numpy as np

popular_df = pickle.load(open('popular.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))

def recommend_books(user_input, top_n=5):
    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:top_n+1]

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)

    return data

def main():
    st.title('Book Recommendation App')

    # User input for recommendation
    user_input = st.text_input('Enter a book title for recommendations:')
    if user_input:
        st.subheader('Recommended Books')
        data = recommend_books(user_input, top_n=5)
        for item in data:
            st.write(f"**Title:** {item[0]}, **Author:** {item[1]}")
            try:
                st.image(item[2], caption=f"Title: {item[0]}", width=96, channels="RGB")
            except Exception as e:
                st.warning(f"Failed to load image from URL: {item[2]}. Error: {e}")

    # Display popular books
    st.subheader('Popular Books')
    for index, row in popular_df.head(5).iterrows():
        st.write(f"**Title:** {row['Book-Title']}, **Author:** {row['Book-Author']}")
        try:
            st.image(row['Image-URL-M'], caption=f"Title: {row['Book-Title']}", width=96, channels="RGB")
        except Exception as e:
            st.warning(f"Failed to load image from URL: {row['Image-URL-M']}. Error: {e}")

if __name__ == '__main__':
    main()

