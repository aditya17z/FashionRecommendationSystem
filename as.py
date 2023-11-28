import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from numpy.linalg import norm
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2

# Load precomputed feature embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Create a new model with GlobalMaxPooling2D layer
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Load the Streamlit page
def main():
    st.title("Fashion Recommendation App")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Process the uploaded image
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)

        # Fit a k-NN model on the feature embeddings
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)

        # Find nearest neighbors for the uploaded image
        distances, indices = neighbors.kneighbors([normalized_result])

        st.subheader("Top 5 Recommendations:")

        # Display the top 5 recommended images
        for file_idx in indices[0][1:6]:
            recommended_img = cv2.imread(filenames[file_idx])
            st.image(recommended_img, caption=f"Recommendation {file_idx + 1}", use_column_width=True)

if __name__ == '__main__':
    main()
