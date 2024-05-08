from flask import Flask, request, render_template
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
import isodate
import requests
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import pickle
import gensim
import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator

from io import BytesIO

w2v_model = gensim.models.Word2Vec.load("word2vec_embed.model")


app = Flask(__name__)

api_key = "AIzaSyDRG4JAdV_Z0OLMbUVwhPAq89HW_uqwvG0"
youtube = build('youtube', 'v3', developerKey=api_key)
adaboost = pickle.load(open('trained_model_ada.pkl', 'rb'))
# cnn = tf.keras.sav.load_model('best_model_cnn.h5', custom_objects={'KerasLayer': hub.KerasLayer})

cnn = keras.models.load_model("best_model_mv2.keras")
image_processor = ImageDataGenerator(rescale=1./255)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_title(text):
    # Tokenization
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens


def fetch_thumbnail(thumbnail_url):
    response = requests.get(thumbnail_url)
    response.raise_for_status()

    img_stream = BytesIO(response.content)
    img = load_img(img_stream, target_size=(512, 512))

    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)

    augmented_img = image_processor.flow(img_arr, batch_size=1)

    return next(augmented_img)


def title_to_vector(title):
    vectors = [w2v_model.wv[word] for word in title if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)


def fetch_video_info(yt_obj, vid_id):
    
    video_info_req = yt_obj.videos().list(
        part='snippet,contentDetails,statistics',
        id=vid_id
    )

    video_info_res = video_info_req.execute()
    return 

def weighted_avg(pred_cnn, pred_ada):
    adaboost_clickbait_prob = np.max(pred_ada, axis=1)
    cnn_clickbait_prob = pred_cnn
    
    # Confidence measures based on distance from the decision boundary (0.5)
    confidence_cnn = abs(cnn_clickbait_prob - 0.5) * 2
    confidence_adaboost = abs(adaboost_clickbait_prob - 0.5) * 2

    # Calculate dynamic weights
    total_confidence = confidence_cnn + confidence_adaboost
    weight_cnn = confidence_cnn / total_confidence
    weight_adaboost = confidence_adaboost / total_confidence
    
    return weight_cnn, weight_adaboost

def custom_weights(cnn_prob, threshold=0.95, base_weight_adaboost=0.95, base_weight_cnn=0.05):
    # Apply dynamic adjustment based on CNN confidence
    if cnn_prob > threshold:
        # Increase CNN weight when super confident
        weight_cnn = 0.7  # More weight to CNN when it's highly confident
        weight_adaboost = 0.3
    else:
        # Default weights
        weight_cnn = base_weight_cnn
        weight_adaboost = base_weight_adaboost
    return weight_cnn, weight_adaboost

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/detection', methods=['POST'])
def detection():
    url = request.form['youtube-url']
    parsed_url = urlparse(url)
    query_vars = parse_qs(parsed_url.query)

    video_id = query_vars["v"][0]

    video_info_req = youtube.videos().list(
        part='snippet,contentDetails,statistics',
        id=video_id
    )

    print("Provided Video ID:", video_id)

    video_info_res = video_info_req.execute()

    video_data_dict = {}

    for video in video_info_res.get('items', []):
        video_data_dict = {
                'videoID': video['id'],
                'url': url,
                'video_title': video['snippet']['title'],
                'thumbnail_link': video['snippet']['thumbnails']['high'],
                'duration': isodate.parse_duration(video['contentDetails']['duration']).seconds,
                'like_count': video['statistics']['likeCount'] if 'likeCount' in video['statistics'] else "",
                'commment_count': video['statistics']['commentCount'] if 'commentCount' in video['statistics'] else "",
                'view_count': video['statistics']['viewCount'] if 'viewCount' in video['statistics'] else ""
            }

    title_tokens = preprocess_title(video_data_dict['video_title'])
    title_vector = np.array(title_to_vector(title_tokens)) # should result in (100,) size
    title_vector = np.hstack((title_vector, video_data_dict['view_count'], video_data_dict['like_count'])).reshape(1, -1) # size = (102,)

    ada_pred = adaboost.predict(title_vector)
    ada_votes = adaboost.predict_proba(title_vector)
    print("Prediction from adaboost:", ada_pred)
    print("Votes:", ada_votes)

    prepared_thumbnail = fetch_thumbnail(video_data_dict['thumbnail_link']['url'])
    cnn_pred = cnn.predict(prepared_thumbnail)

    print("Thumbnail prediction from CNN:", cnn_pred)

    weight_cnn, weight_adaboost = custom_weights(cnn_pred[0])
    ada_prob = ada_votes[0, 1]
    weighted_prediction = (weight_adaboost * ada_prob + weight_cnn * cnn_pred) / (weight_adaboost + weight_cnn)

    print("WEIGHTED PREDICTION:", weighted_prediction)
    

    if weighted_prediction < 0.5:
        classification = False
    else:
        classification = True

    return render_template("classification.html", video_data=video_data_dict, classification=classification)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)