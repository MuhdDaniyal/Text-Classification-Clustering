import json
import glob
import re
import pandas as pd
import numpy as np
from tkinter import Tk, Label, Button, filedialog, Text, END, Toplevel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score
import time

# Load stopwords
with open("Stopword-List.txt", 'r') as f:
    stopwords = set(word_tokenize(f.read().lower()))

# Initialize stemmer
stemmer = PorterStemmer()

def custom_tokenizer(text):
    text = text.lower()
    text = re.sub('[^0-9a-z\s]', ' ', text)  #Removing Punctuation and Special Characters
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stopwords]
    return tokens

# Load class labels from CSV
class_df = pd.read_csv("document_classes.csv")
class_map = dict(zip(class_df['document_id'], class_df['class']))

# Use TfidfVectorizer with the custom tokenizer
vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, stop_words=None, lowercase=False, max_df=0.95, min_df=0.05, ngram_range=(1,2))

# Load documents and match them to their classes
documents = []
doc_labels = []
for file_path in glob.glob("ResearchPapers/*.txt"):
    doc_id = re.sub('[^0-9]', '', file_path.split('/')[-1])
    with open(file_path, 'r') as file:
        documents.append(file.read())
        doc_labels.append(class_map.get(int(doc_id), "Unknown"))  # Default to 'Unknown' if not found

# Create the TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)
df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Split data for training and evaluation
X_train, X_test, y_train, y_test = train_test_split(df, doc_labels, test_size=0.3, random_state=42)

# Train KNN Model
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Calculate metrics
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_train)
y_kmeans = kmeans.predict(X_test)
rand_index = adjusted_rand_score(y_test, y_kmeans)

# Initialize the GUI application
class KNNClassifierApp:
    def __init__(self, master):
        self.master = master
        master.title("Document Classifier")

        self.label = Label(master, text="Choose a document to classify", font=("Helvetica", 12))
        self.label.pack(pady = 10)

        self.classify_button = Button(master, text="Browse Document", command=self.classify_document, font=("Helvetica", 10), height=2, width=20)
        self.classify_button.pack(pady = 20)

        self.result_label = Label(master, text="", fg="green", font=("Times New Roman", 14))
        self.result_label.pack(pady = 30)

        self.metrics_text = Text(master, height=20, width=155)
        self.metrics_text.pack()

        self.classification_time_label = Label(master, text="")
        self.classification_time_label.pack()

        self.kmeans_time_label = Label(master, text="")

        self.show_metrics()
        self.show_kmeans_results()

    def classify_document(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            start_time = time.time()  # Start timing
            with open(file_path, 'r') as file:
                document = file.read()
            # Process the document and classify
            features = vectorizer.transform([document])
            prediction = knn.predict(features)
            self.result_label.config(text=f"Predicted Class: {prediction[0]}")
            end_time = time.time()  # End timing
            duration = end_time - start_time
            self.classification_time_label.config(text=f"KNN Classification Time: {duration:.4f} seconds", font=("Times New Roman", 12))

    def show_metrics(self):
        self.metrics_text.insert(END, f"Confusion Matrix:\n{conf_matrix}\n\n")
        self.metrics_text.insert(END, f"Classification Report:\n{report}\n")
        self.metrics_text.config(state='disabled')  # Disable editing of text widget

    def show_kmeans_results(self):
        # Display KMeans results in a new window
        start_time = time.time()  # Start timing
        top = Toplevel(self.master)
        top.title("KMeans Clustering Results")

        # Adjusted Rand Index
        label = Label(top, text=f"Adjusted Rand Index: {rand_index:.4f}", font=("Times New Roman", 12))
        label.pack(pady = 5)

        # Silhouette Score
        silhouette = silhouette_score(X_train, kmeans.labels_)
        label2 = Label(top, text=f"Silhouette Score: {silhouette:.4f}", font=("Times New Roman", 12))
        label2.pack(pady = 5)

        # Purity
        purity = homogeneity_score(y_test, y_kmeans)
        label3 = Label(top, text=f"Purity: {purity:.4f}", font=("Times New Roman", 12))
        label3.pack(pady = 5)

        # Plotting
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        scatter = ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=kmeans.labels_, cmap='viridis')
        legends = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legends)
        ax.set_title('Cluster Assignments')
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack()

        end_time = time.time()  # End timing
        duration = end_time - start_time
        self.kmeans_time_label.config(text=f"KMeans Processing Time: {duration:.4f} seconds", font=("Times New Roman", 12))
        self.kmeans_time_label.pack()

def main():
    root = Tk()
    app = KNNClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
