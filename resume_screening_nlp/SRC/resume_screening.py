import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

# -----------------------------
# TEXT CLEANING FUNCTION
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# -----------------------------
# SKILL LIST (KNOWLEDGE BASE)
# -----------------------------
skill_list = [
    "python","java","c++","machine learning","deep learning",
    "nlp","sql","mysql","postgresql","aws","docker","kubernetes",
    "pandas","numpy","tensorflow","pytorch","scikit learn",
    "data visualization","power bi","tableau","linux",
    "git","react","node js","html","css"
]

# -----------------------------
# SKILL EXTRACTION FUNCTION
# -----------------------------
def extract_skills(text):

    text = text.lower()
    found_skills = []

    for skill in skill_list:
        if skill in text:
            found_skills.append(skill)

    return found_skills
# -----------------------------
# SKILL MATCHING FUNCTIONS
# -----------------------------
def matched_skills(resume, job):
    return list(set(resume).intersection(set(job)))
def missing_skills(resume, job):
    return list(set(job) - set(resume))

# -----------------------------
# LOAD DATASET
# -----------------------------
data = pd.read_csv('/Users/devanshbansal/Downloads/resume_screening_dataset.csv')
data.drop(columns=["source"], errors="ignore", inplace=True)
print(data.head())
print(data.shape)
print(data.columns)
print(data.isnull().sum())
# -----------------------------
# TEXT PREPROCESSING
# -----------------------------
data['resume_text'] = data['resume_text'].apply(clean_text)
data['job_description'] = data['job_description'].apply(clean_text)

# -----------------------------
# SKILL EXTRACTION
# -----------------------------
data['resume_skills'] = data['resume_text'].apply(extract_skills)
data['job_skills'] = data['job_description'].apply(extract_skills)


# -----------------------------
# SKILL MATCHING
# -----------------------------
data['matched_skills'] = data.apply(
    lambda row: matched_skills(row['resume_skills'], row['job_skills']),
    axis=1
)

data['missing_skills'] = data.apply(
    lambda row: missing_skills(row['resume_skills'], row['job_skills']),
    axis=1
)


# -----------------------------
# LOAD TRANSFORMER MODEL
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')#loading the pre-trained sentence transformer 
#model to #convert the cleaned text data into numerical vectors (embeddings) that can be used for 
#further analysis # or machine learning tasks.


# -----------------------------
# GENERATE EMBEDDINGS
# -----------------------------
resume_embeddings = model.encode(data['resume_text'].tolist())#ok so that those two lines means that 
#first select the column that form the list then list of string then convert them to the numbers through 
# the sentence library
job_embeddings = model.encode(data['job_description'].tolist())


# -----------------------------
# COMPUTE COSINE SIMILARITY
# -----------------------------

#first form an empty list named similarities to compare the similarities of what is needed #and what 
# is there? through cosine similarity
similarities = []

for i in range(len(resume_embeddings)):

    sim = cosine_similarity(
        [resume_embeddings[i]],
        [job_embeddings[i]]
    )[0][0]

    similarities.append(sim)

data['similarity'] = similarities#adding the similarity scores to the original dataset as a new column 
#named 'similarity'


# -----------------------------
# RANK CANDIDATES
# -----------------------------
data_sorted = data.sort_values(
    by='similarity',
    ascending=False
)


# -----------------------------
# DISPLAY RESULTS
# -----------------------------
print(data['similarity'].describe())

print(
    data_sorted[
        ['similarity','resume_skills','job_skills','matched_skills','missing_skills']
    ].head(10)
)


# -----------------------------
# PREDICTION USING THRESHOLD
# -----------------------------
data['prediction'] = data['similarity'].apply( # converts the values to the score of 1 or 0 if the 
                                              #similarity is greater than 0.7 than converts to 1 else 
                                              # converts it to the 0
    lambda x: 1 if x > 0.7 else 0
)


# -----------------------------
# MODEL EVALUATION
# -----------------------------
accuracy = accuracy_score(data['match_label'], data['prediction'])

print("Model Accuracy:", accuracy)
