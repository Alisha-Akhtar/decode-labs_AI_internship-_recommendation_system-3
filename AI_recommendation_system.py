import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Ingestion: Define the job roles and their associated skills
data = {
    'job_role': ['DevOps Engineer', 'Data Scientist', 'Backend Developer', 'Cloud Architect'],
    'skills': ['AWS Docker Kubernetes CI/CD', 'Python SQL Machine Learning', 'Java Python SQL API', 'AWS Cloud Automation']
}
df = pd.DataFrame(data)

# 2. Vector Mapping: Use TF-IDF to transform skills into numerical vectors
# TF-IDF ensures unique skills have more impact than generic terms
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['skills'])

# 3. Scoring: Calculate Cosine Similarity between user profile and job roles
# This measures the mathematical angle between vectors, ignoring magnitude
print("Enter your top 3 skills separated by spaces:")
user_input = input("Skills: ") 
user_vector = tfidf.transform([user_input])

similarity_scores = cosine_similarity(user_vector, tfidf_matrix)
df['score'] = similarity_scores[0]

# 4. Sorting and Filtering: Display the Top-N results
# Sorting in descending order and truncating to prevent choice overload
results = df.sort_values(by='score', ascending=False)
print("\nTop Recommended Roles:")
print(results[['job_role', 'score']].head(3))