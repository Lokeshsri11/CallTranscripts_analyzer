import spacy
from textblob import TextBlob
import pandas as pd

# Sample unstructured feedback
feedback_texts = [
    "hey i bought a loreal shampoo that was awesome for weak hair",
    "hey i bought a redmi note 9 pro phone they have a battery issue I am not satisfied",
    "hi i buy a voltas air conditioner of 1.5 ton but that was a cooling issue",
    "hola i bought a samsung TV which have a screen flickering problem",
    "hey i bought nike court vision shoes that was so much comfortable"
]

# Initialize spaCy for Named Entity Recognition (NER) 
nlp = spacy.load("en_core_web_sm")

# Predefined categories (assumes a simpler approach, advanced version would require a model to classify texts into these categories)
category_map = {
    "shampoo": "Shampoo",
    "phone": "Mobile Phones",
    "air conditioner": "AC Units",
    "TV": "TV",
    "shoes": "Shoes"
}

# Function to infer category from a given text
def infer_category(text):
    for key in category_map.keys():
        if key in text:
            return category_map[key]
    return "Unknown"

# Prepare structured data lists
categories, brands, transcripts, pros, cons = [], [], [], [], []

for text in feedback_texts:
    # NER to find potential brand names
    doc = nlp(text)
    brand = next((ent.text for ent in doc.ents if ent.label_ in ['ORG', 'PRODUCT']), "Unknown")
    
    # Sentiment analysis for pros and cons
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        pros.append(text)
        cons.append("N/A")
    else:
        pros.append("N/A")
        cons.append(text)
    
    # Infer category based on keywords
    category = infer_category(text.lower())
    
    categories.append(category)
    brands.append(brand)
    transcripts.append(text)

# Create and display DataFrame
df = pd.DataFrame({
    'Category': categories,
    'Brand': brands,
    'Transcript': transcripts,
    'Pros': pros,
    'Cons': cons
})

print(df)