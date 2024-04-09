import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np

# Use the provided data dictionary to create a DataFrame
# Original dataset with pros and cons for illustration
data = {
    'Category': [
        'Soap', 'Soap', 'Soap',
        'Shampoo', 'Shampoo',
        'Mobile Phones', 'Mobile Phones', 'Mobile Phones',
        'AC Units', 'AC Units',
        'Bed Sheets', 'Bed Sheets'
    ],
    'Brand': [
        'Dove', 'Tresemme', 'Loreal',
        'Tresemme', 'Loreal',
        'Samsung', 'Apple', 'OnePlus',
        'Whirlpool', 'LG',
        'Brooklinen', 'Parachute'
    ],
    'Transcript': [
        "Dove leaves the skin soft but melts quickly.",
        "Tresemme has a great scent, lasts long.",
        "Loreal feels harsh, not good for sensitive skin.",
        "Tresemme shampoo gives volume but is harsh on the scalp.",
        "Loreal shampoo is great for sensitive scalp but expensive.",
        "Samsung's battery lasts all day but the screen cracks easily.",
        "Apple has an intuitive design but is too expensive.",
        "OnePlus offers great performance but has poor customer service.",
        "Whirlpool's AC cooling is powerful but it's noisy.",
        "LG's AC is energy efficient but has a complicated remote.",
        "Brooklinen sheets are soft and durable but pricey.",
        "Parachute offers great comfort but colors fade after washing."
    ],
    'Pros': [
        'Skin soft', 'Great scent, lasts long', 'N/A',
        'Gives volume', 'Great for sensitive scalp',
        'Battery lasts all day', 'Intuitive design', 'Great performance',
        'Powerful cooling', 'Energy efficient',
        'Soft and durable', 'Great comfort'
    ],
    'Cons': [
        'Melts quickly', 'N/A', 'Harsh for sensitive skin',
        'Harsh on the scalp', 'Expensive',
        'Screen cracks easily', 'Too expensive', 'Poor customer service',
        'Noisy', 'Complicated remote',
        'Pricey', 'Colors fade after washing'
    ]
}
df = pd.DataFrame(data)
df['Sentiment'] = df['Transcript'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Ensure a large figure to accommodate all subplots
plt.figure(figsize=(12, 18))

unique_categories = df['Category'].unique()
total_plots = len(unique_categories) + 1

for i, category in enumerate(unique_categories, 1):
    plt.subplot(total_plots, 1, i)
    category_df = df[df['Category'] == category]
    
    avg_sentiment = category_df.groupby('Brand')['Sentiment'].mean()
    bars = avg_sentiment.plot(kind='bar', color='skyblue', title=f"Average Sentiment for {category} Brands")
    plt.ylabel("Average Sentiment Score")
    
    # Annotate each bar with Pros and Cons
    for idx, brand in enumerate(avg_sentiment.index):
        pros = category_df[category_df['Brand'] == brand]['Pros'].values[0]
        cons = category_df[category_df['Brand'] == brand]['Cons'].values[0]
        plt.text(idx, avg_sentiment[brand], f"Pros: {pros}\nCons: {cons}", ha='center', rotation=45, va='bottom', fontsize=8)

# Adding the overall product graph
plt.subplot(total_plots, 1, total_plots)
overall_avg_sentiment = df.groupby(['Category', 'Brand'])['Sentiment'].mean().unstack().mean(axis=1)
overall_avg_sentiment.plot(kind='bar', color='steelblue', title="Overall Product Sentiment by Category")
plt.ylabel("Average Sentiment Score")
plt.xlabel("Category")

plt.tight_layout()
plt.savefig('overall_and_categories_sentiment_comparison.png')
plt.show()