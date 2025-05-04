#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WhatsApp Sentiment and Intent Analysis Script
This script analyzes the sentiment and intent of WhatsApp messages,
correlating them with resolution status and other metrics.
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Create directories for charts
os.makedirs('charts/Sentiment_Intent_Analysis', exist_ok=True)

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Define intent categories and their keywords
INTENT_CATEGORIES = {
    'Question': ['?', 'who', 'what', 'when', 'where', 'why', 'how', 'is there', 'are there', 'can i', 'could you', 'would you'],
    'Recommendation Request': ['recommend', 'suggestion', 'advice', 'tips', 'best', 'favorite', 'recommend', 'suggestions', 'advise'],
    'Information Sharing': ['fyi', 'just so you know', 'heads up', 'wanted to share', 'thought you might', 'check out', 'look at'],
    'Help Request': ['help', 'assist', 'support', 'guidance', 'struggling', 'trouble', 'problem', 'issue', 'fix', 'solve'],
    'Opinion Request': ['thoughts', 'opinion', 'think', 'feel about', 'perspective', 'view', 'feedback'],
    'Experience Sharing': ['experience', 'happened', 'went through', 'dealt with', 'encountered', 'faced', 'my story'],
    'Gratitude': ['thanks', 'thank you', 'grateful', 'appreciate', 'appreciated', 'thankful'],
    'Confirmation': ['confirm', 'verify', 'sure', 'right', 'correct', 'true', 'valid', 'accurate'],
    'Announcement': ['announcing', 'announcement', 'inform', 'notify', 'notice', 'alert', 'attention'],
    'Other': []  # Default category
}

def load_data():
    """Load the WhatsApp resolution analysis data and message data"""
    print("Loading data...")
    
    # Load resolution analysis data
    resolution_df = pd.read_csv('whatsapp_resolution_analysis.csv')
    
    # Load raw message data if available
    try:
        messages_df = pd.read_csv('whatsapp_final_processed_v4.csv')
        messages_df['datetime'] = pd.to_datetime(messages_df['datetime'])
        print(f"Loaded {len(messages_df)} messages from {len(messages_df['thread_id'].unique())} threads")
    except FileNotFoundError:
        print("Raw message data not found. Some analyses will be limited.")
        messages_df = None
    
    return resolution_df, messages_df

def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    if not isinstance(text, str) or not text.strip():
        return 0, 0  # Neutral sentiment and zero subjectivity for empty text
    
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

def classify_intent(text):
    """Classify the intent of a message based on keywords and patterns"""
    if not isinstance(text, str) or not text.strip():
        return 'Other'
    
    text = text.lower()
    
    # Check each intent category
    for intent, keywords in INTENT_CATEGORIES.items():
        if any(keyword in text for keyword in keywords):
            return intent
    
    # If no specific intent is found, check if it's a question based on punctuation
    if '?' in text:
        return 'Question'
    
    return 'Other'

def analyze_thread_sentiment_intent(thread_messages):
    """Analyze sentiment and intent for a thread"""
    if len(thread_messages) == 0:
        return {
            'initial_sentiment': 0,
            'initial_subjectivity': 0,
            'initial_intent': 'Other',
            'avg_sentiment': 0,
            'response_sentiment': 0,
            'sentiment_shift': 0,
            'dominant_intent': 'Other',
            'intent_diversity': 0
        }
    
    # Get the first message
    first_message = thread_messages.iloc[0]['message']
    initial_sentiment, initial_subjectivity = analyze_sentiment(first_message)
    initial_intent = classify_intent(first_message)
    
    # Analyze all messages
    sentiments = []
    intents = []
    
    for _, row in thread_messages.iterrows():
        sentiment, _ = analyze_sentiment(row['message'])
        intent = classify_intent(row['message'])
        
        sentiments.append(sentiment)
        intents.append(intent)
    
    # Calculate average sentiment
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    
    # Calculate response sentiment (excluding first message)
    response_sentiments = sentiments[1:] if len(sentiments) > 1 else [0]
    response_sentiment = np.mean(response_sentiments)
    
    # Calculate sentiment shift (response sentiment - initial sentiment)
    sentiment_shift = response_sentiment - initial_sentiment
    
    # Determine dominant intent
    intent_counts = Counter(intents)
    dominant_intent = intent_counts.most_common(1)[0][0] if intent_counts else 'Other'
    
    # Calculate intent diversity (number of unique intents / number of messages)
    intent_diversity = len(intent_counts) / len(thread_messages) if len(thread_messages) > 0 else 0
    
    return {
        'initial_sentiment': initial_sentiment,
        'initial_subjectivity': initial_subjectivity,
        'initial_intent': initial_intent,
        'avg_sentiment': avg_sentiment,
        'response_sentiment': response_sentiment,
        'sentiment_shift': sentiment_shift,
        'dominant_intent': dominant_intent,
        'intent_diversity': intent_diversity
    }

def process_threads(resolution_df, messages_df):
    """Process all threads to add sentiment and intent analysis"""
    print("Analyzing sentiment and intent for all threads...")
    
    if messages_df is None:
        print("Raw message data not available. Cannot perform detailed sentiment analysis.")
        return resolution_df
    
    # Create a new dataframe to store results
    sentiment_intent_df = resolution_df.copy()
    
    # Add columns for sentiment and intent metrics
    sentiment_intent_df['initial_sentiment'] = 0.0
    sentiment_intent_df['initial_subjectivity'] = 0.0
    sentiment_intent_df['initial_intent'] = 'Unknown'
    sentiment_intent_df['avg_sentiment'] = 0.0
    sentiment_intent_df['response_sentiment'] = 0.0
    sentiment_intent_df['sentiment_shift'] = 0.0
    sentiment_intent_df['dominant_intent'] = 'Unknown'
    sentiment_intent_df['intent_diversity'] = 0.0
    
    # Process each thread
    for thread_id in sentiment_intent_df['thread_id'].unique():
        # Get messages for this thread
        thread_messages = messages_df[messages_df['thread_id'] == thread_id].sort_values('datetime')
        
        if len(thread_messages) == 0:
            continue
        
        # Analyze sentiment and intent
        analysis = analyze_thread_sentiment_intent(thread_messages)
        
        # Update the dataframe
        idx = sentiment_intent_df['thread_id'] == thread_id
        for key, value in analysis.items():
            sentiment_intent_df.loc[idx, key] = value
    
    # Save the results
    sentiment_intent_df.to_csv('whatsapp_sentiment_intent_analysis.csv', index=False)
    print("Sentiment and intent analysis complete. Results saved to whatsapp_sentiment_intent_analysis.csv")
    
    return sentiment_intent_df

def generate_sentiment_charts(df):
    """Generate charts for sentiment analysis"""
    print("Generating sentiment analysis charts...")
    
    # 1. Overall Sentiment Distribution
    plt.figure(figsize=(12, 8))
    
    # Create bins for sentiment values
    bins = np.linspace(-1, 1, 21)
    labels = [f"{bins[i]:.1f} to {bins[i+1]:.1f}" for i in range(len(bins)-1)]
    
    df['sentiment_bin'] = pd.cut(df['avg_sentiment'], bins=bins, labels=labels)
    sentiment_counts = df['sentiment_bin'].value_counts().sort_index()
    
    sentiment_counts.plot(kind='bar', color=plt.cm.RdYlGn(np.linspace(0, 1, len(sentiment_counts))))
    plt.title('Distribution of Average Sentiment Across All Threads', fontsize=16)
    plt.xlabel('Sentiment Range', fontsize=14)
    plt.ylabel('Number of Threads', fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('charts/Sentiment_Intent_Analysis/overall_sentiment_distribution.png', dpi=300)
    plt.close()
    
    # 2. Sentiment by Resolution Status
    plt.figure(figsize=(14, 8))
    
    # Simplify resolution status
    df['resolution_simple'] = df['resolution_status'].replace({
        'Likely Resolved': 'Resolved',
        'Likely Unresolved': 'Unresolved'
    })
    
    # Calculate average sentiment for each resolution status
    sentiment_by_resolution = df.groupby('resolution_simple')['avg_sentiment'].mean().sort_values()
    
    # Create color map based on sentiment values
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(sentiment_by_resolution)))
    
    ax = sentiment_by_resolution.plot(kind='barh', color=colors)
    plt.title('Average Sentiment by Resolution Status', fontsize=16)
    plt.xlabel('Average Sentiment (-1 = Negative, 1 = Positive)', fontsize=14)
    plt.ylabel('Resolution Status', fontsize=14)
    
    # Add value labels
    for i, value in enumerate(sentiment_by_resolution):
        ax.text(value + 0.01, i, f"{value:.3f}", va='center', fontweight='bold')
    
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('charts/Sentiment_Intent_Analysis/sentiment_by_resolution.png', dpi=300)
    plt.close()
    
    # 3. Sentiment Shift Analysis
    plt.figure(figsize=(14, 8))
    
    # Calculate average sentiment shift for each resolution status
    shift_by_resolution = df.groupby('resolution_simple')['sentiment_shift'].mean().sort_values()
    
    # Create color map based on shift values
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(shift_by_resolution)))
    
    ax = shift_by_resolution.plot(kind='barh', color=colors)
    plt.title('Average Sentiment Shift by Resolution Status', fontsize=16)
    plt.xlabel('Average Sentiment Shift (Negative = Worsening, Positive = Improving)', fontsize=14)
    plt.ylabel('Resolution Status', fontsize=14)
    
    # Add value labels
    for i, value in enumerate(shift_by_resolution):
        ax.text(value + 0.01, i, f"{value:.3f}", va='center', fontweight='bold')
    
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('charts/Sentiment_Intent_Analysis/sentiment_shift_by_resolution.png', dpi=300)
    plt.close()
    
    # 4. Initial Sentiment vs. Resolution Rate
    plt.figure(figsize=(12, 8))
    
    # Create bins for initial sentiment
    bins = np.linspace(-1, 1, 11)
    df['initial_sentiment_bin'] = pd.cut(df['initial_sentiment'], bins=bins)
    
    # Calculate resolution rate for each sentiment bin
    resolution_rate = df[df['resolution_status'] != 'Not a Question'].groupby('initial_sentiment_bin').apply(
        lambda x: (x['resolution_status'].isin(['Resolved', 'Likely Resolved'])).mean() * 100
    ).fillna(0)
    
    resolution_rate.plot(kind='bar', color=plt.cm.Blues(np.linspace(0.3, 0.9, len(resolution_rate))))
    plt.title('Resolution Rate by Initial Message Sentiment', fontsize=16)
    plt.xlabel('Initial Sentiment', fontsize=14)
    plt.ylabel('Resolution Rate (%)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('charts/Sentiment_Intent_Analysis/resolution_rate_by_initial_sentiment.png', dpi=300)
    plt.close()
    
    # 5. Sentiment by Category
    plt.figure(figsize=(14, 10))
    
    # Calculate average sentiment for each category
    sentiment_by_category = df.groupby('category')['avg_sentiment'].mean().sort_values()
    
    # Create color map based on sentiment values
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(sentiment_by_category)))
    
    ax = sentiment_by_category.plot(kind='barh', color=colors)
    plt.title('Average Sentiment by Category', fontsize=16)
    plt.xlabel('Average Sentiment (-1 = Negative, 1 = Positive)', fontsize=14)
    plt.ylabel('Category', fontsize=14)
    
    # Add value labels
    for i, value in enumerate(sentiment_by_category):
        ax.text(value + 0.01, i, f"{value:.3f}", va='center', fontweight='bold')
    
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('charts/Sentiment_Intent_Analysis/sentiment_by_category.png', dpi=300)
    plt.close()
    
    # 6. Sentiment vs. Virality Scatter Plot
    plt.figure(figsize=(12, 8))
    
    plt.scatter(
        df['avg_sentiment'],
        df['virality_combined'],
        c=df['avg_sentiment'],
        cmap='RdYlGn',
        alpha=0.7,
        s=100
    )
    
    plt.colorbar(label='Sentiment Score')
    plt.title('Sentiment vs. Virality', fontsize=16)
    plt.xlabel('Average Sentiment', fontsize=14)
    plt.ylabel('Virality Score', fontsize=14)
    
    # Add trend line
    z = np.polyfit(df['avg_sentiment'], df['virality_combined'], 1)
    p = np.poly1d(z)
    plt.plot(
        sorted(df['avg_sentiment']),
        p(sorted(df['avg_sentiment'])),
        "r--",
        alpha=0.8
    )
    
    # Add correlation coefficient
    corr = df['avg_sentiment'].corr(df['virality_combined'])
    plt.annotate(
        f"Correlation: {corr:.3f}",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    
    plt.tight_layout()
    plt.savefig('charts/Sentiment_Intent_Analysis/sentiment_vs_virality.png', dpi=300)
    plt.close()

def generate_intent_charts(df):
    """Generate charts for intent analysis"""
    print("Generating intent analysis charts...")
    
    # 1. Intent Distribution
    plt.figure(figsize=(14, 8))
    
    intent_counts = df['initial_intent'].value_counts()
    
    # Plot bar chart
    ax = intent_counts.plot(kind='bar', color=plt.cm.tab10(np.linspace(0, 1, len(intent_counts))))
    plt.title('Distribution of Initial Message Intent', fontsize=16)
    plt.xlabel('Intent', fontsize=14)
    plt.ylabel('Number of Threads', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add count labels
    for i, count in enumerate(intent_counts):
        ax.text(i, count + 5, str(count), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/Sentiment_Intent_Analysis/intent_distribution.png', dpi=300)
    plt.close()
    
    # 2. Resolution Rate by Intent
    plt.figure(figsize=(14, 8))
    
    # Calculate resolution rate for each intent
    intent_resolution = df[df['resolution_status'] != 'Not a Question'].groupby('initial_intent').apply(
        lambda x: (x['resolution_status'].isin(['Resolved', 'Likely Resolved'])).mean() * 100
    ).sort_values()
    
    # Plot horizontal bar chart
    ax = intent_resolution.plot(kind='barh', color=plt.cm.Blues(np.linspace(0.3, 0.9, len(intent_resolution))))
    plt.title('Resolution Rate by Initial Message Intent', fontsize=16)
    plt.xlabel('Resolution Rate (%)', fontsize=14)
    plt.ylabel('Intent', fontsize=14)
    
    # Add percentage labels
    for i, value in enumerate(intent_resolution):
        ax.text(value + 1, i, f"{value:.1f}%", va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/Sentiment_Intent_Analysis/resolution_by_intent.png', dpi=300)
    plt.close()
    
    # 3. Intent by Category
    plt.figure(figsize=(16, 10))
    
    # Create a crosstab of category vs intent
    intent_category = pd.crosstab(
        df['category'], 
        df['initial_intent'],
        normalize='index'
    ) * 100
    
    # Plot stacked bar chart
    intent_category.plot(kind='barh', stacked=True, cmap='tab20')
    plt.title('Intent Distribution by Category', fontsize=16)
    plt.xlabel('Percentage', fontsize=14)
    plt.ylabel('Category', fontsize=14)
    plt.legend(title='Intent', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('charts/Sentiment_Intent_Analysis/intent_by_category.png', dpi=300)
    plt.close()
    
    # 4. Intent Diversity vs. Resolution Status
    plt.figure(figsize=(14, 8))
    
    # Calculate average intent diversity for each resolution status
    diversity_by_resolution = df.groupby('resolution_simple')['intent_diversity'].mean().sort_values()
    
    # Plot horizontal bar chart
    ax = diversity_by_resolution.plot(kind='barh', color=plt.cm.viridis(np.linspace(0, 1, len(diversity_by_resolution))))
    plt.title('Average Intent Diversity by Resolution Status', fontsize=16)
    plt.xlabel('Intent Diversity (Higher = More Varied Intents)', fontsize=14)
    plt.ylabel('Resolution Status', fontsize=14)
    
    # Add value labels
    for i, value in enumerate(diversity_by_resolution):
        ax.text(value + 0.01, i, f"{value:.3f}", va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/Sentiment_Intent_Analysis/intent_diversity_by_resolution.png', dpi=300)
    plt.close()
    
    # 5. Intent and Sentiment Combined Analysis
    plt.figure(figsize=(14, 10))
    
    # Calculate average sentiment for each intent
    sentiment_by_intent = df.groupby('initial_intent')['avg_sentiment'].mean().sort_values()
    
    # Create color map based on sentiment values
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(sentiment_by_intent)))
    
    ax = sentiment_by_intent.plot(kind='barh', color=colors)
    plt.title('Average Sentiment by Intent', fontsize=16)
    plt.xlabel('Average Sentiment (-1 = Negative, 1 = Positive)', fontsize=14)
    plt.ylabel('Intent', fontsize=14)
    
    # Add value labels
    for i, value in enumerate(sentiment_by_intent):
        ax.text(value + 0.01, i, f"{value:.3f}", va='center', fontweight='bold')
    
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('charts/Sentiment_Intent_Analysis/sentiment_by_intent.png', dpi=300)
    plt.close()
    
    # 6. Intent vs. Virality
    plt.figure(figsize=(14, 8))
    
    # Calculate average virality for each intent
    virality_by_intent = df.groupby('initial_intent')['virality_combined'].mean().sort_values()
    
    # Plot horizontal bar chart
    ax = virality_by_intent.plot(kind='barh', color=plt.cm.plasma(np.linspace(0, 1, len(virality_by_intent))))
    plt.title('Average Virality by Intent', fontsize=16)
    plt.xlabel('Average Virality Score', fontsize=14)
    plt.ylabel('Intent', fontsize=14)
    
    # Add value labels
    for i, value in enumerate(virality_by_intent):
        ax.text(value + 1, i, f"{value:.2f}", va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/Sentiment_Intent_Analysis/virality_by_intent.png', dpi=300)
    plt.close()

def analyze_high_virality_sentiment(df):
    """Analyze sentiment patterns in high-virality threads"""
    print("Analyzing sentiment patterns in high-virality threads...")
    
    # Create virality ranks if they don't exist
    if 'virality_rank' not in df.columns:
        df['virality_rank'] = df['virality_combined'].rank(method='first')
        total_threads = len(df)
        group_size = total_threads // 4
        
        # Create virality levels based on ranks
        def get_virality_level(rank):
            if rank <= group_size:
                return 'Very Low (Q1)'
            elif rank <= 2 * group_size:
                return 'Low (Q2)'
            elif rank <= 3 * group_size:
                return 'Medium (Q3)'
            else:
                return 'High (Q4)'
        
        df['virality_level'] = df['virality_rank'].apply(get_virality_level)
    
    # Filter for high virality threads
    high_viral = df[df['virality_level'] == 'High (Q4)']
    
    # 1. Sentiment Distribution in High-Virality Threads
    plt.figure(figsize=(14, 8))
    
    # Create bins for sentiment values
    bins = np.linspace(-1, 1, 21)
    high_viral['sentiment_bin'] = pd.cut(high_viral['avg_sentiment'], bins=bins)
    
    sentiment_counts = high_viral['sentiment_bin'].value_counts().sort_index()
    
    sentiment_counts.plot(kind='bar', color=plt.cm.RdYlGn(np.linspace(0, 1, len(sentiment_counts))))
    plt.title('Sentiment Distribution in High-Virality Threads', fontsize=16)
    plt.xlabel('Sentiment Range', fontsize=14)
    plt.ylabel('Number of Threads', fontsize=14)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('charts/Sentiment_Intent_Analysis/high_virality_sentiment_distribution.png', dpi=300)
    plt.close()
    
    # 2. Intent Distribution in High-Virality Threads
    plt.figure(figsize=(14, 8))
    
    intent_counts = high_viral['initial_intent'].value_counts()
    
    # Plot bar chart
    ax = intent_counts.plot(kind='bar', color=plt.cm.tab10(np.linspace(0, 1, len(intent_counts))))
    plt.title('Intent Distribution in High-Virality Threads', fontsize=16)
    plt.xlabel('Intent', fontsize=14)
    plt.ylabel('Number of Threads', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add count labels
    for i, count in enumerate(intent_counts):
        ax.text(i, count + 1, str(count), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/Sentiment_Intent_Analysis/high_virality_intent_distribution.png', dpi=300)
    plt.close()
    
    # 3. Sentiment Shift in High-Virality Threads by Resolution
    plt.figure(figsize=(14, 8))
    
    # Calculate average sentiment shift for each resolution status
    shift_by_resolution = high_viral.groupby('resolution_simple')['sentiment_shift'].mean().sort_values()
    
    # Create color map based on shift values
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(shift_by_resolution)))
    
    ax = shift_by_resolution.plot(kind='barh', color=colors)
    plt.title('Sentiment Shift in High-Virality Threads by Resolution Status', fontsize=16)
    plt.xlabel('Average Sentiment Shift', fontsize=14)
    plt.ylabel('Resolution Status', fontsize=14)
    
    # Add value labels
    for i, value in enumerate(shift_by_resolution):
        ax.text(value + 0.01, i, f"{value:.3f}", va='center', fontweight='bold')
    
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('charts/Sentiment_Intent_Analysis/high_virality_sentiment_shift.png', dpi=300)
    plt.close()

def print_insights(df):
    """Print key insights from the sentiment and intent analysis"""
    print("\n===== KEY INSIGHTS FROM SENTIMENT AND INTENT ANALYSIS =====\n")
    
    # 1. Overall sentiment statistics
    print("Overall Sentiment Statistics:")
    print(f"Average sentiment across all threads: {df['avg_sentiment'].mean():.3f}")
    print(f"Most positive thread sentiment: {df['avg_sentiment'].max():.3f}")
    print(f"Most negative thread sentiment: {df['avg_sentiment'].min():.3f}")
    print(f"Average sentiment shift: {df['sentiment_shift'].mean():.3f}")
    
    # 2. Resolution and sentiment
    resolution_sentiment = df.groupby('resolution_simple')['avg_sentiment'].mean().sort_values(ascending=False)
    print("\nAverage Sentiment by Resolution Status:")
    for status, sentiment in resolution_sentiment.items():
        print(f"  {status}: {sentiment:.3f}")
    
    # 3. Intent distribution
    intent_counts = df['initial_intent'].value_counts()
    print("\nIntent Distribution:")
    for intent, count in intent_counts.items():
        print(f"  {intent}: {count} threads ({count/len(df)*100:.1f}%)")
    
    # 4. Resolution rate by intent
    intent_resolution = df[df['resolution_status'] != 'Not a Question'].groupby('initial_intent').apply(
        lambda x: (x['resolution_status'].isin(['Resolved', 'Likely Resolved'])).mean() * 100
    ).sort_values(ascending=False)
    
    print("\nResolution Rate by Intent:")
    for intent, rate in intent_resolution.items():
        print(f"  {intent}: {rate:.1f}%")
    
    # 5. Sentiment and virality correlation
    sentiment_virality_corr = df['avg_sentiment'].corr(df['virality_combined'])
    print(f"\nCorrelation between sentiment and virality: {sentiment_virality_corr:.3f}")
    
    # 6. Most positive and negative categories
    category_sentiment = df.groupby('category')['avg_sentiment'].mean()
    most_positive = category_sentiment.idxmax()
    most_negative = category_sentiment.idxmin()
    
    print(f"\nMost positive category: {most_positive} ({category_sentiment[most_positive]:.3f})")
    print(f"Most negative category: {most_negative} ({category_sentiment[most_negative]:.3f})")
    
    # 7. Intent diversity insights
    diversity_resolution = df.groupby('resolution_simple')['intent_diversity'].mean().sort_values(ascending=False)
    print("\nIntent Diversity by Resolution Status:")
    for status, diversity in diversity_resolution.items():
        print(f"  {status}: {diversity:.3f}")

def main():
    """Main function to run the sentiment and intent analysis"""
    print("Starting WhatsApp Sentiment and Intent Analysis...")
    
    # Load data
    resolution_df, messages_df = load_data()
    
    # Process threads
    sentiment_intent_df = process_threads(resolution_df, messages_df)
    
    # Generate charts
    generate_sentiment_charts(sentiment_intent_df)
    generate_intent_charts(sentiment_intent_df)
    analyze_high_virality_sentiment(sentiment_intent_df)
    
    # Print insights
    print_insights(sentiment_intent_df)
    
    print("\nAnalysis complete! Charts saved to charts/Sentiment_Intent_Analysis/")

if __name__ == "__main__":
    main()
