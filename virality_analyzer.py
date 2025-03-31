#!/usr/bin/env python3
"""
Virality Analyzer for WhatsApp Conversations

This script analyzes the virality of WhatsApp conversation threads based on:
1. Number of replies in the thread
2. Average response time
3. Total participants

It also adds additional metrics and categorization information.
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime

def calculate_virality_score(df):
    """
    Calculate three virality scores for each thread:
    1. Direct formula: message_count * (1 / avg_response_time)
    2. Normalized approach: equal weights (1/3) for message count, response time, and participant count
    3. Combined score: average of normalized direct score and normalized score
    """
    print("Calculating virality scores...")
    
    # Group by thread_id
    thread_stats = df.groupby('thread_id').agg(
        message_count=('message', 'count'),
        unique_senders=('sender', lambda x: len(set(x))),
        first_message_time=('datetime', 'min'),
        last_message_time=('datetime', 'max')
    ).reset_index()
    
    # Calculate thread duration in minutes
    thread_stats['thread_duration_minutes'] = thread_stats.apply(
        lambda row: (pd.to_datetime(row['last_message_time']) - 
                     pd.to_datetime(row['first_message_time'])).total_seconds() / 60 
        if row['message_count'] > 1 else 0, 
        axis=1
    )
    
    # Calculate average response time (in minutes)
    thread_stats['avg_response_time'] = thread_stats.apply(
        lambda row: row['thread_duration_minutes'] / (row['message_count'] - 1) 
        if row['message_count'] > 1 else float('inf'),
        axis=1
    )
    
    # Calculate direct virality score: message_count * (1 / avg_response_time)
    thread_stats['virality_direct'] = thread_stats.apply(
        lambda row: row['message_count'] * (1 / row['avg_response_time']) if row['avg_response_time'] > 0 else 0,
        axis=1
    )
    
    # Calculate normalized virality score with equal weights
    
    # Normalize message count (0-1 scale)
    max_message_count = thread_stats['message_count'].max()
    thread_stats['normalized_message_count'] = thread_stats['message_count'] / max_message_count
    
    # Normalize response time (0-1 scale, inverted so faster = higher score)
    max_response_time = thread_stats['avg_response_time'].replace([np.inf, -np.inf], 0).max()
    if max_response_time > 0:
        thread_stats['normalized_response_time'] = 1 - (thread_stats['avg_response_time'] / max_response_time)
        # Cap at 0 for very slow responses
        thread_stats['normalized_response_time'] = thread_stats['normalized_response_time'].clip(0, 1)
    else:
        thread_stats['normalized_response_time'] = 0
    
    # Normalize participant count
    max_participants = thread_stats['unique_senders'].max()
    thread_stats['normalized_participants'] = thread_stats['unique_senders'] / max_participants
    
    # Calculate normalized virality score with equal weights (1/3 each)
    thread_stats['virality_normalized'] = (
        (1/3) * thread_stats['normalized_message_count'] + 
        (1/3) * thread_stats['normalized_response_time'] + 
        (1/3) * thread_stats['normalized_participants']
    ) * 100  # Scale to 0-100
    
    # Create a combined score by normalizing the direct score and averaging with the normalized score
    max_direct = thread_stats['virality_direct'].max()
    if max_direct > 0:
        thread_stats['virality_direct_normalized'] = (thread_stats['virality_direct'] / max_direct) * 100
    else:
        thread_stats['virality_direct_normalized'] = 0
    
    # Calculate the combined score (average of the two normalized scores)
    thread_stats['virality_combined'] = (thread_stats['virality_direct_normalized'] + thread_stats['virality_normalized']) / 2
    
    # Round to 2 decimal places
    thread_stats['virality_direct'] = thread_stats['virality_direct'].round(2)
    thread_stats['virality_normalized'] = thread_stats['virality_normalized'].round(2)
    thread_stats['virality_combined'] = thread_stats['virality_combined'].round(2)
    
    return thread_stats

def add_category_concat(df):
    """Add a concatenated category column (category + subcategory)"""
    df['category_full'] = df['topic_category'] + ' - ' + df['topic_subcategory']
    return df

def merge_data(original_df, thread_stats):
    """Merge thread statistics back into the original dataframe"""
    # Get the first topic for each thread
    thread_topics = original_df.groupby('thread_id').agg(
        topic=('gpt4_topic', 'first'),
        category=('topic_category', 'first'),
        subcategory=('topic_subcategory', 'first')
    ).reset_index()
    
    # Merge thread stats with topics
    result = pd.merge(thread_stats, thread_topics, on='thread_id')
    
    # Add concatenated category
    result['category_full'] = result['category'] + ' - ' + result['subcategory']
    
    # Sort by direct virality score (descending)
    result = result.sort_values('virality_direct', ascending=False)
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Calculate virality scores for WhatsApp threads')
    parser.add_argument('--input', default='whatsapp_final_categorized.csv', help='Input CSV file')
    parser.add_argument('--output', default='whatsapp_virality_scores.csv', help='Output CSV file')
    args = parser.parse_args()
    
    print(f"Reading data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
        
        # Convert datetime strings to datetime objects
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Calculate thread statistics
        thread_stats = calculate_virality_score(df)
        
        # Merge with topic information
        result = merge_data(df, thread_stats)
        
        # Save to CSV
        result.to_csv(args.output, index=False)
        print(f"Virality analysis complete. Results saved to {args.output}")
        
        # Print top 10 most viral threads (combined score)
        print("\nTop 10 most viral threads (combined score):")
        top_threads_combined = result.sort_values('virality_combined', ascending=False).head(10)
        for _, row in top_threads_combined.iterrows():
            print(f"Thread {row['thread_id']}: Combined Score {row['virality_combined']}, "
                  f"Direct Score {row['virality_direct']}, "
                  f"Normalized Score {row['virality_normalized']}, "
                  f"{row['message_count']} messages, "
                  f"{row['unique_senders']} participants, "
                  f"Avg response: {row['avg_response_time']:.2f} min, "
                  f"Topic: {row['topic']}, "
                  f"Category: {row['category_full']}")
        
        # Print top 10 most viral threads (direct score)
        print("\nTop 10 most viral threads (direct score):")
        top_threads_direct = result.sort_values('virality_direct', ascending=False).head(10)
        for _, row in top_threads_direct.iterrows():
            print(f"Thread {row['thread_id']}: Direct Score {row['virality_direct']}, "
                  f"Combined Score {row['virality_combined']}, "
                  f"Normalized Score {row['virality_normalized']}, "
                  f"{row['message_count']} messages, "
                  f"{row['unique_senders']} participants, "
                  f"Avg response: {row['avg_response_time']:.2f} min, "
                  f"Topic: {row['topic']}, "
                  f"Category: {row['category_full']}")
        
        # Print top 10 most viral threads (normalized score)
        print("\nTop 10 most viral threads (normalized score):")
        top_threads_norm = result.sort_values('virality_normalized', ascending=False).head(10)
        for _, row in top_threads_norm.iterrows():
            print(f"Thread {row['thread_id']}: Normalized Score {row['virality_normalized']}, "
                  f"Combined Score {row['virality_combined']}, "
                  f"Direct Score {row['virality_direct']}, "
                  f"{row['message_count']} messages, "
                  f"{row['unique_senders']} participants, "
                  f"Avg response: {row['avg_response_time']:.2f} min, "
                  f"Topic: {row['topic']}, "
                  f"Category: {row['category_full']}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
