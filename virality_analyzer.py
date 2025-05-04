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
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# Create charts directory if it doesn't exist
os.makedirs('charts/Virality_Analysis', exist_ok=True)

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

def generate_virality_charts(df):
    """Generate visualizations for virality analysis"""
    print("Generating virality visualizations...")
    
    try:
        # 1. Distribution of virality scores
        plt.figure(figsize=(12, 8))
        plt.hist(df['virality_combined'].dropna(), bins=20, alpha=0.7, color='skyblue')
        plt.title('Distribution of Virality Scores', fontsize=16)
        plt.xlabel('Virality Score (Combined)', fontsize=14)
        plt.ylabel('Number of Threads', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('charts/Virality_Analysis/virality_distribution.png', dpi=300)
        plt.close()
        print("- Created virality distribution chart")
    except Exception as e:
        print(f"Error creating virality distribution chart: {e}")
    
    try:
        # 2. Virality by category
        plt.figure(figsize=(14, 10))
        # Ensure we have category data
        if 'category' in df.columns and not df['category'].isna().all():
            category_virality = df.groupby('category')['virality_combined'].mean().sort_values(ascending=False)
            category_counts = df.groupby('category').size()
            
            # Plot bar chart
            ax = category_virality.plot(kind='bar', color='lightgreen')
            plt.title('Average Virality by Category', fontsize=16)
            plt.xlabel('Category', fontsize=14)
            plt.ylabel('Average Virality Score', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels and thread counts
            for i, (value, count) in enumerate(zip(category_virality, category_counts[category_virality.index])):
                ax.text(i, value + 0.5, f"{value:.1f}", ha='center', fontweight='bold')
                ax.text(i, value/2, f"n={count}", ha='center', color='white', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('charts/Virality_Analysis/virality_by_category.png', dpi=300)
            plt.close()
            print("- Created virality by category chart")
        else:
            print("Skipping virality by category chart - missing category data")
    except Exception as e:
        print(f"Error creating virality by category chart: {e}")
    
    try:
        # 2.1 Virality by category and subcategory combination
        if 'category' in df.columns and 'subcategory' in df.columns:
            # Create category-subcategory combination
            df['category_subcategory'] = df.apply(
                lambda x: f"{x['category']} - {x['subcategory']}" if pd.notna(x['subcategory']) else x['category'], 
                axis=1
            )
            
            # Calculate average virality and count by category-subcategory
            cat_subcat_virality = df.groupby('category_subcategory')['virality_combined'].mean().sort_values(ascending=False)
            cat_subcat_counts = df.groupby('category_subcategory').size()
            
            # Filter to combinations with at least 3 threads for readability
            filtered_cat_subcat = cat_subcat_virality[cat_subcat_counts >= 3]
            filtered_counts = cat_subcat_counts[cat_subcat_counts >= 3]
            
            # Get top 20 for readability
            top_cat_subcat = filtered_cat_subcat.head(20)
            top_counts = filtered_counts[top_cat_subcat.index]
            
            # Create figure
            plt.figure(figsize=(16, 12))
            
            # Plot horizontal bar chart for better readability of labels
            ax = top_cat_subcat.plot(kind='barh', color='coral')
            plt.title('Average Virality by Category-Subcategory Combination (Top 20, min 3 threads)', fontsize=16)
            plt.xlabel('Average Virality Score', fontsize=14)
            plt.ylabel('Category - Subcategory', fontsize=14)
            
            # Add value labels and thread counts
            for i, (value, count) in enumerate(zip(top_cat_subcat, top_counts)):
                ax.text(value + 0.5, i, f"{value:.1f}", va='center', fontweight='bold')
                ax.text(value/2, i, f"n={count}", va='center', ha='center', color='white', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('charts/Virality_Analysis/virality_by_category_subcategory.png', dpi=300)
            plt.close()
            print("- Created virality by category-subcategory chart")
            
            # Create heatmap of virality by category and subcategory
            try:
                # Prepare data for heatmap
                heatmap_data = []
                
                for idx, row in df.iterrows():
                    if pd.notna(row['category']) and pd.notna(row['subcategory']):
                        heatmap_data.append({
                            'Category': row['category'],
                            'Subcategory': row['subcategory'],
                            'Virality': row['virality_combined']
                        })
                
                heatmap_df = pd.DataFrame(heatmap_data)
                
                # Create pivot table
                pivot_data = heatmap_df.pivot_table(
                    values='Virality',
                    index='Category',
                    columns='Subcategory',
                    aggfunc='mean'
                )
                
                # Create thread count pivot for annotations
                thread_pivot = heatmap_df.pivot_table(
                    values='Virality',
                    index='Category',
                    columns='Subcategory',
                    aggfunc='count'
                )
                
                # Create a mask for cells with fewer than 3 threads (for readability)
                mask = thread_pivot < 3
                
                # Plot heatmap
                plt.figure(figsize=(18, 12))
                
                # Create custom colormap
                cmap = plt.cm.YlOrRd.copy()
                cmap.set_bad('lightgray')
                
                # Plot heatmap with mask
                ax = sns.heatmap(
                    pivot_data,
                    annot=True,
                    fmt='.1f',
                    cmap=cmap,
                    mask=mask,
                    linewidths=0.5,
                    cbar_kws={'label': 'Average Virality Score'}
                )
                
                # Add thread count annotations
                for i in range(len(pivot_data.index)):
                    for j in range(len(pivot_data.columns)):
                        try:
                            count = thread_pivot.iloc[i, j]
                            if count >= 3:  # Only add count for cells with at least 3 threads
                                ax.text(j + 0.5, i + 0.85, f"n={int(count)}", 
                                        ha='center', va='center', color='black', fontsize=8)
                        except (IndexError, KeyError):
                            continue
                
                plt.title('Average Virality by Category and Subcategory', fontsize=16)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig('charts/Virality_Analysis/virality_heatmap_by_category_subcategory.png', dpi=300)
                plt.close()
                print("- Created virality heatmap by category and subcategory")
            except Exception as e:
                print(f"Error creating virality heatmap: {e}")
        else:
            print("Skipping category-subcategory charts - missing category or subcategory data")
    except Exception as e:
        print(f"Error creating category-subcategory charts: {e}")
    
    try:
        # 3. Top 10 most viral threads
        plt.figure(figsize=(14, 10))
        top_threads = df.sort_values('virality_combined', ascending=False).head(10)
        
        # Ensure we have topic data
        if 'topic' in top_threads.columns:
            # Create shortened topics for better display
            top_threads['short_topic'] = top_threads['topic'].apply(lambda x: str(x)[:30] + '...' if isinstance(x, str) and len(str(x)) > 30 else x)
            
            # Plot horizontal bar chart
            ax = top_threads.plot(kind='barh', x='short_topic', y='virality_combined', color='coral', legend=False)
            plt.title('Top 10 Most Viral Threads', fontsize=16)
            plt.xlabel('Virality Score (Combined)', fontsize=14)
            plt.ylabel('Thread Topic', fontsize=14)
            
            # Add thread IDs and message counts
            for i, (_, row) in enumerate(top_threads.iterrows()):
                ax.text(row['virality_combined'] + 0.5, i, 
                        f"ID: {row['thread_id']} | {row['message_count']} msgs | {row['unique_senders']} participants", 
                        va='center')
            
            plt.tight_layout()
            plt.savefig('charts/Virality_Analysis/top_viral_threads.png', dpi=300)
            plt.close()
            print("- Created top viral threads chart")
        else:
            print("Skipping top viral threads chart - missing topic data")
    except Exception as e:
        print(f"Error creating top viral threads chart: {e}")
    
    try:
        # 4. Virality components comparison
        plt.figure(figsize=(12, 8))
        
        # Filter out NaN or Inf values
        valid_data = df.dropna(subset=['virality_direct', 'virality_normalized', 'virality_combined', 'message_count'])
        
        # Create a scatter plot of direct vs normalized virality
        plt.scatter(valid_data['virality_direct'], valid_data['virality_normalized'], 
                    alpha=0.6, c=valid_data['virality_combined'], cmap='viridis', s=valid_data['message_count']*3)
        
        plt.title('Virality Components: Direct vs. Normalized Scores', fontsize=16)
        plt.xlabel('Direct Virality Score', fontsize=14)
        plt.ylabel('Normalized Virality Score', fontsize=14)
        plt.colorbar(label='Combined Virality Score')
        
        # Add annotations for top 5 threads
        top5 = valid_data.sort_values('virality_combined', ascending=False).head(5)
        for _, row in top5.iterrows():
            plt.annotate(f"Thread {row['thread_id']}", 
                        (row['virality_direct'], row['virality_normalized']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('charts/Virality_Analysis/virality_components.png', dpi=300)
        plt.close()
        print("- Created virality components chart")
    except Exception as e:
        print(f"Error creating virality components chart: {e}")
    
    try:
        # 5. Virality factors analysis
        # Filter out NaN or Inf values
        valid_data = df.dropna(subset=['message_count', 'unique_senders', 'avg_response_time', 'virality_combined'])
        
        # Set up the figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15))
        
        # Message count vs. virality
        ax1.scatter(valid_data['message_count'], valid_data['virality_combined'], alpha=0.6, c='blue')
        ax1.set_title('Message Count vs. Virality', fontsize=14)
        ax1.set_xlabel('Number of Messages', fontsize=12)
        ax1.set_ylabel('Virality Score', fontsize=12)
        
        # Participant count vs. virality
        ax2.scatter(valid_data['unique_senders'], valid_data['virality_combined'], alpha=0.6, c='green')
        ax2.set_title('Number of Participants vs. Virality', fontsize=14)
        ax2.set_xlabel('Number of Participants', fontsize=12)
        ax2.set_ylabel('Virality Score', fontsize=12)
        
        # Response time vs. virality - filter out extreme outliers
        response_time_data = valid_data[valid_data['avg_response_time'] <= valid_data['avg_response_time'].quantile(0.95)]
        ax3.scatter(response_time_data['avg_response_time'], response_time_data['virality_combined'], alpha=0.6, c='red')
        ax3.set_title('Average Response Time vs. Virality', fontsize=14)
        ax3.set_xlabel('Average Response Time (minutes)', fontsize=12)
        ax3.set_ylabel('Virality Score', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('charts/Virality_Analysis/virality_factors.png', dpi=300)
        plt.close()
        print("- Created virality factors chart")
    except Exception as e:
        print(f"Error creating virality factors chart: {e}")
    
    try:
        # 6. Virality heatmap by time
        # Extract hour from timestamp if available
        if 'timestamp' in df.columns:
            try:
                # Make a copy to avoid modifying the original dataframe
                time_df = df.copy()
                time_df['hour'] = pd.to_datetime(time_df['timestamp']).dt.hour
                time_df['day_of_week'] = pd.to_datetime(time_df['timestamp']).dt.day_name()
                
                # Create pivot table for heatmap
                hour_day_virality = time_df.pivot_table(
                    values='virality_combined', 
                    index='day_of_week', 
                    columns='hour',
                    aggfunc='mean'
                )
                
                # Plot heatmap
                plt.figure(figsize=(14, 8))
                sns.heatmap(hour_day_virality, cmap='YlOrRd', annot=True, fmt='.1f')
                plt.title('Average Virality by Day and Hour', fontsize=16)
                plt.xlabel('Hour of Day', fontsize=14)
                plt.ylabel('Day of Week', fontsize=14)
                plt.tight_layout()
                plt.savefig('charts/Virality_Analysis/virality_time_heatmap.png', dpi=300)
                plt.close()
                print("- Created virality time heatmap")
            except Exception as e:
                print(f"Could not create time-based heatmap: {e}")
    except Exception as e:
        print(f"Error creating time-based heatmap: {e}")

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
        
        # Generate virality visualizations
        generate_virality_charts(result)
        
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
