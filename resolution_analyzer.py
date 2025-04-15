#!/usr/bin/env python3
"""
Resolution Analyzer for WhatsApp Conversations

This script analyzes whether each thread's question was resolved, unresolved, or ambiguous.
It uses a simple rule-based approach to determine resolution status based on:
1. Presence of thank you messages or acknowledgments
2. Number of responses after the initial question
3. Conversation flow patterns

The results are saved to a new CSV file with the resolution status added.

It also generates visualizations for analyzing resolution status across:
1. Categories and subcategories
2. Time dimensions (month, day of week, hour)
3. Virality correlation
"""

import pandas as pd
import re
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import calendar
from collections import defaultdict, Counter
from datetime import datetime, timedelta

# Create necessary directories
os.makedirs('charts/Resolution_Analysis', exist_ok=True)

# Set plot style for charts
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Resolution indicators
RESOLVED_INDICATORS = [
    r'thank(?:s|\s+you)',
    r'appreciate',
    r'got\s+it',
    r'helpful',
    r'perfect',
    r'great',
    r'awesome',
    r'exactly\s+what\s+i\s+need',
    r'solved',
    r'worked',
    r'found\s+it',
    r'that\s+works',
    r'problem\s+solved',
    r'issue\s+resolved',
    r'makes\s+sense',
    r'understood',
    r'clear\s+now',
    r'will\s+do',
    r'sounds\s+good',
    r'perfect',
    r'excellent'
]

UNRESOLVED_INDICATORS = [
    r'still\s+need',
    r'still\s+looking',
    r'any\s+other\s+ideas',
    r'doesn\'t\s+work',
    r'didn\'t\s+work',
    r'not\s+working',
    r'not\s+helpful',
    r'no\s+luck',
    r'no\s+solution',
    r'problem\s+persists',
    r'issue\s+remains',
    r'unresolved',
    r'confused',
    r'unclear',
    r'not\s+sure',
    r'doesn\'t\s+make\s+sense',
    r'can\s+anyone\s+else',
    r'any\s+other\s+suggestions',
    r'help\s+still\s+needed',
    r'not\s+resolved',
    r'anyone\?'
]

def is_question(message):
    """Check if a message is a question."""
    if not isinstance(message, str):
        return False
    
    # Check for question marks
    if '?' in message:
        return True
    
    # Check for question words at the beginning
    question_starters = ['who', 'what', 'when', 'where', 'why', 'how', 'is', 'are', 'was', 'were', 
                         'will', 'would', 'can', 'could', 'should', 'do', 'does', 'did', 'have', 'has', 'had',
                         'anyone', 'anybody', 'any']
    
    first_word = message.lower().split()[0] if message.split() else ""
    
    return first_word in question_starters

def analyze_resolution_status(thread_messages):
    """
    Analyze if a thread's question was resolved, unresolved, or ambiguous.
    
    Args:
        thread_messages: List of message dictionaries for a thread
        
    Returns:
        str: 'Resolved', 'Unresolved', or 'Ambiguous'
    """
    if not thread_messages:
        return 'Ambiguous'
    
    # Check if there's a question in the thread
    has_question = any(is_question(msg['message']) for msg in thread_messages)
    if not has_question:
        return 'Not a Question'
    
    # Get the first question in the thread
    first_question_idx = next((i for i, msg in enumerate(thread_messages) 
                              if is_question(msg['message'])), None)
    
    if first_question_idx is None:
        return 'Ambiguous'
    
    # Check for responses after the question
    responses_after_question = thread_messages[first_question_idx+1:]
    
    if not responses_after_question:
        return 'Unresolved'  # No responses after the question
    
    # Check for resolution indicators in responses
    all_responses_text = ' '.join([msg['message'].lower() for msg in responses_after_question])
    
    # Check for resolved indicators
    resolved_match = any(re.search(pattern, all_responses_text) for pattern in RESOLVED_INDICATORS)
    
    # Check for unresolved indicators
    unresolved_match = any(re.search(pattern, all_responses_text) for pattern in UNRESOLVED_INDICATORS)
    
    # Determine resolution status based on indicators
    if resolved_match and not unresolved_match:
        return 'Resolved'
    elif unresolved_match and not resolved_match:
        return 'Unresolved'
    elif resolved_match and unresolved_match:
        # If both types of indicators are present, check which came last
        last_resolved_idx = max((responses_after_question[i]['message'].lower().find(pattern) 
                               for i in range(len(responses_after_question)) 
                               for pattern in RESOLVED_INDICATORS 
                               if re.search(pattern, responses_after_question[i]['message'].lower())), 
                              default=-1)
        
        last_unresolved_idx = max((responses_after_question[i]['message'].lower().find(pattern) 
                                 for i in range(len(responses_after_question)) 
                                 for pattern in UNRESOLVED_INDICATORS 
                                 if re.search(pattern, responses_after_question[i]['message'].lower())), 
                                default=-1)
        
        if last_resolved_idx > last_unresolved_idx:
            return 'Resolved'
        else:
            return 'Unresolved'
    
    # If no clear indicators, use heuristics
    if len(responses_after_question) >= 3:
        # Multiple responses suggest some engagement
        return 'Likely Resolved'
    elif len(responses_after_question) == 1:
        # Only one response might not be enough
        return 'Likely Unresolved'
    else:
        return 'Ambiguous'

def analyze_threads(input_file, output_file):
    """
    Analyze all threads in the input file and determine resolution status.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    print(f"Reading data from {input_file}...")
    
    # Load the virality scores data
    virality_df = pd.read_csv(input_file)
    
    # Load the original categorized data to get all messages
    categorized_file = 'whatsapp_final_categorized.csv'
    try:
        messages_df = pd.read_csv(categorized_file)
    except FileNotFoundError:
        print(f"Error: {categorized_file} not found. Please ensure this file exists.")
        return
    
    # Convert datetime strings to datetime objects
    messages_df['datetime'] = pd.to_datetime(messages_df['datetime'])
    
    # Group messages by thread_id
    thread_messages = defaultdict(list)
    for _, row in messages_df.iterrows():
        thread_messages[row['thread_id']].append({
            'sender': row['sender'],
            'message': row['message'],
            'datetime': row['datetime']
        })
    
    # Sort messages within each thread by datetime
    for thread_id in thread_messages:
        thread_messages[thread_id].sort(key=lambda x: x['datetime'])
    
    # Analyze resolution status for each thread
    resolution_status = {}
    for thread_id, messages in thread_messages.items():
        resolution_status[thread_id] = analyze_resolution_status(messages)
    
    # Add resolution status to virality dataframe
    virality_df['resolution_status'] = virality_df['thread_id'].map(resolution_status)
    
    # Save to CSV
    virality_df.to_csv(output_file, index=False)
    print(f"Resolution analysis complete. Results saved to {output_file}")
    
    # Print summary statistics
    status_counts = virality_df['resolution_status'].value_counts()
    print("\nResolution Status Summary:")
    for status, count in status_counts.items():
        print(f"{status}: {count} threads ({count/len(virality_df)*100:.1f}%)")
    
    # Print detailed examples of each category
    print("\n===== DETAILED EXAMPLES OF EACH RESOLUTION STATUS =====")
    for status in ['Resolved', 'Unresolved', 'Likely Resolved', 'Likely Unresolved', 'Ambiguous', 'Not a Question']:
        examples = virality_df[virality_df['resolution_status'] == status].head(2)
        print(f"\n\n{'='*20} {status.upper()} EXAMPLES {'='*20}")
        
        for _, row in examples.iterrows():
            thread_id = row['thread_id']
            print(f"\nThread {thread_id}: {row['topic']}")
            print(f"Category: {row['category_full']}")
            print(f"Message count: {row['message_count']}, Unique senders: {row['unique_senders']}")
            print("\nMessages in thread:")
            
            # Print all messages in the thread
            for i, msg in enumerate(thread_messages[thread_id]):
                sender = msg['sender']
                # Truncate sender name for privacy
                if len(sender) > 3:
                    sender = sender[:3] + "..."
                
                message_text = msg['message']
                # Truncate very long messages
                if len(message_text) > 200:
                    message_text = message_text[:200] + "..."
                
                print(f"{i+1}. {sender}: {message_text}")
            
            print("-" * 50)
    
    return virality_df, messages_df

def generate_category_charts(df):
    """Generate charts showing resolution status by category"""
    print("Generating category-based resolution charts...")
    
    # Simplify resolution status for clearer visualization
    df['resolution_simple'] = df['resolution_status'].replace({
        'Likely Resolved': 'Resolved',
        'Likely Unresolved': 'Unresolved'
    })
    
    # 1. Resolution Status Distribution (Overall)
    plt.figure(figsize=(10, 6))
    status_counts = df['resolution_simple'].value_counts()
    status_counts.plot(kind='bar', color=['#2ecc71', '#e74c3c', '#f39c12', '#3498db'])
    plt.title('Overall Distribution of Resolution Status', fontsize=16)
    plt.xlabel('Resolution Status', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add count labels on top of bars
    for i, count in enumerate(status_counts):
        plt.text(i, count + 5, str(count), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/Resolution_Analysis/overall_resolution_distribution.png', dpi=300)
    plt.close()
    
    # 2. Resolution Rate by Category
    plt.figure(figsize=(14, 8))
    
    # Calculate resolution rate for each category
    category_resolution = pd.crosstab(
        df['category'], 
        df['resolution_simple'],
        normalize='index'
    ) * 100
    
    # Sort by resolved percentage
    if 'Resolved' in category_resolution.columns:
        category_resolution = category_resolution.sort_values('Resolved', ascending=False)
    
    # Plot stacked bar chart
    category_resolution.plot(
        kind='barh', 
        stacked=True,
        color=['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    )
    
    plt.title('Resolution Rate by Category', fontsize=16)
    plt.xlabel('Percentage', fontsize=14)
    plt.ylabel('Category', fontsize=14)
    plt.legend(title='Resolution Status')
    plt.tight_layout()
    plt.savefig('charts/Resolution_Analysis/resolution_by_category.png', dpi=300)
    plt.close()
    
    # 3. Top 10 Subcategories with Highest Unresolved Rate
    plt.figure(figsize=(14, 8))
    
    # Get subcategories with at least 5 threads
    subcategory_counts = df.groupby('subcategory').size()
    valid_subcategories = subcategory_counts[subcategory_counts >= 5].index
    
    # Filter and calculate resolution rate
    filtered_df = df[df['subcategory'].isin(valid_subcategories)]
    subcategory_resolution = pd.crosstab(
        filtered_df['subcategory'], 
        filtered_df['resolution_simple'],
        normalize='index'
    ) * 100
    
    # Sort by unresolved percentage and get top 10
    if 'Unresolved' in subcategory_resolution.columns:
        top_unresolved = subcategory_resolution.sort_values('Unresolved', ascending=False).head(10)
        
        # Plot stacked bar chart
        top_unresolved.plot(
            kind='barh', 
            stacked=True,
            color=['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
        )
        
        plt.title('Top 10 Subcategories with Highest Unresolved Rate', fontsize=16)
        plt.xlabel('Percentage', fontsize=14)
        plt.ylabel('Subcategory', fontsize=14)
        plt.legend(title='Resolution Status')
        plt.tight_layout()
        plt.savefig('charts/Resolution_Analysis/top_unresolved_subcategories.png', dpi=300)
        plt.close()

def generate_time_charts(resolution_df, messages_df):
    """Generate time-based resolution status charts"""
    print("Generating time-based resolution charts...")
    
    if messages_df is None:
        print("Skipping time-based charts due to missing message data.")
        return
    
    # Simplify resolution status
    resolution_df['resolution_simple'] = resolution_df['resolution_status'].replace({
        'Likely Resolved': 'Resolved',
        'Likely Unresolved': 'Unresolved'
    })
    
    # Get the first message datetime for each thread
    thread_start_times = messages_df.groupby('thread_id')['datetime'].min().reset_index()
    thread_start_times.columns = ['thread_id', 'start_time']
    
    # Merge with resolution data
    time_df = pd.merge(resolution_df, thread_start_times, on='thread_id')
    
    # Extract time components
    time_df['month'] = time_df['start_time'].dt.month_name()
    time_df['day_of_week'] = time_df['start_time'].dt.day_name()
    time_df['hour'] = time_df['start_time'].dt.hour
    
    # 1. Resolution Status by Month
    plt.figure(figsize=(14, 8))
    
    # Ensure proper month ordering
    months_order = list(calendar.month_name)[1:]
    
    # Calculate resolution counts by month
    month_resolution = pd.crosstab(
        time_df['month'], 
        time_df['resolution_simple']
    )
    
    # Reorder months
    month_resolution = month_resolution.reindex(months_order)
    
    # Plot stacked bar chart
    month_resolution.plot(
        kind='bar', 
        stacked=True,
        color=['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    )
    
    plt.title('Resolution Status by Month', fontsize=16)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Resolution Status')
    plt.tight_layout()
    plt.savefig('charts/Resolution_Analysis/resolution_by_month.png', dpi=300)
    plt.close()
    
    # 2. Resolution Status by Day of Week
    plt.figure(figsize=(14, 8))
    
    # Ensure proper day ordering
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Calculate resolution counts by day
    day_resolution = pd.crosstab(
        time_df['day_of_week'], 
        time_df['resolution_simple']
    )
    
    # Reorder days
    day_resolution = day_resolution.reindex(days_order)
    
    # Plot stacked bar chart
    day_resolution.plot(
        kind='bar', 
        stacked=True,
        color=['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    )
    
    plt.title('Resolution Status by Day of Week', fontsize=16)
    plt.xlabel('Day of Week', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Resolution Status')
    plt.tight_layout()
    plt.savefig('charts/Resolution_Analysis/resolution_by_day.png', dpi=300)
    plt.close()
    
    # 3. Resolution Status by Hour of Day
    plt.figure(figsize=(14, 8))
    
    # Calculate resolution counts by hour
    hour_resolution = pd.crosstab(
        time_df['hour'], 
        time_df['resolution_simple']
    )
    
    # Create hour labels
    hour_labels = [f"{h:02d}:00" for h in range(24)]
    
    # Plot stacked bar chart
    hour_resolution.plot(
        kind='bar', 
        stacked=True,
        color=['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    )
    
    plt.title('Resolution Status by Hour of Day', fontsize=16)
    plt.xlabel('Hour', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(range(24), hour_labels, rotation=45)
    plt.legend(title='Resolution Status')
    plt.tight_layout()
    plt.savefig('charts/Resolution_Analysis/resolution_by_hour.png', dpi=300)
    plt.close()

def generate_virality_charts(df):
    """Generate charts analyzing the relationship between virality and resolution status"""
    print("Generating virality-based resolution charts...")
    
    # Simplify resolution status
    df['resolution_simple'] = df['resolution_status'].replace({
        'Likely Resolved': 'Resolved',
        'Likely Unresolved': 'Unresolved'
    })
    
    # 1. Average Virality Score by Resolution Status
    plt.figure(figsize=(10, 6))
    
    # Calculate average virality for each resolution status
    virality_by_status = df.groupby('resolution_simple')['virality_combined'].mean().sort_values(ascending=False)
    
    # Plot bar chart
    virality_by_status.plot(
        kind='bar',
        color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    )
    
    plt.title('Average Virality Score by Resolution Status', fontsize=16)
    plt.xlabel('Resolution Status', fontsize=14)
    plt.ylabel('Average Virality Score', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add score labels on top of bars
    for i, score in enumerate(virality_by_status):
        plt.text(i, score + 1, f"{score:.2f}", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/Resolution_Analysis/virality_by_resolution.png', dpi=300)
    plt.close()
    
    # 2. Resolution Rate by Virality Level
    plt.figure(figsize=(12, 10))
    
    # Create equal-sized virality groups using rank-based approach
    # This ensures we get exactly equal groups even with duplicate values
    df['virality_rank'] = df['virality_combined'].rank(method='first')
    total_threads = len(df)
    group_size = total_threads // 4
    
    # Create custom labels based on actual virality ranges
    def get_virality_level(rank):
        if rank <= group_size:
            return 'Very Low (Q1)'
        elif rank <= 2 * group_size:
            return 'Low (Q2)'
        elif rank <= 3 * group_size:
            return 'Medium (Q3)'
        else:
            return 'High (Q4)'
    
    # Apply the function to create virality levels
    df['virality_level'] = df['virality_rank'].apply(get_virality_level)
    
    # Calculate resolution rate by virality level (percentages)
    virality_resolution_pct = pd.crosstab(
        df['virality_level'], 
        df['resolution_simple'],
        normalize='index'
    ) * 100
    
    # Calculate raw counts by virality level
    virality_resolution_counts = pd.crosstab(
        df['virality_level'], 
        df['resolution_simple']
    )
    
    # Get total counts per virality level for reference
    total_counts = virality_resolution_counts.sum(axis=1)
    
    # Create a figure with two subplots (percentages and counts)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1, 1]})
    
    # Plot 1: Percentages (stacked bar chart)
    virality_resolution_pct.plot(
        kind='bar', 
        stacked=True,
        color=['#2ecc71', '#e74c3c', '#f39c12', '#3498db'],
        ax=ax1
    )
    
    ax1.set_title('Resolution Rate by Virality Level (Percentage)', fontsize=16)
    ax1.set_xlabel('')  # No x-label for the top plot
    ax1.set_ylabel('Percentage', fontsize=14)
    ax1.set_xticklabels([])  # Hide x-tick labels for the top plot
    ax1.legend(title='Resolution Status')
    
    # Add percentage labels on the bars
    for i, virality_level in enumerate(virality_resolution_pct.index):
        cumulative_height = 0
        for j, col in enumerate(virality_resolution_pct.columns):
            if virality_resolution_pct.loc[virality_level, col] > 5:  # Only show if > 5%
                height = virality_resolution_pct.loc[virality_level, col]
                ax1.text(
                    i, 
                    cumulative_height + height/2, 
                    f"{height:.1f}%", 
                    ha='center', 
                    va='center',
                    fontweight='bold',
                    color='white'
                )
            cumulative_height += virality_resolution_pct.loc[virality_level, col]
    
    # Plot 2: Counts (stacked bar chart)
    virality_resolution_counts.plot(
        kind='bar', 
        stacked=True,
        color=['#2ecc71', '#e74c3c', '#f39c12', '#3498db'],
        ax=ax2
    )
    
    ax2.set_title('Resolution Count by Virality Level (Absolute Numbers)', fontsize=16)
    ax2.set_xlabel('Virality Level', fontsize=14)
    ax2.set_ylabel('Count', fontsize=14)
    ax2.set_xticklabels(labels=['Very Low (Q1)', 'Low (Q2)', 'Medium (Q3)', 'High (Q4)'], rotation=45)
    ax2.legend(title='Resolution Status')
    
    # Add count labels on the bars
    for i, virality_level in enumerate(virality_resolution_counts.index):
        cumulative_height = 0
        for j, col in enumerate(virality_resolution_counts.columns):
            if virality_resolution_counts.loc[virality_level, col] > 0:
                height = virality_resolution_counts.loc[virality_level, col]
                ax2.text(
                    i, 
                    cumulative_height + height/2, 
                    str(height), 
                    ha='center', 
                    va='center',
                    fontweight='bold',
                    color='white' if height > 10 else 'black'
                )
            cumulative_height += virality_resolution_counts.loc[virality_level, col]
    
    # Add total count annotation above each bar
    for i, (level, count) in enumerate(total_counts.items()):
        ax2.text(
            i, 
            count + 5, 
            f"Total: {count}", 
            ha='center', 
            fontweight='bold'
        )
    
    plt.tight_layout()
    plt.savefig('charts/Resolution_Analysis/resolution_by_virality_level.png', dpi=300)
    plt.close()
    
    # 3. Message Count vs. Resolution Status
    plt.figure(figsize=(10, 6))
    
    # Calculate average message count for each resolution status
    msg_count_by_status = df.groupby('resolution_simple')['message_count'].mean().sort_values(ascending=False)
    
    # Plot bar chart
    msg_count_by_status.plot(
        kind='bar',
        color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    )
    
    plt.title('Average Message Count by Resolution Status', fontsize=16)
    plt.xlabel('Resolution Status', fontsize=14)
    plt.ylabel('Average Message Count', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add count labels on top of bars
    for i, count in enumerate(msg_count_by_status):
        plt.text(i, count + 0.5, f"{count:.2f}", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/Resolution_Analysis/message_count_by_resolution.png', dpi=300)
    plt.close()
    
    # 4. Unique Senders vs. Resolution Status
    plt.figure(figsize=(10, 6))
    
    # Calculate average unique senders for each resolution status
    sender_count_by_status = df.groupby('resolution_simple')['unique_senders'].mean().sort_values(ascending=False)
    
    # Plot bar chart
    sender_count_by_status.plot(
        kind='bar',
        color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    )
    
    plt.title('Average Number of Unique Senders by Resolution Status', fontsize=16)
    plt.xlabel('Resolution Status', fontsize=14)
    plt.ylabel('Average Number of Unique Senders', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add count labels on top of bars
    for i, count in enumerate(sender_count_by_status):
        plt.text(i, count + 0.2, f"{count:.2f}", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/Resolution_Analysis/unique_senders_by_resolution.png', dpi=300)
    plt.close()

def analyze_high_virality_unresolved(df, messages_df):
    """Perform detailed analysis of high-virality unresolved questions"""
    print("Analyzing high-virality unresolved questions...")
    
    # Create a subfolder for these specific analyses
    os.makedirs('charts/Resolution_Analysis/High_Virality_Unresolved', exist_ok=True)
    
    # Simplify resolution status
    df['resolution_simple'] = df['resolution_status'].replace({
        'Likely Resolved': 'Resolved',
        'Likely Unresolved': 'Unresolved'
    })
    
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
    
    # Filter for high virality (Q4) unresolved questions
    high_viral_unresolved = df[
        (df['virality_level'] == 'High (Q4)') & 
        (df['resolution_simple'] == 'Unresolved')
    ]
    
    # Print summary
    print(f"\nFound {len(high_viral_unresolved)} high-virality unresolved questions (top 25% by virality)")
    
    if len(high_viral_unresolved) == 0:
        print("No high-virality unresolved questions to analyze.")
        return
    
    # 1. Category Distribution of High-Virality Unresolved Questions (Waterfall Chart)
    plt.figure(figsize=(14, 8))
    category_counts = high_viral_unresolved['category'].value_counts().sort_values(ascending=True)
    
    # Plot horizontal bar chart (waterfall style)
    ax = category_counts.plot(kind='barh', color='#3498db')
    plt.title('Category Distribution of High-Virality Unresolved Questions', fontsize=16)
    plt.xlabel('Count', fontsize=14)
    plt.ylabel('Category', fontsize=14)
    
    # Add count and percentage labels
    total_threads = len(high_viral_unresolved)
    for i, (category, count) in enumerate(category_counts.items()):
        percentage = (count / total_threads) * 100
        ax.text(count + 0.2, i, f"{count} ({percentage:.1f}%)", va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/Resolution_Analysis/High_Virality_Unresolved/category_distribution.png', dpi=300)
    plt.close()
    
    # Print examples of threads for each category
    print("\nExamples of high-virality unresolved threads by category:")
    for category in category_counts.index:
        category_threads = high_viral_unresolved[high_viral_unresolved['category'] == category].sort_values('virality_combined', ascending=False).head(2)
        print(f"\n{category} (Total: {category_counts[category]}):")
        for _, thread in category_threads.iterrows():
            print(f"  - Thread {thread['thread_id']}: {thread['topic']} (Virality: {thread['virality_combined']:.2f})")
    
    # 2. Subcategory Distribution (Top 10)
    plt.figure(figsize=(14, 8))
    subcategory_counts = high_viral_unresolved['subcategory'].value_counts().head(10)
    
    # Plot horizontal bar chart
    subcategory_counts.plot(kind='barh', color='#e74c3c')
    plt.title('Top Subcategories of High-Virality Unresolved Questions', fontsize=16)
    plt.xlabel('Count', fontsize=14)
    plt.ylabel('Subcategory', fontsize=14)
    
    # Add count labels
    for i, count in enumerate(subcategory_counts):
        plt.text(count + 0.1, i, str(count), va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/Resolution_Analysis/High_Virality_Unresolved/top_subcategories.png', dpi=300)
    plt.close()
    
    # 3. Time Analysis
    if messages_df is not None:
        # Get the first message datetime for each thread
        thread_start_times = messages_df.groupby('thread_id')['datetime'].min().reset_index()
        thread_start_times.columns = ['thread_id', 'start_time']
        
        # Merge with high virality unresolved data
        time_df = pd.merge(high_viral_unresolved, thread_start_times, on='thread_id')
        
        # Verify the merge was successful
        print(f"\nTime analysis: Found time data for {len(time_df)} out of {len(high_viral_unresolved)} high-virality unresolved threads")
        
        # Extract time components
        time_df['month'] = time_df['start_time'].dt.month_name()
        time_df['day_of_week'] = time_df['start_time'].dt.day_name()
        time_df['hour'] = time_df['start_time'].dt.hour
        
        # Month distribution
        plt.figure(figsize=(14, 8))
        
        # Ensure proper month ordering
        months_order = list(calendar.month_name)[1:]
        month_counts = time_df['month'].value_counts().reindex(months_order).fillna(0)
        
        # Calculate total for verification
        month_total = month_counts.sum()
        
        month_counts.plot(kind='bar', color='#3498db')
        plt.title(f'Monthly Distribution of High-Virality Unresolved Questions (Total: {int(month_total)})', fontsize=16)
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(rotation=45)
        
        # Add count labels
        for i, count in enumerate(month_counts):
            if count > 0:
                plt.text(i, count + 0.1, str(int(count)), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('charts/Resolution_Analysis/High_Virality_Unresolved/month_distribution.png', dpi=300)
        plt.close()
        
        # Day of week distribution
        plt.figure(figsize=(14, 8))
        
        # Ensure proper day ordering
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = time_df['day_of_week'].value_counts().reindex(days_order).fillna(0)
        
        # Calculate total for verification
        day_total = day_counts.sum()
        
        day_counts.plot(kind='bar', color='#2ecc71')
        plt.title(f'Day of Week Distribution of High-Virality Unresolved Questions (Total: {int(day_total)})', fontsize=16)
        plt.xlabel('Day of Week', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(rotation=45)
        
        # Add count labels
        for i, count in enumerate(day_counts):
            if count > 0:
                plt.text(i, count + 0.1, str(int(count)), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('charts/Resolution_Analysis/High_Virality_Unresolved/day_distribution.png', dpi=300)
        plt.close()
        
        # Hour distribution
        plt.figure(figsize=(14, 8))
        
        # Create hour bins (morning, afternoon, evening, night)
        time_df['time_of_day'] = pd.cut(
            time_df['hour'], 
            bins=[0, 6, 12, 18, 24], 
            labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)']
        )
        
        time_of_day_counts = time_df['time_of_day'].value_counts()
        
        # Calculate total for verification
        time_total = time_of_day_counts.sum()
        
        time_of_day_counts.plot(kind='bar', color='#9b59b6')
        plt.title(f'Time of Day Distribution of High-Virality Unresolved Questions (Total: {int(time_total)})', fontsize=16)
        plt.xlabel('Time of Day', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(rotation=45)
        
        # Add count labels
        for i, count in enumerate(time_of_day_counts):
            if count > 0:
                plt.text(i, count + 0.1, str(int(count)), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('charts/Resolution_Analysis/High_Virality_Unresolved/time_of_day_distribution.png', dpi=300)
        plt.close()
    
    # 4. Virality Score Distribution
    plt.figure(figsize=(12, 8))
    
    plt.hist(high_viral_unresolved['virality_combined'], bins=10, color='#e74c3c', alpha=0.7)
    plt.axvline(high_viral_unresolved['virality_combined'].mean(), color='k', linestyle='dashed', linewidth=1)
    
    plt.title(f'Virality Score Distribution of High-Virality Unresolved Questions (Total: {len(high_viral_unresolved)})', fontsize=16)
    plt.xlabel('Virality Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.text(
        high_viral_unresolved['virality_combined'].mean() + 1, 
        plt.ylim()[1] * 0.9, 
        f'Mean: {high_viral_unresolved["virality_combined"].mean():.2f}', 
        fontsize=12
    )
    
    plt.tight_layout()
    plt.savefig('charts/Resolution_Analysis/High_Virality_Unresolved/virality_distribution.png', dpi=300)
    plt.close()
    
    # 5. Message Count vs. Unique Senders Scatter Plot
    plt.figure(figsize=(12, 8))
    
    plt.scatter(
        high_viral_unresolved['message_count'],
        high_viral_unresolved['unique_senders'],
        c=high_viral_unresolved['virality_combined'],
        cmap='viridis',
        alpha=0.7,
        s=100
    )
    
    plt.colorbar(label='Virality Score')
    plt.title(f'Message Count vs. Unique Senders for High-Virality Unresolved Questions (Total: {len(high_viral_unresolved)})', fontsize=16)
    plt.xlabel('Message Count', fontsize=14)
    plt.ylabel('Unique Senders', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('charts/Resolution_Analysis/High_Virality_Unresolved/message_vs_senders.png', dpi=300)
    plt.close()
    
    # 6. Print top 10 highest virality unresolved questions
    print("\nTop 10 Highest Virality Unresolved Questions:")
    top_unresolved = high_viral_unresolved.sort_values('virality_combined', ascending=False).head(10)
    
    for i, (_, row) in enumerate(top_unresolved.iterrows(), 1):
        print(f"\n{i}. Thread ID: {row['thread_id']}")
        print(f"   Topic: {row['topic']}")
        print(f"   Category: {row['category_full']}")
        print(f"   Virality Score: {row['virality_combined']:.2f}")
        print(f"   Message Count: {row['message_count']}")
        print(f"   Unique Senders: {row['unique_senders']}")

def print_detailed_examples():
    """Print detailed examples of each resolution status type."""
    # Load the resolution analysis data
    try:
        df = pd.read_csv('whatsapp_resolution_analysis.csv')
    except FileNotFoundError:
        print("Resolution analysis file not found. Please run the analysis first.")
        return
    
    # Load the original messages
    try:
        messages_df = pd.read_csv('whatsapp_final_categorized.csv')
    except FileNotFoundError:
        print("Original messages file not found.")
        return
    
    # Group messages by thread_id
    thread_messages = defaultdict(list)
    for _, row in messages_df.iterrows():
        thread_messages[row['thread_id']].append({
            'sender': row['sender'],
            'message': row['message'],
            'datetime': pd.to_datetime(row['datetime'])
        })
    
    # Sort messages within each thread by datetime
    for thread_id in thread_messages:
        thread_messages[thread_id].sort(key=lambda x: x['datetime'])
    
    # Print detailed examples of each category
    print("\n===== DETAILED EXAMPLES OF EACH RESOLUTION STATUS =====")
    for status in ['Resolved', 'Unresolved', 'Likely Resolved', 'Likely Unresolved', 'Ambiguous', 'Not a Question']:
        examples = df[df['resolution_status'] == status].head(2)
        print(f"\n\n{'='*20} {status.upper()} EXAMPLES {'='*20}")
        
        for _, row in examples.iterrows():
            thread_id = row['thread_id']
            print(f"\nThread {thread_id}: {row['topic']}")
            print(f"Category: {row['category_full']}")
            print(f"Message count: {row['message_count']}, Unique senders: {row['unique_senders']}")
            print("\nMessages in thread:")
            
            # Print all messages in the thread
            for i, msg in enumerate(thread_messages[thread_id]):
                sender = msg['sender']
                # Truncate sender name for privacy
                if len(sender) > 3:
                    sender = sender[:3] + "..."
                
                message_text = msg['message']
                # Truncate very long messages
                if len(message_text) > 200:
                    message_text = message_text[:200] + "..."
                
                print(f"{i+1}. {sender}: {message_text}")
            
            print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description='Analyze resolution status of WhatsApp threads')
    parser.add_argument('--input', default='whatsapp_virality_scores.csv', help='Input CSV file')
    parser.add_argument('--output', default='whatsapp_resolution_analysis.csv', help='Output CSV file')
    parser.add_argument('--examples', action='store_true', help='Print detailed examples of each resolution status')
    parser.add_argument('--charts', action='store_true', help='Generate charts for resolution analysis')
    parser.add_argument('--high-virality', action='store_true', help='Analyze high-virality unresolved questions')
    args = parser.parse_args()
    
    if args.examples:
        print_detailed_examples()
    else:
        resolution_df, messages_df = analyze_threads(args.input, args.output)
        
        if args.charts or args.high_virality:
            generate_category_charts(resolution_df)
            generate_time_charts(resolution_df, messages_df)
            generate_virality_charts(resolution_df)
            
        if args.high_virality:
            analyze_high_virality_unresolved(resolution_df, messages_df)

if __name__ == "__main__":
    main()
