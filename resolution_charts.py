#!/usr/bin/env python3
"""
Resolution Analysis Charts Generator

This script creates visualizations for analyzing resolution status across:
1. Categories and subcategories
2. Time dimensions (month, day of week, hour)
3. Virality correlation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import calendar
from datetime import datetime

# Create necessary directory
os.makedirs('charts/Resolution_Analysis', exist_ok=True)

# Set plot style
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_data():
    """Load and prepare data for analysis"""
    print("Loading data...")
    
    # Load resolution analysis data
    try:
        resolution_df = pd.read_csv('whatsapp_resolution_analysis.csv')
    except FileNotFoundError:
        print("Resolution analysis file not found. Please run resolution_analyzer.py first.")
        return None, None
    
    # Load the original messages for time-based analysis
    try:
        messages_df = pd.read_csv('whatsapp_final_categorized.csv')
        messages_df['datetime'] = pd.to_datetime(messages_df['datetime'])
    except FileNotFoundError:
        print("Original messages file not found.")
        return resolution_df, None
    
    return resolution_df, messages_df

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
    plt.figure(figsize=(12, 7))
    
    # Create virality bins instead of quartiles (simpler approach)
    # Define bins for virality scores
    bins = [0, 20, 40, 60, float('inf')]
    labels = ['Very Low (0-20)', 'Low (20-40)', 'Medium (40-60)', 'High (60+)']
    df['virality_level'] = pd.cut(df['virality_combined'], bins=bins, labels=labels)
    
    # Calculate resolution rate by virality level
    virality_resolution = pd.crosstab(
        df['virality_level'], 
        df['resolution_simple'],
        normalize='index'
    ) * 100
    
    # Plot stacked bar chart
    virality_resolution.plot(
        kind='bar', 
        stacked=True,
        color=['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    )
    
    plt.title('Resolution Rate by Virality Level', fontsize=16)
    plt.xlabel('Virality Level', fontsize=14)
    plt.ylabel('Percentage', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Resolution Status')
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

def main():
    """Main function to generate all resolution analysis charts"""
    print("Starting resolution analysis chart generation...")
    
    # Load data
    resolution_df, messages_df = load_data()
    
    if resolution_df is None:
        print("Error: Could not load necessary data.")
        return
    
    # Generate charts
    generate_category_charts(resolution_df)
    generate_time_charts(resolution_df, messages_df)
    generate_virality_charts(resolution_df)
    
    print("All resolution analysis charts generated successfully!")

if __name__ == "__main__":
    main()
