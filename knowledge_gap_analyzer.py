#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WhatsApp Knowledge Gap Analysis
Identifies expertise gaps in WhatsApp conversations by analyzing resolution rates
across different categories, times, and complexity levels.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import calendar

# Create output directories
os.makedirs('charts/Knowledge_Gap_Analysis', exist_ok=True)

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

def analyze_category_gaps(df):
    """Analyze resolution rates and identify gaps by category"""
    print("Analyzing category gaps...")
    
    # Filter out non-questions
    question_df = df[df['resolution_status'] != 'Not a Question'].copy()
    
    # Calculate resolution rates by category
    category_stats = {}
    
    # Group by category
    for category, group in question_df.groupby('category'):
        # Calculate metrics
        total_threads = len(group)
        resolved_threads = len(group[group['resolution_status'].isin(['Resolved', 'Likely Resolved'])])
        unresolved_threads = len(group[group['resolution_status'].isin(['Unresolved', 'Likely Unresolved'])])
        ambiguous_threads = len(group[group['resolution_status'] == 'Ambiguous'])
        
        # Calculate rates
        resolution_rate = (resolved_threads / total_threads) * 100 if total_threads > 0 else 0
        unresolved_rate = (unresolved_threads / total_threads) * 100 if total_threads > 0 else 0
        ambiguous_rate = (ambiguous_threads / total_threads) * 100 if total_threads > 0 else 0
        
        # Calculate average virality
        avg_virality = group['virality_combined'].mean()
        
        # Store results
        category_stats[category] = {
            'total_threads': total_threads,
            'resolved_threads': resolved_threads,
            'unresolved_threads': unresolved_threads,
            'ambiguous_threads': ambiguous_threads,
            'resolution_rate': resolution_rate,
            'unresolved_rate': unresolved_rate,
            'ambiguous_rate': ambiguous_rate,
            'avg_virality': avg_virality
        }
    
    # Convert to DataFrame
    category_df = pd.DataFrame.from_dict(category_stats, orient='index')
    
    # Calculate overall average resolution rate
    overall_resolution_rate = question_df[question_df['resolution_status'].isin(['Resolved', 'Likely Resolved'])].shape[0] / question_df.shape[0] * 100
    
    # Calculate gap score (how far below average)
    category_df['resolution_gap'] = overall_resolution_rate - category_df['resolution_rate']
    
    # Sort by gap score (largest gaps first)
    category_df = category_df.sort_values('resolution_gap', ascending=False)
    
    return category_df

def analyze_subcategory_gaps(df):
    """Analyze resolution rates and identify gaps by subcategory"""
    print("Analyzing subcategory gaps...")
    
    # Filter out non-questions
    question_df = df[df['resolution_status'] != 'Not a Question'].copy()
    
    # Calculate resolution rates by subcategory
    subcategory_stats = {}
    
    # Group by category and subcategory
    for (category, subcategory), group in question_df.groupby(['category', 'subcategory']):
        # Calculate metrics
        total_threads = len(group)
        resolved_threads = len(group[group['resolution_status'].isin(['Resolved', 'Likely Resolved'])])
        unresolved_threads = len(group[group['resolution_status'].isin(['Unresolved', 'Likely Unresolved'])])
        
        # Calculate rates
        resolution_rate = (resolved_threads / total_threads) * 100 if total_threads > 0 else 0
        
        # Only include subcategories with at least 5 threads
        if total_threads >= 5:
            # Store results
            subcategory_stats[f"{category} - {subcategory}"] = {
                'category': category,
                'subcategory': subcategory,
                'total_threads': total_threads,
                'resolved_threads': resolved_threads,
                'unresolved_threads': unresolved_threads,
                'resolution_rate': resolution_rate
            }
    
    # Convert to DataFrame
    subcategory_df = pd.DataFrame.from_dict(subcategory_stats, orient='index')
    
    # Calculate overall average resolution rate
    overall_resolution_rate = question_df[question_df['resolution_status'].isin(['Resolved', 'Likely Resolved'])].shape[0] / question_df.shape[0] * 100
    
    # Calculate gap score (how far below average)
    subcategory_df['resolution_gap'] = overall_resolution_rate - subcategory_df['resolution_rate']
    
    # Sort by gap score (largest gaps first)
    subcategory_df = subcategory_df.sort_values('resolution_gap', ascending=False)
    
    return subcategory_df

def analyze_time_gaps(df, messages_df):
    """Analyze resolution rates by time of day and day of week"""
    print("Analyzing time-based gaps...")
    
    # Filter out non-questions
    question_df = df[df['resolution_status'] != 'Not a Question'].copy()
    
    # Use first_message_time from resolution data
    if 'first_message_time' in question_df.columns:
        # Convert to datetime if it's not already
        question_df['first_message_time'] = pd.to_datetime(question_df['first_message_time'])
        
        # Extract time components
        question_df['hour'] = question_df['first_message_time'].dt.hour
        question_df['day_of_week'] = question_df['first_message_time'].dt.day_name()
        
        # Create time of day categories
        question_df['time_of_day'] = pd.cut(
            question_df['hour'], 
            bins=[0, 6, 12, 18, 24], 
            labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)']
        )
        
        # Analyze by time of day
        time_of_day_stats = {}
        for time_of_day, group in question_df.groupby('time_of_day'):
            total_threads = len(group)
            resolved_threads = len(group[group['resolution_status'].isin(['Resolved', 'Likely Resolved'])])
            
            resolution_rate = (resolved_threads / total_threads) * 100 if total_threads > 0 else 0
            
            time_of_day_stats[time_of_day] = {
                'total_threads': total_threads,
                'resolved_threads': resolved_threads,
                'resolution_rate': resolution_rate
            }
        
        # Analyze by day of week
        day_of_week_stats = {}
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for day in days_order:
            group = question_df[question_df['day_of_week'] == day]
            
            total_threads = len(group)
            resolved_threads = len(group[group['resolution_status'].isin(['Resolved', 'Likely Resolved'])])
            
            resolution_rate = (resolved_threads / total_threads) * 100 if total_threads > 0 else 0
            
            day_of_week_stats[day] = {
                'total_threads': total_threads,
                'resolved_threads': resolved_threads,
                'resolution_rate': resolution_rate
            }
        
        # Convert to DataFrames
        time_of_day_df = pd.DataFrame.from_dict(time_of_day_stats, orient='index')
        day_of_week_df = pd.DataFrame.from_dict(day_of_week_stats, orient='index')
        
        # Calculate overall average resolution rate
        overall_resolution_rate = question_df[question_df['resolution_status'].isin(['Resolved', 'Likely Resolved'])].shape[0] / question_df.shape[0] * 100
        
        # Calculate gap scores
        time_of_day_df['resolution_gap'] = overall_resolution_rate - time_of_day_df['resolution_rate']
        day_of_week_df['resolution_gap'] = overall_resolution_rate - day_of_week_df['resolution_rate']
        
        return {
            'time_of_day': time_of_day_df,
            'day_of_week': day_of_week_df
        }
    else:
        print("No timestamp data available in resolution data. Skipping time gap analysis.")
        return None

def visualize_category_gaps(category_df):
    """Create visualizations for category gaps"""
    print("Generating category gap visualizations...")
    
    # 1. Resolution Rate by Category
    plt.figure(figsize=(14, 10))
    
    # Sort by resolution rate
    sorted_df = category_df.sort_values('resolution_rate')
    
    # Create color map based on resolution rate (lower is red, higher is green)
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(sorted_df)))
    
    # Plot horizontal bar chart
    ax = sorted_df['resolution_rate'].plot(kind='barh', color=colors)
    plt.title('Resolution Rate by Category', fontsize=16)
    plt.xlabel('Resolution Rate (%)', fontsize=14)
    plt.ylabel('Category', fontsize=14)
    
    # Add value labels with thread counts
    for i, (value, count) in enumerate(zip(sorted_df['resolution_rate'], sorted_df['total_threads'])):
        ax.text(value + 1, i, f"{value:.1f}%", va='center', fontweight='bold')
        ax.text(sorted_df['resolution_rate'].max() + 5, i, f"n={int(count)}", va='center', ha='left')
    
    # Add average line
    overall_avg = sorted_df['resolution_rate'].mean()
    plt.axvline(x=overall_avg, color='red', linestyle='--', label=f'Average: {overall_avg:.1f}%')
    plt.legend()
    
    # Adjust x-axis to make room for thread counts
    ax.set_xlim(0, sorted_df['resolution_rate'].max() + 15)
    
    plt.tight_layout()
    plt.savefig('charts/Knowledge_Gap_Analysis/resolution_rate_by_category.png', dpi=300)
    plt.close()
    
    # 2. Resolution Gap by Category
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Sort by gap size
    sorted_df = category_df.sort_values('resolution_gap', ascending=False)
    
    # Create color map based on gap size (larger gap is red, smaller gap is green)
    # Use a direct mapping where the highest gap gets the reddest color
    norm = plt.Normalize(sorted_df['resolution_gap'].min(), sorted_df['resolution_gap'].max())
    colors = plt.cm.RdYlGn_r(norm(sorted_df['resolution_gap']))
    
    # Plot horizontal bar chart with explicit colors
    bars = ax.barh(range(len(sorted_df)), sorted_df['resolution_gap'], color=colors)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df.index)
    ax.set_title('Knowledge Gap by Category (Higher = Worse Performance)', fontsize=16)
    ax.set_xlabel('Resolution Gap (%)', fontsize=14)
    ax.set_ylabel('Category', fontsize=14)
    
    # Add value labels with thread counts on the right side
    for i, (value, count) in enumerate(zip(sorted_df['resolution_gap'], sorted_df['total_threads'])):
        ax.text(value + 0.5, i, f"{value:.1f}%", va='center', fontweight='bold')
        ax.text(sorted_df['resolution_gap'].max() + 2, i, f"n={int(count)}", va='center', ha='left')
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=norm)
    sm.set_array(sorted_df['resolution_gap'])
    fig.colorbar(sm, ax=ax, label='Gap Size (Red = Worse)')
    
    # Adjust x-axis to make room for thread counts
    ax.set_xlim(0, sorted_df['resolution_gap'].max() + 8)
    
    plt.tight_layout()
    plt.savefig('charts/Knowledge_Gap_Analysis/knowledge_gap_by_category.png', dpi=300)
    plt.close()
    
    # 3. Thread Volume vs. Resolution Rate
    plt.figure(figsize=(12, 10))
    
    plt.scatter(
        category_df['total_threads'],
        category_df['resolution_rate'],
        s=category_df['total_threads'] * 5,  # Size based on thread count
        alpha=0.7,
        c=category_df['resolution_gap'],  # Color based on gap
        cmap='RdYlGn_r'  # Red for larger gaps, green for smaller gaps
    )
    
    # Add category labels
    for idx, row in category_df.iterrows():
        plt.annotate(
            idx,
            (row['total_threads'], row['resolution_rate']),
            fontsize=9,
            ha='center'
        )
    
    plt.colorbar(label='Resolution Gap (%) - Higher = Worse')
    plt.title('Thread Volume vs. Resolution Rate by Category', fontsize=16)
    plt.xlabel('Number of Threads', fontsize=14)
    plt.ylabel('Resolution Rate (%)', fontsize=14)
    
    # Add average line
    plt.axhline(y=category_df['resolution_rate'].mean(), color='red', linestyle='--', 
                label=f'Avg Resolution: {category_df["resolution_rate"].mean():.1f}%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('charts/Knowledge_Gap_Analysis/volume_vs_resolution_by_category.png', dpi=300)
    plt.close()

def visualize_subcategory_gaps(subcategory_df):
    """Create visualizations for subcategory gaps"""
    print("Generating subcategory gap visualizations...")
    
    # 1. Top 15 Knowledge Gaps by Subcategory
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Sort by gap size and get top 15
    top_gaps = subcategory_df.sort_values('resolution_gap', ascending=False).head(15)
    
    # Create color map based on gap size (larger gap is red, smaller gap is green)
    norm = plt.Normalize(top_gaps['resolution_gap'].min(), top_gaps['resolution_gap'].max())
    colors = plt.cm.RdYlGn_r(norm(top_gaps['resolution_gap']))
    
    # Plot horizontal bar chart with explicit colors
    bars = ax.barh(range(len(top_gaps)), top_gaps['resolution_gap'], color=colors)
    ax.set_yticks(range(len(top_gaps)))
    ax.set_yticklabels(top_gaps.index)
    ax.set_title('Top 15 Knowledge Gaps by Subcategory (Higher = Worse Performance)', fontsize=16)
    ax.set_xlabel('Resolution Gap (%)', fontsize=14)
    ax.set_ylabel('Subcategory', fontsize=14)
    
    # Add value labels with thread counts on the right side
    for i, (value, count) in enumerate(zip(top_gaps['resolution_gap'], top_gaps['total_threads'])):
        ax.text(value + 0.5, i, f"{value:.1f}%", va='center', fontweight='bold')
        ax.text(top_gaps['resolution_gap'].max() + 2, i, f"n={int(count)}", va='center', ha='left')
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=norm)
    sm.set_array(top_gaps['resolution_gap'])
    fig.colorbar(sm, ax=ax, label='Gap Size (Red = Worse)')
    
    # Adjust x-axis to make room for thread counts
    ax.set_xlim(0, top_gaps['resolution_gap'].max() + 8)
    
    plt.tight_layout()
    plt.savefig('charts/Knowledge_Gap_Analysis/top_subcategory_gaps.png', dpi=300)
    plt.close()
    
    # 2. Resolution Rate vs. Thread Volume for Subcategories
    plt.figure(figsize=(16, 12))
    
    # Create scatter plot
    scatter = plt.scatter(
        subcategory_df['total_threads'],
        subcategory_df['resolution_rate'],
        s=subcategory_df['total_threads'] * 3,  # Size based on thread count
        alpha=0.7,
        c=subcategory_df['resolution_gap'],  # Color based on gap
        cmap='RdYlGn_r'  # Red for larger gaps, green for smaller gaps
    )
    
    # Add labels for all subcategories
    for idx, row in subcategory_df.iterrows():
        # Create shorter labels by removing the category prefix
        short_label = idx.split(' - ')[1] if ' - ' in idx else idx
        
        # Add annotation with background for better visibility
        plt.annotate(
            short_label,
            (row['total_threads'], row['resolution_rate']),
            fontsize=8,
            ha='center',
            va='center',
            xytext=(5, 0),
            textcoords='offset points',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    
    plt.colorbar(scatter, label='Resolution Gap (%) - Higher = Worse')
    plt.title('Resolution Rate vs. Thread Volume by Subcategory', fontsize=16)
    plt.xlabel('Number of Threads', fontsize=14)
    plt.ylabel('Resolution Rate (%)', fontsize=14)
    
    # Add average line
    plt.axhline(y=subcategory_df['resolution_rate'].mean(), color='red', linestyle='--', 
                label=f'Avg Resolution: {subcategory_df["resolution_rate"].mean():.1f}%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('charts/Knowledge_Gap_Analysis/resolution_vs_volume_subcategory.png', dpi=300)
    plt.close()
    
    # 3. Resolution Heatmap by Category and Subcategory
    # Create a pivot table to get resolution rates for each category-subcategory pair
    # First, ensure we have all category-subcategory combinations
    all_categories = subcategory_df.index.str.split(' - ').str[0].unique()
    
    # Create a new DataFrame to hold the heatmap data
    heatmap_data = []
    
    for idx, row in subcategory_df.iterrows():
        if ' - ' in idx:
            category, subcategory = idx.split(' - ', 1)
        else:
            category, subcategory = idx, 'General'
        
        heatmap_data.append({
            'Category': category,
            'Subcategory': subcategory,
            'Resolution Rate': row['resolution_rate'],
            'Thread Count': row['total_threads']
        })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    
    # Create a pivot table
    pivot_data = heatmap_df.pivot_table(
        values='Resolution Rate',
        index='Category',
        columns='Subcategory',
        aggfunc='mean'
    )
    
    # Create thread count pivot for annotations
    thread_pivot = heatmap_df.pivot_table(
        values='Thread Count',
        index='Category',
        columns='Subcategory',
        aggfunc='sum'
    )
    
    # Create a mask for cells with zero threads (N/A values)
    mask = thread_pivot == 0
    
    # Plot the heatmap with masked values
    plt.figure(figsize=(16, 12))
    
    # Create a custom colormap with gray for masked values
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad('lightgray')
    
    # Plot heatmap with mask
    ax = sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.1f',
        cmap=cmap,
        mask=mask,  # Mask cells with no data
        linewidths=0.5,
        cbar_kws={'label': 'Resolution Rate (%)'}
    )
    
    # Add thread count annotations only for non-masked cells
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            try:
                count = thread_pivot.iloc[i, j]
                if count > 0:  # Only add thread count for cells with data
                    ax.text(j + 0.5, i + 0.85, f"n={int(count)}", 
                            ha='center', va='center', color='black', fontsize=8)
            except (IndexError, KeyError):
                continue
    
    plt.title('Resolution Rate by Category and Subcategory', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('charts/Knowledge_Gap_Analysis/resolution_heatmap.png', dpi=300)
    plt.close()

def visualize_time_gaps(time_gaps):
    """Create visualizations for time-based gaps"""
    print("Generating time-based gap visualizations...")
    
    if time_gaps is None:
        print("No time gap data available. Skipping visualizations.")
        return
    
    # 1. Resolution Rate by Time of Day
    plt.figure(figsize=(12, 8))
    
    time_of_day_df = time_gaps['time_of_day']
    
    # Plot bar chart
    ax = time_of_day_df['resolution_rate'].plot(kind='bar', color='skyblue')
    plt.title('Resolution Rate by Time of Day', fontsize=16)
    plt.xlabel('Time of Day', fontsize=14)
    plt.ylabel('Resolution Rate (%)', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add value labels
    for i, value in enumerate(time_of_day_df['resolution_rate']):
        ax.text(i, value + 1, f"{value:.1f}%", ha='center', fontweight='bold')
    
    # Add thread count labels
    for i, count in enumerate(time_of_day_df['total_threads']):
        ax.text(i, 5, f"n={count}", ha='center', fontsize=10)
    
    # Add average line
    overall_avg = time_of_day_df['resolution_rate'].mean()
    plt.axhline(y=overall_avg, color='red', linestyle='--', label=f'Average: {overall_avg:.1f}%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('charts/Knowledge_Gap_Analysis/resolution_by_time_of_day.png', dpi=300)
    plt.close()
    
    # 2. Resolution Rate by Day of Week
    plt.figure(figsize=(14, 8))
    
    day_of_week_df = time_gaps['day_of_week']
    
    # Plot bar chart
    ax = day_of_week_df['resolution_rate'].plot(kind='bar', color='lightgreen')
    plt.title('Resolution Rate by Day of Week', fontsize=16)
    plt.xlabel('Day of Week', fontsize=14)
    plt.ylabel('Resolution Rate (%)', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add value labels
    for i, value in enumerate(day_of_week_df['resolution_rate']):
        ax.text(i, value + 1, f"{value:.1f}%", ha='center', fontweight='bold')
    
    # Add thread count labels
    for i, count in enumerate(day_of_week_df['total_threads']):
        ax.text(i, 5, f"n={count}", ha='center', fontsize=10)
    
    # Add average line
    overall_avg = day_of_week_df['resolution_rate'].mean()
    plt.axhline(y=overall_avg, color='red', linestyle='--', label=f'Average: {overall_avg:.1f}%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('charts/Knowledge_Gap_Analysis/resolution_by_day_of_week.png', dpi=300)
    plt.close()
    
    # 3. Knowledge Gap by Time (Combined View)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Sort time of day by gap
    time_sorted = time_of_day_df.sort_values('resolution_gap', ascending=False)
    
    # Create direct color mapping for time of day
    time_norm = plt.Normalize(time_sorted['resolution_gap'].min(), time_sorted['resolution_gap'].max())
    time_colors = plt.cm.RdYlGn_r(time_norm(time_sorted['resolution_gap']))
    
    # Time of day plot with explicit colors
    bars1 = ax1.bar(range(len(time_sorted)), time_sorted['resolution_gap'], color=time_colors)
    ax1.set_xticks(range(len(time_sorted)))
    ax1.set_xticklabels(time_sorted.index)
    ax1.set_title('Knowledge Gap by Time of Day (Higher = Worse Performance)', fontsize=14)
    ax1.set_ylabel('Resolution Gap (%)', fontsize=12)
    
    # Add value labels
    for i, value in enumerate(time_sorted['resolution_gap']):
        ax1.text(i, value + 1, f"{value:.1f}%", ha='center', fontweight='bold')
    
    # Sort day of week by gap
    day_sorted = day_of_week_df.sort_values('resolution_gap', ascending=False)
    
    # Create direct color mapping for day of week
    day_norm = plt.Normalize(day_sorted['resolution_gap'].min(), day_sorted['resolution_gap'].max())
    day_colors = plt.cm.RdYlGn_r(day_norm(day_sorted['resolution_gap']))
    
    # Day of week plot with explicit colors
    bars2 = ax2.bar(range(len(day_sorted)), day_sorted['resolution_gap'], color=day_colors)
    ax2.set_xticks(range(len(day_sorted)))
    ax2.set_xticklabels(day_sorted.index)
    ax2.set_title('Knowledge Gap by Day of Week (Higher = Worse Performance)', fontsize=14)
    ax2.set_ylabel('Resolution Gap (%)', fontsize=12)
    
    # Add value labels
    for i, value in enumerate(day_sorted['resolution_gap']):
        ax2.text(i, value + 1, f"{value:.1f}%", ha='center', fontweight='bold')
    
    # Add colorbars
    sm1 = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=time_norm)
    sm1.set_array(time_sorted['resolution_gap'])
    fig.colorbar(sm1, ax=ax1, label='Gap Size (Red = Worse)')
    
    sm2 = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=day_norm)
    sm2.set_array(day_sorted['resolution_gap'])
    fig.colorbar(sm2, ax=ax2, label='Gap Size (Red = Worse)')
    
    plt.tight_layout()
    plt.savefig('charts/Knowledge_Gap_Analysis/knowledge_gap_by_time.png', dpi=300)
    plt.close()

def generate_thread_list_for_top_gaps(df, subcategory_df):
    """Generate a list of threads for the top 15 knowledge gaps by subcategory"""
    print("Generating thread list for top knowledge gaps...")
    
    # Get top 15 subcategories with largest gaps
    top_gaps = subcategory_df.sort_values('resolution_gap', ascending=False).head(15)
    
    # Create a dictionary to store threads for each subcategory
    subcategory_threads = {}
    
    # For each top gap subcategory
    for subcategory in top_gaps.index:
        # Split into category and subcategory
        if ' - ' in subcategory:
            category, sub = subcategory.split(' - ', 1)
        else:
            category, sub = subcategory, ''
        
        # Filter threads for this category and subcategory
        if sub:
            filtered_threads = df[(df['category'] == category) & 
                                 (df['subcategory'] == sub) & 
                                 (df['resolution_status'].isin(['Unresolved', 'Likely Unresolved']))]
        else:
            filtered_threads = df[(df['category'] == category) & 
                                 (df['resolution_status'].isin(['Unresolved', 'Likely Unresolved']))]
        
        # Store threads
        subcategory_threads[subcategory] = filtered_threads[['thread_id', 'topic', 'virality_combined', 'resolution_status']]
    
    # Create output file
    with open('top_knowledge_gap_threads.txt', 'w') as f:
        f.write("THREADS FOR TOP 15 KNOWLEDGE GAPS BY SUBCATEGORY\n")
        f.write("=================================================\n\n")
        
        for subcategory, threads in subcategory_threads.items():
            f.write(f"{subcategory} (Gap: {top_gaps.loc[subcategory, 'resolution_gap']:.1f}%, Resolution Rate: {top_gaps.loc[subcategory, 'resolution_rate']:.1f}%)\n")
            f.write("-" * len(subcategory) + "\n")
            
            if len(threads) == 0:
                f.write("No unresolved threads found.\n\n")
                continue
            
            # Sort by virality
            threads = threads.sort_values('virality_combined', ascending=False)
            
            for _, thread in threads.iterrows():
                f.write(f"Thread {thread['thread_id']}: {thread['topic']} (Virality: {thread['virality_combined']:.2f})\n")
            
            f.write("\n")
    
    print(f"Thread list saved to top_knowledge_gap_threads.txt")

def print_key_insights(category_df, subcategory_df, time_gaps):
    """Print key insights from the knowledge gap analysis"""
    print("\n===== KEY INSIGHTS FROM KNOWLEDGE GAP ANALYSIS =====\n")
    
    # 1. Top category gaps
    print("Top 5 Category Knowledge Gaps:")
    for category, row in category_df.sort_values('resolution_gap', ascending=False).head(5).iterrows():
        print(f"  {category}: {row['resolution_gap']:.1f}% gap ({row['resolution_rate']:.1f}% resolution rate, {row['total_threads']} threads)")
    
    # 2. Top subcategory gaps
    print("\nTop 5 Subcategory Knowledge Gaps:")
    for subcategory, row in subcategory_df.sort_values('resolution_gap', ascending=False).head(5).iterrows():
        print(f"  {subcategory}: {row['resolution_gap']:.1f}% gap ({row['resolution_rate']:.1f}% resolution rate, {row['total_threads']} threads)")
    
    # 3. Time-based insights
    if time_gaps is not None:
        print("\nTime-Based Knowledge Gaps:")
        
        # Time of day
        worst_time = time_gaps['time_of_day']['resolution_gap'].idxmax()
        worst_time_gap = time_gaps['time_of_day'].loc[worst_time, 'resolution_gap']
        worst_time_rate = time_gaps['time_of_day'].loc[worst_time, 'resolution_rate']
        
        print(f"  Worst time of day: {worst_time} ({worst_time_gap:.1f}% gap, {worst_time_rate:.1f}% resolution rate)")
        
        # Day of week
        worst_day = time_gaps['day_of_week']['resolution_gap'].idxmax()
        worst_day_gap = time_gaps['day_of_week'].loc[worst_day, 'resolution_gap']
        worst_day_rate = time_gaps['day_of_week'].loc[worst_day, 'resolution_rate']
        
        print(f"  Worst day of week: {worst_day} ({worst_day_gap:.1f}% gap, {worst_day_rate:.1f}% resolution rate)")
    
    # 4. Overall summary
    print("\nOverall Knowledge Gap Summary:")
    
    # Calculate average resolution rate
    avg_resolution = category_df['resolution_rate'].mean()
    
    # Calculate weighted average resolution rate (by thread count)
    weighted_avg = (category_df['resolution_rate'] * category_df['total_threads']).sum() / category_df['total_threads'].sum()
    
    print(f"  Average resolution rate across categories: {avg_resolution:.1f}%")
    print(f"  Weighted average resolution rate: {weighted_avg:.1f}%")
    
    # Identify categories with both high volume and low resolution
    critical_categories = category_df[(category_df['total_threads'] > category_df['total_threads'].median()) & 
                                     (category_df['resolution_rate'] < avg_resolution)]
    
    print(f"\nCritical knowledge gap areas (high volume + low resolution):")
    for category, row in critical_categories.iterrows():
        print(f"  {category}: {row['resolution_gap']:.1f}% gap, {row['total_threads']} threads")

def main():
    # Load data
    resolution_df, messages_df = load_data()
    
    # Perform analyses
    category_gaps = analyze_category_gaps(resolution_df)
    subcategory_gaps = analyze_subcategory_gaps(resolution_df)
    time_gaps = analyze_time_gaps(resolution_df, messages_df)
    
    # Generate visualizations
    visualize_category_gaps(category_gaps)
    visualize_subcategory_gaps(subcategory_gaps)
    visualize_time_gaps(time_gaps)
    
    # Generate thread list for top knowledge gaps
    generate_thread_list_for_top_gaps(resolution_df, subcategory_gaps)
    
    # Print insights
    print_key_insights(category_gaps, subcategory_gaps, time_gaps)

if __name__ == "__main__":
    main()
