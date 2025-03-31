#!/usr/bin/env python3
"""
Generate charts for WhatsApp conversation analysis.

This script creates visualizations for:
1. Time-based analysis (by day, month, hour)
2. Category analysis (count and virality)
3. Subcategory analysis (count and virality)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from collections import defaultdict
import calendar

# Create necessary directories
os.makedirs('charts/Time', exist_ok=True)
os.makedirs('charts/Category', exist_ok=True)
os.makedirs('charts/Subcategory', exist_ok=True)

# Set plot style
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_data():
    """Load and prepare data for analysis"""
    print("Loading data...")
    
    # Load categorized data (for message-level analysis)
    categorized_df = pd.read_csv('whatsapp_final_categorized.csv')
    categorized_df['datetime'] = pd.to_datetime(categorized_df['datetime'])
    
    # Load virality scores data (for thread-level analysis)
    virality_df = pd.read_csv('whatsapp_virality_scores.csv')
    
    return categorized_df, virality_df

def generate_time_charts(df):
    """Generate time-based charts (day, month, hour)"""
    print("Generating time-based charts...")
    
    # Extract time components
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['month'] = df['datetime'].dt.month_name()
    df['hour'] = df['datetime'].dt.hour
    
    # Ensure proper ordering
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    months_order = list(calendar.month_name)[1:]  # Skip empty first element
    hours_order = list(range(24))
    
    # Get top 6 categories for stacking (plus "Other" for the rest)
    top_categories = df['topic_category'].value_counts().nlargest(6).index.tolist()
    df['plot_category'] = df['topic_category'].apply(lambda x: x if x in top_categories else 'Other')
    
    # Set up colors
    categories = top_categories + ['Other']
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    color_map = dict(zip(categories, colors))
    
    # 1. Messages by Day of Week
    plt.figure(figsize=(14, 8))
    day_counts = df.groupby(['day_of_week', 'plot_category']).size().unstack().fillna(0)
    # Reorder days
    day_counts = day_counts.reindex(days_order)
    day_counts.plot(kind='bar', stacked=True, color=[color_map[c] for c in day_counts.columns])
    plt.title('Average Count of Messages by Day of Week', fontsize=16)
    plt.xlabel('Day of Week', fontsize=14)
    plt.ylabel('Message Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('charts/Time/messages_by_day.png', dpi=300)
    plt.close()
    
    # 2. Messages by Month
    plt.figure(figsize=(14, 8))
    month_counts = df.groupby(['month', 'plot_category']).size().unstack().fillna(0)
    # Reorder months
    month_counts = month_counts.reindex(months_order)
    month_counts.plot(kind='bar', stacked=True, color=[color_map[c] for c in month_counts.columns])
    plt.title('Average Count of Messages by Month', fontsize=16)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Message Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('charts/Time/messages_by_month.png', dpi=300)
    plt.close()
    
    # 3. Messages by Hour
    plt.figure(figsize=(14, 8))
    # Create hour labels (0-1, 1-2, etc.)
    hour_labels = [f"{h}-{h+1}" for h in range(24)]
    
    hour_counts = df.groupby(['hour', 'plot_category']).size().unstack().fillna(0)
    # Ensure all hours are present
    for hour in range(24):
        if hour not in hour_counts.index:
            hour_counts.loc[hour] = 0
    hour_counts = hour_counts.sort_index()
    
    hour_counts.plot(kind='bar', stacked=True, color=[color_map[c] for c in hour_counts.columns])
    plt.title('Average Count of Messages by Hour', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=14)
    plt.ylabel('Message Count', fontsize=14)
    plt.xticks(range(24), hour_labels, rotation=90)
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('charts/Time/messages_by_hour.png', dpi=300)
    plt.close()
    
    print("Time-based charts generated successfully.")

def generate_category_charts(virality_df):
    """Generate category-based charts (count and virality)"""
    print("Generating category-based charts...")
    
    # Count of threads per category
    plt.figure(figsize=(14, 8))
    category_counts = virality_df['category'].value_counts().sort_values(ascending=False)
    
    # Filter out categories with very few threads (optional)
    category_counts = category_counts[category_counts >= 5]
    
    bars = category_counts.plot(kind='bar', color=plt.cm.tab10(np.linspace(0, 1, len(category_counts))))
    plt.title('Count of Threads per Category', fontsize=16)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Thread Count', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on top of bars
    for i, count in enumerate(category_counts):
        plt.text(i, count + 0.5, str(count), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/Category/threads_by_category.png', dpi=300)
    plt.close()
    
    # Average virality score per category
    plt.figure(figsize=(14, 8))
    virality_by_category = virality_df.groupby('category')['virality_combined'].mean().sort_values(ascending=False)
    
    # Filter out categories with very few threads (optional)
    virality_by_category = virality_by_category[virality_by_category.index.isin(category_counts.index)]
    
    bars = virality_by_category.plot(kind='bar', color=plt.cm.tab10(np.linspace(0, 1, len(virality_by_category))))
    plt.title('Average Virality Score per Category', fontsize=16)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Average Virality Score', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add score labels on top of bars
    for i, score in enumerate(virality_by_category):
        plt.text(i, score + 0.5, f"{score:.2f}", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/Category/virality_by_category.png', dpi=300)
    plt.close()
    
    print("Category-based charts generated successfully.")

def generate_subcategory_charts(virality_df):
    """Generate subcategory-based charts (count and virality)"""
    print("Generating subcategory-based charts...")
    
    # Create a color map where subcategories of the same category have the same color
    categories = virality_df['category'].unique()
    category_colors = dict(zip(categories, plt.cm.tab10(np.linspace(0, 1, len(categories)))))
    
    # Count of threads per subcategory
    plt.figure(figsize=(16, 10))
    
    # Group by subcategory and count, then sort
    subcategory_data = virality_df.groupby(['category', 'subcategory']).size().reset_index(name='count')
    subcategory_data = subcategory_data.sort_values('count', ascending=False)
    
    # Filter out subcategories with very few threads
    subcategory_data = subcategory_data[subcategory_data['count'] >= 3]
    
    # Create the plot
    bars = plt.bar(subcategory_data['subcategory'], subcategory_data['count'], 
                  color=[category_colors[cat] for cat in subcategory_data['category']])
    
    plt.title('Count of Threads per Subcategory', fontsize=16)
    plt.xlabel('Subcategory', fontsize=14)
    plt.ylabel('Thread Count', fontsize=14)
    plt.xticks(rotation=90)
    
    # Add count labels on top of bars
    for i, row in enumerate(subcategory_data.itertuples()):
        plt.text(i, row.count + 0.5, str(row.count), ha='center', fontweight='bold')
    
    # Create a legend mapping categories to colors
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=color, lw=4, label=cat) 
                      for cat, color in category_colors.items()]
    plt.legend(handles=legend_elements, title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('charts/Subcategory/threads_by_subcategory.png', dpi=300)
    plt.close()
    
    # Average virality score per subcategory
    plt.figure(figsize=(16, 10))
    
    # Group by subcategory and calculate mean virality
    virality_by_subcategory = virality_df.groupby(['category', 'subcategory'])['virality_combined'].mean().reset_index()
    virality_by_subcategory = virality_by_subcategory.sort_values('virality_combined', ascending=False)
    
    # Filter out subcategories with very few threads (using the same filter as above)
    subcategory_counts = subcategory_data.set_index('subcategory')['count']
    virality_by_subcategory = virality_by_subcategory[virality_by_subcategory['subcategory'].isin(subcategory_counts.index)]
    
    # Create the plot
    bars = plt.bar(virality_by_subcategory['subcategory'], virality_by_subcategory['virality_combined'], 
                  color=[category_colors[cat] for cat in virality_by_subcategory['category']])
    
    plt.title('Average Virality Score per Subcategory', fontsize=16)
    plt.xlabel('Subcategory', fontsize=14)
    plt.ylabel('Average Virality Score', fontsize=14)
    plt.xticks(rotation=90)
    
    # Add score labels on top of bars
    for i, row in enumerate(virality_by_subcategory.itertuples()):
        plt.text(i, row.virality_combined + 0.5, f"{row.virality_combined:.2f}", ha='center', fontweight='bold')
    
    # Create a legend mapping categories to colors
    legend_elements = [Line2D([0], [0], color=color, lw=4, label=cat) 
                      for cat, color in category_colors.items()]
    plt.legend(handles=legend_elements, title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('charts/Subcategory/virality_by_subcategory.png', dpi=300)
    plt.close()
    
    print("Subcategory-based charts generated successfully.")

def main():
    """Main function to generate all charts"""
    print("Starting chart generation...")
    
    # Load data
    categorized_df, virality_df = load_data()
    
    # Generate charts
    generate_time_charts(categorized_df)
    generate_category_charts(virality_df)
    generate_subcategory_charts(virality_df)
    
    print("All charts generated successfully!")

if __name__ == "__main__":
    main()
