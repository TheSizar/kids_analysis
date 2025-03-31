# WhatsApp Conversation Analysis

A simple, streamlined toolkit for analyzing WhatsApp group conversations, with a focus on parenting discussions.

## Project Overview

This project provides tools to analyze WhatsApp chat exports, identify conversation topics, categorize them using a MECE (Mutually Exclusive and Collectively Exhaustive) framework, and visualize engagement patterns.

## Key Features

### WhatsApp Analyzer (`whatsapp_analyzer.py`)

The main script that processes WhatsApp chat data through a complete pipeline:

1. **Message Processing**: Extract useful information from raw messages
2. **Thread Identification**: Group related messages into conversations
3. **Topic Analysis**: Use GPT-4 to identify conversation topics
4. **MECE Categorization**: Organize topics into 12 main categories and 49 subcategories

### Virality Analyzer (`virality_analyzer.py`)

Calculates engagement metrics for conversation threads:

1. **Direct Virality Score**: Based on message count and response time
2. **Normalized Virality Score**: Balanced metric including participant count
3. **Combined Virality Score**: Weighted average of both approaches

### Chart Generator (`generate_charts.py`)

Creates visualizations to understand conversation patterns:

1. **Time-Based Analysis**: Message distribution by day, month, and hour
2. **Category Analysis**: Thread counts and virality by category
3. **Subcategory Analysis**: Detailed breakdown with color-coding by parent category

## MECE Categorization System

The project uses a comprehensive categorization system with 12 main categories:

1. Parenting Challenges & Support
2. Child Care & Education
3. Activities & Recreation
4. Products & Shopping
5. Marketplace & Exchange
6. Health & Wellness
7. Nutrition & Feeding
8. Local Community
9. Travel & Transportation
10. Group Communication
11. Home & Family Management
12. Uncategorized

Each main category contains multiple subcategories for detailed analysis.

## Usage

```bash
# Process WhatsApp data and categorize topics
python whatsapp_analyzer.py --mode full

# Calculate virality scores
python virality_analyzer.py

# Generate visualization charts
python generate_charts.py
```

## Requirements

- Python 3.6+
- OpenAI API key (for GPT-4 analysis)
- Required packages: pandas, matplotlib, nltk, openai

## Notes

- CSV data files are excluded from the repository due to privacy concerns
- The `.env` file should contain your OpenAI API key
