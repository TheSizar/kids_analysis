import pandas as pd
import os
import time
import re
from datetime import timedelta
from collections import defaultdict, Counter
from tqdm import tqdm
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    print("You can create a .env file with the content: OPENAI_API_KEY=your_api_key_here")
    exit(1)

# Set up OpenAI client
client = openai.OpenAI(api_key=api_key)

# Common English stopwords and less useful words
stop_words = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'with', 'on', 'at', 'from', 
              'by', 'about', 'as', 'an', 'are', 'was', 'were', 'be', 'been', 'being', 
              'have', 'has', 'had', 'do', 'does', 'did', 'but', 'or', 'if', 'because', 
              'as', 'until', 'while', 'that', 'this', 'these', 'those', 'then', 'than',
              'so', 'not', 'can', 'will', 'just', 'should', 'now', 'also', 'very', 'our',
              'your', 'my', 'their', 'his', 'her', 'its', 'they', 'them', 'she', 'he', 'it',
              'you', 'me', 'we', 'us', 'who', 'what', 'when', 'where', 'why', 'how',
              'thank', 'thanks', 'would', 'could', 'may', 'might', 'must', 'shall', 'here',
              'there', 'one', 'two', 'three', 'four', 'five', 'first', 'second', 'third',
              'com', 'www', 'http', 'https', 'html', 'php', 'org', 'net', 'edu', 'gov',
              'going', 'know', 'like', 'want', 'need', 'get', 'got', 'getting', 'anyone',
              'someone', 'everyone', 'anybody', 'somebody', 'everybody', 'any', 'some', 'every',
              'say', 'said', 'saying', 'says', 'tell', 'told', 'telling', 'tells',
              'think', 'thought', 'thinking', 'thinks', 'see', 'saw', 'seeing', 'sees',
              'use', 'used', 'using', 'uses', 'make', 'made', 'making', 'makes'}

# ===== PART 1: MESSAGE PROCESSING FUNCTIONS =====

def extract_useful_tokens(message):
    """Extract useful tokens from a message, removing stopwords and punctuation."""
    if not isinstance(message, str):
        return []
    
    # Convert to lowercase and split into words
    tokens = re.findall(r'\b\w+\b', message.lower())
    
    # Remove stopwords
    useful_tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    return useful_tokens

def is_question(message):
    """Check if a message is a question."""
    if not isinstance(message, str):
        return False
    
    # Check for question marks
    if '?' in message:
        return True
    
    # Check for question words at the beginning
    question_starters = ['who', 'what', 'when', 'where', 'why', 'how', 'is', 'are', 'was', 'were', 
                         'will', 'would', 'can', 'could', 'should', 'do', 'does', 'did', 'have', 'has', 'had']
    
    first_word = message.lower().split()[0] if message.split() else ""
    
    return first_word in question_starters

def identify_threads(df, time_threshold_minutes=30):
    """Identify conversation threads based on timing."""
    print("Identifying conversation threads...")
    
    # Sort by datetime
    df = df.sort_values('datetime')
    
    # Initialize thread tracking
    current_thread_id = 1
    last_message_time = None
    
    # Add thread_id column if it doesn't exist
    if 'thread_id' not in df.columns:
        df['thread_id'] = 0
    
    # Iterate through messages
    for idx, row in df.iterrows():
        current_time = pd.to_datetime(row['datetime'])
        
        # Check if this is the first message or if the time gap is large
        if last_message_time is None or (current_time - last_message_time).total_seconds() / 60 > time_threshold_minutes:
            current_thread_id += 1
        
        # Assign thread ID
        df.at[idx, 'thread_id'] = current_thread_id
        
        # Update last message time
        last_message_time = current_time
    
    print(f"Identified {current_thread_id} conversation threads.")
    return df

# ===== PART 2: GPT-4 ANALYSIS FUNCTIONS =====

def analyze_thread_with_openai(messages, thread_id):
    """Analyze a thread using OpenAI's GPT-4 to identify the main topic."""
    # Construct the prompt
    prompt = f"Below is a WhatsApp conversation thread. Identify the main topic of this conversation in a concise phrase (5-7 words max).\n\n"
    
    for msg in messages:
        sender = msg['sender']
        message = msg['message']
        prompt += f"{sender}: {message}\n"
    
    prompt += "\nMain topic (5-7 words max): "
    
    try:
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that identifies the main topic of conversations."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the topic from the response
        topic = response.choices[0].message.content.strip()
        
        # Remove quotes if present
        topic = topic.strip('"\'')
        
        return topic
    except Exception as e:
        print(f"Error analyzing thread {thread_id}: {e}")
        return "Error: Could not analyze"

def analyze_threads_with_openai(df, max_threads=None, skip_short=True, save_interval=10):
    """Analyze all threads using OpenAI's GPT-4."""
    print("Analyzing conversation threads with GPT-4...")
    
    # Add topic column if it doesn't exist
    if 'gpt4_topic' not in df.columns:
        df['gpt4_topic'] = None
    
    # Get unique thread IDs
    thread_ids = df['thread_id'].unique()
    
    # Limit the number of threads to analyze if specified
    if max_threads:
        thread_ids = thread_ids[:max_threads]
    
    # Create output file if it doesn't exist
    output_file = 'whatsapp_gpt4_analyzed.csv'
    if not os.path.exists(output_file):
        df.to_csv(output_file, index=False)
    else:
        # Load existing analysis to avoid re-analyzing threads
        existing_df = pd.read_csv(output_file)
        analyzed_threads = existing_df.dropna(subset=['gpt4_topic'])['thread_id'].unique()
        print(f"Found {len(analyzed_threads)} already analyzed threads.")
        
        # Update the dataframe with existing analysis
        for thread_id in analyzed_threads:
            thread_topic = existing_df[existing_df['thread_id'] == thread_id]['gpt4_topic'].iloc[0]
            df.loc[df['thread_id'] == thread_id, 'gpt4_topic'] = thread_topic
        
        # Filter out already analyzed threads
        thread_ids = [tid for tid in thread_ids if tid not in analyzed_threads]
    
    print(f"Analyzing {len(thread_ids)} threads...")
    
    # Track progress
    threads_analyzed = 0
    last_save_time = time.time()
    
    # Analyze each thread
    for thread_id in tqdm(thread_ids):
        # Get messages for this thread
        thread_messages = df[df['thread_id'] == thread_id].copy()
        
        # Skip very short threads if requested
        if skip_short and len(thread_messages) < 2:
            continue
        
        # Prepare messages for analysis
        messages = []
        for _, row in thread_messages.iterrows():
            messages.append({
                'sender': row['sender'],
                'message': row['message']
            })
        
        # Analyze the thread
        topic = analyze_thread_with_openai(messages, thread_id)
        
        # Update the dataframe with the topic
        df.loc[df['thread_id'] == thread_id, 'gpt4_topic'] = topic
        
        # Increment counter
        threads_analyzed += 1
        
        # Save periodically
        current_time = time.time()
        if threads_analyzed % save_interval == 0 or (current_time - last_save_time) > 300:  # Save every 5 minutes or after save_interval threads
            print(f"Saving progress after analyzing {threads_analyzed} threads...")
            df.to_csv(output_file, index=False)
            last_save_time = current_time
    
    # Save the final results
    print(f"Analysis complete. Analyzed {threads_analyzed} threads.")
    df.to_csv(output_file, index=False)
    
    return df

# ===== PART 3: MECE CATEGORIZATION FUNCTIONS =====

def categorize_topics(input_file='whatsapp_gpt4_analyzed.csv', output_file='whatsapp_final_categorized.csv'):
    """Categorize topics using the MECE framework."""
    print("Categorizing topics using MECE framework...")
    
    # Load the analyzed data
    df = pd.read_csv(input_file)
    
    # Define MECE categories and subcategories
    categories = {
        'Child Care & Education': {
            'Daycare & Preschools': ['daycare', 'preschool', 'childcare', 'child care', 'early education', 'montessori', 'nursery'],
            'Babysitters & Nannies': ['babysit', 'nanny', 'au pair', 'caregiver', 'sitter'],
            'Schools & Education': ['school', 'education', 'teacher', 'class', 'learning', 'tutor', 'homework', 'academic'],
            'Child Development & Milestones': ['development', 'milestone', 'skill', 'growth', 'cognitive', 'motor skills', 'language development', 'transition', 'phase', 'stage']
        },
        
        'Health & Wellness': {
            'Medical Care & Providers': ['doctor', 'pediatrician', 'hospital', 'medical', 'prescription', 'clinic', 'specialist', 'appointment', 'urgent care', 'emergency'],
            'Illness & Symptoms': ['sick', 'illness', 'fever', 'cold', 'flu', 'rash', 'cough', 'infection', 'virus', 'symptom'],
            'Dental & Vision Care': ['dental', 'dentist', 'teeth', 'orthodontist', 'braces', 'eye', 'vision', 'optometrist'],
            'Mental & Behavioral Health': ['therapy', 'counseling', 'mental health', 'anxiety', 'depression', 'stress', 'emotional', 'behavior', 'discipline'],
            'Preventive Care & Safety': ['vaccine', 'vaccination', 'checkup', 'wellness', 'prevention', 'screening', 'safety', 'childproof', 'babyproof', 'secure', 'protection'],
            'Special Needs & Therapy': ['special needs', 'disability', 'therapy', 'intervention', 'diagnosis', 'assessment', 'speech', 'occupational']
        },
        
        'Nutrition & Feeding': {
            'Breastfeeding & Formula': ['breastfeed', 'nursing', 'formula', 'pump', 'lactation', 'feeding', 'breast milk'],
            'Baby & Toddler Food': ['baby food', 'puree', 'solid food', 'toddler meal', 'finger food', 'feeding', 'eat'],
            'Family Meals & Recipes': ['recipe', 'meal', 'cooking', 'food prep', 'dinner', 'lunch', 'breakfast', 'food'],
            'Restaurants & Dining Out': ['restaurant', 'dining', 'cafe', 'kid-friendly', 'family dining', 'eatery'],
            'Food Allergies & Dietary Needs': ['allergy', 'allergic', 'intolerance', 'dietary restriction', 'gluten', 'dairy', 'nut']
        },
        
        'Products & Shopping': {
            'Baby Gear & Equipment': ['stroller', 'car seat', 'carseat', 'carrier', 'crib', 'bassinet', 'monitor', 'diaper bag', 'equipment'],
            'Clothing & Accessories': ['clothes', 'clothing', 'shoe', 'outfit', 'dress', 'jacket', 'hat', 'accessory'],
            'Toys, Books & Entertainment': ['toy', 'book', 'game', 'puzzle', 'educational toy', 'stuffed animal', 'blocks', 'entertainment'],
            'Home & Furniture': ['furniture', 'decor', 'bed', 'chair', 'table', 'storage', 'organization']
        },
        
        'Marketplace & Exchange': {
            'Buying & Selling': ['buy', 'sell', 'purchase', 'sale', 'selling', 'marketplace', 'shop', 'store'],
            'Borrowing & Lending': ['borrow', 'lend', 'loan', 'share'],
            'Secondhand & Donations': ['used', 'secondhand', 'hand-me-down', 'donate', 'donation', 'charity', 'free'],
            'Services & Providers': ['service', 'provider', 'professional', 'contractor', 'cleaner', 'photographer']
        },
        
        'Activities & Recreation': {
            'Classes & Programs': ['class', 'lesson', 'program', 'workshop', 'instruction', 'teaching'],
            'Sports & Physical Activities': ['sport', 'swim', 'gymnastics', 'soccer', 'dance', 'baseball', 'basketball', 'physical activity'],
            'Arts & Music': ['art', 'craft', 'drawing', 'painting', 'creative', 'music', 'instrument'],
            'Social Events & Playdates': ['playdate', 'play date', 'meetup', 'get together', 'social', 'friend', 'party'],
            'Parks & Playgrounds': ['park', 'playground', 'play area', 'outdoor play', 'swing', 'slide'],
            'Seasonal & Holiday Activities': ['holiday', 'christmas', 'halloween', 'easter', 'thanksgiving', 'seasonal', 'festival', 'celebration']
        },
        
        'Parenting Challenges & Support': {
            'Sleep & Bedtime': ['sleep', 'nap', 'bedtime', 'night', 'waking', 'insomnia', 'sleep training'],
            'Potty Training & Hygiene': ['potty', 'toilet', 'training', 'diaper', 'accident', 'bathroom', 'hygiene', 'clean'],
            'Behavior Management': ['tantrum', 'timeout', 'punishment', 'reward', 'consequence', 'misbehavior'],
            'Family Dynamics': ['sibling', 'brother', 'sister', 'jealousy', 'rivalry', 'relationship', 'family'],
            'Parenting Support & Advice': ['support', 'advice', 'help', 'tip', 'suggestion', 'guidance', 'experience', 'question']
        },
        
        'Travel & Transportation': {
            'Family Vacations': ['vacation', 'resort', 'hotel', 'destination', 'family trip', 'holiday trip'],
            'Day Trips & Outings': ['outing', 'visit', 'museum', 'zoo', 'aquarium', 'day trip', 'excursion'],
            'Transportation & Commuting': ['car', 'drive', 'commute', 'transit', 'bus', 'train', 'transportation', 'traffic'],
            'Travel Planning & Tips': ['travel', 'plan', 'itinerary', 'packing', 'luggage', 'flight', 'airplane', 'passport']
        },
        
        'Home & Family Management': {
            'Work-Life Balance': ['work', 'balance', 'job', 'career', 'working parent', 'time management'],
            'Household Management': ['cleaning', 'chore', 'organization', 'routine', 'schedule', 'management', 'housekeeping'],
            'Family Celebrations': ['birthday', 'celebration', 'party', 'milestone', 'achievement', 'ceremony', 'anniversary']
        },
        
        'Local Community': {
            'Neighborhood Information': ['neighborhood', 'community', 'area', 'boston', 'north end', 'local', 'nearby'],
            'Local Events & Activities': ['event', 'community event', 'fair', 'festival', 'local event', 'happening'],
            'Local Facilities & Resources': ['facility', 'resource', 'center', 'library', 'pool', 'rink', 'field'],
            'Community Support Networks': ['support group', 'parent group', 'community group', 'network', 'forum', 'assistance', 'aid', 'help'],
            'Environmental & Public Issues': ['pollution', 'air quality', 'water quality', 'environmental', 'chemical', 'toxin', 'public health', 'safety']
        },
        
        'Group Communication': {
            'Announcements & Updates': ['announcement', 'notice', 'inform', 'update', 'news'],
            'Greetings & Introductions': ['welcome', 'introduction', 'greeting', 'hello', 'new member', 'join', 'thank', 'appreciation', 'grateful'],
            'Group Guidelines & Coordination': ['rule', 'guideline', 'policy', 'regulation', 'requirement', 'procedure', 'coordinate', 'plan', 'organize', 'schedule'],
            'Questions & Information Requests': ['question', 'inquiry', 'asking', 'information', 'where', 'how', 'what', 'when']
        },
        
        'Uncategorized': {
            'Unclear or Insufficient': ['unclear', 'insufficient', 'unknown', 'no topic', 'not specified'],
            'Other Topics': []  # Catch-all for anything else
        }
    }
    
    def categorize_topic(topic):
        """Categorize a topic into category and subcategory."""
        if not isinstance(topic, str):
            return "Uncategorized", "Unclear or Insufficient"
        
        topic_lower = topic.lower().strip('"\'')
        
        # Check for unclear topics first
        unclear_patterns = ['unclear', 'insufficient', 'unknown', 'no topic', 'not specified']
        if any(pattern in topic_lower for pattern in unclear_patterns) or len(topic_lower) < 3:
            return "Uncategorized", "Unclear or Insufficient"
        
        # Check each category and subcategory
        for category, subcategories in categories.items():
            for subcategory, keywords in subcategories.items():
                for keyword in keywords:
                    if keyword.lower() in topic_lower:
                        return category, subcategory
        
        # Additional specific categorizations
        
        # Check for recommendations (which could fall into multiple categories)
        if any(term in topic_lower for term in ['recommendation', 'recommend', 'suggestion', 'advice']):
            if any(term in topic_lower for term in ['doctor', 'pediatrician', 'medical', 'health']):
                return 'Health & Wellness', 'Medical Care & Providers'
            if any(term in topic_lower for term in ['daycare', 'childcare', 'nanny', 'babysit']):
                return 'Child Care & Education', 'Daycare & Preschools'
            if any(term in topic_lower for term in ['toy', 'stroller', 'product', 'gear']):
                return 'Products & Shopping', 'Baby Gear & Equipment'
            if any(term in topic_lower for term in ['restaurant', 'food', 'meal', 'recipe']):
                return 'Nutrition & Feeding', 'Family Meals & Recipes'
            if any(term in topic_lower for term in ['class', 'activity', 'sport', 'lesson']):
                return 'Activities & Recreation', 'Classes & Programs'
            
            # Default for recommendations
            return 'Parenting Challenges & Support', 'Parenting Support & Advice'
        
        # Check for specific health issues
        if any(term in topic_lower for term in ['cold', 'flu', 'fever', 'rash', 'cough', 'sick']):
            return 'Health & Wellness', 'Illness & Symptoms'
        
        # Check for sleep-related topics
        if any(term in topic_lower for term in ['sleep', 'nap', 'bedtime', 'night']):
            return 'Parenting Challenges & Support', 'Sleep & Bedtime'
        
        # Check for behavior topics
        if any(term in topic_lower for term in ['behavior', 'tantrum', 'discipline']):
            return 'Parenting Challenges & Support', 'Behavior Management'
        
        # Check for travel topics
        if any(term in topic_lower for term in ['travel', 'vacation', 'trip', 'flight']):
            return 'Travel & Transportation', 'Travel Planning & Tips'
        
        # Check for food topics
        if any(term in topic_lower for term in ['food', 'eat', 'meal', 'recipe', 'restaurant']):
            return 'Nutrition & Feeding', 'Family Meals & Recipes'
        
        # Check for product topics
        if any(term in topic_lower for term in ['buy', 'purchase', 'shop', 'product']):
            return 'Marketplace & Exchange', 'Buying & Selling'
        
        # Check for local information
        if any(term in topic_lower for term in ['boston', 'north end', 'local', 'area', 'neighborhood']):
            return 'Local Community', 'Neighborhood Information'
        
        # Check for activity topics
        if any(term in topic_lower for term in ['activity', 'play', 'fun', 'entertainment']):
            return 'Activities & Recreation', 'Social Events & Playdates'
        
        # Check for thank you messages
        if any(term in topic_lower for term in ['thank', 'thanks', 'appreciation']):
            return 'Group Communication', 'Greetings & Introductions'
        
        # Check for welcome messages
        if any(term in topic_lower for term in ['welcome', 'introduction', 'new member']):
            return 'Group Communication', 'Greetings & Introductions'
        
        # Default to Other Topics
        return 'Uncategorized', 'Other Topics'
    
    # Apply categorization to all topics
    df['topic_category'] = None
    df['topic_subcategory'] = None
    
    for idx, row in df.iterrows():
        if pd.notna(row['gpt4_topic']):
            category, subcategory = categorize_topic(row['gpt4_topic'])
            df.at[idx, 'topic_category'] = category
            df.at[idx, 'topic_subcategory'] = subcategory
    
    # Count topics by category and subcategory
    thread_df = df.drop_duplicates('thread_id')
    category_counts = thread_df['topic_category'].value_counts()
    subcategory_counts = thread_df.groupby(['topic_category', 'topic_subcategory']).size().reset_index(name='count')
    
    print("\nTopic distribution by category:")
    for category, count in category_counts.items():
        print(f"{category}: {count} topics")
    
    print("\nTopic distribution by subcategory:")
    for _, row in subcategory_counts.sort_values(['topic_category', 'count'], ascending=[True, False]).iterrows():
        print(f"{row['topic_category']} - {row['topic_subcategory']}: {row['count']} topics")
    
    # Save the categorized data
    df.to_csv(output_file, index=False)
    print(f"\nCategorized data saved to '{output_file}'")
    
    return df

# ===== PART 4: MAIN FUNCTIONS =====

def process_raw_data(input_file='whatsapp_final_processed_v4.csv', output_file='whatsapp_processed.csv'):
    """Process raw WhatsApp data."""
    print(f"Processing raw data from {input_file}...")
    
    # Load the data
    df = pd.read_csv(input_file)
    
    # Add processed columns
    df['is_question'] = df['message'].apply(is_question)
    
    # Extract tokens
    df['processed_tokens'] = df['message'].apply(extract_useful_tokens)
    df['token_count'] = df['processed_tokens'].apply(len)
    
    # Identify threads
    df = identify_threads(df)
    
    # Save the processed data
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    
    return df

def analyze_with_gpt4(input_file='whatsapp_processed.csv', output_file='whatsapp_gpt4_analyzed.csv', max_threads=None):
    """Analyze WhatsApp data with GPT-4."""
    print(f"Analyzing data with GPT-4 from {input_file}...")
    
    # Load the data
    df = pd.read_csv(input_file)
    
    # Analyze threads
    df = analyze_threads_with_openai(df, max_threads=max_threads)
    
    # Save the analyzed data
    df.to_csv(output_file, index=False)
    print(f"Analyzed data saved to {output_file}")
    
    return df

def full_pipeline(raw_file='whatsapp_final_processed_v4.csv', max_threads=None):
    """Run the full analysis pipeline."""
    # 1. Process raw data
    processed_df = process_raw_data(input_file=raw_file)
    
    # 2. Analyze with GPT-4
    analyzed_df = analyze_with_gpt4(input_file='whatsapp_processed.csv', max_threads=max_threads)
    
    # 3. Categorize topics
    categorized_df = categorize_topics(input_file='whatsapp_gpt4_analyzed.csv')
    
    print("\nFull analysis pipeline completed successfully!")
    return categorized_df

def analyze_existing_data():
    """Analyze existing GPT-4 analyzed data without rerunning GPT-4."""
    # Just run the categorization step
    categorized_df = categorize_topics()
    return categorized_df

# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='WhatsApp Conversation Analyzer')
    parser.add_argument('--mode', type=str, default='categorize', 
                        choices=['full', 'process', 'analyze', 'categorize'],
                        help='Mode to run: full pipeline, process only, analyze only, or categorize only')
    parser.add_argument('--input', type=str, default='whatsapp_final_processed_v4.csv',
                        help='Input file for raw data')
    parser.add_argument('--max-threads', type=int, default=None,
                        help='Maximum number of threads to analyze with GPT-4')
    
    args = parser.parse_args()
    
    if args.mode == 'full':
        full_pipeline(raw_file=args.input, max_threads=args.max_threads)
    elif args.mode == 'process':
        process_raw_data(input_file=args.input)
    elif args.mode == 'analyze':
        analyze_with_gpt4(max_threads=args.max_threads)
    elif args.mode == 'categorize':
        analyze_existing_data()
