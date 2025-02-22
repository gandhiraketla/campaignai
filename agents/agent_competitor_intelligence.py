import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import tweepy
import praw
import requests
from googleapiclient.discovery import build
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langsmith import traceable
import sys

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from util.envutils import EnvUtils

###############################################################################
# 1. Environment Setup & Configuration
###############################################################################

# LangSmith Setup
LANGSMITH_API_KEY = EnvUtils().get_required_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = EnvUtils().get_required_env("LANGSMITH_PROJECT")

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger("WearablesCompetitorAgent")

# Product Keywords Configuration
PRODUCT_KEYWORDS = {
    "fitbit": ["Fitbit", "FitbitSense", "FitbitVersa", "FitbitCharge", "FitbitInspire", "FitbitLuxe"],
    "apple watch": ["AppleWatch", "WatchOS", "Apple Watch Series", "Apple Watch Ultra", "Apple Watch SE"]
}

# Load API keys from environment
X_BEARER_TOKEN = EnvUtils().get_required_env("X_BEARER_TOKEN")
REDDIT_CLIENT_ID = EnvUtils().get_required_env("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = EnvUtils().get_required_env("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = EnvUtils().get_required_env("REDDIT_USER_AGENT")
YOUTUBE_API_KEY = EnvUtils().get_required_env("YOUTUBE_API_KEY")

###############################################################################
# 2. API Client Initialization
###############################################################################

# Initialize X (Twitter) client
try:
    x_client = tweepy.Client(bearer_token=X_BEARER_TOKEN)
    logger.info("X (Twitter) client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize X client: {e}")
    x_client = None

# Initialize Reddit client
try:
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    logger.info("Reddit client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Reddit client: {e}")
    reddit = None

# Initialize YouTube client
try:
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    logger.info("YouTube client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize YouTube client: {e}")
    youtube = None

###############################################################################
# 3. Analysis Tools
###############################################################################

@traceable(name="twitter_analysis")
def analyze_twitter_sentiment(product: str, days_back: int = 7) -> str:
    """
    Analyze recent Twitter sentiment for a specific product using Twitter API v2
    """
    if not x_client:
        return "X (Twitter) analysis is not available due to client initialization failure"
        
    try:
        # Get product-specific keywords
        product_lower = product.lower()
        keywords = PRODUCT_KEYWORDS.get(product_lower, [product])
        
        # Build a comprehensive search query
        query = f"({' OR '.join(keywords)}) -is:retweet lang:en"
        
        # Search tweets using Twitter API v2
        response = x_client.search_recent_tweets(
            query=query,
            max_results=100,
            tweet_fields=['created_at', 'public_metrics', 'context_annotations']
        )
        
        if not response.data:
            return f"No recent tweets found about {product}"
            
        tweets = response.data
        
        # Basic sentiment categorization
        positive_keywords = ["great", "love", "awesome", "perfect", "amazing"]
        negative_keywords = ["bad", "hate", "terrible", "poor", "issues"]
        
        sentiments = []
        engagement_total = 0
        example_tweets = []
        
        for tweet in tweets:
            text = tweet.text.lower()
            metrics = tweet.public_metrics
            
            # Calculate engagement
            engagement = (metrics.get('retweet_count', 0) + 
                        metrics.get('reply_count', 0) + 
                        metrics.get('like_count', 0))
            engagement_total += engagement
            
            # Sentiment analysis
            pos_count = sum(1 for word in positive_keywords if word in text)
            neg_count = sum(1 for word in negative_keywords if word in text)
            
            if pos_count > neg_count:
                sentiments.append(1)
            elif neg_count > pos_count:
                sentiments.append(-1)
            else:
                sentiments.append(0)
                
            # Collect high-engagement tweets as examples
            if len(example_tweets) < 3 and len(text) > 50 and engagement > 10:
                example_tweets.append({
                    'text': text,
                    'engagement': engagement
                })

        if not sentiments:
            return f"No relevant tweets found about {product}"

        sentiment_score = sum(sentiments) / len(sentiments)
        sentiment_label = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
        
        response = [
            f"Twitter Analysis for {product}:",
            f"- Analyzed {len(sentiments)} recent tweets",
            f"- Overall sentiment: {sentiment_label} (score: {sentiment_score:.2f})",
            f"- Total engagement: {engagement_total} (likes + retweets + replies)",
            f"- Average engagement per tweet: {engagement_total/len(tweets):.1f}",
            "\nMost engaging discussions:"
        ]
        
        # Sort example tweets by engagement
        example_tweets.sort(key=lambda x: x['engagement'], reverse=True)
        for tweet in example_tweets:
            response.append(
                f"- [Engagement: {tweet['engagement']}] {tweet['text'][:200]}..."
            )

        return "\n".join(response)

    except Exception as e:
        logger.error(f"X API error: {e}")
        return f"Error analyzing X data: {str(e)}"

@traceable(name="reddit_analysis")
def analyze_reddit_discussions(product: str) -> str:
    """
    Analyze Reddit discussions about a specific product
    """
    if not reddit:
        return "Reddit analysis is not available due to client initialization failure"
        
    try:
        # Refined subreddit mapping with more flexible matching
        subreddit_map = {
            "fitbit": "fitbit",
            "fitbit versa": "fitbit",
            "apple watch": "AppleWatch",
            "apple watch ultra": "AppleWatch"
        }
        
        product_lower = product.lower()
        subreddit_name = subreddit_map.get(product_lower, product_lower.replace(" ", ""))
        
        try:
            subreddit = reddit.subreddit(subreddit_name)
            
            # Get top posts from the past week
            top_posts = list(subreddit.top(time_filter="week", limit=10))
        except Exception as e:
            logger.error(f"Error accessing subreddit {subreddit_name}: {e}")
            return f"Could not access r/{subreddit_name}. The subreddit might be private or not exist."
        
        if not top_posts:
            return f"No recent posts found in r/{subreddit_name}"
            
        response = [f"Reddit Analysis for {product} (from r/{subreddit_name}):"]
        
        total_comments = 0
        total_score = 0
        
        for post in top_posts:
            total_comments += post.num_comments
            total_score += post.score
            
            response.append(
                f"\nPost: {post.title}"
                f"\nUpvotes: {post.score:,} | Comments: {post.num_comments:,}"
                f"\nURL: reddit.com{post.permalink}"
            )
            
            # Get top 2 comments with error handling
            try:
                post.comments.replace_more(limit=0)
                top_comments = list(post.comments)[:2]
                
                for comment in top_comments:
                    response.append(
                        f"- Top comment ({comment.score:,} points): "
                        f"{comment.body[:200]}..."
                    )
            except Exception as e:
                logger.error(f"Error fetching comments for post {post.id}: {e}")
                response.append("- Could not fetch comments for this post")
            
            response.append("")  # Empty line between posts
        
        # Add summary statistics
        response.insert(1, f"Analysis of top {len(top_posts)} posts:")
        response.insert(2, f"- Total engagement: {total_comments:,} comments, {total_score:,} upvotes")
        response.insert(3, f"- Average engagement: {total_comments/len(top_posts):,.0f} comments/post")
        response.insert(4, "")  # Empty line before posts
        
        return "\n".join(response)

    except Exception as e:
        logger.error(f"Reddit API error: {e}")
        return f"Error analyzing Reddit data: {str(e)}"

@traceable(name="youtube_analysis")
def analyze_youtube_content(product: str) -> str:
    """
    Analyze recent YouTube videos about a specific product
    """
    if not youtube:
        return "YouTube analysis is not available due to client initialization failure"
        
    try:
        # Set up search parameters
        search_terms = {
            "fitbit": "Fitbit review|Fitbit features|Fitbit comparison",
            "apple watch": "Apple Watch review|Apple Watch features|Apple Watch comparison"
        }
        
        product_lower = product.lower()
        search_query = search_terms.get(product_lower, product)
        
        # Search for recent videos about the product
        search_response = youtube.search().list(
            q=search_query,
            part="id,snippet",
            maxResults=5,
            type="video",
            order="date",
            relevanceLanguage="en"
        ).execute()

        if not search_response.get('items'):
            return f"No recent YouTube videos found about {product}"

        response = [f"Recent YouTube Coverage of {product}:"]
        total_views = 0
        total_likes = 0
        
        for item in search_response['items']:
            video_id = item['id']['videoId']
            
            # Get detailed video statistics
            video_response = youtube.videos().list(
                part="statistics,snippet",
                id=video_id
            ).execute()
            
            if video_response['items']:
                video = video_response['items'][0]
                stats = video['statistics']
                
                # Convert statistics to integers
                views = int(stats.get('viewCount', 0))
                likes = int(stats.get('likeCount', 0))
                comments = int(stats.get('commentCount', 0))
                
                total_views += views
                total_likes += likes
                
                # Format numbers with commas
                response.append(
                    f"\nVideo: {video['snippet']['title']}"
                    f"\nChannel: {video['snippet']['channelTitle']}"
                    f"\nStatistics:"
                    f"\n- Views: {views:,}"
                    f"\n- Likes: {likes:,}"
                    f"\n- Comments: {comments:,}"
                    f"\nURL: youtube.com/watch?v={video_id}\n"
                )

        # Add summary statistics
        response.insert(1, f"Analysis of top {len(search_response['items'])} videos:")
        response.insert(2, f"Total Reach: {total_views:,} views, {total_likes:,} likes")
        response.insert(3, f"Average Views: {total_views/len(search_response['items']):,.0f} per video\n")

        return "\n".join(response)

    except Exception as e:
        logger.error(f"YouTube API error: {e}")
        return f"Error analyzing YouTube data: {str(e)}"

@traceable(name="feature_comparison")
def compare_features(query: str) -> str:
    """
    Compare features between Fitbit and Apple Watch models
    """
    # Latest model comparison data
    comparison_data = {
        "Fitbit Sense 2": {
            "price": "$299.95",
            "battery_life": "6+ days",
            "health_features": [
                "Heart rate monitoring",
                "ECG",
                "EDA stress tracking",
                "Skin temperature",
                "SpO2"
            ],
            "fitness_features": [
                "40+ exercise modes",
                "Built-in GPS",
                "Active Zone Minutes"
            ],
            "notable_features": [
                "Stress management",
                "Sleep tracking",
                "Amazon Alexa"
            ]
        },
        "Apple Watch Series 9": {
            "price": "$399",
            "battery_life": "18 hours",
            "health_features": [
                "Heart rate monitoring",
                "ECG",
                "Blood oxygen",
                "Temperature sensing"
            ],
            "fitness_features": [
                "Workout detection",
                "GPS + GLONASS",
                "Fitness+ integration"
            ],
            "notable_features": [
                "Always-on display",
                "Fall detection",
                "Siri integration",
                "Double tap gesture"
            ]
        }
    }
    
    query_lower = query.lower()
    response = []
    
    if "compare" in query_lower:
        response.append("Feature Comparison:\n")
        for model, features in comparison_data.items():
            response.append(f"{model}:")
            response.append(f"- Price: {features['price']}")
            response.append(f"- Battery: {features['battery_life']}")
            response.append("- Health Features: " + ", ".join(features['health_features']))
            response.append("- Fitness Features: " + ", ".join(features['fitness_features']))
            response.append("- Notable Features: " + ", ".join(features['notable_features']))
            response.append("")
    else:
        # Look for specific feature queries
        for model, features in comparison_data.items():
            if model.lower() in query_lower:
                response.append(f"{model} Features:")
                if "health" in query_lower:
                    response.append("Health Features: " + ", ".join(features['health_features']))
                elif "fitness" in query_lower:
                    response.append("Fitness Features: " + ", ".join(features['fitness_features']))
                elif "battery" in query_lower:
                    response.append(f"Battery Life: {features['battery_life']}")
                elif "price" in query_lower:
                    response.append(f"Price: {features['price']}")
                else:
                    response.append(f"Price: {features['price']}")
                    response.append(f"Battery Life: {features['battery_life']}")
                    response.append("Key Features:")
                    response.extend([f"- {f}" for f in features['notable_features']])
    
    return "\n".join(response) if response else "No specific feature information found for this query."

###############################################################################
# 4. Agent Setup
###############################################################################

def build_wearables_competitor_agent():
    """
    Creates an agent specialized in wearables competitor analysis
    """
    llm = EnvUtils().get_llm()
    # Define the tools
    twitter_tool = Tool(
        name="twitter_analysis",
        func=analyze_twitter_sentiment,
        description="Analyze recent Twitter sentiment and discussions about a specific product"
    )

    reddit_tool = Tool(
        name="reddit_analysis",
        func=analyze_reddit_discussions,
        description="Analyze Reddit discussions, top posts, and community sentiment about a product"
    )

    youtube_tool = Tool(
        name="youtube_analysis",
        func=analyze_youtube_content,
        description="Analyze recent YouTube videos, reviews, and coverage about a product"
    )

    feature_tool = Tool(
        name="feature_comparison",
        func=compare_features,
        description="Compare features, prices, and specifications between Fitbit and Apple Watch models"
    )

    tools = [twitter_tool, reddit_tool, youtube_tool, feature_tool]

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Wearables Market Intelligence Specialist focused on analyzing 
        Fitbit and Apple Watch wearable fitness devices. 
        Provide detailed insights about product reception, user sentiment, and trending discussions.
        When analyzing data, focus on:
        1. Overall sentiment trends
        2. Specific user feedback and complaints
        3. Feature comparisons
        4. Community engagement levels
        5. Don't hallucinate, rely only on the data provided.
        5. If you cannot fins information in one platform, try in another one, for example if reddit returns no results try in twitter or youtube.
        Always cite your sources (Twitter, Reddit, YouTube) when providing insights."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Create the agent executor with tracing
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        tags=["wearables_competitor_agent"],
        metadata={
            "agent_type": "wearables_intelligence",
            "tool_count": len(tools)
        }
    )

    logger.info("Wearables Competitor Intelligence Agent built with social media analysis tools.")
    return agent_executor

###############################################################################
# 5. Main Demo
###############################################################################

if __name__ == "__main__":
    logger.info("Starting the Wearables Competitor Intelligence Agent Demo...")

    agent = build_wearables_competitor_agent()

    # Example queries
    test_queries = [
        "How are people responding to the new Apple Watch Ultra features?"
    ]

    for query in test_queries:
        logger.info("USER: %s", query)
        try:
            response = agent.invoke({"input": query})
            print(f"\nUSER: {query}\nAGENT:\n{response['output']}\n---\n")
        except Exception as exc:
            logger.exception("Agent failed on query='%s'", query)
            print(f"Error: {exc}")