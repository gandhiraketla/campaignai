# CampaignAI: AI-Powered Marketing Intelligence Platform

## 🌟 Project Overview
CampaignAI is an advanced marketing intelligence platform that leverages AI to provide deep insights into marketing campaigns, competitor analysis, and social media trends.

## 📋 Prerequisites
- Python 3.8+
- Node.js 16+
- MySQL


## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/gandhiraketla/campaignai
cd campaignai
```

### 2. Set Up Python Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Environment Configuration

#### 3.1 Create Environment Variables File
Create a `.env` file in the project root with the following configuration:

```env
# Database Configuration
DB_HOST=localhost
DB_USER=your_mysql_username
DB_PASSWORD=your_mysql_password
DB_NAME=campaignai

# AI Model Configuration
MODEL_TYPE=openai  # Options: openai, azure, etc.
OPENAI_API_KEY=your_openai_api_key

# LangSmith Tracing (Optional)
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=campaignai_project

# Chroma Vector Database
CHROMA_PERSIST_DIR=./chroma_store

# Social Media and Content APIs
# Twitter (X) API
X_BEARER_TOKEN=your_twitter_bearer_token

# YouTube API
YOUTUBE_API_KEY=your_youtube_api_key

# Reddit API
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_unique_user_agent_string
```

#### 3.2 Obtaining API Keys

##### Twitter (X) API
1. Visit [Twitter Developer Portal](https://developer.twitter.com/en/portal/dashboard)
2. Create a new application
3. Generate a Bearer Token

##### Reddit API
1. Go to [Reddit Apps](https://www.reddit.com/prefs/apps)
2. Create a new application
3. Note down Client ID, Client Secret, and create a unique User Agent

##### YouTube API
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable YouTube Data API v3
4. Create credentials and obtain API Key

##### OpenAI API
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an account or log in
3. Generate an API key from your account settings

### 4. Database Setup
```bash
# Create MySQL database
mysql -u root -p
CREATE DATABASE campaignai;
exit;

# Execute database scripts
mysql -u root -p campaignai < campaigndb.sql
python data/insert_campaign_data.py
```

### 5. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd campaignai-frontend
npm install
```

### 6. Start Backend API
```bash
# From api directory
cd ../api
python campaginapi.py
# API docs available at http://localhost:8000/docs
```

### 7. Start Frontend
```bash
# From campaignai-frontend directory
cd ../campaignai-frontend
npm run dev
# Frontend available at http://localhost:3000
```

## 🔒 Security Considerations
- Never commit `.env` file to version control
- Keep API keys confidential
- Use environment-specific configurations

## 📂 Project Structure
```
campaignai/
│
├── api/ # Backend FastAPI application     
├── agent/ # All agents        
├── campaignai-frontend/  # React frontend
├── data/               # Database scripts
├── util/               # Utility scripts
└── .env               # Environment configuration (git-ignored)
```

## 🛠 Troubleshooting
- Verify all API keys are valid
- Check database connection settings
- Ensure all dependencies are installed
- Review error logs for specific issues

## 📦 Key Dependencies
- Backend: FastAPI, LangChain, OpenAI
- Frontend: React, Tailwind CSS
- Database: MySQL
- Vector Store: Chroma DB

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request
## 📜 License
This project is under MIT Open Licencse