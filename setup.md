# Game Recommender Setup Guide

This guide will help you set up the Game Recommender application with social login and MongoDB integration.

## Prerequisites

- Node.js 18+ installed
- Python 3.8+ installed
- MongoDB instance (local or cloud)
- Google Developer Account

## Step 1: Frontend Setup

### 1.1 Install Dependencies
```bash
cd game-recommender-frontend
npm install
```

### 1.2 Create Environment File
Create a `.env.local` file in the `game-recommender-frontend` directory:

```env
# NextAuth Configuration
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-super-secret-key-here-make-it-long-and-random

# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/game-recommender

# Google OAuth (you'll get these in step 2)
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret



# API Configuration
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

### 1.3 Generate NextAuth Secret
You can generate a secure secret using:
```bash
openssl rand -base64 32
```

## Step 2: Google OAuth Setup

### 2.1 Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google+ API:
   - Go to "APIs & Services" > "Library"
   - Search for "Google+ API" and enable it

### 2.2 Create OAuth Credentials
1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth 2.0 Client IDs"
3. Choose "Web application"
4. Set the following:
   - Name: "Game Recommender"
   - Authorized JavaScript origins: `http://localhost:3000`
   - Authorized redirect URIs: `http://localhost:3000/api/auth/callback/google`
5. Click "Create"
6. Copy the Client ID and Client Secret to your `.env.local` file



## Step 3: MongoDB Setup

### Option A: Local MongoDB
1. Install MongoDB Community Edition
2. Start MongoDB service
3. Create database: `game-recommender`

### Option B: MongoDB Atlas (Cloud)
1. Go to [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Create a free cluster
3. Create a database user
4. Get your connection string
5. Update `MONGODB_URI` in `.env.local`

## Step 4: Backend Setup

### 4.1 Install Python Dependencies
```bash
cd ..  # Go back to project root
pip install -r requirements.txt
```

### 4.2 Create Backend Environment File
Create a `.env` file in the project root:

```env
MONGODB_URI=mongodb://localhost:27017/game-recommender
```

## Step 5: Run the Application

### 5.1 Start the Backend
```bash
python main.py
```
The backend will start on `http://localhost:8000`

### 5.2 Start the Frontend
```bash
cd game-recommender-frontend
npm run dev
```
The frontend will start on `http://localhost:3000`

## Step 6: Test the Application

1. Open `http://localhost:3000` in your browser
2. Click "Sign In" and choose Google
3. Complete the OAuth flow
4. You should see your profile in the top right
5. Start rating games to see personalized recommendations

## Troubleshooting

### Common Issues

1. **"Invalid redirect URI" error**
   - Make sure the redirect URI in your OAuth app matches exactly: `http://localhost:3000/api/auth/callback/google`

2. **MongoDB connection error**
   - Check if MongoDB is running
   - Verify the connection string in `.env.local`

3. **"Module not found" errors**
   - Run `npm install` in the frontend directory
   - Run `pip install -r requirements.txt` in the backend directory

4. **CORS errors**
   - Make sure both frontend and backend are running
   - Check that `NEXT_PUBLIC_API_BASE_URL` is set correctly

### Environment Variables Checklist

Make sure all these are set in your `.env.local`:

- ✅ `NEXTAUTH_URL`
- ✅ `NEXTAUTH_SECRET`
- ✅ `MONGODB_URI`
- ✅ `GOOGLE_CLIENT_ID`
- ✅ `GOOGLE_CLIENT_SECRET`

- ✅ `NEXT_PUBLIC_API_BASE_URL`

## Production Deployment

For production deployment:

1. Update `NEXTAUTH_URL` to your production domain
2. Update OAuth redirect URIs to your production domain
3. Use a production MongoDB instance
4. Set up proper environment variables on your hosting platform
5. Consider using environment-specific configuration files

## Support

If you encounter issues:
1. Check the browser console for errors
2. Check the backend logs for errors
3. Verify all environment variables are set correctly
4. Ensure MongoDB is running and accessible 