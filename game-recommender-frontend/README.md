# Game Recommender Frontend

A modern Next.js application with social login and AI-powered game recommendations.

## Features

- 🔐 Social Login (Google)
- 🎮 AI-powered game recommendations
- 📊 User interaction tracking
- 🎨 Modern UI with Tailwind CSS
- 🔄 Real-time recommendations

## Prerequisites

- Node.js 18+ 
- MongoDB instance
- Google OAuth credentials

## Setup

### 1. Install Dependencies

```bash
npm install
```

### 2. Environment Variables

Create a `.env.local` file in the root directory:

```env
# NextAuth Configuration
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-nextauth-secret-key-here

# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/game-recommender

# Google OAuth
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret



# API Configuration
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

### 3. OAuth Setup

#### Google OAuth
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Google+ API
4. Go to Credentials → Create Credentials → OAuth 2.0 Client IDs
5. Set authorized redirect URI: `http://localhost:3000/api/auth/callback/google`



### 4. MongoDB Setup

1. Install MongoDB locally or use MongoDB Atlas
2. Create a database named `game-recommender`
3. The application will automatically create the required collections

### 5. Run the Application

```bash
npm run dev
```

The application will be available at `http://localhost:3000`

## Backend Setup

Make sure the Python backend is running with MongoDB support:

1. Install backend dependencies:
```bash
pip install -r requirements.txt
```

2. Set up backend environment variables:
```env
MONGODB_URI=mongodb://localhost:27017/game-recommender
```

3. Run the backend:
```bash
python main.py
```

## Usage

1. Visit the application
2. Click "Sign In" to authenticate with Google
3. Browse game recommendations
4. Like/dislike games to improve recommendations
5. View your personalized recommendations

## Architecture

- **Frontend**: Next.js 15 with TypeScript
- **Authentication**: NextAuth.js with MongoDB adapter
- **Database**: MongoDB for user data and interactions
- **Styling**: Tailwind CSS
- **Backend**: FastAPI with Python

## API Endpoints

- `POST /api/auth/[...nextauth]` - Authentication routes
- `GET /api/recommend/*` - Game recommendations
- `POST /api/newInteraction` - Save user interactions
- `GET /api/user/stats/{user_id}` - User statistics
- `GET /api/user/interactions/{user_id}` - User interactions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License
