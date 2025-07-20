# Game Recommendation System

A full-stack, hybrid game recommendation platform that leverages content-based, collaborative, and deep learning models to provide personalized game suggestions. Features a FastAPI backend with MongoDB integration and a modern Next.js frontend with authentication.

---

## 📋 Table of Contents
- [Abstract](#abstract)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Dataset Requirements](#dataset-requirements)
- [Environment Setup](#environment-setup)
- [Installation & Setup](#installation--setup)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Authentication](#authentication)
- [Database Schema](#database-schema)
- [Deployment](#deployment)
- [Customization](#customization)
- [Contributing](#contributing)
- [Credits](#credits)

---

## 📄 Abstract

This project implements a comprehensive game recommendation system that combines multiple AI approaches to deliver personalized gaming suggestions. The system addresses the cold-start problem for new users while providing sophisticated recommendations for returning users through:

- **Content-based filtering** using TF-IDF and BERT embeddings
- **Collaborative filtering** based on user similarity
- **Hybrid models** that combine content and collaborative approaches
- **Real-time user interaction tracking** with MongoDB
- **Modern web interface** with authentication and user profiles

The system demonstrates how different recommendation algorithms can be integrated into a single platform, providing users with multiple recommendation strategies and allowing for comparison of their effectiveness.

---

## ✨ Features

### 🎯 Recommendation Algorithms
- **Cold Start Recommendations:** Popular games for new users
- **TF-IDF Content-based Filtering:** Text-based game similarity
- **BERT Semantic Analysis:** Deep learning embeddings for semantic understanding
- **Collaborative Filtering:** User-based similarity recommendations
- **Hybrid Models:** Combination of content and collaborative approaches

### 🔐 Authentication & User Management
- **NextAuth.js Integration:** Secure authentication system
- **MongoDB User Storage:** Persistent user data and interactions
- **User Profiles:** Personalized experience with interaction history
- **User Statistics:** Detailed analytics and recommendation insights

### 🎨 Modern Web Interface
- **Responsive Design:** Works on desktop and mobile devices
- **Real-time Updates:** Live interaction tracking and recommendations
- **Interactive UI:** Like/dislike buttons and rating system
- **Admin Panel:** Debug tools and system monitoring

### 🔄 Real-time Features
- **Live Data Updates:** Automatic refresh of recommendations
- **User Interaction Tracking:** Instant feedback processing
- **Performance Monitoring:** Real-time system statistics

---

## 🛠 Tech Stack

### Backend
- **Python 3.8+** - Core programming language
- **FastAPI** - Modern, fast web framework
- **MongoDB** - NoSQL database for user data and interactions
- **Pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning algorithms
- **SentenceTransformers** - BERT embeddings for semantic analysis
- **APScheduler** - Background task scheduling
- **Pydantic** - Data validation and serialization

### Frontend
- **Next.js 15** - React framework with App Router
- **React 19** - User interface library
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS 4** - Utility-first CSS framework
- **NextAuth.js** - Authentication solution
- **MongoDB Adapter** - Database integration for auth

### DevOps
- **Render** - Cloud deployment platform
- **Git** - Version control

---

## 🏗 Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   Database      │
│   (Next.js)     │◄──►│   (FastAPI)     │◄──►│   (MongoDB)     │
│                 │    │                 │    │                 │
│ • Authentication│    │ • API Endpoints │    │ • User Data     │
│ • UI Components │    │ • ML Models     │    │ • Interactions  │
│ • Real-time UI  │    │ • DataProcessing│    │ • Analytics     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 📊 Dataset Requirements

### Game Data
- **File:** `dataset/rawg_games.csv`
- **Required Columns:**
  - `id` - Unique game identifier
  - `name` - Game title
  - `genre_text` or `genres` - Game genres (comma-separated)
- **Optional Columns:**
  - `rating` - Average rating
  - `ratings_count` - Number of ratings
  - `background_image` - Game cover image URL
  - `description` - Game description

### User Interactions (Auto-generated)
- Stored in MongoDB collections
- **Fields:**
  - `user_id` - User identifier
  - `game_id` - Game identifier
  - `liked` - Boolean like/dislike
  - `rating` - Numeric rating (1-5)
  - `timestamp` - Interaction timestamp

---

## 🔧 Environment Setup

### Backend Environment Variables

Create a `.env` file in the project root:

```env
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB=game-recommender

# Server Configuration
PORT=8000
HOST=0.0.0.0

# Optional: Production Settings
NODE_ENV=development
LOG_LEVEL=INFO
```

### Frontend Environment Variables

Create a `.env.local` file in `game-recommender-frontend/`:

```env
# Authentication
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-secret-key-here

# MongoDB for NextAuth
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB=game-recommender

# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Production Environment Variables

For production deployment (e.g., Render):

```env
# MongoDB Atlas (Production)
MONGODB_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/<dbname>?retryWrites=true&w=majority
MONGODB_DB=game-recommender-prod

# NextAuth (Production)
NEXTAUTH_URL=https://your-domain.com
NEXTAUTH_SECRET=your-production-secret-key

# API Configuration
NEXT_PUBLIC_API_URL=https://your-api-domain.com
```

---

## 🚀 Installation & Setup

### Prerequisites
- **Python 3.8+**
- **Node.js 18+**
- **MongoDB** (local or Atlas)
- **Git**

### Backend Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Game_Rec_redefined
   ```


2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your MongoDB connection details
   ```


4. **Start the backend server:**
   ```bash
   python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd game-recommender-frontend
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env.local
   # Edit .env.local with your configuration
   ```

4. **Start the development server:**
   ```bash
   npm run dev
   ```

5. **Access the application:**
   - Frontend: [http://localhost:3000](http://localhost:3000)
   - Backend API: [http://localhost:8000](http://localhost:8000)
   - API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📖 Usage

### Web Interface
1. **Visit the application** at `http://localhost:3000`
2. **Sign in** using the authentication system
3. **Explore different recommendation algorithms:**
   - **Popular Games:** Cold start recommendations
   - **TF-IDF AI:** Content-based filtering
   - **BERT AI:** Semantic analysis
   - **Collaborative:** User-based filtering
   - **Hybrid Models:** Combined approaches
4. **Interact with games:** Like/dislike and rate games
5. **View your statistics:** Track your interaction history

### API Usage
```bash
# Get cold start recommendations
curl http://localhost:8000/cold-start

# Get personalized recommendations
curl "http://localhost:8000/recommend/tfidf?user_id=user123"

# Record user interaction
curl -X POST http://localhost:8000/newInteraction \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "game_id": 123, "liked": true, "rating": 5, "timestamp": "2024-01-01T00:00:00Z"}'
```

---

## 🔌 API Endpoints

### Authentication Endpoints
- `POST /api/auth/signin` - User sign in
- `POST /api/auth/signout` - User sign out
- `GET /api/auth/session` - Get current session

### Recommendation Endpoints
- `GET /cold-start` - Get popular games for new users
- `GET /recommend/tfidf?user_id={user_id}` - TF-IDF recommendations
- `GET /recommend/bert?user_id={user_id}` - BERT-based recommendations
- `GET /recommend/collaborative?user_id={user_id}` - Collaborative filtering
- `GET /recommend/hybrid-tfidf?user_id={user_id}` - Hybrid TF-IDF
- `GET /recommend/hybrid-bert?user_id={user_id}` - Hybrid BERT

### User Interaction Endpoints
- `POST /newInteraction` - Record user interaction
- `GET /user/stats/{user_id}` - Get user statistics
- `GET /user/interactions/{user_id}` - Get user interactions

### Response Format
```json
{
  "success": true,
  "data": [
    {
      "id": 123,
      "name": "Game Title",
      "genre_text": "Action, Adventure",
      "rating": 4.5,
      "background_image": "https://example.com/image.jpg",
      "score": 0.85
    }
  ]
}
```

---

## 🔐 Authentication

The system uses NextAuth.js for secure authentication:

### Features
- **Multiple Providers:** Email/password, OAuth (configurable)
- **Session Management:** Secure session handling
- **User Profiles:** Persistent user data
- **Protected Routes:** Secure access to personalized features

### Configuration
```javascript
// pages/api/auth/[...nextauth].js
export default NextAuth({
  providers: [
    CredentialsProvider({
      // Email/password authentication
    }),
    // Add OAuth providers as needed
  ],
  adapter: MongoDBAdapter(clientPromise),
  session: {
    strategy: "jwt",
  },
})
```

---

## 🗄 Database Schema

### Users Collection
```javascript
{
  _id: ObjectId,
  email: String,
  name: String,
  image: String,
  createdAt: Date,
  updatedAt: Date
}
```

### User Interactions Collection
```javascript
{
  _id: ObjectId,
  user_id: String,
  game_id: Number,
  liked: Boolean,
  rating: Number,
  timestamp: Date
}
```

### Indexes
- `user_id` + `game_id` (unique compound index)
- `user_id` + `timestamp` (for user history queries)

---

## 🚀 Deployment

### Render Deployment

1. **Backend Deployment:**
   ```yaml
   # render.yaml
   services:
     - type: web
       name: game-recommender-api
       env: python
       buildCommand: pip install -r requirements.txt
       startCommand: python -m uvicorn main:app --host 0.0.0.0 --port $PORT
       envVars:
         - key: MONGODB_URI
           value: mongodb+srv://...
         - key: MONGODB_DB
           value: game-recommender-prod
   ```

2. **Frontend Deployment:**
   ```yaml
   services:
     - type: web
       name: game-recommender-frontend
       env: node
       buildCommand: npm install && npm run build
       startCommand: npm start
       envVars:
         - key: NEXTAUTH_URL
           value: https://your-domain.com
         - key: MONGODB_URI
           value: mongodb+srv://...
   ```

### Environment Variables for Production
- Set all required environment variables in your deployment platform
- Use strong, unique secrets for NEXTAUTH_SECRET
- Configure CORS origins for production domains
- Set up MongoDB Atlas for production database

---

## 🎨 Customization

### Adding New Recommendation Algorithms
1. **Implement algorithm** in `recommender.py`
2. **Add API endpoint** in `main.py`
3. **Create frontend component** in `src/components/`
4. **Update navigation** in `src/app/page.tsx`

### Customizing the UI
- **Styling:** Modify Tailwind classes in components
- **Layout:** Update `src/app/layout.tsx` and page components
- **Themes:** Add custom CSS variables for theming

### Database Customization
- **Additional Fields:** Extend MongoDB schemas
- **New Collections:** Add collections for additional features
- **Indexes:** Optimize queries with custom indexes

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/amazing-feature`
3. **Commit your changes:** `git commit -m 'Add amazing feature'`
4. **Push to the branch:** `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 for Python code
- Use TypeScript for frontend components
- Add tests for new features
- Update documentation for API changes

---

## 👥 Credits

**Developed by:**
- [Joel Joy](https://github.com/Joeljoy1237) - Backend Development & ML Algorithms
- [Bidhun B](https://github.com/BidhunB/) - Frontend Development & UI/UX
- [Varghese Francis](https://github.com/VargheeseFrancis) - Database Design & API Development
- [Ashok Xavier](https://github.com/AshokXavier) - System Architecture & Deployment

**Special Thanks:**
- [RAWG API](https://rawg.io/apidocs) - Game data source
- [NextAuth.js](https://next-auth.js.org/) - Authentication framework
- [FastAPI](https://fastapi.tiangolo.com/) - Backend framework
- [Tailwind CSS](https://tailwindcss.com/) - Styling framework

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**🎮 Enjoy exploring game recommendations with AI!** 
