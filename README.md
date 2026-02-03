# Cloud9 Assistant Coach - API

> **Related Repositories:**
> - 🖥️ Frontend UI: [c9-assistant-ui](https://github.com/AvishKaushik/c9-assistant-ui)

FastAPI backend providing esports analytics and AI-powered coaching insights for League of Legends and VALORANT.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip (Python package manager)
- GRID API key ([Get access](https://grid.gg/get-access/))
- Groq API key ([Get free key](https://console.groq.com/))

### Installation

```bash
# Clone the repository
cd c9-assistant-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r category1-assistant-coach/requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running the Server

```bash
cd category1-assistant-coach
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

---

## ⚙️ Environment Variables

Create a `.env` file in the root directory:

| Variable | Description | Required |
|----------|-------------|----------|
| `GRID_API_KEY` | Your GRID API key for match data | ✅ |
| `GROQ_API_KEY` | Groq API key for LLM features | ✅ |
| `USE_MOCK_DATA` | Set to `true` for development without API | ❌ |
| `CORS_ORIGINS` | Allowed origins (default: `*`) | ❌ |

Example `.env`:
```env
GRID_API_KEY=your_grid_api_key_here
GROQ_API_KEY=your_groq_api_key_here
USE_MOCK_DATA=false
CORS_ORIGINS=*
```

---

## 📡 API Endpoints

### Insights

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/insights/player` | Get AI-generated player improvement insights |
| `POST` | `/api/v1/insights/team` | Get team-level pattern analysis |

### Macro Review

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/macro-review` | Generate structured VOD review agenda |

**Request Body:**
```json
{
  "match_id": "2843069",
  "game": "Valorant",
  "game_number": 1
}
```

### What-If Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/what-if` | Analyze hypothetical scenarios with AI |
| `POST` | `/api/v1/what-if/quick` | Quick scenario analysis without match context |

**Request Body:**
```json
{
  "match_id": "2843069",
  "game": "Valorant",
  "scenario_description": "What if we had attacked more on Lotus instead of defense?",
  "game_number": 1
}
```

**Response:**
```json
{
  "match_id": "2843069",
  "game": "Valorant",
  "original_outcome": "Cloud9 victory (13-10)",
  "prediction": {
    "success_probability": 0.65,
    "confidence": "medium",
    "key_factors": ["map control", "utility usage", "team coordination"],
    "risks": ["Overextending", "Economy management"],
    "rewards": ["Early round advantage", "Map control"],
    "reasoning": "..."
  },
  "alternative_scenarios": [...]
}
```

---

## 🏗️ Project Structure

```
c9-assistant-api/
├── .env                          # Environment variables
├── category1-assistant-coach/
│   ├── app/
│   │   ├── main.py               # FastAPI application entry
│   │   ├── routers/
│   │   │   ├── insights.py       # Player/team insights endpoints
│   │   │   ├── macro_review.py   # VOD review agenda generation
│   │   │   └── what_if.py        # Hypothetical scenario analysis
│   │   ├── services/
│   │   │   ├── insight_generator.py   # LLM insight generation
│   │   │   └── scenario_predictor.py  # What-if prediction logic
│   │   └── models/
│   │       └── schemas.py        # Pydantic request/response models
│   └── requirements.txt
└── shared/                       # Shared utilities (GRID client, LLM)
    ├── grid_client/              # GraphQL client for GRID API
    └── utils/
        └── llm.py                # LLM integration (Groq/Anthropic)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | FastAPI |
| Language | Python 3.10+ |
| LLM | Groq (llama-3.3-70b-versatile) |
| Data Source | GRID GraphQL API |
| Validation | Pydantic v2 |
| Server | Uvicorn |

---

## 📚 API Documentation

Once running, access the interactive API docs:
- **Swagger UI**: https://c9-assistant-api.onrender.com/docs

---

## 🧪 Development

```bash
# Run with auto-reload
uvicorn app.main:app --reload

# Run tests
pytest

# Check code style
flake8 app/
```
