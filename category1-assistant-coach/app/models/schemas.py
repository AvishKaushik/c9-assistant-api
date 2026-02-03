"""Pydantic schemas for Assistant Coach API."""

from datetime import datetime
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field


class GameType(str, Enum):
    LOL = "lol"
    VALORANT = "Valorant"


class Pattern(BaseModel):
    """Detected pattern in player/team data."""

    pattern_type: str
    description: str
    frequency: float = Field(ge=0.0, le=1.0)
    impact: str  # "positive", "negative", "neutral"
    games_observed: int = 0
    examples: list[dict] = Field(default_factory=list)
    recommendation: Optional[str] = None


class Insight(BaseModel):
    """Individual coaching insight."""

    title: str
    category: str  # "mechanical", "strategic", "mental", "teamwork"
    priority: str  # "high", "medium", "low"
    description: str
    data_points: list[dict] = Field(default_factory=list)
    actionable_steps: list[str] = Field(default_factory=list)


class PlayerInsightRequest(BaseModel):
    """Request for player insights."""

    player_id: str
    match_ids: list[str] = Field(default_factory=list, max_length=50)
    game: GameType
    focus_areas: list[str] = Field(default_factory=list)
    limit: int = Field(default=10, ge=1, le=50, description="Number of series to analyze")


class PlayerStats(BaseModel):
    """Aggregated player statistics."""

    games_played: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_kills: int = 0
    total_deaths: int = 0
    total_assists: int = 0
    avg_kills: float = 0.0
    avg_deaths: float = 0.0
    avg_assists: float = 0.0
    avg_kda: float = 0.0
    best_kda_game: Optional[dict] = None
    worst_kda_game: Optional[dict] = None


class AgentStats(BaseModel):
    """Stats for a specific agent/champion."""

    agent_id: str
    agent_name: str
    games_played: int = 0
    wins: int = 0
    win_rate: float = 0.0
    avg_kills: float = 0.0
    avg_deaths: float = 0.0
    avg_assists: float = 0.0
    avg_kda: float = 0.0


class RecentForm(BaseModel):
    """Recent performance trend."""

    last_5_results: list[str] = Field(default_factory=list)  # ["W", "L", "W", "W", "L"]
    form_rating: str = "neutral"  # "hot", "cold", "neutral"
    trend: str = "stable"  # "improving", "declining", "stable"
    recent_avg_kda: float = 0.0


class PlayerInsightResponse(BaseModel):
    """Response containing player insights."""

    player_id: str
    player_name: str
    game: GameType
    analysis_period: str
    stats: Optional[PlayerStats] = None
    agent_pool: list[AgentStats] = Field(default_factory=list)
    recent_form: Optional[RecentForm] = None
    patterns: list[Pattern] = Field(default_factory=list)
    insights: list[Insight] = Field(default_factory=list)
    recent_matches: list[dict] = Field(default_factory=list)
    summary: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class TeamInsightRequest(BaseModel):
    """Request for team insights."""

    team_id: str
    match_ids: list[str] = Field(default_factory=list, max_length=50)
    game: GameType
    focus_areas: list[str] = Field(default_factory=list)
    limit: int = Field(default=10, ge=1, le=50, description="Number of series to analyze")


class TeamStats(BaseModel):
    """Aggregated team statistics."""

    games_played: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_kills: int = 0
    total_deaths: int = 0
    avg_kills_per_game: float = 0.0
    avg_deaths_per_game: float = 0.0
    team_kd: float = 0.0


class PlayerSummary(BaseModel):
    """Brief player summary for roster."""

    player_id: str
    player_name: str
    games_played: int = 0
    avg_kda: float = 0.0
    win_rate: float = 0.0
    main_agents: list[str] = Field(default_factory=list)
    recent_form: str = "neutral"  # "hot", "cold", "neutral"


class TeamInsightResponse(BaseModel):
    """Response containing team insights."""

    team_id: str
    team_name: str
    game: GameType
    analysis_period: str
    team_stats: Optional[TeamStats] = None
    roster: list[PlayerSummary] = Field(default_factory=list)
    patterns: list[Pattern] = Field(default_factory=list)
    insights: list[Insight] = Field(default_factory=list)
    player_highlights: dict[str, list[str]] = Field(default_factory=dict)
    recent_matches: list[dict] = Field(default_factory=list)
    summary: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# New schemas for roster and profile endpoints
class RosterRequest(BaseModel):
    """Request for team roster."""

    game: GameType
    limit: int = Field(default=10, ge=1, le=50, description="Number of series to analyze for stats")


class RosterResponse(BaseModel):
    """Response containing team roster."""

    team_id: str
    team_name: str
    game: GameType
    players: list[PlayerSummary] = Field(default_factory=list)
    team_stats: Optional[TeamStats] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class PlayerProfileRequest(BaseModel):
    """Request for player profile."""

    game: GameType
    limit: int = Field(default=10, ge=1, le=50, description="Number of series to analyze")


class ReviewAgendaItem(BaseModel):
    """Single item in a review agenda."""

    timestamp: Optional[str] = None
    title: str
    description: str
    category: str  # "objective", "teamfight", "rotation", "economy", "execution"
    priority: str  # "critical", "important", "notable"
    players_involved: list[str] = Field(default_factory=list)
    discussion_points: list[str] = Field(default_factory=list)
    suggested_duration_minutes: int = 5


class ReviewAgenda(BaseModel):
    """Complete macro review agenda."""

    match_id: str
    game_number: int = 1
    match_outcome: str
    total_duration_minutes: int = 30
    executive_summary: str
    key_moments: list[ReviewAgendaItem] = Field(default_factory=list)
    team_level_observations: list[str] = Field(default_factory=list)
    individual_notes: dict[str, list[str]] = Field(default_factory=dict)
    priority_topics: list[str] = Field(default_factory=list)


class MacroReviewRequest(BaseModel):
    """Request for macro review generation."""

    match_id: str
    game: GameType
    game_number: int = 1
    team_id: Optional[str] = None  # Perspective for the review
    review_duration_minutes: int = Field(default=30, ge=10, le=120)


class MacroReviewResponse(BaseModel):
    """Response containing macro review."""

    agenda: ReviewAgenda
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ScenarioPrediction(BaseModel):
    """Prediction for a what-if scenario."""

    scenario_description: str
    success_probability: float = Field(ge=0.0, le=1.0)
    confidence: str  # "high", "medium", "low"
    key_factors: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    rewards: list[str] = Field(default_factory=list)
    historical_precedents: list[dict] = Field(default_factory=list)
    reasoning: str


class WhatIfRequest(BaseModel):
    """Request for what-if analysis."""

    match_id: str
    game: GameType
    timestamp: Optional[str] = None  # Game timestamp for scenario
    scenario_description: str
    game_number: int = 1


class WhatIfResponse(BaseModel):
    """Response containing what-if analysis."""

    match_id: str
    game: GameType
    original_outcome: str
    prediction: ScenarioPrediction
    alternative_scenarios: list[ScenarioPrediction] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# New schemas for additional endpoints

class MatchSummary(BaseModel):
    """Brief match/series summary for listing."""

    series_id: str
    opponent_name: str
    opponent_id: str
    date: Optional[str] = None
    result: str  # "Win", "Loss", "Ongoing"
    score: str  # "2-1", "1-2", etc.
    tournament: Optional[str] = None
    maps: list[str] = Field(default_factory=list)  # Map names for Valorant


class MatchListResponse(BaseModel):
    """Response containing list of matches."""

    team_id: str
    team_name: str
    game: GameType
    matches: list[MatchSummary] = Field(default_factory=list)
    total_count: int = 0
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class TimelineEvent(BaseModel):
    """Single event in match timeline."""

    timestamp: Optional[str] = None
    round_number: Optional[int] = None
    event_type: str  # "kill", "objective", "round_end", "spike_plant", etc.
    description: str
    team: Optional[str] = None
    players_involved: list[str] = Field(default_factory=list)
    score_after: Optional[str] = None


class MatchTimelineResponse(BaseModel):
    """Response containing match timeline."""

    series_id: str
    game_number: int = 1
    map_name: Optional[str] = None
    events: list[TimelineEvent] = Field(default_factory=list)
    final_score: str = ""
    winner: Optional[str] = None
    duration: Optional[str] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class PlayerComparisonRequest(BaseModel):
    """Request for player comparison."""

    player_ids: list[str] = Field(..., min_length=2, max_length=5)
    game: GameType
    limit: int = Field(default=10, ge=1, le=50)


class PlayerComparisonStats(BaseModel):
    """Comparison stats for a single player."""

    player_id: str
    player_name: str
    games_played: int = 0
    wins: int = 0
    win_rate: float = 0.0
    avg_kills: float = 0.0
    avg_deaths: float = 0.0
    avg_assists: float = 0.0
    avg_kda: float = 0.0
    main_agents: list[str] = Field(default_factory=list)
    recent_form: str = "neutral"

    # Valorant-specific stats
    avg_acs: Optional[float] = None  # Average Combat Score
    avg_adr: Optional[float] = None  # Average Damage per Round
    headshot_pct: Optional[float] = None  # Headshot percentage
    first_kills: Optional[int] = None  # Total first kills
    first_deaths: Optional[int] = None  # Total first deaths
    fk_fd_ratio: Optional[float] = None  # First Kill / First Death ratio
    kast_pct: Optional[float] = None  # Kill/Assist/Survive/Trade percentage
    clutch_win_pct: Optional[float] = None  # Clutch win percentage
    plants: Optional[int] = None  # Spike plants
    defuses: Optional[int] = None  # Spike defuses
    multi_kills: Optional[int] = None  # 3k+ rounds

    # LoL-specific stats
    cs_per_min: Optional[float] = None  # CS per minute
    gold_per_min: Optional[float] = None  # Gold per minute
    damage_per_min: Optional[float] = None  # Damage per minute
    vision_score: Optional[float] = None  # Average vision score
    kill_participation: Optional[float] = None  # Kill participation percentage
    damage_share: Optional[float] = None  # Damage share percentage
    gold_share: Optional[float] = None  # Gold share percentage


class PlayerComparisonResponse(BaseModel):
    """Response containing player comparisons."""

    players: list[PlayerComparisonStats] = Field(default_factory=list)
    comparison_highlights: list[str] = Field(default_factory=list)
    best_performer: Optional[str] = None
    most_consistent: Optional[str] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class PerformanceDataPoint(BaseModel):
    """Single data point for trend chart."""

    date: str
    series_id: str
    kda: float
    kills: int
    deaths: int
    assists: int
    result: str
    agent: Optional[str] = None


class TrendResponse(BaseModel):
    """Response containing performance trends."""

    player_id: str
    player_name: str
    game: GameType
    period: str
    data_points: list[PerformanceDataPoint] = Field(default_factory=list)
    trend_direction: str = "stable"  # "improving", "declining", "stable"
    avg_kda_trend: float = 0.0
    win_rate_trend: float = 0.0
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class TeamComparisonRequest(BaseModel):
    """Request for team comparison."""

    team_ids: list[str] = Field(..., min_length=2, max_length=2)
    game: GameType
    limit: int = Field(default=10, ge=1, le=50)


class TeamComparisonStats(BaseModel):
    """Comparison stats for a single team."""

    team_id: str
    team_name: str
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_kills: float = 0.0
    avg_deaths: float = 0.0
    team_kd: float = 0.0
    playstyle: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class HeadToHead(BaseModel):
    """Head-to-head record between two teams."""

    team_a_wins: int = 0
    team_b_wins: int = 0
    recent_matches: list[dict] = Field(default_factory=list)


class TeamComparisonResponse(BaseModel):
    """Response containing team comparisons."""

    teams: list[TeamComparisonStats] = Field(default_factory=list)
    head_to_head: Optional[HeadToHead] = None
    comparison_highlights: list[str] = Field(default_factory=list)
    advantage: Optional[str] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)
