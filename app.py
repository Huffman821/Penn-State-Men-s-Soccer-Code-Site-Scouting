from flask import Flask, render_template, abort, request, redirect, url_for, flash, jsonify, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
import pandas as pd
import numpy as np
import os
import random
from dotenv import load_dotenv

# Attempt to import the archetype assignment utility (pandas port of your R logic).
# If it's not present the app will still run, but archetypes won't be computed.
try:
    from archetypes import assign_archetypes
    ASSIGN_ARCHETYPES_AVAILABLE = True
except Exception as e:
    print(f"archetypes module not available: {e}")
    assign_archetypes = None
    ASSIGN_ARCHETYPES_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

# Try to import TiDB database
try:
    from database_tidb import TiDBConfig, load_data_from_tidb
    TIDB_AVAILABLE = True
except Exception as e:
    print(f"TiDB dependencies not available: {e}")
    TIDB_AVAILABLE = False


app = Flask(__name__)

# Disable Jinja template caching to ensure fresh templates on every reload
app.jinja_env.cache = None

# Configure Flask with environment variables
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')

# Session security configurations
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)  # Session expires after 2 hours
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevents JavaScript access to session cookie
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection
# Set SESSION_COOKIE_SECURE to True when using HTTPS in production
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True if using HTTPS

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.refresh_view = 'login'
login_manager.needs_refresh_message = None  # Disable session refresh message
login_manager.login_message = None  # Disable "Please log in to access this page" message

# User Model
class User(UserMixin):
    def __init__(self, id, username, password_hash, is_admin=False):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.is_admin = is_admin

# Temporary in-memory user store (replace with database in production)
# Default user: username="admin", password from environment variable
users = {
    1: User(1, 'admin', generate_password_hash(os.getenv('ADMIN_PASSWORD', 'changeme')), is_admin=True)
}

@login_manager.user_loader
def load_user(user_id):
    return users.get(int(user_id))

# Check if session has expired
@app.before_request
def check_session_timeout():
    # Skip check for static files to avoid spam
    if request.path.startswith('/static/'):
        return
    
    if current_user.is_authenticated:
        from datetime import datetime
        now = datetime.now()
        
        # Check if session has a last_activity timestamp
        last_activity = session.get('last_activity')
        if last_activity:
            last_activity_time = datetime.fromisoformat(last_activity)
            timeout = app.config['PERMANENT_SESSION_LIFETIME']
            time_since_activity = now - last_activity_time
            
            print(f"[SESSION CHECK] Last activity: {last_activity_time}")
            print(f"[SESSION CHECK] Time since: {time_since_activity.total_seconds()} seconds")
            print(f"[SESSION CHECK] Timeout: {timeout.total_seconds()} seconds")
            
            # If more than timeout has passed, log out the user
            if time_since_activity > timeout:
                print(f"[SESSION EXPIRED] Logging out user")
                logout_user()
                session.clear()
                return redirect(url_for('login'))
        
        # Update last activity timestamp AFTER checking expiry
        session['last_activity'] = now.isoformat()

# Load and process the data
def load_data():
    """Load data from TiDB or fallback to CSV and compute per90 metrics"""
    if TIDB_AVAILABLE:
        try:
            query = """
    SELECT
      pi.Player                        AS player_name,
      pi.`Full_Name`                   AS full_name,
      pi.Team                          AS team,
      pi.Position                      AS position,
      pi.Role                          AS role,
      pi.Number                        AS number,
      pi.Year                          AS year,
      COALESCE(SUM(s.Minutes_played), 0)                 AS minutes_played,
      COALESCE(SUM(s.Goals), 0)                          AS goals,
      COALESCE(SUM(s.xG), 0)                             AS xg,
      COALESCE(SUM(s.Assists), 0)                        AS assists,
      COALESCE(SUM(s.xA), 0)                             AS xa,

      COALESCE(SUM(s.Actions_total), 0)                  AS actions_total,
      COALESCE(SUM(s.Actions_successful), 0)             AS actions_successful,

      COALESCE(SUM(s.Shots_total), 0)                    AS shots_total,
      COALESCE(SUM(s.Shots_on_target), 0)                AS shots_on_target,
      COALESCE(SUM(s.Shots_blocked), 0)                  AS shots_blocked,

      COALESCE(SUM(s.Passes_total), 0)                   AS passes_total,
      COALESCE(SUM(s.Passes_accurate), 0)                AS passes_accurate,

      COALESCE(SUM(s.Crosses_total), 0)                  AS crosses_total,
      COALESCE(SUM(s.Crosses_accurate), 0)               AS crosses_accurate,

      COALESCE(SUM(s.Dribbles_total), 0)                 AS dribbles_total,
      COALESCE(SUM(s.Dribbles_successful), 0)            AS dribbles_successful,

      COALESCE(SUM(s.Duels_total), 0)                    AS duels_total,
      COALESCE(SUM(s.Duels_won), 0)                      AS duels_won,

      COALESCE(SUM(s.Defensive_duels_total), 0)          AS defensive_duels_total,
      COALESCE(SUM(s.Defensive_duels_won), 0)            AS defensive_duels_won,
      COALESCE(SUM(s.Offensive_duels_total), 0)          AS offensive_duels_total,
      COALESCE(SUM(s.Offensive_duels_won), 0)            AS offensive_duels_won,
      COALESCE(SUM(s.Aerial_duels_total), 0)             AS aerial_duels_total,
      COALESCE(SUM(s.Aerial_duels_won), 0)               AS aerial_duels_won,
      COALESCE(SUM(s.Loose_ball_duels_total), 0)         AS loose_ball_duels_total,
      COALESCE(SUM(s.Loose_ball_duels_won), 0)           AS loose_ball_duels_won,

      COALESCE(SUM(s.Losses), 0)                         AS losses,
      COALESCE(SUM(s.Losses_own_half), 0)                AS losses_own_half,

      COALESCE(SUM(s.Recoveries_total), 0)               AS recoveries_total,
      COALESCE(SUM(s.Recoveries_opponent_half), 0)       AS recoveries_opponent_half,

      COALESCE(SUM(s.Touches_in_penalty_area), 0)        AS touches_in_penalty_area,
      COALESCE(SUM(s.Offsides), 0)                       AS offsides,

      COALESCE(SUM(s.Yellow_cards), 0)                   AS yellow_cards,
      COALESCE(SUM(s.Red_cards), 0)                      AS red_cards,
      COALESCE(SUM(s.Fouls_committed), 0)                AS fouls_committed,
      COALESCE(SUM(s.Fouls_suffered), 0)                 AS fouls_suffered,

      COALESCE(SUM(s.Interceptions), 0)                  AS interceptions,
      COALESCE(SUM(s.Clearances), 0)                     AS clearances,
      COALESCE(SUM(s.Sliding_tackles_total), 0)          AS sliding_tackles_total,
      COALESCE(SUM(s.Sliding_tackles_won), 0)            AS sliding_tackles_won,

      COALESCE(SUM(s.Free_kicks), 0)                     AS free_kicks,
      COALESCE(SUM(s.Direct_free_kicks), 0)              AS direct_free_kicks,
      COALESCE(SUM(s.Corners_served), 0)                 AS corners_served,
      COALESCE(SUM(s.Throw_ins), 0)                      AS throw_ins,

      COALESCE(SUM(s.Forward_passes_total), 0)           AS forward_passes_total,
      COALESCE(SUM(s.Forward_passes_accurate), 0)        AS forward_passes_accurate,
      COALESCE(SUM(s.Back_passes_total), 0)              AS back_passes_total,
      COALESCE(SUM(s.Back_passes_accurate), 0)           AS back_passes_accurate,
      COALESCE(SUM(s.Lateral_passes_total), 0)           AS lateral_passes_total,
      COALESCE(SUM(s.Lateral_passes_accurate), 0)        AS lateral_passes_accurate,
      COALESCE(SUM(s.Short_medium_passes_total), 0)      AS short_medium_passes_total,
      COALESCE(SUM(s.Short_medium_passes_accurate), 0)   AS short_medium_passes_accurate,
      COALESCE(SUM(s.Long_passes_total), 0)              AS long_passes_total,
      COALESCE(SUM(s.Long_passes_accurate), 0)           AS long_passes_accurate,
      COALESCE(SUM(s.Progressive_passes_total), 0)       AS progressive_passes_total,
      COALESCE(SUM(s.Progressive_passes_accurate), 0)    AS progressive_passes_accurate,
      COALESCE(SUM(s.Passes_final_third_total), 0)       AS passes_final_third_total,
      COALESCE(SUM(s.Passes_final_third_accurate), 0)    AS passes_final_third_accurate,
      COALESCE(SUM(s.Through_passes_total), 0)           AS through_passes_total,
      COALESCE(SUM(s.Through_passes_accurate), 0)        AS through_passes_accurate,
      COALESCE(SUM(s.Deep_completions), 0)               AS deep_completions,
      COALESCE(SUM(s.Key_passes), 0)                     AS key_passes,
      COALESCE(SUM(s.Second_third_assists), 0)           AS second_third_assists,
            COALESCE(AVG(s.Average_pass_length), 0)            AS average_pass_length,

            /* Ratings from analytics.player_ratings */
            COALESCE(MAX(pr.Offensive_Grade), 0)               AS offensive_grade,
            COALESCE(MAX(pr.Passing_Grade), 0)                 AS passing_grade,
            COALESCE(MAX(pr.Defensive_Grade), 0)               AS defensive_grade,
            COALESCE(MAX(pr.Duel_Grade), 0)                    AS duel_grade,
            COALESCE(MAX(pr.Discipline_Grade), 0)              AS discipline_grade,
            COALESCE(MAX(pr.Rating), 0)                        AS rating
        FROM player_info pi
        LEFT JOIN player_totals s
      ON s.Player = pi.Player AND s.Team = pi.Team
            LEFT JOIN analytics.player_ratings pr
                ON LOWER(TRIM(pr.Player)) = LOWER(TRIM(pi.Player))
    WHERE pi.Team = 'Penn State'
    GROUP BY pi.Player, pi.`Full_Name`, pi.Team, pi.Position, pi.Role, pi.Number, pi.Year;
    """
            df = load_data_from_tidb(query)
            data_source = "tidb"

        except Exception as e:
            print(f"TiDB not available, falling back to CSV: {e}")
            try:
                df = pd.read_csv('data/pennstate_mercyhurst.csv')
                print("📄 Data loaded from local CSV")
                data_source = "csv"
            except FileNotFoundError:
                print("⚠️ No data files available - creating empty dataset")
                df = pd.DataFrame(columns=[
                    'player_name','minutes_played','goals','xg','assists','xa',
                    'actions_attempted','successful_actions','shots_attempted',
                    'shots_on_target','passes_attempted','passes_completed',
                    'crosses_attempted','crosses_successful','team'
                ])
                data_source = "empty"
    else:
        print("📄 TiDB dependencies not available, checking for CSV")
        try:
            df = pd.read_csv('data/pennstate_mercyhurst.csv')
            print("📄 Data loaded from local CSV")
            data_source = "csv"
        except FileNotFoundError:
            print("⚠️ No data files available - creating empty dataset")
            df = pd.DataFrame(columns=[
                'player_name','minutes_played','goals','xg','assists','xa',
                'actions_attempted','successful_actions','shots_attempted',
                'shots_on_target','passes_attempted','passes_completed',
                'crosses_attempted','crosses_successful','team'
            ])
            data_source = "empty"

    # Only rename columns if data comes from CSV
    if data_source == "csv":
        column_mapping = {
            'Player': 'player_name',
            'Minutes_played': 'minutes_played',
            'Goals': 'goals',
            'xG': 'xg',
            'Assists': 'assists',
            'xA': 'xa',
            'Actions_total': 'actions_attempted',
            'Actions_successful': 'successful_actions',
            'Shots_total': 'shots_attempted',
            'Shots_on_target': 'shots_on_target',
            'Passes_total': 'passes_attempted',
            'Passes_accurate': 'passes_completed',
            'Crosses_total': 'crosses_attempted',
            'Crosses_accurate': 'crosses_successful',
            'Dribbles_total': 'dribbles_attempted',
            'Dribbles_successful': 'dribbles_successful',
            'Duels_total': 'duels_attempted',
            'Duels_won': 'duels_won',
            'Losses_own_half': 'losses_own_half',
            'Recoveries_own_half': 'recoveries_own_half',
            'Recoveries_opponent_half': 'recoveries_opponent_half',
            'Touches_in_penalty_area': 'touches_penalty_area',
            'Offsides': 'offsides',
            'Yellow_cards': 'cards',
            'Red_cards': 'red_cards',
            'Defensive_duels_total': 'defensive_duels_total',
            'Defensive_duels_won': 'defensive_duels_won',
            'Offensive_duels_total': 'offensive_duels_total',
            'Offensive_duels_won': 'offensive_duels_won',
            'Aerial_duels_total': 'aerial_duels_total',
            'Aerial_duels_won': 'aerial_duels_won',
            'Loose_ball_duels_total': 'loose_ball_duels_total',
            'Loose_ball_duels_won': 'loose_ball_duels_won',
            'Shots_blocked': 'shots_blocked',
            'Interceptions': 'interceptions',
            'Clearances': 'clearances',
            'Sliding_tackles_total': 'sliding_tackles_total',
            'Sliding_tackles_won': 'sliding_tackles_won',
            'Fouls_committed': 'fouls_committed',
            'Fouls_suffered': 'fouls_suffered',
            'Free_kicks': 'free_kicks',
            'Direct_free_kicks': 'direct_free_kicks',
            'Corners_served': 'corners_served',
            'Throw_ins': 'throw_ins',
            'Forward_passes_total': 'forward_passes_total',
            'Forward_passes_accurate': 'forward_passes_accurate',
            'Back_passes_total': 'back_passes_total',
            'Back_passes_accurate': 'back_passes_accurate',
            'Lateral_passes_total': 'lateral_passes_total',
            'Lateral_passes_accurate': 'lateral_passes_accurate',
            'Short_medium_passes_total': 'short_med_passes_total',
            'Short_medium_passes_accurate': 'short_med_passes_accurate',
            'Long_passes_total': 'long_passes_total',
            'Long_passes_accurate': 'long_passes_accurate',
            'Progressive_passes_total': 'progressive_passes_total',
            'Progressive_passes_accurate': 'progressive_passes_accurate',
            'Passes_final_third_total': 'passes_final_third_total',
            'Passes_final_third_accurate': 'passes_final_third_accurate',
            'Through_passes_total': 'through_passes_total',
            'Through_passes_accurate': 'through_passes_accurate',
            'Deep_completions': 'deep_completions',
            'Key_passes': 'key_passes',
            'Second_third_assists': 'second_third_assists',
            'Shot_assists': 'shot_assists',
            'Average_pass_length': 'average_pass_length'
        }
        df = df.rename(columns=column_mapping)
        # Ensure rating columns exist with default 0 in CSV mode
    for col in ['offensive_grade','passing_grade','defensive_grade','duel_grade','discipline_grade','rating']:
            if col not in df.columns:
                df[col] = 0.0
    else:
        db_mapping = {
            'player': 'player_name',
            'shots_total': 'shots_attempted',
            'actions_total': 'actions_attempted',
            'actions_successful': 'successful_actions',
            'passes_total': 'passes_attempted',
            'passes_accurate': 'passes_completed',
            'crosses_total': 'crosses_attempted',
            'crosses_accurate': 'crosses_successful',
            'dribbles_total': 'dribbles_attempted',
            'duels_total': 'duels_attempted',
            'touches_in_penalty_area': 'touches_penalty_area',
            'yellow_cards': 'cards'
        }
        df = df.rename(columns=db_mapping)

    # Convert numeric columns with error handling
    numeric_columns = [
        'number', 'minutes_played', 'goals', 'xg', 'assists', 'xa',
        'actions_attempted', 'successful_actions',
        'shots_attempted', 'shots_on_target', 'shots_blocked',
        'passes_attempted', 'passes_completed',
        'crosses_attempted', 'crosses_successful',
        'dribbles_attempted', 'dribbles_successful',
        'duels_attempted', 'duels_won',
        'defensive_duels_total', 'defensive_duels_won',
        'offensive_duels_total', 'offensive_duels_won',
        'aerial_duels_total', 'aerial_duels_won',
        'loose_ball_duels_total', 'loose_ball_duels_won',
        'losses_own_half', 'recoveries_own_half', 'recoveries_opponent_half',
        'touches_penalty_area', 'offsides', 'cards', 'red_cards',
        'interceptions', 'clearances', 'sliding_tackles_total', 'sliding_tackles_won',
        'fouls_committed', 'fouls_suffered', 'free_kicks', 'direct_free_kicks',
        'corners_served', 'throw_ins',
        'forward_passes_total', 'forward_passes_accurate',
        'back_passes_total', 'back_passes_accurate',
        'lateral_passes_total', 'lateral_passes_accurate',
        'short_med_passes_total', 'short_med_passes_accurate',
        'long_passes_total', 'long_passes_accurate',
        'progressive_passes_total', 'progressive_passes_accurate',
        'passes_final_third_total', 'passes_final_third_accurate',
        'through_passes_total', 'through_passes_accurate',
        'deep_completions', 'key_passes', 'second_third_assists',
    'shot_assists', 'average_pass_length',
    # Ratings
    'offensive_grade','passing_grade','defensive_grade','duel_grade','discipline_grade','rating'
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if data_source == "empty":
        print("⚠️ Running with empty dataset - website will show 'no data available'")

    # ---------------------
    # Compute per90 columns
    # ---------------------
    # Define candidate raw stat names we want per90 for (we will only compute those present)
    candidate_stats = [
        "goals","xg","shots_on_target","shots_attempted","shots_total","assists","xa",
        "touches_penalty_area","touches_in_penalty_area","dribbles_successful","dribbles_attempted","key_passes",
        "progressive_passes_accurate","progressive_passes_total","passes_final_third_accurate","passes_final_third_total",
        "duels_won","defensive_duels_won","recoveries_opponent_half","recoveries_total","interceptions",
        "aerial_duels_won","aerial_duels_total","clearances","crosses_attempted","crosses_total","passes_attempted",
        "passes_total","forward_passes_total","dribbles_total","duels_total"
    ]

    # Normalize column names in df to ensure we catch variants
    existing_cols = set(df.columns)

    per90_created = []
    for raw in candidate_stats:
        if raw in existing_cols and 'minutes_played' in existing_cols:
            per90_col = raw
            # normalize the per90 column name to a stable suffix
            per90_name = per90_col if per90_col.endswith('_per90') else f"{per90_col}_per90"
            # compute per90 safely
            df[per90_name] = np.where(df['minutes_played'] > 0,
                                      df[raw].fillna(0) / df['minutes_played'].fillna(0) * 90,
                                      0.0)
            per90_created.append(per90_name)

    # If no per90 columns were created, attempt to compute from common alternative stat names
    # (e.g., if 'shots_total' is not present but 'shots_attempted' is)
    if not per90_created:
        fallback_map = {
            'shots_total': ['shots_attempted', 'shots_total'],
            'touches_penalty_area': ['touches_in_penalty_area', 'touches_penalty_area'],
            'passes_total': ['passes_attempted','passes_total'],
        }
        for out_col, candidates in fallback_map.items():
            for c in candidates:
                if c in existing_cols and 'minutes_played' in existing_cols:
                    per90_name = f"{out_col}_per90"
                    df[per90_name] = np.where(df['minutes_played'] > 0,
                                              df[c].fillna(0) / df['minutes_played'].fillna(0) * 90,
                                              0.0)
                    per90_created.append(per90_name)
                    break

    # Ensure string columns exist for matching
    if 'player_name' not in df.columns and 'player' in df.columns:
        df = df.rename(columns={'player': 'player_name'})

    # Fill NaNs for safe templating later
    df['player_name'] = df['player_name'].fillna('').astype(str)
    if 'full_name' in df.columns:
        df['full_name'] = df['full_name'].fillna('').astype(str)

    return df

# Load data once when the app starts
players_df = load_data()
print(f"📊 Loaded {len(players_df)} Penn State players for the app")
print(f"🔍 Available columns: {list(players_df.columns)}")
if len(players_df) > 0:
    print(f"🔍 Sample player data: {players_df.iloc[0].to_dict()}")

# ---------------------------------------------------------------------
# Integration: compute archetypes (on-demand or cached) and use them in
# the player detail route. This follows the snippet you requested.
# ---------------------------------------------------------------------
# Cache for dataframe with archetypes applied (computed once)
_df_with_archetypes = None

def load_players_df():
    """
    Adapter used by the archetype integration snippet.
    Returns the main players dataframe used by the app.
    """
    global players_df
    return players_df

# ---- Projection / Predictive model (server-side) ----
def project_lineup(df, players, minutes_default=90):
    """
    Simple coach-facing projection:
      - baseline: mean of each per90 stat across selected players
      - synergy_mult: random uniform(0.9, 1.1) (placeholder)
      - minutes_factor: avg(minutes_played) / minutes_default
    Returns dict ready for JSON serialization.
    """
    if df is None or df.empty:
        raise ValueError("Player database is empty")

    # Normalize lookup keys
    players_normalized = [p.strip().lower() for p in players if isinstance(p, str) and p.strip()]
    if not players_normalized:
        raise ValueError("No player names provided")

    # Create helper lowercase name columns for case-insensitive matching
    df['_name_key'] = df['player_name'].astype(str).str.strip().str.lower()
    if 'full_name' in df.columns:
        df['_full_name_key'] = df['full_name'].astype(str).str.strip().str.lower()
    else:
        df['_full_name_key'] = ''

    # Select rows that match either player_name or full_name (case-insensitive)
    mask = df['_name_key'].isin(players_normalized) | df['_full_name_key'].isin(players_normalized)
    lineup_df = df[mask].copy()

    if lineup_df.empty:
        # Try fuzzy fallback: include rows whose names contain provided tokens
        possible = []
        for token in players_normalized:
            possible.extend(df[df['_name_key'].str.contains(token, na=False)].index.tolist())
            possible.extend(df[df['_full_name_key'].str.contains(token, na=False)].index.tolist())
        if possible:
            lineup_df = df.loc[sorted(set(possible))].copy()

    if lineup_df.empty:
        raise ValueError("No matching players found in data.")

    # Find per90 columns
    per90_cols = [c for c in lineup_df.columns if c.endswith('_per90')]
    if not per90_cols:
        raise ValueError("No per90 columns available in dataset. Ensure per90 stats were computed.")

    # Baseline mean across players
    baseline = lineup_df[per90_cols].mean(skipna=True)

    synergy_mult = float(random.uniform(0.9, 1.1))
    avg_minutes = float(lineup_df['minutes_played'].mean() if 'minutes_played' in lineup_df.columns else minutes_default)
    minutes_factor = avg_minutes / float(minutes_default) if minutes_default and minutes_default != 0 else 1.0

    projected = baseline * synergy_mult * minutes_factor

    # Convert to JSON-serializable dict of floats
    projected_dict = {}
    for k, v in projected.items():
        try:
            projected_dict[k] = None if pd.isna(v) else float(v)
        except Exception:
            projected_dict[k] = None

    result = {
        "projected_team_per90": projected_dict,
        "synergy_mult": float(synergy_mult),
        "avg_minutes": float(avg_minutes),
        "minutes_factor": float(minutes_factor),
        "players_used": lineup_df['player_name'].astype(str).tolist()
    }

    # Clean up helper columns in original df (optional)
    df.drop(columns=['_name_key','_full_name_key'], errors='ignore', inplace=True)

    return result

# ---- Authentication Routes ----
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Find user by username
        user = None
        for u in users.values():
            if u.username == username:
                user = u
                break
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=False)  # Don't use persistent "remember me" cookie
            session.permanent = True  # Makes session respect PERMANENT_SESSION_LIFETIME
            from datetime import datetime
            session['last_activity'] = datetime.now().isoformat()  # Initialize timestamp
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()  # Clear all session data
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login'))

# ---- Main Routes ----
@app.route('/')
@login_required
def index():
    return render_template('index.html', players=players_df.to_dict('records'))

# Replace the existing player_detail route in your app.py with the code below.
# This version derives an 'sPosition' column from available 'position'/'role'
# fields and ensures the raw stat columns expected by the archetype code exist
# (filling missing ones with zeros). That fixes the "Unknown" archetype result
# caused by missing/incorrect sPosition or missing raw stats.

@app.route('/player/<player_name>')
@login_required
def player_detail(player_name):
    """
    Player detail route updated to compute archetypes (using assign_archetypes)
    and cache the dataframe with archetypes applied. Falls back to the original
    players_df if archetype computation is unavailable.

    Important: This function now prepares the dataframe for assign_archetypes by:
      - creating an sPosition column (ST, AM, WM, CM, FB, CB) from position/role
      - making sure required raw stat columns exist (set to 0 if missing)
    """
    global _df_with_archetypes

    # Compute dataframe with archetypes once (cached)
    if _df_with_archetypes is None:
        try:
            base_df = load_players_df().copy()

            # --- Ensure sPosition exists and maps to archetype groups ---
            def derive_sposition(row):
                pos = str(row.get('position') or '').strip().upper()
                role = str(row.get('role') or '').strip().upper()

                # Normalize common abbreviations
                if pos in ['F', 'FW', 'FORWARD', 'FORW']:
                    # Try role hints
                    if any(k in role for k in ['WING', 'WINGER', 'RM', 'LM', 'RW', 'LW']):
                        return 'WM'
                    if any(k in role for k in ['CAM', 'AM', 'ATTACKING', 'SUPPORT']):
                        return 'AM'
                    return 'ST'
                if pos in ['M', 'MF', 'MID', 'MIDFIELDER', 'MIDFIELD']:
                    if any(k in role for k in ['CAM', 'AM', 'ATTACKING']):
                        return 'AM'
                    if any(k in role for k in ['WING', 'WINGER', 'RM', 'LM', 'RW', 'LW']):
                        return 'WM'
                    return 'CM'
                if pos in ['D', 'DEF', 'DEFENDER', 'DF']:
                    # Role hints for fullbacks/wingbacks
                    if any(k in role for k in ['FB', 'FULLBACK', 'WB', 'WINGBACK', 'LB', 'RB', 'LEFT BACK', 'RIGHT BACK']):
                        return 'FB'
                    return 'CB'
                # If the pos isn't a single-letter code, inspect role text
                if any(k in role for k in ['STRIKER', 'ST', 'POACHER', 'TARGET']):
                    return 'ST'
                if any(k in role for k in ['WING', 'WINGER', 'WIDE', 'OUTSIDE']):
                    return 'WM'
                if any(k in role for k in ['CAM', 'ATTACK', 'ATTACKING', 'SUPPORT STRIKER']):
                    return 'AM'
                if any(k in role for k in ['CB', 'CENTRE BACK', 'CENTER BACK', 'CENTER-BACK', 'CENTRE-BACK']):
                    return 'CB'
                if any(k in role for k in ['FULLBACK', 'BACK', 'LB', 'RB', 'WINGBACK']):
                    return 'FB'
                # Reasonable default: treat generic midfielders as CM, forwards as ST, defenders as CB
                if pos.startswith('F'):
                    return 'ST'
                if pos.startswith('M'):
                    return 'CM'
                if pos.startswith('D'):
                    return 'CB'
                # Fallback to CM to keep player within a non-Unknown category for archetype scoring
                return 'CM'

            if 'sPosition' not in base_df.columns:
                base_df['sPosition'] = base_df.apply(derive_sposition, axis=1)
            else:
                # Normalize existing sPosition strings to expected uppercase tokens
                base_df['sPosition'] = base_df['sPosition'].astype(str).str.strip().str.upper().replace({
                    'FORWARD': 'ST', 'FW': 'ST', 'ATTACKER': 'ST',
                    'MID': 'CM', 'MIDFIELDER': 'CM',
                    'DEFENDER': 'CB', 'FULLBACK': 'FB'
                })

            # --- Ensure required raw stat columns exist (assign_archetypes expects these names) ---
            required_raw_cols = [
                'goals','xg','shots_on_target','shots_total','shots_attempted',
                'touches_penalty_area','touches_in_penalty_area',
                'dribbles_successful','dribbles_attempted','key_passes','xa',
                'progressive_passes_accurate','progressive_passes_total',
                'passes_final_third_accurate','passes_final_third_total',
                'duels_won','defensive_duels_won','recoveries_opponent_half',
                'recoveries_total','interceptions','aerial_duels_won','aerial_duels_total',
                'clearances','crosses_total','crosses_attempted','passes_attempted','passes_total'
            ]

            for col in required_raw_cols:
                if col not in base_df.columns:
                    # create missing numeric columns as zeros to avoid KeyErrors and NaNs
                    base_df[col] = 0.0

            # Also create numeric defaults for any columns that could be passed as ints/floats but have NA
            numeric_defaults = ['minutes_played']
            for col in numeric_defaults:
                if col not in base_df.columns:
                    base_df[col] = 0
                base_df[col] = pd.to_numeric(base_df[col], errors='coerce').fillna(0)

            # If archetype assignment is available, compute and cache it
            if ASSIGN_ARCHETYPES_AVAILABLE and assign_archetypes is not None:
                try:
                    _df_with_archetypes = assign_archetypes(base_df.copy())
                    print("✅ Archetypes computed and cached for players_df")
                except Exception as e:
                    print(f"Failed to compute archetypes, continuing without them: {e}")
                    _df_with_archetypes = base_df
            else:
                _df_with_archetypes = base_df

        except Exception as e:
            print(f"Error loading players for archetypes: {e}")
            _df_with_archetypes = players_df

    # Try exact match first (player_name)
    player_data = _df_with_archetypes[_df_with_archetypes['player_name'] == player_name]
    if player_data.empty:
        # Case-insensitive lookup & full_name fallback
        name_key = str(player_name).strip().lower()
        _df_with_archetypes['_name_key'] = _df_with_archetypes['player_name'].astype(str).str.strip().str.lower()
        if 'full_name' in _df_with_archetypes.columns:
            _df_with_archetypes['_full_name_key'] = _df_with_archetypes['full_name'].astype(str).str.strip().str.lower()
            player_data = _df_with_archetypes[
                (_df_with_archetypes['_name_key'] == name_key) | (_df_with_archetypes['_full_name_key'] == name_key)
            ]
        else:
            player_data = _df_with_archetypes[_df_with_archetypes['_name_key'] == name_key]

    if player_data.empty:
        abort(404)

    player = player_data.iloc[0].to_dict()

    # Fetch percentile data for pizza plot (same behaviour as before)
    percentiles_data = {}
    if TIDB_AVAILABLE:
        try:
            percentile_query = """
            SELECT *
            FROM analytics.player_percentiles
            WHERE LOWER(TRIM(player)) = LOWER(TRIM(%s))
            LIMIT 1
            """
            percentiles_df = load_data_from_tidb(percentile_query, params=(player_name,))
            if not percentiles_df.empty:
                percentiles_data = percentiles_df.iloc[0].to_dict()
        except Exception as e:
            print(f"Could not load percentiles: {e}")
            percentiles_data = {}

    # Derived stats (kept the same)
    stats = {}
    stats['shot_accuracy'] = round((player.get('shots_on_target',0) / player.get('shots_attempted',0)) * 100, 1) if player.get('shots_attempted',0) > 0 else 0
    stats['pass_completion'] = round((player.get('passes_completed',0) / player.get('passes_attempted',0)) * 100, 1) if player.get('passes_attempted',0) > 0 else 0
    stats['cross_success'] = round((player.get('crosses_successful',0) / player.get('crosses_attempted',0)) * 100, 1) if player.get('crosses_attempted',0) > 0 else 0
    stats['dribble_success'] = round((player.get('dribbles_successful',0) / player.get('dribbles_attempted',0)) * 100, 1) if player.get('dribbles_attempted',0) > 0 else 0
    stats['duel_win_rate'] = round((player.get('duels_won',0) / player.get('duels_attempted',0)) * 100, 1) if player.get('duels_attempted',0) > 0 else 0
    stats['action_success'] = round((player.get('successful_actions',0) / player.get('actions_attempted',0)) * 100, 1) if player.get('actions_attempted',0) > 0 else 0
    stats['defensive_duel_win_rate'] = round((player.get('defensive_duels_won',0) / player.get('defensive_duels_total',0)) * 100, 1) if player.get('defensive_duels_total',0) > 0 else 0
    stats['offensive_duel_win_rate'] = round((player.get('offensive_duels_won',0) / player.get('offensive_duels_total',0)) * 100, 1) if player.get('offensive_duels_total',0) > 0 else 0
    stats['aerial_duel_win_rate'] = round((player.get('aerial_duels_won',0) / player.get('aerial_duels_total',0)) * 100, 1) if player.get('aerial_duels_total',0) > 0 else 0
    stats['progressive_pass_success'] = round((player.get('progressive_passes_accurate',0) / player.get('progressive_passes_total',0)) * 100, 1) if player.get('progressive_passes_total',0) > 0 else 0
    stats['sliding_tackle_success'] = round((player.get('sliding_tackles_won',0) / player.get('sliding_tackles_total',0)) * 100, 1) if player.get('sliding_tackles_total',0) > 0 else 0

    def calculate_percentile(player_value, stat_column):
        if stat_column not in players_df.columns or stat_column is None:
            return 0
        active_players = players_df[players_df['minutes_played'] > 0]
        if len(active_players) <= 1:
            return 50
        rank = (active_players[stat_column] < player_value).sum()
        return min(max(round((rank / len(active_players)) * 100), 1), 99)

    percentiles = {}
    key_stats = ['goals', 'xg', 'assists', 'xa', 'shots_attempted', 'shots_on_target',
                 'passes_completed', 'key_passes', 'crosses_successful',
                 'interceptions', 'clearances', 'duels_won']
    for stat in key_stats:
        percentiles[f'{stat}_percentile'] = calculate_percentile(player.get(stat,0), stat) if stat in player else 0

    if player.get('passes_attempted',0) > 0:
        pass_accuracy = (player.get('passes_completed',0) / player.get('passes_attempted',0)) * 100
        active_players = players_df[players_df['minutes_played'] > 0]
        pass_accuracies = [(p['passes_completed'] / p['passes_attempted']) * 100 for _, p in active_players.iterrows() if p.get('passes_attempted',0) > 0]
        if pass_accuracies:
            rank = sum(1 for acc in pass_accuracies if acc < pass_accuracy)
            percentiles['pass_accuracy_percentile'] = min(max(round((rank / len(pass_accuracies)) * 100), 1), 99)

    if player.get('duels_attempted',0) > 0:
        duel_win_rate = (player.get('duels_won',0) / player.get('duels_attempted',0)) * 100
        active_players = players_df[players_df['minutes_played'] > 0]
        duel_rates = [(p['duels_won'] / p['duels_attempted']) * 100 for _, p in active_players.iterrows() if p.get('duels_attempted',0) > 0]
        if duel_rates:
            rank = sum(1 for rate in duel_rates if rate < duel_win_rate)
            percentiles['duel_win_rate_percentile'] = min(max(round((rank / len(duel_rates)) * 100), 1), 99)

    # Render using the same template and variables as before
    return render_template('player.html', player=player, stats=stats, percentiles=percentiles, percentiles_data=percentiles_data)

@app.route('/player-comparison')
@login_required
def player_comparison():
    return render_template('player_comparison.html', players=players_df.to_dict('records'))

@app.route('/api/player/<player_name>')
@login_required
def api_player(player_name):
    """API endpoint to get player data with percentiles"""
    player_data = players_df[players_df['player_name'] == player_name]
    if player_data.empty:
        return {"error": "Player not found"}, 404
    
    player = player_data.iloc[0].to_dict()
    
    # Fetch percentiles
    percentiles = {}
    if TIDB_AVAILABLE:
        try:
            percentile_query = """
            SELECT *
            FROM analytics.player_percentiles
            WHERE LOWER(TRIM(player)) = LOWER(TRIM(%s))
            LIMIT 1
            """
            percentiles_df = load_data_from_tidb(percentile_query, params=(player_name,))
            if not percentiles_df.empty:
                percentiles = percentiles_df.iloc[0].to_dict()
        except Exception as e:
            print(f"Could not load percentiles: {e}")
    
    # Convert numpy types to native Python types for JSON serialization
    player_dict = {}
    for key, value in player.items():
        if pd.isna(value):
            player_dict[key] = None
        elif isinstance(value, (np.integer, np.floating)):
            player_dict[key] = float(value)
        else:
            player_dict[key] = value
    
    player_dict['percentiles'] = percentiles
    
    return player_dict, 200

@app.route('/squad-builder')
@login_required
def squad_builder():
    # squad_builder.html will receive the full players_df records (including any computed per90 columns)
    return render_template('squad_builder.html', players=players_df.to_dict('records'))

@app.route('/scouting')
def scouting():
    return render_template('scouting.html', players=players_df.to_dict('records'))


@app.route('/teams/b1g')
def big_ten():
    """Simple Big Ten overview page."""
    # Example team list — template can expand this with logos/links
    teams = [
        {'slug': 'rutgers', 'name': 'Rutgers'},
        {'slug': 'michigan', 'name': 'Michigan'},
        {'slug': 'maryland', 'name': 'Maryland'},
        {'slug': 'ohio-state', 'name': 'Ohio State'},
        {'slug': 'indiana', 'name': 'Indiana'},
        {'slug': 'michigan-state', 'name': 'Michigan State'},
        {'slug': 'northwestern', 'name': 'Northwestern'},
        {'slug': 'wisconsin', 'name': 'Wisconsin'},
        {'slug': 'washington', 'name': 'Washington'},
    ]
    # Pass players so the template can render sample player cards and lists
    return render_template('big10.html', teams=teams, players=players_df.to_dict('records'))

@app.route('/health/db')
def database_health():
    try:
        cfg = TiDBConfig()
        cfg.test_connection()
        return {"status": "healthy", "database": "connected"}, 200
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500



# ---- Server-side prediction endpoint ----
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """
    Accepts JSON payload: { "players": ["A Name","B Name",...], "minutes_default": 90 }
    Returns projected per90 team stats, synergy multiplier, avg minutes, etc.
    """
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "Invalid JSON payload"}), 400
    players = payload.get('players') or []
    minutes_default = payload.get('minutes_default', 90)

    try:
        result = project_lineup(players_df, players, minutes_default=minutes_default)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"[PREDICT ERROR] {e}")
        return jsonify({"error": "Server error during projection"}), 500

    return jsonify(result), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)