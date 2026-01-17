# =============================================================================
# SECTION 1: IMPORTS AND CONFIGURATION
# =============================================================================
import os
import io
import logging
import urllib.request

import dash
from dash import dcc, html
from dash.dependencies import Output, Input

import plotly.express as px
import plotly.graph_objects as go

import dash_bootstrap_components as dbc
import pandas as pd
from flask_caching import Cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# SECTION 2: CONFIGURATION MAPPINGS
# =============================================================================

# Firm size mapping
LABEL_TO_SIZE = {
    "A": "Small", "B": "Small", "C": "Small", "D": "Small",
    "E": "Medium", "F": "Medium",
    "G": "Large"
}

# State name to abbreviation mapping for choropleth
STATE_CODES = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
    'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
    'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
    'District of Columbia': 'DC'
}

# Reverse lookup: abbreviation to state name (pre-computed for efficiency)
CODE_TO_STATE = {v: k for k, v in STATE_CODES.items()}

# =============================================================================
# SECTION 2.5: BTOS PERIOD CODE -> END DATE (NO DATES_MAP)
# =============================================================================

def btos_period_to_end_date(period_series: pd.Series) -> pd.Series:
    """
    Convert BTOS biweekly period codes (YYYYPP) into realistic end dates.

    Input examples:
      "202319"  -> year=2023, period=19
      "202526"  -> year=2025, period=26
      "202601"  -> year=2026, period=01

    We anchor period 01 as the first Sunday on or after Jan 1 of that year,
    then add 14-day steps for each subsequent period.

    end_date = first_sunday(year) + (period - 1) * 14 days
    """
    s = period_series.astype(str).str.strip()
    valid = s.str.fullmatch(r"\d{6}", na=False)

    out = pd.Series(pd.NaT, index=s.index)

    years = s[valid].str.slice(0, 4).astype(int)
    periods = s[valid].str.slice(4, 6).astype(int)

    jan1 = pd.to_datetime(years.astype(str) + "-01-01")
    days_until_sunday = (6 - jan1.dt.weekday) % 7
    first_sunday = jan1 + pd.to_timedelta(days_until_sunday, unit="D")

    end_dates = first_sunday + pd.to_timedelta((periods - 1) * 14, unit="D")

    out.loc[valid] = end_dates.values
    return out


# =============================================================================
# SECTION 2.6: ROBUST AI QUESTION NORMALIZATION (NO EXACT STRING MATCH)
# =============================================================================

def normalize_ai_question(question_series: pd.Series) -> pd.Series:
    """
    The Census sometimes changes wording slightly, adds extra spaces,
    or updates "producing goods or services" -> "any of its business functions".

    This function classifies the AI question into one of two labels using
    'contains' checks rather than exact text matches.
    """
    q = question_series.astype(str)

    used_mask = q.str.contains(
        r"In the last two weeks, did this business use Artificial Intelligence",
        case=False, na=False
    )

    intend_mask = q.str.contains(
        r"During the next six months, do you think this business will be using Artificial Intelligence",
        case=False, na=False
    )

    out = pd.Series(pd.NA, index=q.index, dtype="object")
    out.loc[used_mask] = "Used AI last 2 weeks"
    out.loc[intend_mask] = "Intend to use AI next 6 months"
    return out


# =============================================================================
# SECTION 3: APP INITIALIZATION AND CACHING
# =============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LITERA],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
        {"http-equiv": "Cache-Control", "content": "no-cache, no-store, must-revalidate"},
        {"http-equiv": "Pragma", "content": "no-cache"},
        {"http-equiv": "Expires", "content": "0"}
    ]
)
server = app.server

# Disable browser caching
@server.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

cache = Cache(app.server, config={
    "CACHE_TYPE": "filesystem",
    "CACHE_DIR": os.path.join(os.path.dirname(__file__), ".cache"),
    "CACHE_DEFAULT_TIMEOUT": 3600,  # 1 hour
    "CACHE_THRESHOLD": 50
})

# =============================================================================
# SECTION 4: DATA LOADING FUNCTIONS
# =============================================================================

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_excel_from_url(url: str) -> io.BytesIO:
    """Fetch Excel file from URL with proper headers."""
    req = urllib.request.Request(url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=60) as response:
        return io.BytesIO(response.read())


def safe_load_data(loader_func, data_name: str, fallback_df=None) -> pd.DataFrame:
    """Wrapper for safe data loading with error handling."""
    try:
        logger.info(f"Loading {data_name}...")
        data = loader_func()
        logger.info(f"Successfully loaded {data_name}: {len(data)} rows")
        return data
    except Exception as e:
        logger.error(f"Failed to load {data_name}: {str(e)}")
        return fallback_df if fallback_df is not None else pd.DataFrame()


@cache.memoize(timeout=3600)
def load_historical_ai_data() -> pd.DataFrame:
    """Load historical AI Core Questions from local file (202319-202520)."""
    return pd.read_excel("assets/AI Core Questions.xlsx").dropna(subset=["Question", "Answer"])


@cache.memoize(timeout=3600)
def load_national_data() -> pd.DataFrame:
    """
    Load national data from Census Bureau and merge with historical AI data.

    The Census Bureau changed the AI question wording in late 2025:
    - Old (202319-202520): "producing goods or services"
    - New (202521+): "any of its business functions"

    This function splices the historical local data with current remote data
    to provide a complete time series.
    """
    # Load current data from Census Bureau
    data = fetch_excel_from_url("https://www.census.gov/hfp/btos/downloads/National.xlsx")
    new_df = pd.read_excel(data)

    # Load historical AI data from local file
    try:
        old_ai = load_historical_ai_data()

        # Get period columns from each dataset
        old_periods = [c for c in old_ai.columns if str(c).isdigit() and len(str(c)) == 6]
        new_periods = [c for c in new_df.columns if str(c).isdigit() and len(str(c)) == 6]

        # Find periods only in old data (where new data has '.' for AI questions)
        # Old data covers 202319-202520, new data has values starting ~202524
        old_only_periods = [p for p in old_periods if int(p) <= 202520]
        new_only_periods = [p for p in new_periods if int(p) > 202520]

        # Normalize old AI questions to match the mapping
        old_ai = old_ai.copy()

        # Create merged AI rows with data from both sources
        merged_rows = []
        for _, old_row in old_ai.iterrows():
            # Find matching row in new data by Answer (Yes/No/Do not know)
            answer = old_row["Answer"]
            question_type = "used" if "In the last two weeks" in str(old_row["Question"]) else "intend"

            # Get matching new row
            if question_type == "used":
                new_match = new_df[
                    (new_df["Question"].str.contains("In the last two weeks", case=False, na=False)) &
                    (new_df["Answer"] == answer)
                ]
            else:
                new_match = new_df[
                    (new_df["Question"].str.contains("During the next six months", case=False, na=False)) &
                    (new_df["Answer"] == answer)
                ]

            if len(new_match) > 0:
                new_row = new_match.iloc[0].copy()
                # Copy old period values into the new row
                for p in old_only_periods:
                    if p in new_row.index:
                        new_row[p] = old_row[p]
                merged_rows.append(new_row)
            else:
                logger.warning(f"No match found for national AI data: {question_type}, Answer={answer}")

        if merged_rows:
            # Replace AI rows in new_df with merged rows
            merged_ai_df = pd.DataFrame(merged_rows)

            # Remove old AI rows from new_df
            non_ai_df = new_df[~new_df["Question"].str.contains("Artificial Intelligence", case=False, na=False)]

            # Combine non-AI rows with merged AI rows
            result = pd.concat([non_ai_df, merged_ai_df], ignore_index=True)
            logger.info(f"Merged historical AI data: {len(old_only_periods)} old periods + {len(new_only_periods)} new periods")
            return result
    except Exception as e:
        logger.warning(f"Could not load historical AI data: {e}. Using current data only.")

    return new_df


@cache.memoize(timeout=3600)
def load_states_data() -> pd.DataFrame:
    """
    Load state-level data from Census Bureau and merge with historical data.
    Historical data from AI Core Questions.xlsx covers 202319-202520.
    """
    # Load current data from Census Bureau
    data = fetch_excel_from_url("https://www.census.gov/hfp/btos/downloads/State.xlsx")
    new_df = pd.read_excel(data, na_values="S")

    try:
        # Load historical state data
        old_states = pd.read_excel(
            "assets/AI Core Questions.xlsx",
            sheet_name="State Estimates"
        ).dropna(subset=["State", "Question", "Answer"])

        # Filter to AI questions only
        old_ai = old_states[old_states["Question"].str.contains("Artificial Intelligence", case=False, na=False)]

        if old_ai.empty:
            return new_df

        # Get period columns
        old_periods = [c for c in old_ai.columns if str(c).isdigit() and len(str(c)) == 6]
        old_only_periods = [p for p in old_periods if int(p) <= 202520]

        # Merge historical data into new data
        merged_rows = []
        for _, old_row in old_ai.iterrows():
            state = old_row["State"]
            answer = old_row["Answer"]
            question_type = "used" if "In the last two weeks" in str(old_row["Question"]) else "intend"

            # Find matching row in new data
            if question_type == "used":
                new_match = new_df[
                    (new_df["State"] == state) &
                    (new_df["Question"].str.contains("In the last two weeks", case=False, na=False)) &
                    (new_df["Answer"] == answer)
                ]
            else:
                new_match = new_df[
                    (new_df["State"] == state) &
                    (new_df["Question"].str.contains("During the next six months", case=False, na=False)) &
                    (new_df["Answer"] == answer)
                ]

            if len(new_match) > 0:
                new_row = new_match.iloc[0].copy()
                for p in old_only_periods:
                    if p in new_row.index:
                        new_row[p] = old_row[p]
                merged_rows.append(new_row)
            else:
                logger.warning(f"No match found for state AI data: State={state}, {question_type}, Answer={answer}")

        if merged_rows:
            merged_ai_df = pd.DataFrame(merged_rows)
            non_ai_df = new_df[~new_df["Question"].str.contains("Artificial Intelligence", case=False, na=False)]
            result = pd.concat([non_ai_df, merged_ai_df], ignore_index=True)
            logger.info(f"Merged historical State AI data: {len(old_only_periods)} old periods")
            return result

    except Exception as e:
        logger.warning(f"Could not load historical State data: {e}")

    return new_df


@cache.memoize(timeout=3600)
def load_sector_employment_data() -> pd.DataFrame:
    """
    Load sector/employment data from Census Bureau and merge with historical data.
    Historical data from AI Core Questions.xlsx covers 202319-202520.
    """
    # Load current data from Census Bureau
    data = fetch_excel_from_url(
        "https://www.census.gov/hfp/btos/downloads/Sector%20by%20Employment%20Size%20Class.xlsx"
    )
    new_df = pd.read_excel(data, na_values="S")

    try:
        # Load historical sector x employment data
        old_sector = pd.read_excel(
            "assets/AI Core Questions.xlsx",
            sheet_name="Sector x Employment Estimates"
        ).dropna(subset=["Sector", "Empsize", "Question", "Answer"])

        # Filter to AI questions only
        old_ai = old_sector[old_sector["Question"].str.contains("Artificial Intelligence", case=False, na=False)]

        if old_ai.empty:
            return new_df

        # Get period columns
        old_periods = [c for c in old_ai.columns if str(c).isdigit() and len(str(c)) == 6]
        old_only_periods = [p for p in old_periods if int(p) <= 202520]

        # Merge historical data into new data
        merged_rows = []
        for _, old_row in old_ai.iterrows():
            sector = old_row["Sector"]
            empsize = old_row["Empsize"]
            answer = old_row["Answer"]
            question_type = "used" if "In the last two weeks" in str(old_row["Question"]) else "intend"

            # Find matching row in new data
            if question_type == "used":
                new_match = new_df[
                    (new_df["Sector"] == sector) &
                    (new_df["Empsize"] == empsize) &
                    (new_df["Question"].str.contains("In the last two weeks", case=False, na=False)) &
                    (new_df["Answer"] == answer)
                ]
            else:
                new_match = new_df[
                    (new_df["Sector"] == sector) &
                    (new_df["Empsize"] == empsize) &
                    (new_df["Question"].str.contains("During the next six months", case=False, na=False)) &
                    (new_df["Answer"] == answer)
                ]

            if len(new_match) > 0:
                new_row = new_match.iloc[0].copy()
                for p in old_only_periods:
                    if p in new_row.index:
                        new_row[p] = old_row[p]
                merged_rows.append(new_row)
            else:
                logger.warning(f"No match found for sector AI data: Sector={sector}, Empsize={empsize}, {question_type}, Answer={answer}")

        if merged_rows:
            merged_ai_df = pd.DataFrame(merged_rows)
            non_ai_df = new_df[~new_df["Question"].str.contains("Artificial Intelligence", case=False, na=False)]
            result = pd.concat([non_ai_df, merged_ai_df], ignore_index=True)
            logger.info(f"Merged historical Sector AI data: {len(old_only_periods)} old periods")
            return result

    except Exception as e:
        logger.warning(f"Could not load historical Sector data: {e}")

    return new_df


@cache.memoize(timeout=3600)
def load_naics_codes() -> pd.DataFrame:
    """Load local NAICS codes."""
    return pd.read_excel("assets/2022_NAICS_Descriptions (6).xlsx")


# =============================================================================
# SECTION 5: DATA TRANSFORMATION FUNCTIONS
# =============================================================================

def clean_percentage(series: pd.Series) -> pd.Series:
    """
    Convert percent strings like "12.3%" into numeric 12.3.
    Handles '.' or blanks by converting them to NaN.
    """
    s = series.astype(str).str.strip()
    s = s.str.rstrip("%").replace(".", pd.NA)
    return pd.to_numeric(s, errors="coerce")


def tweak_national(national_df: pd.DataFrame) -> pd.DataFrame:
    """Transform national survey data into long format."""
    return (
        national_df
        .dropna(subset=["Question", "Answer"])  # ✅ do not global dropna()
        .loc[lambda df_: df_["Question"].str.contains("Artificial Intelligence", case=False, na=False)]
        .drop(["Question ID", "Answer ID"], axis="columns", errors="ignore")
        .melt(id_vars=["Question", "Answer"], value_name="percentage", var_name="end_date")
        .assign(
            end_date=lambda df_: btos_period_to_end_date(df_["end_date"]),
            percentage=lambda df_: clean_percentage(df_["percentage"]),
            Question=lambda df_: normalize_ai_question(df_["Question"])
        )
        .dropna(subset=["end_date", "percentage", "Question"])
    )


def tweak_states(states_raw: pd.DataFrame) -> pd.DataFrame:
    """Transform state-level survey data into long format with monthly aggregation."""
    return (
        states_raw
        .drop(["Question ID", "Answer ID"], axis="columns", errors="ignore")
        .dropna(subset=["State", "Question", "Answer"])
        .loc[lambda df_: df_["Question"].str.contains("Artificial Intelligence", case=False, na=False)]
        .assign(Question=lambda df_: normalize_ai_question(df_["Question"]))
        .melt(id_vars=["State", "Question", "Answer"], value_name="percentage", var_name="end_date")
        .assign(
            end_date=lambda df_: btos_period_to_end_date(df_["end_date"]),
            percentage=lambda df_: clean_percentage(df_["percentage"])
        )
        .dropna(subset=["end_date", "percentage", "Question"])
        .groupby(["State", "Question", "Answer", pd.Grouper(key="end_date", freq="ME")])["percentage"]
        .mean()
        .reset_index()
    )


def tweak_sector_employment(sector_raw: pd.DataFrame, naics_codes: pd.DataFrame) -> pd.DataFrame:
    """Transform sector/employment survey data into long format with NAICS industry names."""
    naics_clean = (
        naics_codes
        .dropna(subset=["Code", "Title"])
        .assign(
            sector=lambda df_: df_["Code"].astype(str).str.strip(),
            title=lambda df_: df_["Title"].astype(str)
            .str.replace("T$", "", regex=True)
            .str.replace("and", "&")
        )
        .loc[lambda df_: df_["sector"].str.len() == 2]
        .drop(["Code", "Title", "Description"], axis="columns", errors="ignore")
        .reset_index(drop=True)
    )

    return (
        sector_raw
        .drop(["Question ID", "Answer ID"], axis="columns", errors="ignore")
        .dropna(subset=["Sector", "Empsize", "Question", "Answer"])
        .loc[lambda df_: df_["Question"].str.contains("Artificial Intelligence", case=False, na=False)]
        .loc[lambda df_: df_["Sector"] != "XX"]
        .assign(
            question=lambda df_: normalize_ai_question(df_["Question"]),
            emp_size=lambda df_: df_["Empsize"].map(LABEL_TO_SIZE),
            sector=lambda df_: df_["Sector"].astype(str).str.strip()
        )
        .drop(["Empsize", "Question"], axis="columns", errors="ignore")
        .melt(
            id_vars=["sector", "emp_size", "question", "Answer"],
            value_name="percentage",
            var_name="end_date"
        )
        .assign(
            end_date=lambda df_: btos_period_to_end_date(df_["end_date"]),
            percentage=lambda df_: clean_percentage(df_["percentage"])
        )
        .dropna(subset=["end_date", "percentage", "question", "emp_size", "sector"])
        .groupby(["sector", "emp_size", "question", "Answer", pd.Grouper(key="end_date", freq="ME")])["percentage"]
        .mean()
        .round(2)
        .reset_index()
        .pipe(lambda df_: pd.merge(naics_clean, df_, on="sector", how="inner"))
        .rename(columns={"title": "industry"})
        .drop("sector", axis="columns", errors="ignore")
    )


# =============================================================================
# SECTION 6: LOAD AND TRANSFORM DATA
# =============================================================================

df_national_raw = safe_load_data(load_national_data, "National Data")
states_df_raw = safe_load_data(load_states_data, "States Data")
sector_empl_raw = safe_load_data(load_sector_employment_data, "Sector Employment Data")
naics_codes = safe_load_data(load_naics_codes, "NAICS Codes")

ai_df = tweak_national(df_national_raw) if not df_national_raw.empty else pd.DataFrame()
states_df = tweak_states(states_df_raw) if not states_df_raw.empty else pd.DataFrame()
sector_empl = tweak_sector_employment(sector_empl_raw, naics_codes) if not sector_empl_raw.empty else pd.DataFrame()

ai_df_yes = ai_df.loc[ai_df["Answer"] == "Yes"] if not ai_df.empty else pd.DataFrame()
ai_df_no = ai_df.loc[ai_df["Answer"] == "No"] if not ai_df.empty else pd.DataFrame()

# Compute dynamic date range for UI header
if not ai_df.empty:
    DATA_DATE_MIN = ai_df["end_date"].min().strftime("%b %Y")
    DATA_DATE_MAX = ai_df["end_date"].max().strftime("%b %Y")
    DATA_DATE_RANGE = f"{DATA_DATE_MIN} - {DATA_DATE_MAX}"
else:
    DATA_DATE_RANGE = "No data available"

# Startup logging for debugging
logger.info(f"=== DATA LOAD SUMMARY ===")
logger.info(f"ai_df: {len(ai_df)} rows")
logger.info(f"states_df: {len(states_df)} rows")
logger.info(f"sector_empl: {len(sector_empl)} rows")
if not states_df.empty:
    logger.info(f"states_df Questions: {states_df['Question'].unique().tolist()}")
    logger.info(f"states_df date range: {states_df['end_date'].min()} to {states_df['end_date'].max()}")
    logger.info(f"states_df unique states: {states_df['State'].nunique()}")

# =============================================================================
# SECTION 7: FIGURE CREATION FUNCTIONS
# =============================================================================

def create_empty_figure(message="Select options above to view data"):
    """Create an empty figure with a centered message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template="plotly_white",
        height=400
    )
    return fig


def create_national_figure(df: pd.DataFrame, title: str):
    """Create national trend figure."""
    if df.empty:
        return create_empty_figure("No data available")

    grouped = (
        df
        .groupby(["Question", pd.Grouper(key="end_date", freq="ME")])["percentage"]
        .mean()
        .round(2)
        .reset_index()
    )

    fig = px.line(
        grouped,
        x="end_date",
        y="percentage",
        color="Question",
        template="plotly_white",
        labels={"percentage": "% of Firms", "end_date": "Month/Year"}
    )

    fig.update_layout(
        height=400,
        title_text=title,
        font=dict(family="Times New Roman", size=16),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        autosize=True,
        margin=dict(l=40, r=40, t=50, b=80)
    )
    fig.update_traces(line=dict(width=3.5))
    fig.update_xaxes(tickformat="%b %Y", title_text="Month/Year")
    return fig


fig_yes = create_national_figure(ai_df_yes, "Did you use AI? Yes")
fig_no = create_national_figure(ai_df_no, "Did you use AI? No")


# =============================================================================
# SECTION 8: LAYOUT DEFINITION
# =============================================================================

app.layout = dbc.Container([

    dbc.Row([
        dbc.Col(
            html.H1("National AI Adoption Tracker", className="text-center text-success mb-4"),
            width=12
        ),
        dbc.Col(
            html.Small(f"Data: {DATA_DATE_RANGE} ({len(ai_df)} records)", className="text-muted text-center d-block mb-2"),
            width=12
        ),
        dbc.Col(
            html.P(
                "AI is one of the transformative technologies of our times. "
                "How US businesses adopt this technology is of utmost importance. "
                "The US Census Bureau added supplemental content on AI to its Business Trends and Outlook Survey. "
                "The top two graphs show businesses responding Yes and No to the question of using AI. "
                "The map and bottom graphs allow you to explore the data by US states and by industry sectors and firm sizes.",
                className="text-primary"
            ),
            width=12
        )
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Did you use AI? Yes")),
                dbc.CardBody([
                    dcc.Loading(
                        type="circle",
                        children=[dcc.Graph(id="national-yes-fig", figure=fig_yes, config={"responsive": True})]
                    )
                ])
            ])
        ], width=6, xs=12, sm=12, md=12, lg=6, xl=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Did you use AI? No")),
                dbc.CardBody([
                    dcc.Loading(
                        type="circle",
                        children=[dcc.Graph(id="national-no-fig", figure=fig_no, config={"responsive": True})]
                    )
                ])
            ])
        ], width=6, xs=12, sm=12, md=12, lg=6, xl=6),
    ], justify="center", className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("AI Adoption by State", className="d-inline"),
                    html.Small(id="map-subtitle", children=" (Mean across all available periods)", className="text-muted ms-2")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id="map-question-dropdown",
                                placeholder="Select a question",
                                options=[
                                    {"label": "Intend to use AI next 6 months", "value": "Intend"},
                                    {"label": "Used AI last 2 weeks", "value": "Used"}
                                ],
                                value="Used",
                                className="mb-2"
                            ),
                        ], md=4),
                        dbc.Col([
                            dcc.Dropdown(
                                id="map-aggregation-dropdown",
                                placeholder="Select aggregation",
                                options=[
                                    {"label": "Mean", "value": "mean"},
                                    {"label": "Median", "value": "median"},
                                    {"label": "Max", "value": "max"}
                                ],
                                value="mean",
                                className="mb-2"
                            ),
                        ], md=4),
                        dbc.Col([
                            dbc.RadioItems(
                                id="map-answer-radio",
                                options=[
                                    {"label": "Yes", "value": "Yes"},
                                    {"label": "No", "value": "No"}
                                ],
                                value="Yes",
                                inline=True,
                                className="mt-2"
                            )
                        ], md=4),
                    ]),
                    dcc.Loading(
                        type="circle",
                        children=[
                            dcc.Graph(
                                id="choropleth-map",
                                figure=create_empty_figure("Select options to view the map"),
                                config={"responsive": True}
                            )
                        ]
                    )
                ])
            ])
        ], width=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("US States Explorer")),
                dbc.CardBody([
                    dcc.Dropdown(
                        id="state-dropdown",
                        placeholder="Select US states",
                        options=[{"label": s, "value": s} for s in sorted(states_df["State"].unique())] if not states_df.empty else [],
                        multi=True,
                        className="mb-2"
                    ),
                    dcc.Dropdown(
                        id="question-dropdown-state",
                        placeholder="Select a question",
                        options=[
                            {"label": "Intend to use AI next 6 months", "value": "Intend"},
                            {"label": "Used AI last 2 weeks", "value": "Used"}
                        ],
                        className="mb-2"
                    ),
                    html.Label("Select Answer", className="mb-1"),
                    dbc.RadioItems(
                        id="answer-radio-state",
                        options=[
                            {"label": "Yes", "value": "Yes"},
                            {"label": "No", "value": "No"},
                            {"label": "Do not know", "value": "Do not know"}
                        ],
                        value=None,
                        inline=True,
                        className="mb-3"
                    ),
                    dcc.Loading(
                        type="circle",
                        children=[
                            dcc.Graph(id="states-plot", figure=create_empty_figure(), config={"responsive": True})
                        ]
                    )
                ])
            ])
        ], width=6, xs=12, sm=12, md=12, lg=6, xl=6, className="p-2"),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Industries and Firm Sizes")),
                dbc.CardBody([
                    dcc.Dropdown(
                        id="industry-dropdown",
                        placeholder="Select an industry",
                        options=[{"label": industry, "value": industry} for industry in sorted(sector_empl["industry"].unique())] if not sector_empl.empty else [],
                        className="mb-2"
                    ),
                    dcc.Dropdown(
                        id="question-dropdown-sector",
                        placeholder="Select a question",
                        options=[{"label": q, "value": q} for q in sorted(sector_empl["question"].unique())] if not sector_empl.empty else [],
                        className="mb-2"
                    ),
                    html.Label("Select Answer", className="mb-1"),
                    dbc.RadioItems(
                        id="answer-radio-sector",
                        options=[
                            {"label": "Yes", "value": "Yes"},
                            {"label": "No", "value": "No"},
                            {"label": "Do not know", "value": "Do not know"}
                        ],
                        value=None,
                        inline=True,
                        className="mb-3"
                    ),
                    dcc.Loading(
                        type="circle",
                        children=[
                            dcc.Graph(id="sector-empl-plot", figure=create_empty_figure(), config={"responsive": True})
                        ]
                    )
                ])
            ])
        ], width=6, xs=12, sm=12, md=12, lg=6, xl=6, className="p-2")
    ], justify="center"),

    html.Div([
        dbc.Row([
            dbc.Col(
                html.Button("Data Details PDF", id="btn-pdf", className="btn btn-success"),
                width=6,
                style={"textAlign": "left"}
            ),
            dbc.Col(
                html.Img(src="/assets/aatiny.jpg", style={"marginRight": "50px"}, height="50px"),
                width=6,
                style={"textAlign": "right"}
            ),
        ], style={"position": "fixed", "bottom": 8, "left": 8, "right": 8, "zIndex": 999}),
        dcc.Download(id="download-link"),
    ])

], fluid=True)


# =============================================================================
# SECTION 9: CALLBACKS - Choropleth Map
# =============================================================================

@app.callback(
    [Output("choropleth-map", "figure"),
     Output("map-subtitle", "children")],
    [Input("map-question-dropdown", "value"),
     Input("map-answer-radio", "value"),
     Input("map-aggregation-dropdown", "value")]
)
def update_choropleth(question, answer, aggregation):
    """Update choropleth map showing aggregated data across all available periods (2023-2025)."""
    logger.info(f"update_choropleth called: question={question}, answer={answer}, aggregation={aggregation}")
    logger.info(f"states_df shape: {states_df.shape if not states_df.empty else 'EMPTY'}")

    if not all([question, answer, aggregation]) or states_df.empty:
        logger.warning("Returning empty figure - missing inputs or empty states_df")
        return create_empty_figure("Select options to view the map"), " (Select options above)"

    # Filter by question and answer - use all data from 2023-2025
    map_df = states_df[
        (states_df["Question"].str.contains(question, na=False)) &
        (states_df["Answer"] == answer)
    ].copy()

    logger.info(f"After filtering: map_df has {len(map_df)} rows")
    if not map_df.empty:
        logger.info(f"Date range: {map_df['end_date'].min()} to {map_df['end_date'].max()}")
        logger.info(f"Unique states: {map_df['State'].nunique()}")

    if map_df.empty:
        logger.warning("map_df is empty after filtering")
        return create_empty_figure("No data available for selection"), " (No data available)"

    # Get date range for subtitle
    date_min = map_df["end_date"].min()
    date_max = map_df["end_date"].max()
    date_range_str = f"{date_min.strftime('%b %Y')} - {date_max.strftime('%b %Y')}" if pd.notna(date_min) and pd.notna(date_max) else "all periods"

    # Calculate aggregated percentage per state across all time periods
    agg_label = aggregation.capitalize()
    if aggregation == "mean":
        map_df = (
            map_df
            .groupby("State")["percentage"]
            .mean()
            .round(1)
            .reset_index()
        )
    elif aggregation == "median":
        map_df = (
            map_df
            .groupby("State")["percentage"]
            .median()
            .round(1)
            .reset_index()
        )
    elif aggregation == "max":
        map_df = (
            map_df
            .groupby("State")["percentage"]
            .max()
            .round(1)
            .reset_index()
        )

    # State column already contains state codes (e.g., 'AL', 'CA')
    # Red to Yellow color scale (low values = red, high values = yellow)
    fig = px.choropleth(
        map_df,
        locations="State",
        locationmode="USA-states",
        color="percentage",
        scope="usa",
        color_continuous_scale=[[0, "red"], [0.5, "orange"], [1, "yellow"]],
        labels={"percentage": "% of Firms", "State": "State"},
        hover_data={"percentage": ":.1f"}
    )

    fig.update_layout(
        geo=dict(showlakes=True, lakecolor="rgb(255, 255, 255)"),
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_colorbar=dict(title="% of Firms", ticksuffix="%"),
        height=450
    )

    subtitle = f" ({agg_label} across {date_range_str})"
    return fig, subtitle


@app.callback(
    Output("state-dropdown", "value"),
    Input("choropleth-map", "clickData"),
    prevent_initial_call=True
)
def map_click_to_dropdown(click_data):
    """When user clicks a state on map, add it to state dropdown."""
    if click_data is None:
        return dash.no_update

    state_code = click_data["points"][0].get("location")
    if state_code:
        state_name = CODE_TO_STATE.get(state_code)
        return [state_name] if state_name else dash.no_update

    return dash.no_update


# =============================================================================
# SECTION 10: CALLBACKS - States Explorer
# =============================================================================

@app.callback(
    Output("states-plot", "figure"),
    [Input("state-dropdown", "value"),
     Input("question-dropdown-state", "value"),
     Input("answer-radio-state", "value")]
)
def update_states_plot(selected_states, selected_question, selected_answer):
    """Update states comparison plot."""
    if not (selected_states and selected_question and selected_answer) or states_df.empty:
        return create_empty_figure()

    filtered_df = states_df[
        (states_df["State"].isin(selected_states)) &
        (states_df["Question"].str.contains(selected_question, na=False)) &
        (states_df["Answer"] == selected_answer)
    ]

    if filtered_df.empty:
        return create_empty_figure("No data available for selection")

    median_value = filtered_df["percentage"].median()

    fig = px.line(
        filtered_df,
        x="end_date",
        y="percentage",
        color="State",
        title="",
        color_discrete_sequence=px.colors.qualitative.Light24,
        height=400,
        template="plotly_white",
        labels={"percentage": "Percentage", "end_date": "Month/Year", "State": "States"}
    )
    fig.update_layout(autosize=True, margin=dict(l=40, r=40, t=40, b=40))
    fig.update_xaxes(tickformat="%b %Y", title_text="Month/Year")
    fig.update_traces(line=dict(width=3.5))

    fig.add_shape(
        type="line",
        x0=filtered_df["end_date"].min(), y0=median_value,
        x1=filtered_df["end_date"].max(), y1=median_value,
        line=dict(color="red", width=2, dash="dash")
    )
    fig.add_annotation(
        x=filtered_df["end_date"].max(), y=median_value - 0.5,
        text="Selection Median",
        showarrow=False,
        font=dict(family="Times New Roman", size=12, color="red")
    )

    return fig


# =============================================================================
# SECTION 11: CALLBACKS - Sector/Industry Explorer
# =============================================================================

@app.callback(
    Output("sector-empl-plot", "figure"),
    [Input("industry-dropdown", "value"),
     Input("question-dropdown-sector", "value"),
     Input("answer-radio-sector", "value")]
)
def update_sector_plot(selected_industry, selected_question, selected_answer):
    """Update sector/industry comparison plot."""
    if not (selected_industry and selected_question and selected_answer) or sector_empl.empty:
        return create_empty_figure()

    filtered_df = sector_empl[
        (sector_empl["industry"] == selected_industry) &
        (sector_empl["question"] == selected_question) &
        (sector_empl["Answer"] == selected_answer)
    ]

    if filtered_df.empty:
        return create_empty_figure("No data available for selection")

    median_value = filtered_df["percentage"].median()

    fig = px.line(
        filtered_df,
        x="end_date",
        y="percentage",
        color="emp_size",
        color_discrete_sequence=px.colors.qualitative.Light24,
        title="",
        labels={"percentage": "Percentage", "end_date": "Month/Year", "emp_size": "Firm Size"},
        template="plotly_white",
        height=400
    )
    fig.update_layout(autosize=True, margin=dict(l=40, r=40, t=40, b=40))
    fig.update_xaxes(tickformat="%b %Y", title_text="Month/Year")
    fig.update_traces(line=dict(width=3.5))

    fig.add_shape(
        type="line",
        x0=filtered_df["end_date"].min(), y0=median_value,
        x1=filtered_df["end_date"].max(), y1=median_value,
        line=dict(color="red", width=2, dash="dash")
    )
    fig.add_annotation(
        x=filtered_df["end_date"].max(), y=median_value - 0.5,
        text="Selection Median",
        showarrow=False,
        font=dict(family="Times New Roman", size=12, color="red")
    )

    return fig


# =============================================================================
# SECTION 12: CALLBACKS - PDF Download
# =============================================================================

@app.callback(
    Output("download-link", "data"),
    Input("btn-pdf", "n_clicks"),
    prevent_initial_call=True
)
def trigger_download(n_clicks):
    """Trigger PDF download."""
    if n_clicks:
        file_path = "assets/BTOS_AI_Data_Description.pdf"
        if os.path.exists(file_path):
            return dcc.send_file(file_path)
        else:
            logger.error(f"PDF file not found: {file_path}")
            return None


# =============================================================================
# SECTION 13: APP ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)
