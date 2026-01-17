# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

AI Dashboard displaying US Census Bureau Business Trends and Outlook Survey (BTOS) data on AI adoption by US businesses. Built with Dash/Plotly for interactive data visualization.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server (port 8050)
python app.py

# Production deployment uses Procfile with gunicorn
```

## Architecture

**Single-file Dash application** (`app.py`) with ~440 lines organized as:

1. **Data Loading & Transformation (Lines 1-159)**: Fetches three Excel datasets from US Census Bureau URLs plus local NAICS lookup, then cleans and transforms using pandas pipelines
2. **Layout Definition (Lines 247-301)**: Bootstrap grid with 4 cards (national yes/no graphs, state explorer, sector/firm size explorer) plus fixed footer
3. **Callbacks (Lines 303-438)**: Interactive filtering for state/sector visualizations with median reference lines

**Data Sources:**
- Remote: Census BTOS national, state, and sector/employment size Excel files
- Local: `assets/NAICS_descriptions.xlsx` for industry code lookup

## Key Patterns

- Pandas method chaining with `.assign()`, `.loc[]`, `.pipe()` for data transformations
- Single-select enforcement via callback override on checklists
- Median reference lines on all time-series plots
- Environment variable `PORT` for deployment flexibility (defaults to 8050)

## Dependencies

Core: Dash 2.16.1, Plotly 5.20.0, Dash Bootstrap Components 1.5.0 (LITERA theme), Pandas 2.2.1, Openpyxl 3.1.2
