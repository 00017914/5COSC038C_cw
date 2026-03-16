# AI-Assisted Data Wrangler & Visualizer

This coursework project is a Streamlit application for module `5COSC038C Data Wrangling and visualization`. It supports dataset upload, profiling, cleaning, validation, visualization, and export with a reproducible transformation log.

## Features

- Upload CSV, XLSX, and JSON datasets
- Optional Google Sheets import by shareable URL
- Dataset overview with shape, number of columns, dtypes, summary statistics, missing values, and duplicate count
- Cleaning studio for:
  - missing values
  - duplicates
  - type conversion and datetime parsing
  - categorical standardization and mapping
  - outlier treatment
  - min-max and z-score scaling
  - column rename, drop, formulas, and binning
  - validation rules and violations export
- Visualization builder with 6 chart types:
  - histogram
  - box plot
  - scatter plot
  - line chart
  - bar chart
  - heatmap / correlation matrix
- Export cleaned data as CSV and Excel
- Export transformation log as CSV and a JSON recipe
- Undo last step and reset workflow support

## Project structure

- `app.py`: main Streamlit app
- `requirements.txt`: Python dependencies
- `sample_data/`: demo datasets with missing values, mixed types, and duplicates
- `outputs/transformation_report_example.csv`: example transformation report
- `AI_USAGE.md`: AI usage disclosure and manual verification notes
- `PROMPTS_USED.md`: development prompt log

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Google Sheets import

The app includes optional Google Sheets support for the coursework bonus feature.

- Paste a Google Sheets tab URL into the `Optional Google Sheets URL` field on Page A
- The app converts the link into a CSV export URL and loads that tab
- This works best when the sheet is public or shared so the deployed app can access it

Example:

`https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit#gid=0`

## Demo datasets

The repository includes two sample datasets:

- `sample_data/retail_sales_dirty.csv`
- `sample_data/hr_records_dirty.json`

Both satisfy the coursework testing constraints with more than 1,000 rows, at least 8 columns, mixed data types, missing values, and duplicates.

## Deployment URL

https://5cosc038ccw-zroaqngcusurccvaesuqin.streamlit.app/




