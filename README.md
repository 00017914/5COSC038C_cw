# AI-Assisted Data Wrangler & Visualizer

A professional-grade, interactive workbench designed for data analysts and researchers to perform end-to-end data preparation, exploratory data analysis (EDA), and AI-driven visualization. 

This application focuses on a **minimalistic, high-utility UI** that treats data cleaning as a repeatable "recipe," ensuring transparency and efficiency in the research pipeline.

---

## 🚀 Key Features

### 1. Multi-Source Data Ingestion
* **File Support:** Seamlessly upload `CSV`, `XLSX`, or `JSON` files.
* **Cloud Integration:** Direct import from **Google Sheets** via URL (public access files only).
* **Automated Inference:** Smart data type detection (Numeric, Datetime, and Categorical) upon ingestion to reduce manual configuration.

### 2. The Cleaning & Preparation Studio
The core of the app is a high-performance transformation engine that tracks every change in a **Transformation Log**.

* **Missing Value Management:** Strategies include row/column dropping based on percentage thresholds or filling (Mean, Median, Mode, Forward/Backward Fill).
* **Outlier & Numeric Cleaning:** Statistical outlier detection using the **Interquartile Range (IQR)** method with options for Winsorization (capping) or removal.
* **Categorical Engineering:** * Standardization (Trim, Lower, or Title case).
    * Dynamic category mapping and grouping of "Rare" categories.
    * **One-Hot Encoding** for machine learning readiness.
* **Feature Engineering:** * **Formula Engine:** Create new columns using Python/NumPy expressions (e.g., `log(sales) / qty`).
    * **Binning:** Discretize continuous data using Equal-Width or Quantile-based binning.

### 3. Data Validation & Profiling
* **Deep Profiling:** Instant metrics on missingness, duplicates, and descriptive statistics.
* **Custom Validation Rules:** Define range constraints for numeric data or "Allowed Value" lists for categorical data to identify pipeline violations.

### 4. AI-Powered Visual Insights
* **GPT-Driven Suggestions:** Leveraging OpenAI to analyze your dataset schema and recommend the most statistically relevant chart based on your research goals.
* **Dynamic UI/UX:** Professional dashboards that focus on data-driven storytelling rather than just raw charts.

---

## 🛠️ Technical Stack
* **Frontend/App Framework:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Plotly
* **AI Integration:** OpenAI API

---

## ⚙️ Installation & Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

- `app.py`: main Streamlit app
- `requirements.txt`: Python dependencies
- `sample_data/`: demo datasets with missing values, mixed types, and duplicates
- `outputs/transformation_report_example.csv`: example transformation report
- `AI_USAGE.md`: AI usage disclosure and manual verification notes
- `PROMPTS_USED.md`: development prompt log
- `README.md`: this file

## Demo datasets

The repository includes two sample datasets:

- `sample_data/retail_sales_dirty.csv`
- `sample_data/hr_records_dirty.json`

Both satisfy the testing constraints with more than 1,000 rows, at least 8 columns, mixed data types, missing values, and duplicates.

## Deployment URL

https://5cosc038ccw-zroaqngcusurccvaesuqin.streamlit.app/




