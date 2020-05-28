python_utils
=========================================================
Repository for functions that are useful in my daily analisys in Data Science


Templates and useful functions for a Data Science Workflow
==========================================================

The following libs are included in this repository:

- `analysis_template.py` Is the main template to use for data science analysis
- `cloud_data_utils.py` Functions to use .pdcsv data and to easily gather data
- `eda_utils.py` Functions of exploratory data analysis

Recommended Project Structure
===========================================================
We recommend to keep Datasets available in central folders, produced project data in other folder and code in a Github project.
We can see a scheme below

DRIVER_BASE
    |
    ------ Dataset1 ----- Dataset1_20200501.csv
    |                     Dataset1_20200401.csv
    |                     Dataset1_20200301.csv
    |----- Dataset2 ----- Dataset2_20100301.xlsx
    
REPORTS_BASE
    |
    ------- Project1 ---- Report.pdf
    |                     plot1.jpg
    ------- Project2 ---- Docs.docx
    
 Github
    |
    ------- python_utils --- cloud_data_utils.py
    |                        eda_utils.py
    ------- Project1 ------- analysis.py
 
