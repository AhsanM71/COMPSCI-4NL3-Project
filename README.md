# COMPSCI-4NL3-Project

# Stock Market Tweets Dataset

## Overview
This dataset consists of stock market about major technology company stocks, primarily focusing on companies like Tesla ($TSLA), Apple ($AAPL), Amazon ($AMZN), and Google ($GOOGL).

## Data Collection
- **Source**: Twitter
- **Collection Period**: Tweets published between 01/01/2015 and 31/12/2019.
- **Collection Method**: Downloaded the dataset from [Hugging Face](https://huggingface.co/datasets/mjw/stock_market_tweets).

## Dataset Format
- The dataset is split into 8 files (data1.csv through data8.csv)
- Each file contains an equal portion of the total dataset
- Each CSV file has 3 columns:
  - `id`: Unique identifier for each post
  - `body`: The text content of the post
  - `label`: Currently empty, to be annotated

## Dataset Statistics
- Total instances: 960 posts
  - 816 unique posts
  - 144 duplicate posts (15% of total) for annotator agreement calculation
- The duplicate instances are distributed across files data1.csv through data7.csv
- data8.csv contains only unique instances
- Each file contains 120 posts

## Annotation Time
- Each instance represents a single tweet about stock market activity
- Annotators will need to label the sentiment of the tweet as either positive, negative, or neutral
- Estimated annotation time per instance: 10 seconds (based on test annotations by team members)
- Total annotation capacity for 8-hour day: 2880 instances
  - Calculation: (8 hours * 60 minutes * 60 seconds) / 10 seconds per instance = 2880 instances

## Missing Data
None.

## Data Distribution
- The posts cover various tech companies but may have different proportions of coverage
- Primary stock tickers mentioned:
  - $TSLA (Tesla)
  - $AAPL (Apple)
  - $AMZN (Amazon)
  - $GOOGL (Google)
  - And others

## Notes
- The duplicate instances are used for calculating inter-annotator agreement
- Duplicates do not appear twice in the same file
- The 8th file (data8.csv) contains no duplicate instances

## Annotation Process
Each dataset will be annotated by 7-8 different annotators, with each annotator spending approximately 8 hours on the task.