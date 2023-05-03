# Twitter-Health-News-Mining
Work for a data mining class on health news data.

## Dataset
We used the [Health News in Twitter Dataset](https://archive.ics.uci.edu/ml/datasets/Health+News+in+Twitter) for this project.

## Instructions

 1. Create Python environment with `make env`
 2. Download and generate the data with `make generate-data`

## ToDo

 - [x] Combine datasets into single csv
 - [ ] Clean dataset of links, hashtags, cashtags, etc. (this is accomplished but bugged)
 - [ ] Remove punctuation & convert words to lowercase (same as above)
 - [ ] Determine how many tokens are in the dataset
 - [x] Embed tweets into vectors (or graphs????)
 - [ ] Cluster tweet embeddings (Half done, just needs a more formal workflow)
 - [ ] Cluster tweets on sentiment
 - [ ] Track trends through time of each cluster
 - [ ] Identify qualities of each cluster (top-k words, semantic themes, sentiment, etc.)

