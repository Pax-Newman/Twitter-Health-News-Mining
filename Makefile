
env:
	@# Create new venv if there isn't one
ifeq ($(wildcard .venv),)
	@python -m venv .venv
endif
	@# Install/update venv with necessary packages
	./.venv/bin/python -m pip install -r requirements.txt

python := ./.venv/bin/python

generate-data:
	@# Fetch dataset archive from UCI if it hasn't already been
ifeq ($(wildcard datasets.zip),)
	@echo 'downloading dataset...'
	curl https://archive.ics.uci.edu/ml/machine-learning-databases/00438/Health-News-Tweets.zip --output datasets.zip
endif	
	@# Create data directories and extract the datasets there
	@mkdir -p data
	@$(python) utils/unzip.py datasets.zip data/
	
	@# Merge datasets into a single csv file
	@$(python) utils/format_data.py ./data/Health-Tweets/ ./data/dataset.csv

	@# Remove subdirectories from data/
	@rm -r data/*/

clean:
	rm -rf data

