
default: env generate-data train-model

env:
	@# Create new venv if there isn't one
ifeq ($(wildcard .venv),)
	@python -m venv .venv
endif
	@# Install/update venv with necessary packages
	./.venv/bin/python -m pip install -r requirements.txt

#python := ./.venv/bin/python
python = python

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
	@$(python) utils/format_data.py data/Health-Tweets/ data/original_dataset.pickle

	@# Remove subdirectories from data/
	@rm -r data/*/

	@# Clean data and turn it into a pandas dataframe
	@python create_dataframe.py \
		--data_path data/original_dataset.pickle \
		--save_path data/beegframe.pickle \
		--n_clusters 6 \
		--device mps # Change this to whatever pytorch device is best for your system

train-model:
	@# Train the feature reduction network
	@python models/reduction_net.py \
		--device mps \
		--data_path data/beegframe.pickle \
		--embedding_col 'bert' \
		--label_col 'bert label' \
		--input_features 1024 \
		--layers 512,256,256,64 \
		--classes 11 \
		--activation relu \
		--lr 0.004 \
		--batch_size 64 \
		--epochs 10 \
		--optimizer adamw \
		--save_path models/reduction_net


clean:
	rm -rf data

