.PHONY: dirs news churn housing

dirs:
	mkdir data

news:
	cd data; \
	curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip; \
	unzip OnlineNewsPopularity.zip;

churn:
	@echo "Kaggle datasets require an API. Download this dataset https://www.kaggle.com/blastchar/telco-customer-churn, and put it at data/churn.csv"

housing:
	@echo "Kaggle datasets require an API. Download this dataset https://www.kaggle.com/camnugent/california-housing-price and put it at data/housing.csv"

bikes:
	cd data; \
	curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip; \
	unzip Bike-Sharing-Dataset.zip;

london-bikes:
	@echo "Kaggle datasets require an API. Download this dataset https://www.kaggle.com/hmavrodiev/london-bike-sharing-dataset and put it at data/london-bikes.csv"

student:
	cd data; \
	curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip; \
	unzip student.zip;
