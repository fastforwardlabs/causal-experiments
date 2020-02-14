.PHONY: dirs news

dirs:
	mkdir data

news:
	cd data; \
	curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip; \
	unzip OnlineNewsPopularity.zip;
