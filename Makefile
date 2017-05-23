
# Default to blank target
default: ;

baseurl = https://homes.cs.washington.edu/~ijchen/cse481n/team-butterfly/
.PHONY: data
data:
	mkdir -p data
	rm data/*
	wget --user team-butterfly --password cse481n $(baseurl)data/listing -O data/listing
	for file in `cat data/listing`; do \
		wget --user team-butterfly --password cse481n $(baseurl)data/$$file -O data/$$file; \
	done

.PHONY: preprocess
preprocess:
	ruby preprocess/preprocess-main.rb data/tweets-preprocessed-v1.txt data/tweets.v3.part*.txt
	ruby preprocess/preprocess-main.rb --fix data/tweets-preprocessed-v2.txt data/tweets.v3.part*.txt
