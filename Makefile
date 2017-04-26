
# Default to blank target
default: ;

baseurl = https://homes.cs.washington.edu/~ijchen/cse481n/team-butterfly/
.PHONY: data
data:
	mkdir -p data
	wget --user team-butterfly --password cse481n $(baseurl)data/listing -O data/listing
	for file in `cat data/listing`; do \
		wget --user team-butterfly --password cse481n $(baseurl)data/$$file -O data/$$file; \
	done
