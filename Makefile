
# Default to blank target
default: ;

.PHONY: data
data:
	mkdir -p data
	wget --user team-butterfly --password cse481n https://homes.cs.washington.edu/~ijchen/cse481n/team-butterfly/data/tweets-small.txt -O data/tweets-small.txt