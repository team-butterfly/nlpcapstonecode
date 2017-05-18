# USAGE: ruby preprocess-main.rb <output> <inputs ...>
#   e.g. ruby preprocess-main.rb myfile.txt tweets.v3.part*.txt

require 'json'
require_relative 'preprocess-twitter'

save_name = ARGV[0]
File.open(save_name, "w") do |save_file|
    save_file.write("v.5/16-tok\n")
    save_file.flush

    ARGV.drop(1).each do |filename|
        lines = File.readlines(filename)
        if lines[0].strip != "v.4/21"
            puts "Skipping #{filename} (unexpected format)"
        end
        lines.drop(1).each do |line|
            line = line.strip
            tweet = JSON.parse(line)
            tweet["text"] = tokenize(tweet["text"])
            save_file.write(tweet.to_json)
            save_file.write("\n")
            save_file.flush
        end
    end
end
