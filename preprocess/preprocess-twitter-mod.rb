# Ruby 2.0
#
# Script for preprocessing tweets by Romain Paulus
# with small modifications by Jeffrey Pennington

def tokenize2 input

	# Different regex parts for smiley faces
	eyes = "[8:=;]"
	nose = "['`\-]?"

	input = input
		.gsub(/https?:\/\/\S+\b|www\.(\w+\.)+\S*/,"<url>")
		.gsub("/"," / ") # Force splitting words appended with slashes (once we tokenized the URLs, of course)
		.gsub(/@\w+/, "<user>")
		.gsub(/#{eyes}#{nose}[)d]+|[)d]+#{nose}#{eyes}/i, "<smile>")
		.gsub(/#{eyes}#{nose}p+/i, "<lolface>")
		.gsub(/#{eyes}#{nose}\(+|\)+#{nose}#{eyes}/, "<sadface>")
		.gsub(/#{eyes}#{nose}[\/|l*]/, "<neutralface>")
		.gsub(/<3/,"<heart>")
		.gsub(/[-+]?[.\d]*[\d]+[:,.\d]*/, "<number>")
		.gsub(/#\S+/){ |hashtag| # Split hashtags on uppercase letters
			# TODO: also split hashtags with lowercase letters (requires more work to detect splits...)

			hashtag_body = hashtag[1..-1]
			if hashtag_body.upcase == hashtag_body
				result = "<hashtag> #{hashtag_body} <allcaps>"
			else
				result = (["<hashtag>"] + hashtag_body.split(/(?=[A-Z])/)).join(" ")
			end
			result
		}
		.gsub(/([!?.]){2,}/){ # Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
			"#{$~[1]} <repeat>"
		}
		.gsub(/\b(\S*?)(.)\2{2,}\b/){ # Mark elongated words (eg. "wayyyy" => "way <ELONG>")
			# TODO: determine if the end letter should be repeated once or twice (use lexicon/dict)
			$~[1] + $~[2] + " <elong>"
		}
		.gsub(/([A-Z]){2,}/){ |word|
			"#{word.downcase} <allcaps>"
		}
		.downcase

	return input
end
