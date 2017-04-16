
class FakeDataSource(object):

    def get_inputs(self):
        return [
            "I love eggs for breakfast",
            "Flies love fruit",
            "I love bacon",
            "I love coffee",
            "I love chocolate",
            "I love , love chocolate and coffee"
            "I like spam sandwiches",
            "I like peas and onions",
            "I like water",
            "I like ice cream , especially chocolate",
            "I hate strawberry milk",
            "I hate swiss cheese",
            "I hate vegetables , especially asparagus",
            "I hate fruit",
            "I am Python",
            "I have no brain",
            "Chickens lay eggs",
            "Sam eats spam",
            "I drink water all day",
            "I drink coffee in the morning"
        ]

    def num_labels(self):
        return 4

    def get_labels(self):
        return [
            3 if sent.find("love") != -1 else
            2 if sent.find("like") != -1 else
            1 if sent.find("hate") != -1 else
            0
            for sent in self.get_inputs()
        ]

    def decode_labels(self, labels, meanings=("neutral", "hatred", "enjoyment", "endearment")):
        return [meanings[i] for i in labels]
