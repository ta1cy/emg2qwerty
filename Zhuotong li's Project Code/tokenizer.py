import string

class Tokenizer:

    def __init__(self):

        chars = list(string.ascii_lowercase + " '")

        self.char2idx = {c:i+1 for i,c in enumerate(chars)}
        self.char2idx["<blank>"] = 0

        self.idx2char = {v:k for k,v in self.char2idx.items()}

    def encode(self, text):

        text = text.lower()

        tokens = []

        for c in text:

            if c in self.char2idx:
                tokens.append(self.char2idx[c])

        return tokens

    def decode(self, tokens):

        chars = []

        for token in tokens:
            if token == 0:
                continue
            if token in self.idx2char:
                chars.append(self.idx2char[token])

        return "".join(chars)
