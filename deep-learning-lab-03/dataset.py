import numpy as np
import os


class Dataset:
    def __init__(self, batch_size, sequence_length):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.char2id, self.id2char, self.x, self.sorted_chars = None, None, None, None
        self.num_batches, self.batch_pointer, self.size = 0, 0, 0

    def preprocess(self, input_file):
        with open(input_file, "r") as f:
            data = f.read()

        # count and sort most frequent characters
        char_dict = {}
        for char in data:
            if char in char_dict:
                char_dict[char] += 1
            else:
                char_dict[char] = 1

        self.sorted_chars = list(c for c, _ in sorted(char_dict.items(), key=lambda kv: (kv[1], kv[0])))

        # self.sorted chars contains just the characters ordered descending by frequency
        self.char2id = dict(zip(self.sorted_chars, range(len(self.sorted_chars))))
        # reverse the mapping
        self.id2char = {k: v for v, k in self.char2id.items()}
        # convert the data to ids
        self.x = np.array(list(map(self.char2id.get, data)))

    def encode(self, sequence):
        return np.array(list(map(self.char2id.get, sequence)))

    def decode(self, encoded_sequence):
        return list(map(self.id2char.get, encoded_sequence))

    def create_minibatches(self):
        self.num_batches = int(
            len(self.x) / (self.batch_size * self.sequence_length))  # calculate the number of batches

        # Is all the data going to be present in the batches? Why?
        # What happens if we select a batch size and sequence length larger than the length of the data?

        #######################################
        #       Convert data to batches       #
        #######################################
        self.size = len(self.x)
        #if self.size == self.num_batches * self.batch_size * self.sequence_length:
            #self.x.append(0)

        pass

    def next_minibatch(self):
        # handling batch pointer & reset
        # new_epoch is a boolean indicating if the batch pointer was reset
        # in this function call
        new_epoch = False

        if self.batch_pointer == self.num_batches:
            self.batch_pointer = 0
            new_epoch = True

        begin_x = self.batch_pointer * self.batch_size * self.sequence_length
        end_x = (self.batch_pointer + 1) * self.batch_size * self.sequence_length
        begin_y = self.batch_pointer * self.batch_size * self.sequence_length + 1
        end_y = (self.batch_pointer + 1) * self.batch_size * self.sequence_length + 1
        batch_x = np.expand_dims(self.x, 1)[begin_x:end_x, :].reshape((self.batch_size, self.sequence_length))
        batch_y = np.expand_dims(self.x, 1)[begin_y:end_y, :].reshape((self.batch_size, self.sequence_length))

        self.batch_pointer += 1

        return new_epoch, batch_x, batch_y


if __name__ == '__main__':
    root = 'data'
    input_destination = 'selected_conversations.txt'
    dataset = Dataset(500, 30)
    dataset.preprocess(os.path.join(root, input_destination))
    dataset.create_minibatches()
    new_epoch, batch_x, batch_y = dataset.next_minibatch()
    while not new_epoch:
        print(batch_x)
        print(batch_y)
        new_epoch, batch_x, batch_y = dataset.next_minibatch()
