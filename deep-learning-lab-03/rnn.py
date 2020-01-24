import numpy as np
import dataset
import os

btch_sz = 200
lrn_rt = 1e-1
epoch_cnt = 10


class RNN:
    def __init__(self, hidden_size, sequence_length, vocab_size):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size

        self.U = np.random.normal(0, 1e-2, (vocab_size, hidden_size))  # ... input projection
        self.W = np.random.normal(0, 1e-2, (hidden_size, hidden_size))  # ... hidden-to-hidden projection
        self.b = np.zeros((1, hidden_size))  # ... input bias

        self.V = np.random.normal(0, 1e-2, (hidden_size, vocab_size))  # ... output projection
        self.c = np.zeros((1, vocab_size))  # ... output bias

    def step_forward(self, x, h_prev):
        # A single time step forward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity.

        # x - input data (minibatch size x input dimension)
        # h_prev - previous hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        # return the new hidden state and a tuple of values needed for the backward step

        h_current = np.tanh(np.dot(h_prev, self.W) + np.dot(x, self.U) + self.b)

        return h_current, (h_prev, x, h_current)

    def forward(self, x, h0):
        # Full unroll forward of the recurrent neural network with a
        # hyperbolic tangent nonlinearity

        # x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
        # h0 - initial hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        h, cache = [], []
        h_prev = h0

        for i in range(x.shape[1]):
            h_i, cache_i = self.step_forward(x[:, i], h_prev)
            h.append(h_i)
            cache.append(cache_i)
            h_prev = h_i

        # return the hidden states for the whole time series (T+1) and a tuple of values needed for the backward step
        return h, cache

    def step_backward(self, grad_next, cache):
        # A single time step backward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity.

        # grad_next - upstream gradient of the loss with respect to the next hidden state and current output
        # cache - cached information from the forward pass

        dU, dW, db = None, None, None

        # compute and return gradients with respect to each parameter
        # HINT: you can use the chain rule to compute the derivative of the
        # hyperbolic tangent function and use it to compute the gradient
        # with respect to the remaining parameters

        da = grad_next * (1 - cache[2]**2)

        dW = np.dot(cache[0].T, da)
        dU = np.dot(cache[1].T, da)
        db = da.sum(axis=0)
        grad_prev = np.dot(da, self.W.T)

        return grad_prev, dU, dW, db

    def backward(self, dh, cache):
        # Full unroll forward of the recurrent neural network with a
        # hyperbolic tangent nonlinearity

        dU, dW, db = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.b)

        # compute and return gradients with respect to each parameter
        # for the whole time series.
        # Why are we not computing the gradient with respect to inputs (x)?

        dh_i = 0

        for i in reversed(range(len(dh))):
            dh[i] += dh_i
            dh_i, dU_i, dW_i, db_i = self.step_backward(dh[i], cache[i])
            dU += dU_i
            dW += dW_i
            db += db_i

        return dU, dW, db

    def output(self, h):
        # Calculate the output probabilities of the network

        probs = []

        for i in range(len(h)):
            probs.append(np.dot(h[i], self.V) + self.c)

        return probs

    def output_loss_and_grads(self, h, y):
        # Calculate the loss of the network for each of the outputs

        # h - hidden states of the network for each timestep.
        #     the dimensionality of h is (batch size x sequence length x hidden size (the initial state is irrelevant for the output)
        # V - the output projection matrix of dimension hidden size x vocabulary size
        # c - the output bias of dimension vocabulary size x 1
        # y - the true class distribution - a tensor of dimension
        #     batch_size x sequence_length x vocabulary size - you need to do this conversion prior to
        #     passing the argument. A fast way to create a one-hot vector from
        #     an id could be something like the following code:

        #   y[batch_id][timestep] = np.zeros((vocabulary_size, 1))
        #   y[batch_id][timestep][batch_y[timestep]] = 1

        #     where y might be a list or a dictionary.

        loss, dh, dV, dc = None, [], np.zeros_like(self.V), np.zeros_like(self.c)
        # calculate the output (o) - unnormalized log probabilities of classes
        # calculate yhat - softmax of the output
        # calculate the cross-entropy loss
        # calculate the derivative of the cross-entropy softmax loss with respect to the output (o)
        # calculate the gradients with respect to the output parameters V and c
        # calculate the gradients with respect to the hidden layer h

        N = y.shape[0]
        logprobs = np.zeros_like(y)

        o = self.output(h)
        for i in range(len(o)):
            # softmax(o)
            expscores = np.exp(o[i] - np.max(o[i]))
            sumexp = np.sum(expscores, axis=1, keepdims=True)
            probs = expscores / sumexp

            # log(softmax(o))
            logprobs[:, i] = np.log(probs)

            # backprop
            do = probs - y[:, i]
            dV += np.dot(h[i].T, do)
            dc += do.sum(axis=0)
            dh.append(np.dot(do, self.V.T))

        loss = - 1 / N * np.sum(logprobs * y)

        return loss, dh, dV, dc

    def update(self, dU, dW, dV, db, dc):
        self.U += dU
        self.W += dW
        self.V += dV
        self.b += db
        self.c += dc


class Adagrad:
    def __init__(self, rnn, hidden_size, vocab_size, learning_rate, delta=1e-7):
        self.learning_rate = learning_rate
        self.delta = delta
        self.rnn = rnn

        self.memory_U, self.memory_W, self.memory_V = np.zeros((vocab_size, hidden_size)), np.zeros(
            (hidden_size, hidden_size)), np.zeros((hidden_size, vocab_size))
        self.memory_b, self.memory_c = np.zeros((1, hidden_size)), np.zeros((1, vocab_size))

    def step(self, h0, x_oh, y_oh):
        h, cache = self.rnn.forward(x_oh, h0)
        loss, gh, gV, gc = self.rnn.output_loss_and_grads(h, y_oh)
        gU, gW, gb = self.rnn.backward(gh, cache)
        N = x_oh.shape[0]
        gU /= N
        gW /= N
        gV /= N
        gb /= N
        gc /= N
        np.clip(gU, -5, 5, out=gU)
        np.clip(gW, -5, 5, out=gW)
        np.clip(gV, -5, 5, out=gV)
        np.clip(gb, -5, 5, out=gb)
        np.clip(gc, -5, 5, out=gc)

        self.memory_U += gU**2
        self.memory_W += gW**2
        self.memory_V += gV**2
        self.memory_b += gb**2
        self.memory_c += gc**2

        dU = - (self.learning_rate / (self.delta + np.sqrt(self.memory_U))) * gU
        dW = - (self.learning_rate / (self.delta + np.sqrt(self.memory_W))) * gW
        dV = - (self.learning_rate / (self.delta + np.sqrt(self.memory_V))) * gV
        db = - (self.learning_rate / (self.delta + np.sqrt(self.memory_b))) * gb
        dc = - (self.learning_rate / (self.delta + np.sqrt(self.memory_c))) * gc

        self.rnn.update(dU, dW, dV, db, dc)

        return loss, h[-1]


def to_one_hot(x, vocab_size):
    # x_one_hot = np.zeros((x.shape[0], x.shape[1], vocab_size))
    #
    # for batch_id in range(x.shape[0]):
    #     for timestep in range(x.shape[1]):
    #         x_one_hot[batch_id][timestep][x[batch_id][timestep]] = 1
    #
    # return x_one_hot
    return (np.arange(vocab_size) == x[..., None]-1).astype(int)


def sample(dataset, rnn, seed, n_sample, hidden_size, sequence_length, batch_size):
    vocab_size = len(dataset.sorted_chars)
    h0 = np.zeros((batch_size, hidden_size))
    seed_one_hot = to_one_hot(np.expand_dims(dataset.encode(seed), axis=0), vocab_size)
    sample = ""
    while len(sample) < n_sample:
        h0, _ = rnn.forward(seed_one_hot, h0)
        o = rnn.output(h0)
        for i in range(len(o)):
            sample += ''.join(dataset.decode(o[i].argmax(axis=1)))

    return sample[len(seed):]


def run_language_model(dataset, max_epochs, hidden_size=100, sequence_length=30, learning_rate=1e-1, sample_every=100, batch_size=196):
    vocab_size = len(dataset.sorted_chars)
    rnn = RNN(hidden_size, sequence_length, vocab_size)  # initialize the recurrent network
    optimizer = Adagrad(rnn, hidden_size, vocab_size, learning_rate)

    current_epoch = 0
    batch = 0

    h0 = np.zeros((batch_size, hidden_size))

    average_loss = 0

    while current_epoch < max_epochs:
        e, x, y = dataset.next_minibatch()

        if e:
            print('Average loss %f0.2' % (average_loss / dataset.num_batches))
            print()
            current_epoch += 1
            average_loss = 0
            h0 = np.zeros((batch_size, hidden_size))
            # why do we reset the hidden state here?

        # One-hot transform the x and y batches
        x_oh, y_oh = to_one_hot(x, vocab_size), to_one_hot(y, vocab_size)

        # Run the recurrent network on the current batch
        # Since we are using windows of a short length of characters,
        # the step function should return the hidden state at the end
        # of the unroll. You should then use that hidden state as the
        # input for the next minibatch. In this way, we artificially
        # preserve context between batches.
        loss, h0 = optimizer.step(h0, x_oh, y_oh)
        average_loss += loss

        if batch % sample_every == 0:
            s = sample(dataset, rnn, "HAN:\nIs that good or bad?\n\n", 300, hidden_size, sequence_length, batch_size)
            print(s)
        print('Epoch=%d, batch=%d/%d, loss=%f0.2' % (
            current_epoch, (batch % dataset.num_batches) + 1, dataset.num_batches, loss))
        batch += 1


if __name__ == '__main__':
    np.random.seed(100)
    root = 'data'
    input_destination = 'selected_conversations.txt'
    dataset = dataset.Dataset(btch_sz, 30)
    dataset.preprocess(os.path.join(root, input_destination))
    dataset.create_minibatches()
    run_language_model(dataset, epoch_cnt, batch_size=btch_sz, learning_rate=lrn_rt)
    # root = 'data'
    # input_destination = 'selected_conversations.txt'
    # dataset = dataset.Dataset(500, 30)
    # dataset.preprocess(os.path.join(root, input_destination))
    # dataset.create_minibatches()
    # vocab_size = len(dataset.sorted_chars)
    # new_epoch, batch_x, batch_y = dataset.next_minibatch()
    # f_x = open("batch_x.txt", "w+")
    # f_y = open("batch_y.txt", "w+")
    # f_x_one = open("batch_x_one.txt", "w+")
    # f_y_one = open("batch_y_one.txt", "w+")
    # while not new_epoch:
    #     x_oh, y_oh = to_one_hot(batch_x, vocab_size), to_one_hot(batch_y, vocab_size)
    #     for i in range(dataset.num_batches):
    #         np.savetxt(f_x_one, x_oh[i], '%d', delimiter='')
    #         np.savetxt(f_y_one, y_oh[i], '%d', delimiter='')
    #     np.savetxt(f_x, batch_x, '%d')
    #     np.savetxt(f_y, batch_y, '%d')
    #     new_epoch, batch_x, batch_y = dataset.next_minibatch()
