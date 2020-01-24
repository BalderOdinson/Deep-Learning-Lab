import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from utils import tile_raster_images
import math
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'jet'

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, \
                     mnist.test.labels


def weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias(shape):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial)


def sample_prob(probs):
    """Uzorkovanje vektora x prema vektoru vjerojatnosti p(x=1) = probs"""
    return tf.to_float(tf.random_uniform(tf.shape(probs)) <= probs)


def draw_weights(W, shape, N, stat_shape, interpolation="bilinear"):
    """Vizualizacija težina
    W -- vektori težina
    shape -- tuple dimenzije za 2D prikaz težina - obično dimenzije ulazne slike, npr. (28,28)
    N -- broj vektora težina
    shape_state -- dimezije za 2D prikaz stanja (npr. za 100 stanja (10,10)
    """
    image = (tile_raster_images(
        X=W.T,
        img_shape=shape,
        tile_shape=(int(math.ceil(N / stat_shape[0])), stat_shape[0]),
        tile_spacing=(1, 1)))
    plt.figure(figsize=(10, 14))
    plt.imshow(image, interpolation=interpolation)
    plt.axis('off')


def draw_reconstructions(ins, outs, states, shape_in, shape_state, N):
    """Vizualizacija ulaza i pripadajućih rekonstrukcija i stanja skrivenog sloja
    ins -- ualzni vektori
    outs -- rekonstruirani vektori
    states -- vektori stanja skrivenog sloja
    shape_in -- dimezije ulaznih slika npr. (28,28)
    shape_state -- dimezije za 2D prikaz stanja (npr. za 100 stanja (10,10)
    N -- broj uzoraka
    """
    plt.figure(figsize=(8, int(2 * N)))
    for i in range(N):
        plt.subplot(N, 4, 4 * i + 1)
        plt.imshow(ins[i].reshape(shape_in), vmin=0, vmax=1, interpolation="nearest")
        plt.title("Test input")
        plt.axis('off')
        plt.subplot(N, 4, 4 * i + 2)
        plt.imshow(outs[i][0:784].reshape(shape_in), vmin=0, vmax=1, interpolation="nearest")
        plt.title("Reconstruction")
        plt.axis('off')
        plt.subplot(N, 4, 4 * i + 3)
        plt.imshow(states[i].reshape(shape_state), vmin=0, vmax=1, interpolation="nearest")
        plt.title("States")
        plt.axis('off')
    plt.tight_layout()


def draw_generated(stin, stout, gen, shape_gen, shape_state, N):
    """Vizualizacija zadanih skrivenih stanja, konačnih skrivenih stanja i pripadajućih rekonstrukcija
    stin -- početni skriveni sloj
    stout -- rekonstruirani vektori
    gen -- vektori stanja skrivenog sloja
    shape_gen -- dimezije ulaznih slika npr. (28,28)
    shape_state -- dimezije za 2D prikaz stanja (npr. za 100 stanja (10,10)
    N -- broj uzoraka
    """
    plt.figure(figsize=(8, int(2 * N)))
    for i in range(N):
        plt.subplot(N, 4, 4 * i + 1)
        plt.imshow(stin[i].reshape(shape_state), vmin=0, vmax=1, interpolation="nearest")
        plt.title("set state")
        plt.axis('off')
        plt.subplot(N, 4, 4 * i + 2)
        plt.imshow(stout[i][0:784].reshape(shape_state), vmin=0, vmax=1, interpolation="nearest")
        plt.title("final state")
        plt.axis('off')
        plt.subplot(N, 4, 4 * i + 3)
        plt.imshow(gen[i].reshape(shape_gen), vmin=0, vmax=1, interpolation="nearest")
        plt.title("generated visible")
        plt.axis('off')
    plt.tight_layout()


Nh = 100  # Broj elemenata prvog skrivenog sloja
h1_shape = (10, 10)
Nv = 784  # Broj elemenata vidljivog sloja
v_shape = (28, 28)
Nu = 5000  # Broj uzoraka za vizualizaciju rekonstrukcije
Nh2 = Nh  # Broj elemenata drugog skrivenog sloja
h2_shape = h1_shape
gibbs_sampling_steps = 2
alpha = 0.1
beta = 0.01

g1 = tf.Graph()
with g1.as_default():
    X1 = tf.placeholder("float", [None, Nv])
    r1_up = weights([Nv, Nh])
    w1_down = tf.Variable(tf.transpose(r1_up))
    w2 = weights([Nh, Nh2])
    hb1_up = bias([Nh])
    hb1_down = bias([Nh2])
    vb1_down = bias([Nv])
    hb2 = bias([Nh2])

    # wake pass
    h1_up_prob = tf.add(tf.matmul(X1, r1_up), hb1_up)
    h1_up = sample_prob(h1_up_prob)  # s^{(n)} u pripremi
    v1_up_down_prob = tf.add(tf.matmul(h1_up, w1_down), vb1_down)
    v1_up_down = sample_prob(v1_up_down_prob)  # s^{(n-1)\mathit{novo}} u tekstu pripreme

    # top RBM Gibs passes
    h2_up_prob = tf.add(tf.matmul(h1_up, w2), hb2)
    h2_up = sample_prob(h2_up_prob)
    h2_down = h2_up
    for step in range(gibbs_sampling_steps):
        h1_down_prob = tf.add(tf.matmul(h2_down, w2, transpose_b=True), hb1_down)
        h1_down = sample_prob(h1_down_prob)
        h2_down_prob = tf.add(tf.matmul(h1_down, w2), hb2)
        h2_down = sample_prob(h2_down_prob)

        # sleep pass
    v1_down_prob = tf.add(tf.matmul(h1_down, w1_down), vb1_down)
    v1_down = sample_prob(v1_down_prob)  # s^{(n-1)} u pripremi
    h1_down_up_prob = tf.add(tf.matmul(v1_down, r1_up), hb1_up)
    h1_down_up = sample_prob(h1_down_up_prob)  # s^{(n)\mathit{novo}} u pripremi

    # generative weights update during wake pass
    update_w1_down = tf.assign_add(w1_down, beta * tf.matmul(tf.transpose(h1_up), X1 - v1_up_down_prob) / tf.to_float(
        tf.shape(X1)[0]))
    update_vb1_down = tf.assign_add(vb1_down, beta * tf.reduce_mean(X1 - v1_up_down_prob, 0))

    # top RBM update
    w2_positive_grad = tf.matmul(h1_up, h2_up, transpose_a=True)
    w2_negative_grad = tf.matmul(h1_down, h2_down, transpose_a=True)
    dw2 = (w2_positive_grad - w2_negative_grad) / tf.to_float(tf.shape(h1_up)[0])
    update_w2 = tf.assign_add(w2, beta * dw2)
    update_hb1_down = tf.assign_add(hb1_down, beta * tf.reduce_mean(h1_up - h1_down, 0))
    update_hb2 = tf.assign_add(hb2, beta * tf.reduce_mean(h2_up - h2_down, 0))

    # recognition weights update during sleep pass
    update_r1_up = tf.assign_add(r1_up,
                                 beta * tf.matmul(tf.transpose(v1_down_prob), h1_down - h1_down_up) / tf.to_float(
                                     tf.shape(X1)[0]))
    update_hb1_up = tf.assign_add(hb1_up, beta * tf.reduce_mean(h1_down - h1_down_up, 0))

    out1 = (update_w1_down, update_vb1_down, update_w2, update_hb1_down, update_hb2, update_r1_up, update_hb1_up)

    err1 = X1 - v1_down_prob
    err_sum1 = tf.reduce_mean(err1 * err1)

    initialize1 = tf.global_variables_initializer()

batch_size = 100
epochs = 100
n_samples = mnist.train.num_examples
total_batch = int(n_samples / batch_size) * epochs

sess1 = tf.Session(graph=g1)
sess1.run(initialize1)

for i in range(total_batch):
    batch, label = mnist.train.next_batch(batch_size)
    err, _ = sess1.run([err_sum1, out1], feed_dict={X1: batch})

    if i % (int(total_batch / 10)) == 0:
        print(i, err)

w2ss, r1_ups, w1_downs, hb2ss, hb1_ups, hb1_downs, vb1_downs, vr, h2_downs, h2_down_probs = sess1.run(
    [w2, r1_up, w1_down, hb2, hb1_up, hb1_down, vb1_down, v1_down_prob, h2_down, h2_down_prob],
    feed_dict={X1: teX[0:Nu, :]})

# vizualizacija težina
draw_weights(r1_ups, v_shape, Nh, h1_shape)
plt.show(block=True)
draw_weights(w1_downs.T, v_shape, Nh, h1_shape)
plt.show(block=True)
draw_weights(w2ss, h1_shape, Nh2, h2_shape, interpolation="nearest")
plt.show(block=True)

# vizualizacija rekonstrukcije i stanja
draw_reconstructions(teX, vr, h2_downs, v_shape, h2_shape, 50)
plt.show(block=True)


# vizualizacija jedne rekonstrukcije s postepenim dodavanjem doprinosa aktivnih skrivenih elemenata
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def draw_rec(inp, title, size, Nrows, in_a_row, j):
    """ Iscrtavanje jedne iteracije u kreiranju vidljivog sloja
    inp - vidljivi sloj
    title - naslov sličice
    size - 2D dimenzije vidljiovg sloja
    Nrows - maks. broj redaka sličica
    in-a-row . broj sličica u jednom redu
    j - pozicija sličice u gridu
    """
    plt.subplot(Nrows, in_a_row, j)
    plt.imshow(inp.reshape(size), vmin=0, vmax=1, interpolation="nearest")
    plt.title(title)
    plt.axis('off')


def reconstruct(ind, states, orig, weights, biases):
    """ Slijedno iscrtavanje rekonstrukcije vidljivog sloja
    ind - indeks znamenke u orig (matrici sa znamenkama kao recima)
    states - vektori stanja ulaznih vektora
    orig - originalnalni ulazni vektori
    weights - matrica težina
    biases - vektori pomaka vidljivog sloja
    """
    j = 1
    in_a_row = 6
    Nimg = states.shape[1] + 3
    Nrows = int(np.ceil(float(Nimg + 2) / in_a_row))

    plt.figure(figsize=(12, 2 * Nrows))

    draw_rec(states[ind], 'states', h1_shape, Nrows, in_a_row, j)
    j += 1
    draw_rec(orig[ind], 'input', v_shape, Nrows, in_a_row, j)

    reconstr = biases.copy()
    j += 1
    draw_rec(sigmoid(reconstr), 'biases', v_shape, Nrows, in_a_row, j)

    for i in range(Nh):
        if states[ind, i] > 0:
            j += 1
            reconstr = reconstr + weights[:, i]
            titl = '+= s' + str(i + 1)
            draw_rec(sigmoid(reconstr), titl, v_shape, Nrows, in_a_row, j)
    plt.tight_layout()


# Generiranje uzoraka iz slučajnih vektora
r_input = np.random.rand(100, Nh2)
r_input[r_input > 0.9] = 1  # postotak aktivnih - slobodno varirajte
r_input[r_input < 1] = 0
r_input = r_input * 20  # pojačanje za slučaj ako je mali postotak aktivnih

s = 10
i = 0
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s
i += 1
r_input[i, :] = 0
r_input[i, i] = s

out_1 = sess1.run(v1_down, feed_dict={h2_up: r_input})

# Emulacija dodatnih Gibbsovih uzorkovanja pomoću feed_dict
for i in range(4000):
    out_1_prob, out_1, hout1 = sess1.run((v1_down_prob, v1_down, h2_down), feed_dict={X1: out_1})
draw_generated(r_input, hout1, out_1_prob, v_shape, h2_shape, 50)
plt.show(block=True)
