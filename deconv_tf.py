import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from layers import deconv, conv, dense, batch_norm
from covers import covers
import pickle
from utils import *

matplotlib.rc('xtick', labelsize=10)

session = tf.InteractiveSession()
COVERS_PATH = "C:/Users/Alex/Documents/PhD/spotify-dataset2"


# load_session=True to continue training an existing session
load_session= True
batch_size = 128
stride = [1, 2, 2, 1]
con_dim = 2
rand_dim = 100
cat_dim = 2
noise_dim = cat_dim + con_dim + rand_dim
cover_set = covers(COVERS_PATH, batch_size, cat_dim)

def generator(cat, con, rand):
    with tf.variable_scope("g_") as scope:
        noise = tf.concat([cat, con, rand], 1)
        g0 = dense(noise_dim, 512*4*4, "g1", activation='linear')(noise)
        g1 = tf.reshape(g0, (batch_size, 4, 4, 512))
        g1_b = batch_norm(scope='g1_norm', activation=tf.nn.relu)(g1)
        g2 = deconv(5, 5, 512, 256, stride, "g2")(g1_b)
        g2_b = batch_norm(scope='g2_norm', activation=tf.nn.relu)(g2)
        g3 = deconv(5, 5, 256, 128, stride, "g3")(g2_b)
        g3_b = batch_norm(scope='g3_norm', activation=tf.nn.relu)(g3)
        g4 = deconv(5, 5, 128, 64, stride, "g4")(g3_b)
        g4_b = batch_norm(scope='g4_norm', activation=tf.nn.relu)(g4)
        g5 = deconv(5, 5, 64, 3, stride, "g5")(g4_b)
        return tf.nn.tanh(g5)

# copy of generator to not effect training of generator
def sampler(cat, con, rand):
    with tf.variable_scope("g_") as scope:
        scope.reuse_variables()
        noise = tf.concat([cat, con, rand], 1)
        g0 = dense(noise_dim, 512*4*4, "g1", activation='linear')(noise)
        g1 = tf.reshape(g0, (batch_size, 4, 4, 512))
        g1_b = batch_norm(scope='g1_norm', activation=tf.nn.relu)(g1)
        g2 = deconv(5, 5, 512, 256, stride, "g2")(g1_b)
        g2_b = batch_norm(scope='g2_norm', activation=tf.nn.relu)(g2)
        g3 = deconv(5, 5, 256, 128, stride, "g3")(g2_b)
        g3_b = batch_norm(scope='g3_norm', activation=tf.nn.relu)(g3)
        g4 = deconv(5, 5, 128, 64, stride, "g4")(g3_b)
        g4_b = batch_norm(scope='g4_norm', activation=tf.nn.relu)(g4)
        g5 = deconv(5, 5, 64, 3, stride, "g5")(g4_b)
        return tf.nn.tanh(g5)

def discriminator(image, reuse=False):
    with tf.variable_scope("d_") as scope:
        if reuse:
            scope.reuse_variables()
        d1 = conv(5, 5, 3, 64, stride, "d1")(image)
        d1_b = batch_norm(scope='d1_norm', activation='lrelu')(d1)
        d2 = conv(5, 5, 64, 128, stride, "d2")(d1_b)
        d2_b = batch_norm(scope='d2_norm', activation='lrelu')(d2)
        d3 = conv(5, 5, 128, 256,  stride, "d3")(d2_b)
        d3_b = batch_norm(scope='d3_norm', activation='lrelu')(d3)
        d4 = conv(5, 5, 256, 512,  stride, "d4")(d3_b)
        d4_b = batch_norm(scope='d4_norm', activation='lrelu')(d4)
        d4_flat = tf.reshape(d4_b, (batch_size, 512*4*4))
        d5 = dense(512*4*4, 1, "d5", activation='linear')(d4_flat)
        d5_cat = dense(512*4*4, cat_dim, "d5_cat", activation='linear')(d4_flat)
        d5_cont = dense(512*4*4, con_dim, "d5_cont", activation='linear')(d4_flat)
        return d5, d5_cat, d5_cont

# DEFINE ALL TENSORFLOW VARIABLES NEEDED
z_cat = tf.placeholder(tf.float32, (batch_size, cat_dim))
z_con = tf.placeholder(tf.float32, (batch_size, con_dim))
z_rand = tf.placeholder(tf.float32, (batch_size, rand_dim))

real_inputs = tf.placeholder(tf.float32, (batch_size, 64, 64, 3))
real_labels = tf.placeholder(tf.float32, (batch_size, cat_dim))

G = generator(z_cat, z_con, z_rand)
D_logits, D_cat, _= discriminator(real_inputs)
D_logits_, D_cat_, D_cont_  = discriminator(G, reuse=True)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D_logits)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.zeros_like(D_logits)))
d_loss = (d_loss_real + d_loss_fake) / 2

d_loss_real_cat = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_cat, labels=real_labels))
d_loss_fake_cat = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_cat_, labels=z_cat))
d_loss_cat = (d_loss_real_cat + d_loss_fake_cat) / 2
d_cont_loss = tf.reduce_mean(tf.square(D_cont_ - z_con))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.ones_like(D_logits)))

d_score = d_loss + 10 * d_loss_cat
g_score = g_loss + 10 * d_loss_cat

variables = tf.trainable_variables()
g_vars = [v for v in variables if v.name.startswith("g_/")]
d_vars = [v for v in variables if v.name.startswith("d_/")]

g_op = tf.train.AdamOptimizer(learning_rate=0.0003, beta1=0.5).minimize(g_score, var_list=g_vars)
d_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5).minimize(d_score, var_list=d_vars)

example_cat = tf.placeholder(tf.float32, (batch_size, cat_dim))
example_con = tf.placeholder(tf.float32, (batch_size, con_dim))
example_rand = tf.placeholder(tf.float32, (batch_size, rand_dim))
example_generated = sampler(example_cat, example_con, example_rand)

init = tf.global_variables_initializer()
session.run(init)

def train_generator(photos, photo_labels, cat, con, rand):
    _, score = session.run([g_op, g_loss], {
        z_cat: cat,
        z_con: con,
        z_rand: rand,
        real_inputs: photos,
        real_labels : photo_labels
    })
    return score

def train_discrim(photos, photo_labels, cat, con, rand):
    _, score, cat_score, cont_score = session.run([d_op, d_loss, d_loss_cat, d_cont_loss], {
        z_cat: cat,
        z_con: con,
        z_rand: rand,
        real_inputs: photos,
        real_labels: photo_labels
    })
    return score, cat_score, cont_score

def plotgenerated(num, const_cat, const_con, const_rand):
    img = session.run(example_generated, {
        example_cat: const_cat,
        example_con: const_con,
        example_rand: const_rand
    })
    save_images(img, [5, 5], './Images/test_%s.png' % num)

def train_on_batch(batch, batch_cat):
    batch_con = np.random.uniform(-1, 1, size=[batch_size, con_dim]).astype(np.float32)
    batch_rand = np.random.uniform(-1, 1, size=[batch_size, rand_dim]).astype(np.float32)
    batch_fake_cat = np.random.uniform(0, 1, size=[batch_size, cat_dim]).astype(np.float32)
    dis_err, cat_err, cont_err = train_discrim(batch, batch_cat, batch_fake_cat, batch_con, batch_rand)
    gen_err = train_generator(batch, batch_cat, batch_fake_cat, batch_con, batch_rand)
    return dis_err, gen_err, cat_err, cont_err

def plot_loss():
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='all')
    number_of_batches = len(losses["dis"])
    plot_batch = np.linspace(0, number_of_batches, number_of_batches)/100
    ax1.plot(plot_batch, losses["dis"], label="Discriminative loss")
    ax1.set_title('Discriminative Loss')
    ax2.plot(plot_batch, losses["gen"], label="Generative loss")
    ax2.set_title('Generative Loss')
    ax3.plot(plot_batch,losses["cat"], label="Catagorical loss")
    ax3.set_title('Catagorical Loss')
    ax4.plot(plot_batch,losses["cont"], label="Continuous loss")
    ax4.set_title('Latent Loss')
    ax4.set_xlabel('Number of Batches (hundreds)')
    plt.savefig('Images/losses.png')
    plt.close()

if load_session:
    losses = pickle.load(open("losses.p", "rb"))
    const = pickle.load(open("rand_vectors.p", "rb"))
    const_cat = const["cat"]
    const_con = const["con"]
    const_rand = const["rand"]
else:
    dis_graph, gen_graph, cat_graph, cont_graph = [], [], [], []
    losses = {"dis":dis_graph, "gen":gen_graph, "cat":cat_graph, "cont":cont_graph}
    # see the same noise changes as the network trains
    const_cat = np.random.uniform(0, 1, size=[batch_size, cat_dim]).astype(np.float32)
    const_con = np.random.uniform(-1, 1, size=[batch_size, con_dim]).astype(np.float32)
    const_rand = np.random.uniform(-1, 1, size=[batch_size, rand_dim]).astype(np.float32)
    const = {"cat":const_cat, "con":const_con, "rand":const_rand}
    pickle.dump(const, open("rand_vectors.p", "wb"))

start_epoch, start_indx = 0, 0
saver = tf.train.Saver()
if load_session==True:
    ckpt = tf.train.get_checkpoint_state("C:/Users/Alex/Documents/PhD/deconv-softlabels/save")
    saver.restore(session, ckpt.model_checkpoint_path)
    print ('Loaded session')

num_epochs = 10000
num_batches = 0
for epoch in range(1, num_epochs):
    for next_idx, batch , batch_cat in cover_set.batched_images(0):
        START_EPOCH, START_IDX = epoch, next_idx
        num_batches += 1
        dis_err, gen_err, cat_err, cont_err = train_on_batch(batch, batch_cat)
        losses["dis"].append(dis_err)
        losses["gen"].append(gen_err)
        losses["cat"].append(cat_err)
        losses["cont"].append(cont_err)
        print ("Dis_err: %f  Gen_err: %f  Cat_err: %f" % (dis_err, gen_err, cat_err))
        if num_batches % 100 == 0 or num_batches==1:
            plot_loss()
            plotgenerated(i, const_cat, const_con, const_rand)
            pickle.dump(losses, open("losses.p", "wb"))
            saver.save(session, "C:/Users/Alex/Documents/PhD/deconv-softlabels/save/gan_batch%d.ckpt" % (num_batches))
        print("Epoch: %d  Batch: %d  image %d" % (epoch, num_batches, next_idx*num_batches))
    START_IDX = 0

session.close()
