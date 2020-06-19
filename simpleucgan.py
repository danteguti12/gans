#simple Unconditional Generative Adversarial Network trained on MNIST Fashion dataset

import tensorflow as tf
import numpy as np

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

print('Train', trainX.shape, trainY.shape)
print('Test', testX.shape, testY.shape)

import matplotlib.pyplot as plt
for i in range(10):
  plt.subplot(2, 5, 1+i)
  plt.axis('off')
  plt.imshow(trainX[i], cmap='gray_r')
plt.show()

def define_discriminator(in_shape=(28,28,1)):
  model = tf.keras.Sequential([       
                               #downsample
                               tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same', input_shape=in_shape),
                               tf.keras.layers.LeakyReLU(alpha=0.2),
                               #downsample
                               tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same'),
                               tf.keras.layers.LeakyReLU(alpha=0.2),
                               #classifier
                               tf.keras.layers.Flatten(),
                               tf.keras.layers.Dropout(0.4),
                               tf.keras.layers.Dense(1, activation='sigmoid')])
 
  opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
  model.compile(
      loss='binary_crossentropy', optimizer=opt, metrics=['acc']
  )
  
  return model

def define_generator(latent_dim):
  n_nodes = 128*7*7
  model = tf.keras.Sequential([
                              tf.keras.layers.Dense(n_nodes, input_dim=latent_dim),
                              tf.keras.layers.LeakyReLU(alpha=0.2),
                              tf.keras.layers.Reshape((7, 7, 128)),
                              #upsample
                              tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'),
                              tf.keras.layers.LeakyReLU(alpha=0.2),
                              tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'),
                              tf.keras.layers.LeakyReLU(alpha=0.2),
                              #generate
                              tf.keras.layers.Conv2D(1, (7,7), activation='tanh', padding='same')
  ])
  
  return model

def define_gan(generator, discriminator):
  discriminator.trainable = False
  model = tf.keras.Sequential([
                               generator,
                               discriminator,
  ])
  opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt)

  return model

print("Networks defined")

def load_real_samples():
  (trainX, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
  X = np.expand_dims(trainX, axis=-1)
  X = np.float32(X)
  X = (X-127.5)/127.5

  return X

def generate_real_samples(dataset, n_samples):
  ix = np.random.randint(0, dataset.shape[0], n_samples)
  X = dataset[ix]
  Y = np.ones((n_samples, 1))

  return X, Y

def generate_latent_points(latent_dim, n_samples):
  x_input = np.random.randn(latent_dim*n_samples)
  x_input = x_input.reshape(n_samples, latent_dim)

  return x_input

def generate_fake_samples(generator, latent_dim, n_samples):
  x_input = generate_latent_points(latent_dim, n_samples)
  X = generator.predict(x_input)
  Y = np.zeros((n_samples, 1))

  return X, Y

print('Auxilary functions defined')

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
  bat_per_epo = int(dataset.shape[0]/n_batch)
  half_batch = int(n_batch/2)
  for i in range(n_epochs):
    for j in range(bat_per_epo):
      X_real, Y_real = generate_real_samples(dataset, half_batch)
      d_loss1, _ = d_model.train_on_batch(X_real, Y_real)
      X_fake, Y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
      d_loss2, _ = d_model.train_on_batch(X_fake, Y_real)
      X_gan = generate_latent_points(latent_dim, n_batch)
      Y_gan = np.ones((n_batch, 1))
      g_loss = gan_model.train_on_batch(X_gan, Y_gan)
      print('>%d, %d/%d, d1=%.3f, d2=%.3f, g=%.3f' %
            (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
  g_model.save('generator.h5')

print('Training function defined')

latent_dim = 100
generator = define_generator(latent_dim)
discriminator = define_discriminator()
gan_model = define_gan(generator, discriminator)
dataset = load_real_samples()
train(generator, discriminator, gan_model, dataset, latent_dim)
