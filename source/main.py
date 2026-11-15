import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

BATCH_SIZE = 256
BUFFER_SIZE = 60000

# Chargement des données dans le dataset
face_dataset = tfds.load(
  'lfw',
  split='train',
  batch_size= BATCH_SIZE,
  as_supervised=True,
  shuffle_files=True
)

face_dataset.shuffle(BUFFER_SIZE)

for image, label in face_dataset.take(1):
  print(image.shape)

def createGenerator():
    generator = keras.Sequential()

    # Entrée : vecteur de bruit de taille 100. Projection vers 8 * 8 * 512
    generator.add(keras.layers.Dense(8 * 8 * 512, use_bias=False, input_shape=(100,)))
    generator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.LeakyReLU())

    # Redimensionnement vers un volume 3D
    generator.add(keras.layers.Reshape((8, 8, 512)))

    # Taille : 16 * 16 * 256
    generator.add(keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    generator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.LeakyReLU())

    # Taille : 32 * 32 * 128
    generator.add(keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    generator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.LeakyReLU())

    # Taille : 64 * 64 * 64
    generator.add(keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    generator.add(keras.layers.BatchNormalization())
    generator.add(keras.layers.LeakyReLU())

    # Taille finale : 128 * 128 * 3 (image RGB couleur pour les visages)
    generator.add(keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return generator

def createDiscriminator():
    discriminator = keras.Sequential()

    # Entrée : Image de taille 128 * 128 * 3
    discriminator.add(keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[128, 128, 3]))
    discriminator.add(keras.layers.LeakyReLU())
    discriminator.add(keras.layers.Dropout(0.3))

    # Taille : 64 * 64 * 128
    discriminator.add(keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    discriminator.add(keras.layers.LeakyReLU())
    discriminator.add(keras.layers.Dropout(0.3))

    # Taille : 32 * 32 * 256
    discriminator.add(keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    discriminator.add(keras.layers.LeakyReLU())
    discriminator.add(keras.layers.Dropout(0.3))

    # Taille : 16 * 16 * 512
    discriminator.add(keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    discriminator.add(keras.layers.LeakyReLU())
    discriminator.add(keras.layers.Dropout(0.3))

    # Aplatissement et classification finale (1 pour vrai, 0 pour faux)
    discriminator.add(keras.layers.Flatten())
    discriminator.add(keras.layers.Dense(1, activation='sigmoid'))

    return discriminator

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

import time
from IPython import display

# Instanciation des modèles
generator = createGenerator()
discriminator = createDiscriminator()

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# Nous gardons un bruit constant pour visualiser l'évolution des performances du générateur
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def generate_and_save_images(model, epoch, test_input):
    # Nous mettons training à False pour désactiver le mode entraînement des couches de Batch Normalization
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        # Re-convertir de [-1, 1] vers [0, 1] pour l'affichage de l'image
        plt.imshow((predictions[i] + 1.0) / 2.0)
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        # Inversion ici : le dataset LFW renvoie (label, image) avec as_supervised=True
        for _, image_batch in dataset:
            # Les images d'origine doivent être redimensionnées à 128x128 et normalisées dans [-1, 1]
            # correspondant au format attendu par le discriminateur et produit par le générateur
            image_batch = tf.image.resize(image_batch, [128, 128])
            image_batch = (tf.cast(image_batch, tf.float32) - 127.5) / 127.5

            # Pour s'assurer que le dernier batch incomplet ne lève pas d'erreur sur la taille de BATCH_SIZE
            if image_batch.shape[0] == BATCH_SIZE:
                train_step(image_batch)

        # Produire des images pour visualiser l'apprentissage
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        print ('Temps pour l\'époque {} est de {} sec'.format(epoch + 1, time.time()-start))

    # Générer après la dernière époque
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

import os

# Configuration de la sauvegarde et du chargement des checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Restauration du dernier checkpoint disponible
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint)
    print("Checkpoint restauré avec succès depuis :", latest_checkpoint)
else:
    print("Aucun checkpoint trouvé. Initialisation d'un nouvel entraînement.")

# Lancement de l'entraînement
train(face_dataset, EPOCHS)