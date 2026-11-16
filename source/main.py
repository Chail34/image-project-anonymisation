import os
import time
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from IPython import display

# Paramètres globaux
BATCH_SIZE = 64
BUFFER_SIZE = 15000
EPOCHS = 300
noise_dim = 100
num_examples_to_generate = 16

# Fonction de prétraitement : redimensionnement 128x128 et normalisation dans [-1, 1]
def preprocess_image(label, image):
    image = tf.image.resize(image, [128, 128])
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image

# Chargement, prétraitement, mélange et création des batchs
face_dataset = tfds.load(
    'lfw',
    split='train',
    as_supervised=True,
    shuffle_files=True
)

face_dataset = (
    face_dataset
    .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

# Vérification de la forme d'un batch
for image_batch in face_dataset.take(1):
    print("Forme du batch d'images :", image_batch.shape)

# Création du Générateur
def createGenerator():
    generator = keras.Sequential([
        # Entrée : vecteur de bruit de taille 100 -> 8 * 8 * 512
        keras.layers.Dense(8 * 8 * 512, use_bias=False, input_shape=(100,)),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),
        keras.layers.Reshape((8, 8, 512)),

        # 16 * 16 * 256
        keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        # 32 * 32 * 128
        keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        # 64 * 64 * 64
        keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(),

        # Sortie finale : 128 * 128 * 3 (valeurs entre -1 et 1 grâce à tanh)
        keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return generator

# Création du Discriminateur
def createDiscriminator():
    discriminator = keras.Sequential([
        # Entrée : 128 * 128 * 3 -> 64 * 64 * 64
        keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[128, 128, 3]),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.3),

        # 32 * 32 * 128
        keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.3),

        # 16 * 16 * 256
        keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.3),

        # 8 * 8 * 512
        keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'),
        keras.layers.LeakyReLU(),
        keras.layers.Dropout(0.3),

        # Classification finale (logits bruts, sans activation sigmoid)
        keras.layers.Flatten(),
        keras.layers.Dense(1)
    ])
    return discriminator

# Fonctions de perte (Loss)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    # Label smoothing (0.9) pour éviter que le discriminateur ne devienne trop confiant
    real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimiseurs Adam réglés avec beta_1=0.5 (recommandé pour les DCGAN)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

# Instanciation des modèles
generator = createGenerator()
discriminator = createDiscriminator()

# Graine fixe pour suivre l'évolution visuelle sur les mêmes vecteurs latents
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Étape d'entraînement compilée
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

    return gen_loss, disc_loss

# Génération et sauvegarde des images de suivi
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        # Re-convertir de [-1, 1] vers [0, 1] et sécuriser les bornes avec clip_by_value
        img = tf.clip_by_value((predictions[i] + 1.0) / 2.0, 0.0, 1.0)
        plt.imshow(img)
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

# Configuration des checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator
)

# Boucle principale d'entraînement
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        total_gen_loss = 0.0
        total_disc_loss = 0.0
        num_batches = 0

        for image_batch in dataset:
            g_loss, d_loss = train_step(image_batch)
            total_gen_loss += g_loss
            total_disc_loss += d_loss
            num_batches += 1

        # Affichage des images générées à chaque époque
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        # Sauvegarde d'un checkpoint toutes les 5 époques
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f"Époque {epoch + 1}/{epochs} - Temps: {time.time()-start:.2f}s | "
              f"Gen Loss: {total_gen_loss/num_batches:.4f} | "
              f"Disc Loss: {total_disc_loss/num_batches:.4f}")

# Restauration éventuelle d'un checkpoint existant (à vider si d'anciens poids corrompus s'y trouvent)
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint)
    print("Checkpoint restauré avec succès depuis :", latest_checkpoint)
else:
    print("Aucun checkpoint trouvé. Initialisation d'un nouvel entraînement.")

# Lancement de l'entraînement
train(face_dataset, EPOCHS)
