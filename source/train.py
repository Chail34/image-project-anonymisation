import os
import time
import math


# Modules de machine learning
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds


# Modules d'affichage
import matplotlib.pyplot as plt
from IPython import display


# Variables globales d'entrainement
BATCH_SIZE = 64
NUMBER_OF_EPOCHS = 256
NOISE_SIZE = 128
NUMBER_OF_EXAMPLES = 16
BUFFER_SIZE = 15000


# Dossiers de données
DATASET_IMAGE_DIRECTORY = "./dataset/image"
GENERATED_IMAGE_DIRECTORY = "./generated-image"
MODEL_CHECKPOINT_DIRECTORY = "./training-checkpoint"


# Générations des dossiers
os.makedirs(
	name = GENERATED_IMAGE_DIRECTORY, 
	exist_ok = True
)
os.makedirs(
	MODEL_CHECKPOINT_DIRECTORY, 
	exist_ok = True
)


# Fonction de normalisation de [0, 255] -> [-1, 1]
def normalize(image):
	image = tf.image.resize(image, [128, 128])
	image = (tf.cast(image, tf.float32) - 127.5) / 127.5
	return image


# Chargement et normalisation du dataset
dataset = keras.utils.image_dataset_from_directory(
	DATASET_IMAGE_DIRECTORY,
	labels = None,                 # GAN non-supervisé
	image_size = (128, 128),       # Redimensionnement direct à 128 * 128
	batch_size = BATCH_SIZE,
	shuffle = True,
	color_mode = "rgb"
)
dataset = (
	dataset
	.map(normalize, num_parallel_calls = tf.data.AUTOTUNE)
	.prefetch(tf.data.AUTOTUNE)
)


# Verification des dimension des batchs
for image_batch in dataset.take(1):
    print("Dimenssions du batch d'images :", image_batch.shape)


# Fonction de création du décodeur :
# Entrée: Vecteur latent de taille 100
# Sortie: Image de taille 128 * 128 sur 3 canaux RGB
def createDecoder():
	decoder = keras.Sequential(
		[
			# Entrée : Vecteur latent de taille 128 -> 8 * 8 * 512
			layers.Input(shape = (NOISE_SIZE, )),
			layers.Dense(
				units = 8 * 8 * 512,
				use_bias = False
			),
			layers.BatchNormalization(),
			layers.LeakyReLU(negative_slope = 0.30),
			layers.Reshape((8, 8, 512)),


			# (8 * 8 * 512) -> (16 * 16 * 256)
			layers.UpSampling2D(
				size = (2, 2),
				interpolation = "nearest"
			),
			layers.Conv2D(
				filters = 256, 
				kernel_size = (4, 4), 
				strides = (1, 1), 
				padding = "same", 
				use_bias = False
			),
			layers.BatchNormalization(),
			layers.LeakyReLU(negative_slope = 0.30),


			# (16 * 16 * 256) -> (32 * 32 * 128)
			layers.UpSampling2D(
				size = (2, 2),
				interpolation = "nearest"
			),
			layers.Conv2D(
				filters = 128, 
				kernel_size = (4, 4), 
				strides = (1, 1), 
				padding = "same", 
				use_bias = False
			),
			layers.BatchNormalization(),
			layers.LeakyReLU(negative_slope = 0.30),


			# (32 * 32 * 128) -> (64 * 64 * 64)
			layers.UpSampling2D(
				size = (2, 2),
				interpolation = "nearest"
			),
			layers.Conv2D(
				filters = 64, 
				kernel_size = (4, 4), 
				strides = (1, 1), 
				padding = "same", 
				use_bias = False
			),
			layers.BatchNormalization(),
			layers.LeakyReLU(negative_slope = 0.30),


			# Sortie: Image de taille 128 * 128 sur 3 canaux RGB
			layers.UpSampling2D(
				size = (2, 2),
				interpolation = "nearest"
			),
			layers.Conv2D(
				filters = 3, 
				kernel_size = (3, 3), 
				strides = (1, 1), 
				padding = "same", 
				use_bias = False,
				activation = "tanh"
			)
		]
	)

	return decoder


# Fonction de création du discriminateur :
# Entrée: Image de taille 128 * 128 sur 3 canaux RGB
# Sortie: Valeur de probabilité
def createDiscriminator():
	discriminator = keras.Sequential(
		[
			# Entrée : Image de taille 128 * 128 sur 3 canaux RGB -> 64 * 64 * 64
			keras.Input(shape = (128, 128, 3)),
			layers.Conv2D(
				filters = 64, 
				kernel_size = (3, 3), 
				strides = (2, 2), 
				padding = "same"
			),
			layers.LeakyReLU(negative_slope = 0.30),
			layers.Dropout(0.25),


			# (64 * 64 * 64) -> (32 * 32 * 128)
			layers.Conv2D(
				filters = 128, 
				kernel_size = (3, 3), 
				strides = (2, 2), 
				padding = "same"
			),
			layers.LeakyReLU(negative_slope = 0.30),
			layers.Dropout(0.25),


			# (32 * 32 * 128) -> (16 * 16 * 256)
			layers.Conv2D(
				filters = 256, 
				kernel_size = (3, 3), 
				strides = (2, 2), 
				padding = "same"
			),
			layers.LeakyReLU(negative_slope = 0.30),
			layers.Dropout(0.25),


			# (16 * 16 * 256) -> (8 * 8 * 512)
			layers.Conv2D(
				filters = 512, 
				kernel_size = (3, 3), 
				strides = (2, 2), 
				padding = "same"
			),
			layers.LeakyReLU(negative_slope = 0.30),
			layers.Dropout(0.25),


			# Sortie : Valeur scalaire de probabilité.
			keras.layers.Flatten(),
			keras.layers.Dense(1)
		]
	)

	return discriminator


# Création des métriques de loss
crossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)

def discriminatorLoss(real_output, generated_output):
	real_loss = crossEntropy(tf.ones_like(real_output) * 0.9, real_output)
	fake_loss = crossEntropy(tf.zeros_like(generated_output), generated_output)
	return real_loss + fake_loss

def decoderLoss(generated_output):
	return crossEntropy(tf.ones_like(generated_output), generated_output)


# Création des optimizateurs
decoderOptimizer = tf.keras.optimizers.Adam(learning_rate = 2e-4, beta_1 = 0.5)
discriminatorOptimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4, beta_1 = 0.5)


# Instanciation des models et des vecteurs d'examples
decoder = createDecoder()
discriminator = createDiscriminator()
example_input = tf.random.normal([NUMBER_OF_EXAMPLES, NOISE_SIZE])


# Logique d'étape d'entrainement
@tf.function
def trainStep(images):
	current_batch_size = tf.shape(images)[0]
	noise = tf.random.normal([current_batch_size, NOISE_SIZE])

	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		generated_images = decoder(noise, training = True)

		real_output = discriminator(images, training = True)
		fake_output = discriminator(generated_images, training = True)

		gen_loss = decoderLoss(fake_output)
		disc_loss = discriminatorLoss(real_output, fake_output)

	gradients_of_generator = gen_tape.gradient(gen_loss, decoder.trainable_variables)
	gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

	decoderOptimizer.apply_gradients(zip(gradients_of_generator, decoder.trainable_variables))
	discriminatorOptimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

	return gen_loss, disc_loss


# Visualisation d'entrainements
def createExampleImages(model, epoch, test_input):
	predictions = model(test_input, training = False)
	grid_size: int = 4 # math.sqrt(float(NUMBER_OF_EXAMPLES))

	fig = plt.figure(figsize=(grid_size, grid_size))
	for i in range(predictions.shape[0]):
		plt.subplot(grid_size, grid_size, i + 1)
		img = tf.clip_by_value((predictions[i] + 1.0) / 2.0, 0.0, 1.0)
		plt.imshow(img.numpy())
		plt.axis('off')

	plt.tight_layout()
	plt.savefig(os.path.join(GENERATED_IMAGE_DIRECTORY, f'examples_at_epoch{epoch:04d}.png'))
	plt.show()
	plt.close(fig)


# Mise en place des check points
checkpoint_prefix = os.path.join(MODEL_CHECKPOINT_DIRECTORY, "ckpt")
checkpoint = tf.train.Checkpoint(
    decoderOptimizer = decoderOptimizer,
    discriminatorOptimizer = discriminatorOptimizer,
    decoder = decoder,
    discriminator = discriminator
)


# Logique d'entrainement
def train(dataset, epochs):
	for epoch in range(epochs):
		start = time.time()
		total_decoder_loss = 0.0
		total_discriminator_loss = 0.0
		batch_index = 0

		for image_batch in dataset:
			decoder_loss, discriminator_loss = trainStep(image_batch)
			total_decoder_loss += decoder_loss
			total_discriminator_loss += discriminator_loss
			batch_index += 1

		display.clear_output(wait=True)
		createExampleImages(decoder, epoch + 1, example_input)

		# Sauvegarde d'un checkpoint toutes les 2 époques
		if (epoch + 1) % 2 == 0:
			checkpoint.save(file_prefix=checkpoint_prefix)

		print(
			f"Époque {epoch + 1}/{epochs} | Temps: {time.time() - start:.2f}s | "
			f"Gen Loss: {total_decoder_loss / batch_index:.4f} | "
			f"Disc Loss: {total_discriminator_loss / batch_index:.4f}"
		)


# Restauration du dernier checkpoint
latest_checkpoint = tf.train.latest_checkpoint(MODEL_CHECKPOINT_DIRECTORY)
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint)
    print("Checkpoint restauré avec succès depuis :", latest_checkpoint)
else:
    print("Aucun checkpoint trouvé. Initialisation d'un nouvel entraînement.")


train(dataset, NUMBER_OF_EPOCHS)


