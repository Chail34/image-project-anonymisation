#let course-code = "HAI927I"
#let course-title = "Projet Image"
#let document-type = "Compte-rendu de projet"
#let school-level = "Master 2"




#let university-name = "Université de Montpellier"
#let college-name = "Faculté des sciences"
#let school-year = "2026 - 2027"




#let author-name = "Groupe 5"
#let page-number = context counter(page).get().at(0)




#let document-title = "Compte-rendu semaine 3 : Entrainement à la génération de visages d'un GAN"




#let basic-space = v(0.25in)
#let double-space = v(0.5in)





#let academic-header = [
	#grid(
		columns: 2,
		rows: 1,
		gutter: 0in,

		grid.cell(
			align: top + left,
			rect(
				width: 100%,
				height: 80%,
				stroke: 0pt
			)[
				#strong[#course-code - #course-title]\
				#emph[#document-type]\
				#school-level
			]
		),

		grid.cell(
			align: top + right,
			rect(
				width: 100%,
				height: 80%,
				stroke: 0pt
			)[
				#strong[#university-name]\
				#emph[#college-name]\
				#school-year
			]
		)
	)

	#v(0%, weak: true)
	#line(
		stroke: black + 1pt,
		length: 100%
	)
]




#let academic-footer = [
		#grid(
		columns: 2,
		rows: 1,
		gutter: 0in,

		grid.cell(
			align: bottom + left,
			rect(
				width: 100%,
				height: 0.5in,
				stroke: 0pt
			)[
				#emph[#author-name]
			]
		),

		grid.cell(
			align: bottom + right,
			rect(
				width: 100%,
				height: 0.5in,
				stroke: 0pt
			)[
				#strong[#page-number]
			]
		)
	)
]




#set text(
	size: 14pt, 
	font: "Crimson Pro"
)




#set page(
	width: 21cm,
	height:29.7cm,
	margin: (top: 1.5in, bottom: 1in, x:1in),
	header-ascent: 0.25in,
	header: academic-header,
	footer-descent: 0.25in,
	footer: academic-footer
)




#double-space
#align(center)[
	#heading(
		level: 1,
		outlined: false,
		[#document-title]
	)
]
#double-space



// ---------------- Introduction Begin --------------------
#heading(
	level: 2,
	outlined: false,
	"Introduction"
)
L'objectif de cette semaine de projet était de concevoir et de débuter l'entrainement d'un réseau de neurones antagoniste génératif profond _(Deep Convolutional Generative Adversarial Network)_ capable de générer des images réalistes de visages humains. Le projet s'appuie sur le jeu de données _LFW - (Labeled Faces in the Wild)_ dont une implémentation exploitable est fourni par TensorFlow Datasets.
#double-space





#show outline.entry: set text(fill: rgb(0, 0, 255))
#show outline.entry.where(
	level: 1
): set text(weight: "bold")
#outline(
	title: "Table des matières"
)





#set heading(
	numbering: "1.",
)
#pagebreak()




// ---------------- Document Begin --------------------

= Préparation et Pipeline de Données
Le pipeline de chargement de données a été configuré de la manière suivante :
- *Source* : Jeu de données `lfw` chargé via `tensorflow_datasets`.
- *Hyperparamètres de données* : #[
  - `BATCH_SIZE` = 256
  - `BUFFER_SIZE` = 60000
]

- *Prétraitement à la volée* lors de l'entraînement : #[
  * Redimensionnement des images de leur taille d'origine (250x250) vers une taille standardisée de *128x128 pixels* afin d'adapter l'entrée aux dimensions de notre discriminateur.
  * Normalisation des pixels dans l'intervalle *[-1, 1]* (requis par l'activation 
  `tanh` de la couche de sortie du générateur).
]


= Architecture du Modèle
Le système repose sur la confrontation de deux modèles distincts :

== Le Générateur
Son rôle est de transformer un vecteur de bruit aléatoire de dimension 100 en une image de visage de taille 128x128x3.
- *Entrée* : Vecteur aléatoire (taille 100).
- *Projection & Reshape* : Passage par une couche dense projetée en `8 * 8 * 512`, puis remodelée en un volume 3D.
- *Suréchantillonnage (Upsampling)* : Quatre couches successives de convolutions transposées (`Conv2DTranspose`) doublant la taille spatiale à chaque étape ($8 times 8 -->; 16 times 16 -->; 32 times 32 -->; 64 times 64 -->; 128 times 128$).
- *Régularisation et Activation* : Utilisation de `BatchNormalization` et de fonctions d'activation `LeakyReLU` pour stabiliser l'apprentissage. La couche finale utilise une activation `tanh`.

== Le Discriminateur
Son rôle est de classer les images reçues comme "réelles" (issues du dataset) ou "fausses" (créées par le générateur).
- *Entrée* : Image de taille 128x128x3.
- *Sous-échantillonnage (Downsampling)* : Quatre couches de convolutions standards (`Conv2D`) avec un pas (*stride*) de 2.
- *Régularisation* : Utilisation de `Dropout(0.3)` pour éviter le surapprentissage du discriminateur face au générateur.
- *Sortie* : Une valeur scalaire unique avec une activation `sigmoid` représentant la probabilité que l'image soit réelle.


= Stratégie d'Entraînement et Fonctions de Perte
- *Fonction de perte* : Entropie croisée binaire (`BinaryCrossentropy`).
- *Optimiseurs* : Deux instances de l'optimiseur *Adam* avec un taux d'apprentissage de `1e-4` pour les deux réseaux.
- *Boucle d'entraînement (`train_step`)* : #[
  - Optimisation alternée et simultanée grâce aux `tf.GradientTape()` pour calculer et appliquer les gradients de manière indépendante pour le générateur et le discriminateur.
  - Utilisation de checkpoints (`tf.train.Checkpoint`) permettant de sauvegarder et de restaurer l'état du modèle afin de prévenir les interruptions d'apprentissage.
]


= Observations
L'entrainement prenant un certain temps, nous n'avons actuellement pas de résultat probant à montrer.

De plus, Lors du lancement de l'entraînement, l'avertissement suivant a été émis par le framework :
*`UserWarning: "binary_crossentropy" received "from_logits=True", but the "output" argument was produced by a Sigmoid activation and thus does not represent logits.`*
Recherche active d'une solution à cette avertissement