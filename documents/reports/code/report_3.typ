#let course-code = "HAI927I"
#let course-title = "Projet Image"
#let document-type = "Compte-rendu de projet"
#let school-level = "Master 2"




#let university-name = "Université de Montpellier"
#let college-name = "Faculté des sciences"
#let school-year = "2026 - 2027"




#let author-name = "Groupe 5"
#let page-number = context counter(page).get().at(0)




#let document-title = "Compte-rendu semaine 4 : Amélioration de l'architecture existante"




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
L'objectif de cette semaine de projet était d'ameliorer l'entrainement de notre réseau de neurones antagoniste génératif profond _(Deep Convolutional Generative Adversarial Network)_ capable de générer des images réalistes de visages humains et d'y ajouter les parties de traitement de clés et d'encodage. Le temps nous a un peu manqué mais nous avons pu a moins eprouver le modèle existant et commencer la programationd des autres réseaux.
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
= Changements des modèles existent
	== Changements de Datasets
		Nous utilisions à l'origine le jeu de données `lfw` chargé grâce au module `tensorflow_datasets`
		Maintenant, nous utilisons le jeu de données `celebA` directement chargé depuis un dossier d'image.
		Ce changement nous permet neanmoins de faire des époques plus éfficace, le datasets étant 15 fois plus grand.

	== Pré-traitements
		Les images sont toutes redimensionnées à une taille de $128 times 128$ et les 3 canaux RGB sont changés d'intervalle 
		$[0, 255] -> [-1, 1]$

	== Modification des modèles 
		Les modèles ont été modifié. Pour évité les effets de damiers, les `ConvTranspose2D` ont été remplacées par une combinaison de 
		`UpSampling2D` avec un filtrage `nearest`pour éviter le lissage des données et de `Conv2D`.

	== Remarques
		Les changements de datasets et des modèles ont entrainés une amélioration de la qualité des images générées mais on alourdi le temps
		d'entrainement. Nous avons donc changé la fréquence d'enregistremnet des checkpoints de 5 a 2 épochs.

	== Visualisation des changements

	#grid(
		columns: (1fr),
		gutter: 1em,
		figure(
			image("image/epoch100.webp", width: 100%),
			caption:[Modèle d'origine après 50 épochs],
		),
		figure(
			image("image/epoch15.png", width: 90%),
			caption: [Nouveau modèle après 15 épochs],
		),
		figure(
			image("image/epoch50.png", width: 100%),
			caption: [Nouveau modèle après 50 épochs],
		),
	)