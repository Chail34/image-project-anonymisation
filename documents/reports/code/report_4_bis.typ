#let course-code = "HAI927I"
#let course-title = "Projet Image"
#let document-type = "Compte-rendu de projet"
#let school-level = "Master 2"




#let university-name = "Université de Montpellier"
#let college-name = "Faculté des sciences"
#let school-year = "2026 - 2027"




#let author-name = "Groupe 5"
#let page-number = context counter(page).get().at(0)




#let document-title = "Compte-rendu 4 bis : Entraînements de notre chiffreur latent"




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
	font: "Crimson Pro",
	lang: "fr"
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




// Figures : légendes, tableaux et bibliographie en plus petit que le texte courant
#show figure.caption: set text(size: 11pt)
#show table: set text(size: 11pt)
#show bibliography: set text(size: 11pt)
#set par(justify: true)

// Libellés de colonnes au-dessus d'une grille d'images
#let entetes(..noms) = {
	grid(
		columns: (1fr,) * noms.pos().len(),
		align: center,
		..noms.pos().map(n => text(size: 9pt, n)),
	)
	v(-0.5em)
}




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
Après avoir fait tourner le code de _RiDDLE_ @riddle-article, nous avons réécrit son *chiffreur latent* en PyTorch (`source/anon/`), sans reprendre le code des auteurs, et l'avons entraîné six fois. Le premier entraînement (`exp1`) fonctionne mais laisse fuir l'identité avec une mauvaise clé ; tous les suivants cherchent à corriger ce défaut. Ce compte-rendu présente en détail `exp1` et `exp2` (pipeline, données, résultats, conclusion) et les compare, puis résume les quatre essais courts qui ont suivi (`exp3a` à `exp4a_ter`) et le prochain essai prévu.
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
#show bibliography: set heading(numbering: none)
#pagebreak()




// ---------------- Document Begin --------------------

= Premier entraînement (`exp1`)

== Architecture et pipeline
Un visage aligné est inversé par e4e @e4e en un code latent $w$ ($14 times 512$) de StyleGAN2 @stylegan2. Un réseau $E$ mélange ce code avec une *clé* $p$, elle aussi un code latent, tiré à partir d'une graine :

$ w_e = E(w, p) quad "(chiffrement)", quad quad w_d = E(w_e, p) quad "(déchiffrement)". $

Le *même* réseau sert dans les deux sens. StyleGAN2 transforme $w_e$ en visage anonyme et $w_d$ en visage retrouvé. Seul $E$ est entraîné (9,7 M de paramètres) : e4e, StyleGAN2 et le réseau d'identité restent figés. Comme dans l'article, $E$ a un bloc transformer par niveau de StyleGAN2 (4/4/6 couches), avec une cross-attention sur la clé. Tout le code est le nôtre : de RiDDLE, nous n'utilisons que les poids publiés.

À chaque itération, un visage est chiffré avec deux clés $p_1, p_2$, puis $w_(e 1)$ est déchiffré avec la bonne clé et deux mauvaises $q_1, q_2$. Avec $f$ l'embedding ArcFace @arcface de l'original et $f_k$ celui de l'image $k$ :

#text(size: 12pt)[
$ cal(L)_1 = underbrace(overline(max(0, cos(f, f_k))), "dé-identification") + underbrace(1 - cos(f, f_d), "récupération") + underbrace(overline(cos(f_k, f_l)), "diversité") + 0.05 cal(L)_"L1" + cal(L)_"LPIPS" + 0.01 cal(L)_"latent" $
]

où $k, l$ parcourent les quatre visages qui doivent être anonymes. Les termes L1, LPIPS @lpips et latent ($norm(w_k - w)$) s'appliquent aux *cinq* images générées et les gardent proches de l'original, pour conserver pose, expression et décor.

== Données d'entraînement
Nous n'utilisons pas d'images, mais les 70 000 visages de FFHQ déjà inversés par e4e et publiés par les auteurs. 68 000 servent à l'entraînement ; 2 000 sont réservés à la validation (64 suivis pendant l'entraînement avec des clés fixes, 512 pour l'évaluation finale). Les clés sont tirées à chaque itération par le réseau de *mapping* de StyleGAN2. Réglages : 20 000 itérations, batch 4, Ranger ($5 dot 10^(-4)$), environ 10 h sur une RTX 5060 (8 Go) grâce au _gradient checkpointing_.

== Résultats
Sur la @fig-courbes, la récupération monte jusqu'à 0,93 et la dé-identification est acquise dès le début. En revanche, la mauvaise clé *plafonne vers 0,4 dès 2 000 itérations*, et la diversité entre deux clés *se dégrade* (de 0,27 à 0,43).


Nous avons évalué notre modèle et le checkpoint des auteurs avec le même protocole : 512 visages de validation, 5 bonnes et 5 mauvaises clés. Le seuil de reconnaissance (« seuil ») est fixé pour que seulement 0,1 % des paires de personnes différentes le dépassent.

#figure(
	table(
		columns: 4,
		align: (left, center, center, center),
		table.header([*Mesure (ArcFace)*], [*`exp1` (20 k it.)*], [*RiDDLE (90 k it.)*], [*Objectif*]),
		[Original / chiffré], [*0,04*], [0,07], [bas],
		[Chiffrés non reconnus], [100 %], [99,9 %], [haut],
		[Original / déchiffré], [*0,92*], [0,91], [haut],
		[Déchiffrés reconnus], [100 %], [100 %], [haut],
		[Original / mauvaise clé], [0,44], [*0,23*], [bas],
		[Mauvaises clés non reconnues], [54,9 %], [*82,6 %*], [haut],
		[Même personne, deux clés], [0,39], [*0,22*], [bas],
		[Deux personnes, même clé], [*0,45*], [0,52], [bas],
		[LPIPS original / chiffré], [0,186], [*0,149*], [bas],
	),
	caption: [`exp1` et modèle des auteurs : similarités ArcFace moyennes et taux de reconnaissance (512 visages × 5 clés).]
) <tab-exp1>

#figure(
	placement: none,
	kind: image,
	image("image/exp1_courbes.png", width: 68%),
	caption: [`exp1` : similarités ArcFace sur 64 visages de validation (clés fixes). Au-dessus du seuil, ArcFace considère les deux visages comme la même personne.]
) <fig-courbes>

#figure(
	placement: none,
	kind: image,
	block(width: 40%)[
		#entetes[Original][Chiffré (clé 1)][Chiffré (clé 2)][Déchiffré][Mauvaise clé]
		#image("image/exp1_validation.png", width: 100%)
	],
	caption: [`exp1` sur cinq visages de validation, jamais vus pendant l'entraînement.]
) <fig-validation>

#figure(
	placement: none,
	kind: image,
	grid(
		columns: (2cm, 10cm),
		column-gutter: 0.2cm,
		stack(
			v(0.7em),
			..([`exp1` : \ chiffré], [`exp1` : \ mauvaise clé], [RiDDLE : \ chiffré], [RiDDLE : \ mauvaise clé]).map(
				l => box(height: 1.663cm, align(horizon + right, text(size: 9pt, l)))
			),
		),
		[
			#entetes[Photo][Clé 1][Clé 2][Clé 3][Clé 4][Clé 5]
			#image("image/exp1_multi_cle.png", width: 10cm)
		],
	),
	caption: [Un visage, cinq clés : `exp1` et le modèle des auteurs. Avec une mauvaise clé, `exp1` laisse souvent reconnaître la personne.]
) <fig-multi-cle>

== Conclusion
- *Ce qui marche* : réversibilité (0,92, 100 % reconnus, un peu mieux que les auteurs avec 4,5 fois moins d'itérations), dé-identification (aucun chiffré reconnu), réalisme et conservation de la pose, de l'expression et du décor (@fig-validation).
- *Ce qui ne marche pas* : *une mauvaise clé révèle l'identité* (45 % des mauvais déchiffrements reconnus, contre 17 % chez les auteurs, @fig-multi-cle) ; la diversité est insuffisante (0,38) ; l'âge et le genre changent ; et ArcFace sert à la fois à l'entraînement et à la mesure, ce qui biaise l'évaluation.
- *Cause probable* : L1, LPIPS et le terme latent tirent aussi les deux mauvais déchiffrements vers l'original. Le terme latent vaut à lui seul environ 0,47 en fin d'entraînement, plus que tout le terme de dé-identification (≈ 0,22).

= Deuxième entraînement (`exp2`)

== Architecture et pipeline
Le réseau $E$, StyleGAN2, e4e et le schéma « 2 chiffrements, 1 bon déchiffrement, 2 mauvais » sont *inchangés*. Seuls le réseau d'identité et les pertes changent, pour corriger les défauts d'`exp1` :

+ *FaceNet* @facenet (InceptionResnetV1, VGGFace2) remplace ArcFace dans les pertes. ArcFace ne sert plus qu'à l'évaluation, qui devient indépendante de l'entraînement.
+ *Terme latent supprimé* ($lambda_"latent" = 0$, comme les auteurs).
+ *L1 et LPIPS* ne s'appliquent plus qu'aux chiffrés et au bon déchiffrement, plus aux mauvais.
+ *Nouvelle perte « mauvaise clé »*, de poids 2, qui éloigne les mauvais déchiffrements $f_q$ de l'original *et* du bon déchiffrement :

#text(size: 12pt)[
$ cal(L)_2 = & overline(max(0, cos(f, f_e))) + 2 dot overline(max(0, cos(f, f_q)) + max(0, cos(f_d, f_q))) / 2 \
  & + (1 - cos(f, f_d)) + overline(cos(f_k, f_l)) + 0.05 cal(L)_"L1" + cal(L)_"LPIPS" $
]

== Données d'entraînement
Mêmes latents, même découpage et mêmes clés de validation qu'`exp1`, pour que les deux courbes soient comparables. Faute de temps, l'entraînement est limité à *10 000 itérations* (environ 4 h à 0,68 it/s), avec un checkpoint toutes les 1 000 itérations. On le compare donc au checkpoint d'`exp1` à 10 000 itérations.

== Résultats
*L'entraînement a échoué : le déchiffrement ne redonne plus la personne.* Sur la @fig-courbes-2, la récupération ne dépasse jamais 0,12 (contre 0,89 pour `exp1` au même stade). Les autres mesures paraissent excellentes (mauvaise clé à 0,02), mais pour une mauvaise raison, visible sur la @fig-validation-2 :

- *Les visages chiffrés ne dépendent presque plus que de la clé* : la clé 1 donne la même femme aux yeux clairs pour les cinq personnes, la clé 2 un même visage plus neutre. La pose, l'âge, le genre et les lunettes sont perdus.
- *Le déchiffré est un visage « moyen »*, identique pour tout le monde, puisque l'information sur l'original a été jetée au chiffrement.
- *Les mauvaises clés donnent des images hors du domaine de StyleGAN2* (couleurs saturées, visages fantomatiques). Leur cosinus de 0,02 ne prouve donc pas qu'elles protègent mieux : ce ne sont plus des visages.

Dès 1 000 itérations, toutes les images générées étaient déjà dégénérées : sans terme latent, les codes produits quittent immédiatement le domaine de StyleGAN2. La distance latente moyenne $norm(w_e - w)$ est de 40,8 contre 13,2 pour `exp1` au même stade, et elle *augmente* tout au long de l'entraînement au lieu de baisser.

#figure(
	placement: none,
	kind: image,
	image("image/exp2_courbes.png", width: 68%),
	caption: [`exp2` : mêmes mesures ArcFace que la @fig-courbes, sur les mêmes visages et clés de validation.]
) <fig-courbes-2>

#figure(
	placement: none,
	kind: image,
	block(width: 40%)[
		#entetes[Original][Chiffré (clé 1)][Chiffré (clé 2)][Déchiffré][Mauvaise clé]
		#image("image/exp2_validation.png", width: 100%)
	],
	caption: [`exp2` sur les mêmes visages et les mêmes clés que la @fig-validation.]
) <fig-validation-2>

== Conclusion
Les pertes qui tiraient les images vers l'original étaient la *cause* de la fuite par mauvaise clé, mais aussi ce qui *forçait le réseau à garder l'information* sur la personne. Sans elles, le minimum le plus facile consiste à ignorer $w$ : un visage qui ne dépend que de la clé est toujours anonyme et toujours différent pour deux clés. La perte de récupération (poids 1) ne suffit pas à sortir de ce minimum, d'autant que la perte « mauvaise clé » (poids 2) la contredit tant que le réseau ne sait pas distinguer la bonne clé des mauvaises.

Réseau d'identité et pertes ont changé en même temps : on ne peut donc pas exclure que FaceNet y contribue. FaceNet lui-même confirme l'échec (récupération 0,26 contre 0,91 pour `exp1`).

= Comparaison des deux entraînements

== Mesures
Le @tab-comparaison compare les deux entraînements à budget égal (10 000 itérations), avec `exp1` à 20 000 itérations et le modèle des auteurs en référence. Chaque modèle est mesuré par ArcFace *et* par FaceNet, donc toujours aussi par un réseau qu'il n'a jamais vu.

#figure(
	table(
		columns: 6,
		align: (left, center, center, center, center, center),
		table.header([*Mesure*], [*Réseau*], [*`exp1` 10 k*], [*`exp2` 10 k*], [*`exp1` 20 k*], [*RiDDLE 90 k*]),
		table.cell(rowspan: 2)[Original / chiffré ↓], [ArcFace], [0,01], [0,01], [0,04], [0,07],
		[FaceNet], [0,06], [−0,11], [0,08], [0,13],
		table.cell(rowspan: 2)[Original / déchiffré ↑], [ArcFace], [*0,88*], [0,11], [0,92], [0,91],
		[FaceNet], [*0,91*], [0,26], [0,94], [0,93],
		table.cell(rowspan: 2)[Déchiffrés reconnus ↑], [ArcFace], [*100 %*], [0 %], [100 %], [100 %],
		[FaceNet], [*100 %*], [1,1 %], [100 %], [100 %],
		table.cell(rowspan: 2)[Original / mauvaise clé ↓], [ArcFace], [0,38], [*0,03*], [0,44], [0,23],
		[FaceNet], [0,36], [*0,03*], [0,45], [0,23],
		table.cell(rowspan: 2)[Mauvaises clés non reconnues ↑], [ArcFace], [64,6 %], [*100 %*], [54,9 %], [82,6 %],
		[FaceNet], [86,3 %], [*100 %*], [69,5 %], [95,9 %],
		table.cell(rowspan: 2)[Même personne, deux clés ↓], [ArcFace], [0,45], [0,55], [0,39], [0,22],
		[FaceNet], [0,48], [0,44], [0,41], [0,24],
		table.cell(rowspan: 2)[Deux personnes, même clé ↓], [ArcFace], [0,39], [0,59], [0,45], [0,52],
		[FaceNet], [0,50], [0,64], [0,55], [0,58],
		[LPIPS original / chiffré ↓], [—], [*0,211*], [0,393], [0,186], [0,149],
	),
	caption: [Évaluation finale (512 visages × 5 clés). En gras, les écarts marquants entre les deux entraînements à 10 000 itérations. Seuils de reconnaissance : 0,434 (ArcFace), 0,651 (FaceNet).]
) <tab-comparaison>

== Courbes et visages

#figure(
	placement: none,
	kind: image,
	image("image/comparaison_courbes.png", width: 100%),
	caption: [Courbes de validation ArcFace d'`exp1` et d'`exp2` sur les 10 000 premières itérations (mêmes visages, mêmes clés). Pointillés : seuil de reconnaissance.]
) <fig-comparaison-courbes>

#figure(
	placement: none,
	kind: image,
	block(width: 100%)[
		#entetes[Photo][Inversion e4e][`exp1` chiffré][`exp1` déchiffré][`exp1` mauv. clé][`exp2` chiffré][`exp2` déchiffré][`exp2` mauv. clé]
		#image("image/comparaison_photos.png", width: 100%)
	],
	caption: [Nos photos chiffrées (clé 1234), déchiffrées et déchiffrées avec une mauvaise clé (999), par `exp1` et `exp2` à 10 000 itérations. Cosinus ArcFace original / déchiffré : 0,76 / 0,80 / 0,63 pour `exp1`, 0,01 / 0,01 / 0,03 pour `exp2`.]
) <fig-comparaison-photos>

== Bilan
- *Amélioré* : en apparence seulement, les mauvaises clés (100 % non reconnues contre 64,6 %), car elles ne produisent plus des visages. En revanche, l'évaluation est devenue honnête : mesuré par FaceNet, `exp1` reste bon (chiffré 0,08, déchiffré 0,94) ; il ne faisait donc pas que tromper ArcFace.
- *Inchangé* : la dé-identification (au moins 99,9 % des chiffrés non reconnus) et une diversité entre deux clés insuffisante (ArcFace 0,45 → 0,55, FaceNet 0,48 → 0,44).
- *Moins bien* : la réversibilité (0,11 contre 0,88), la conservation des attributs (LPIPS 0,393 contre 0,211) et l'indépendance à la clé (même clé, deux personnes : 0,59 contre 0,39).

*`exp1` reste donc notre meilleur modèle.* Nous en avons tiré trois règles, appliquées dans la section suivante :

+ *Garder l'ancrage* (latent, L1, LPIPS), mais seulement sur les chiffrés et le bon déchiffrement.
+ *Repartir des poids d'`exp1`*, qui sait déjà se réinverser, en introduisant la perte « mauvaise clé » progressivement.
+ *Tester un changement à la fois* sur 2 000 itérations (environ 1 h) : `exp1` y dépassait déjà 0,7 en récupération, assez pour détecter un échec comme celui d'`exp2`.

= Corriger la fuite : quatre essais courts

== Protocole commun
Les quatre essais repartent des poids d'`exp1` (20 000 itérations), gardent ArcFace dans les pertes et durent 2 000 itérations. L'ancrage ne porte plus que sur $e_1, e_2, d$, et la perte « mauvaise clé » d'`exp2` monte de 0 à 2 pendant les 1 000 premières itérations. Chaque essai ajoute *un* élément, pour corriger le défaut du précédent (@tab-essais).

#figure(
	table(
		columns: (auto, 1fr, auto),
		align: (left, left, left),
		table.header([*Essai*], [*Ajout*], [*Défaut observé*]),
		[`exp3a`], [protocole commun seul], [mauvaises clés hors du domaine : taches],
		[`exp4a`], [perte de *domaine* (distance de chaque couche de $w_q$ au latent moyen) et marge de 0,2 sur la perte « mauvaise clé »], [perte contournée : toujours des taches],
		[`exp4a_bis`], [perte de *réalisme* : discriminateur de StyleGAN2, figé, sur l'image bornée], [divergence : images jaunes],
		[`exp4a_ter`], [vrai bornage devant le discriminateur et rappel des pixels dans $[-1, 1]$], [discriminateur trompé : textures],
	),
	caption: [Les quatre essais de correction, chacun construit sur le précédent.]
) <tab-essais>

== Résultats
*La fuite est corrigée dans tous les essais*, et la récupération reste correcte (0,89 à 0,90) dès qu'on ajoute la perte de domaine (@tab-essais-mesures). *Mais les mauvais déchiffrements ne sont plus des visages* (@fig-mauvaises-cles) : la distance LPIPS entre l'original et le mauvais déchiffrement, d'environ 0,16 pour un visage réaliste (`exp1`, auteurs), monte à 0,44 à 0,66. Un non-visage n'est reconnu par personne : le cosinus faible ne prouve donc pas la protection. La @fig-compromis résume la situation : nos essais passent d'un coin à l'autre (visage réaliste qui fuit, ou non-visage anonyme) sans atteindre la zone du modèle des auteurs.

#figure(
	text(size: 9pt, table(
		columns: (1fr,) + (auto,) * 7,
		align: (left, center, center, center, center, center, center, center),
		table.header([*Mesure*], [*`exp1`*], [*`exp3a`*], [*`exp4a` 1 k*], [*`exp4a` 2 k*], [*`exp4a_bis`*], [*`exp4a_ter`\**], [*RiDDLE*]),
		[Orig. / déchiffré ↑], [*0,92*], [0,77], [0,89], [0,89], [0,01], [0,90], [0,91],
		[Orig. / mauv. clé ↓], [0,44], [0,07], [*0,04*], [0,19], [0,01], [0,10], [0,23],
		[Mauv. non reconnues ↑], [54,9 %], [95,3 %], [*99,3 %*], [81,0 %], [100 %], [—], [82,6 %],
		[#h(0.8em) idem, FaceNet ↑], [69,5 %], [98,9 %], [*100 %*], [85,2 %], [100 %], [—], [95,9 %],
		[LPIPS orig. / mauv. clé], [*0,157*], [0,609], [0,660], [0,441], [1,081], [0,59], [0,159],
		[Score $D$, mauv. clé], [−0,54], [—], [—], [−0,99], [−1,07], [+0,43], [—],
		[Pixels saturés ↓], [*0,3 %*], [—], [—], [0,4 %], [100 %], [6,0 %], [—],
	)),
	caption: [Essais de correction (ArcFace sauf mention, 512 visages × 5 clés). LPIPS : ≈ 0,16 pour un visage réaliste. Score $D$ : −0,37 pour de vraies images. Pixels saturés : hors de $[-1, 1]$, mauvaises clés. \* `exp4a_ter` : validation seulement (64 visages), l'évaluation complète a été interrompue. — : non mesuré.]
) <tab-essais-mesures>

#figure(
	placement: none,
	kind: image,
	block(width: 100%)[
		#entetes[Original][`exp1`][`exp3a`][`exp4a` 2 k][`exp4a_bis`][`exp4a_ter` 2 k]
		#image("image/exp34_mauvaises_cles.png", width: 100%)
	],
	caption: [Le même visage déchiffré avec la même *mauvaise* clé par chaque essai (visages de validation). `exp1` donne un visage réaliste mais reconnaissable ; les corrections donnent des taches, une image jaune, puis des textures.]
) <fig-mauvaises-cles>

#figure(
	placement: none,
	kind: image,
	image("image/compromis_mauvaises_cles.png", width: 85%),
	caption: [Fuite d'identité et écart visuel des mauvais déchiffrements, pour chaque essai. En vert, la zone visée : un autre visage, réaliste. \* `exp4a_ter` : validation.]
) <fig-compromis>

== Ce que chaque essai nous a appris
- *`exp3a`* : rien ne retient plus les mauvais déchiffrements dans le domaine de StyleGAN2, et *le chemin le plus court pour ne plus ressembler à l'original est de ne plus être un visage*. Le défaut est localisé dans les couches moyennes (4 à 7, celles des traits du visage) : leur distance au latent moyen atteint 17 à 21 contre 13 à 15 pour de vrais latents. La moyenne sur les 14 couches le cache. La récupération en pâtit (0,77, pire visage à −0,07).
- *`exp4a`* : la perte de domaine est satisfaite, mieux même qu'avec `exp1`, mais elle ne mesure qu'une *distance* au centre. Un code peut être à distance normale dans une *direction* qu'aucun vrai latent ne prend. Une distance de Mahalanobis ne sépare pas mieux les cas : les bons chiffrés, pourtant réalistes, en sont plus loin (7 fois le rayon des vrais latents) que les taches (4,8 fois). Un discriminateur *latent* pénaliserait donc aussi les bons chiffrés ; nous l'avons abandonné. Le discriminateur *image* de StyleGAN2, lui, sépare les cas : 41 à 47 % des taches sont sous le 5#super[e] centile des vraies images, contre 8 % des mauvaises clés d'`exp1`.
- *`exp4a_bis`* : divergence vers 300 itérations. Le bornage laissait passer le gradient sur les pixels saturés, que le discriminateur ne voit pas : 93 à 96 % de ces gradients poussaient les pixels encore plus loin. L'amplitude des images a atteint $10^12$, toutes les sorties sont devenues jaunes et le modèle a cessé de dépendre de la clé. Nous avons corrigé le bornage, ajouté une perte de rappel des pixels, une mesure de saturation et un arrêt d'urgence.
- *`exp4a_ter`* : plus de divergence, fuite supprimée (0,10) et récupération de 0,90, mais *le discriminateur est trompé* : les mauvais déchiffrements obtiennent un meilleur score (+0,43) que de vraies photos (−0,37), avec des fonds texturés très saturés (6 % de pixels saturés, 20 fois `exp1`). Figé, un discriminateur laisse au chiffreur le temps de trouver ses angles morts.

= Bilan et prochain essai

== Leçons
+ Les chiffrés et le bon déchiffrement *doivent rester ancrés* sur l'original, sinon le chiffreur ignore $w$ (`exp2`).
+ Les mauvais déchiffrements *ne doivent pas l'être*, sinon l'identité fuit (`exp1`)…
+ … mais sans ancrage, ils sortent du domaine (`exp3a`), et *les critères globaux de réalisme sont contournés* : distance au latent moyen (`exp4a`), Mahalanobis, discriminateur figé (`exp4a_ter`). Il faut une *cible précise* pour les mauvais déchiffrements, pas un critère de plus.
+ Le chiffreur est le même dans les deux sens : tout ce qu'on impose à $E(w_e, q)$ agit aussi sur $E(w_e, p)$, et chaque correction coûte un peu de récupération (0,92 → 0,89-0,90).
+ Ne jamais conclure sur un cosinus seul : regarder les images, LPIPS et la saturation des mauvais déchiffrements. Un score de réalisme supérieur à celui des vraies images est un symptôme, pas une réussite.

== Prochain essai : ancrer les mauvaises clés sur le chiffré
Le chiffré $w_(e 1) = E(w, p_1)$ est déjà ce que l'on veut pour une mauvaise clé : un visage *réaliste* et *anonyme*. Nous ancrerons donc les mauvais déchiffrements $w_q = E(w_(e 1), q)$ sur lui, sans gradient vers le chiffré ($"sg"$) pour ne pas l'abîmer :

$ cal(L)_"qanc" = "LPIPS"(X_q, "sg"(X_(e 1))) + alpha norm(w_q - "sg"(w_(e 1))), quad alpha approx 0","03 $

- *Pas de retour de la fuite d'`exp1`* : la cible n'est plus l'original mais un visage déjà anonyme, donc l'ancrage et la perte « mauvaise clé » vont dans le même sens.
- *Pas une simple copie du chiffré* : la perte de diversité éloigne déjà l'identité de $q$ de celle de $e_1$, pendant que l'ancrage garde la pose, le fond et l'éclairage. On attend un autre visage dans la même scène, comme dans RiDDLE.
- *Le discriminateur est retiré* : c'est lui qui poussait vers ses angles morts. La perte de domaine, la marge et le rappel des pixels sont conservés.

Même protocole (départ d'`exp1`, 2 000 itérations). Réussite si LPIPS original / mauvaise clé < 0,25, moins de 1 % de pixels saturés, mauvaise clé < 0,30, récupération ≥ 0,85, et un mauvais déchiffrement qui ne recopie pas le chiffré. Ensuite : le même essai avec FaceNet dans les pertes, puis le meilleur prolongé à 10 000 itérations.

#pagebreak(weak: true)
#bibliography("bibliography/anonymisation-bib.yaml", title: "Bibliographie")
