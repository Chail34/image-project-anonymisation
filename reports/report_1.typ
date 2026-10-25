#import "template.typ": template


#show: template.with(
    title: [Rapport 1 - Projet Image : #linebreak() Anonymisation],
    short_title: [Rapport 1 - Anonymisation],
    subtitle: [Les débuts du projet],
    author: ([Rémi Bonnafé], [Mickaël Longlade], [Killian Texier-Grailhes]),
)


= État de l'art

On va tout d'abord regarder l'article de référence pour voir l'objectif final, puis on regardera un article pour chacune des trois briques que l'on doit développer dans le projet : l'*encodeur*, l'*auto-encodeur* et le *GAN*. Tous les articles qui suivent sont cités dans l'article de référence.

== Article de référence : RiDDLE @riddle-article

Le titre nous donne déjà beaucoup d'informations : *Reversible and Diversified De-identification with Latent Encryptor* signifie, pris littéralement, une dé-identification réversible et diversifiée avec un chiffreur latent.

#figure(
    image("images/RiDDLE_figure2.png", width: 100%),
    caption: [Pipeline complet de la méthode *RiDDLE*.]
) <fig-riddle-pipeline>

On peut observer sur la @fig-riddle-pipeline toutes les étapes permettant d'*anonymiser* et de *désanonymiser* un visage. Voici ces étapes :
+ un *encodeur* d'inversion projette l'image dans un espace latent de StyleGAN2 ;
+ le *latent encryptor*, qui va combiner le code latent obtenu plus tôt avec un *mot de passe*, afin de rendre l'anonymisation réversible (ce mot de passe est lui aussi issu d'un espace latent) ;
+ le *générateur* StyleGAN2, qui va créer un nouveau visage à partir du code latent chiffré.

À noter que, dans cette méthode, seul le *latent encryptor* est entraîné : l'encodeur d'inversion et le générateur StyleGAN2 sont pré-entraînés.

== Encodeur : ArcFace @arcface

ArcFace est un réseau de *reconnaissance faciale* : il transforme un visage en un vecteur de 512 valeurs. Deux photos de la même personne donnent des vecteurs proches. Attention, ce n'est pas l'encodeur d'inversion de RiDDLE (e4e), qui projette l'image dans l'espace latent de StyleGAN2 : ArcFace est un encodeur d'*identité*.

#figure(
    image("images/arcFace_figure2.png", width: 100%),
    caption: [Entraînement d'un réseau de reconnaissance faciale avec *ArcFace*.]
) <fig-arcface-pipeline>

Le réseau est entraîné à classer des identités (@fig-arcface-pipeline). L'idée d'ArcFace est d'ajouter une marge $m$ à l'angle $theta_y$ entre le visage et sa bonne classe $y$ :

$ cal(L) = -log frac(e^(s cos(theta_y + m)), e^(s cos(theta_y + m)) + sum_(j != y) e^(s cos theta_j)) $

où $cal(L)$ est la perte à minimiser, $theta_j$ l'angle entre le vecteur du visage et celui de la classe $j$, et $s$ un facteur d'échelle fixe. Cette marge oblige les identités à être bien séparées (@fig-arcface-toy).

#figure(
    image("images/arcFace_figure3.png", width: 60%),
    caption: [8 identités en 2D : softmax classique à gauche, *ArcFace* à droite.]
) <fig-arcface-toy>

Ce réseau est utilisé dans la méthode *RiDDLE* afin de vérifier que le visage anonymisé ne ressemble pas au visage original, et sert surtout à entraîner le *latent encryptor*.

== Auto-encodeur : Live Face De-Identification in Video @live-face-deid

Cette méthode anonymise des visages dans des vidéos, en temps réel, avec un *auto-encodeur* entraîné de bout en bout. Contrairement à RiDDLE, il n'y a pas de StyleGAN pré-entraîné : le réseau apprend lui-même à décoder le visage. Il en sort deux choses : un nouveau visage et un *masque*, qui sert à le fusionner avec l'image d'origine pour ne pas redessiner le fond (@fig-gafni-archi).

#figure(
    image("images/LiveFace_figure2.png", width: 100%),
    caption: [Architecture de l'auto-encodeur de Gafni _et al._]
) <fig-gafni-archi>

Pour enlever l'identité, elle utilise une perte avec un réseau de reconnaissance faciale :
- ses *premières couches* (pose, expression, éclairage) doivent rester proches de l'original ;
- ses *dernières couches* (l'identité) doivent s'en éloigner.

Le visage change donc de personne, mais garde la même attitude (@fig-gafni-resultats). En revanche, cette méthode n'est pas réversible et ne donne qu'un seul résultat par visage.

#figure(
    image("images/LiveFace_figure1.png", width: 60%),
    caption: [Image source (en haut) et image anonymisée (en bas).]
) <fig-gafni-resultats>

== GAN : CIAGAN @ciagan

*CIAGAN* est un GAN *conditionnel*. Le générateur reçoit l'image avec le visage masqué, les *landmarks* (les points clés du visage) et une *identité cible*, puis il dessine un nouveau visage. Deux discriminateurs l'entraînent : l'un vérifie que l'image est réaliste, l'autre qu'elle a bien l'identité cible (@fig-ciagan-archi).

#figure(
    image("images/CIAGAN_figure2.png", width: 100%),
    caption: [Architecture de *CIAGAN*.]
) <fig-ciagan-archi>

Ainsi, en changeant l'identité cible, on obtient des visages différents (@fig-ciagan-resultats).

#figure(
    image("images/CIAGAN_figure3.png", width: 60%),
    caption: [Un même visage anonymisé avec plusieurs identités cibles.]
) <fig-ciagan-resultats>

= Suite du projet

Pour la semaine prochaine, on compte commencer l'implémentation d'un *encodeur* en Python. C'est le langage le plus utilisé et le mieux documenté pour les réseaux de neurones, et c'est celui du code de RiDDLE et des modèles pré-entraînés qu'il utilise (ArcFace, StyleGAN2).

#bibliography("anonymisation-bib.yaml")
