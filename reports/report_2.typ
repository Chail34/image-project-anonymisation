#import "template.typ": template


#show: template.with(
    title: [Rapport 2 - Projet Image : #linebreak() Anonymisation],
    short_title: [Rapport 2 - Anonymisation],
    subtitle: [Prise en main de RiDDLE],
    author: ([Rémi Bonnafé], [Mickaël Longlade], [Killian Texier-Grailhes]),
)


= Ce qu'on a fait cette semaine

On a fait tourner le code de *RiDDLE* @riddle-article : mise à jour de l'environnement, récupération des poids pré-entraînés, et un script qui aligne et inverse (avec e4e) nos propres photos. On l'a testé sur des visages de FFHQ et sur 4 photos à nous.

Pour mesurer l'identité, on utilise la similarité cosinus entre vecteurs ArcFace @arcface (1 = même personne) :

#figure(
    table(
        columns: 2,
        align: (left, center),
        table.header([*Visages comparés*], [*Similarité*]),
        [Original / inversion e4e], [0,83],
        [Original / chiffré], [0,02],
        [Original / déchiffré (bon mot de passe)], [0,73],
        [Original / déchiffré (mauvais mot de passe)], [0,12],
        [Deux personnes différentes (originaux)], [0,10],
        [Même personne, deux mots de passe (chiffrés)], [0,13],
        [Deux personnes, même mot de passe (chiffrés)], [*0,60*],
    ),
    caption: [Similarité ArcFace moyenne (3 personnes, 5 mots de passe).]
) <tab-mesures>

Ces chiffres restent indicatifs : peu de visages, et ArcFace est aussi utilisé pour entraîner RiDDLE.


= Ce que RiDDLE fait bien

#figure(
    block(width: 100%)[
        #grid(
            columns: (1fr,) * 5,
            align: center,
            text(size: 9pt)[Original], text(size: 9pt)[Inversion], text(size: 9pt)[Chiffré], text(size: 9pt)[Déchiffré], text(size: 9pt)[Mauvais mdp],
        )
        #v(-0.4em)
        #image("images/riddle_nos_photos.png", width: 100%)
    ],
    caption: [RiDDLE sur deux de nos photos.]
) <fig-nos-photos>

- *Identité effacée* : 0,02 entre l'original et le visage chiffré.
- *Attributs conservés* : pose, expression, coiffure, vêtements, fond (@fig-nos-photos).
- *Réalisme* avec le bon mot de passe.
- *Réversible* (0,73) et *diversifié* : deux mots de passe donnent deux identités distinctes (@fig-multi-mdp).
- *Rapide* : seul le *latent encryptor* est entraîné, \~6 s pour 8 visages.

#figure(
    image("images/riddle_multi_mdp.png", width: 100%),
    caption: [Un visage, 5 mots de passe. Lignes : chiffré, déchiffré (bon mot de passe), déchiffré (mauvais mot de passe).]
) <fig-multi-mdp>


= Ce que RiDDLE fait moins bien

- *Mauvais mot de passe → visages irréalistes* : yeux jaunes, nez gonflé, traits « collés » (dernière ligne de la @fig-multi-mdp). Le chiffreur est appliqué deux fois, ce qui sort le code latent de la zone réaliste de StyleGAN2, et RiDDLE n'a *pas de discriminateur* pour l'en empêcher.
- *Un mot de passe → presque le même visage pour tous* (0,60, @fig-meme-mdp). La perte de diversité sépare les mots de passe, pas les personnes : une base anonymisée avec un seul mot de passe ne permet plus de distinguer les gens.
- *Âge, genre et origine non conservés* : un bébé devient une femme adulte (@fig-bebes).
- *Inversion e4e imparfaite* (0,83) : le fond, la barbe et la texture de peau sont perdus, donc le déchiffrement ne redonne jamais la photo exacte.
- *Autres limites* : cheveux, vêtements et fond restent reconnaissables ; le visage (256×256, aligné) n'est pas réinséré dans la photo ; la détection rate les profils.

#figure(
    image("images/riddle_meme_mdp.png", width: 56%),
    caption: [Trois de nos photos (haut) chiffrées avec le *même* mot de passe (bas).]
) <fig-meme-mdp>

#figure(
    image("images/riddle_bebes.png", width: 75%),
    caption: [Deux enfants de FFHQ et leur version chiffrée.]
) <fig-bebes>


= Objectifs pour la suite

+ *Évaluer* avec un autre réseau de reconnaissance faciale, sur plus de visages (partie *encodeur*).
+ *Ajouter un discriminateur* sur les visages chiffrés et mal déchiffrés pour garantir leur réalisme (partie *GAN*).
+ *Séparer les personnes* : une perte qui éloigne les visages chiffrés de personnes différentes avec le même mot de passe.
+ *Conserver l'âge et le genre* via un classifieur.
+ *Réinsérer le visage* dans la photo avec un masque, comme Gafni _et al._ @live-face-deid (partie *auto-encodeur*).

Les deux premiers points sont prioritaires.

#pagebreak(weak: true)
#bibliography("anonymisation-bib.yaml")
