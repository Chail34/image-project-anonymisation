// Présentation courte du chiffreur latent (6 diapos, 16:9). Peu de texte : support pour l'oral.
// Compilation : typst compile crypting_latent_presentation.typ ../pdf/crypting_latent_presentation.pdf
// Figures : source/anon/runs/exp1/eval/make_figures.py, source/anon/runs/exp4a_ter/eval/make_figures.py

#let ink = rgb("#1f1f1f")
#let muted = rgb("#6b6a66")
#let line-col = rgb("#cfcec8")
#let blue = rgb("#2a78d6")
#let orange = rgb("#eb6834")
#let green = rgb("#1baf7a")
#let frozen = rgb("#efeeea")

#set page(width: 33.867cm, height: 19.05cm, margin: (x: 1.6cm, top: 1.2cm, bottom: 1.2cm), fill: white,
	footer: context {
		set text(size: 11pt, fill: muted)
		if counter(page).get().first() > 1 [Groupe 5 · HAI927I #h(1fr) #counter(page).display()]
	})
#set text(font: ("Inter", "Inter Display"), size: 20pt, fill: ink, lang: "fr")
#show raw: set text(size: 1.1em)

#let slide(title, body) = {
	pagebreak(weak: true)
	text(size: 30pt, weight: "bold", title)
	v(0.2cm)
	line(length: 3cm, stroke: 3pt + blue)
	v(0.4cm)
	body
}

// Briques des schémas
#let I(name) = "image/" + name
#let pic(path, h: 3.4cm, stroke: none) = box(stroke: stroke, radius: 4pt, clip: true, image(path, height: h))
#let node(body, fill: frozen, stroke: line-col) = box(fill: fill, stroke: 1.5pt + stroke, radius: 6pt,
	inset: (x: 0.4cm, y: 0.3cm), align(center + horizon, body))
#let enc(key, col: blue) = node(fill: col.lighten(85%), stroke: col)[
	#text(weight: "bold", fill: col)[$E$] #h(0.1cm) #text(size: 16pt)[🔑 #key]]
#let arr(label: none, w: 2.3cm) = box(width: w, align(center + horizon, stack(spacing: 0.12cm,
	text(size: 12pt, fill: muted, if label == none [#h(0pt)] else [#label]),
	stack(dir: ltr, line(length: w - 0.55cm, stroke: 2pt + muted),
		polygon(fill: muted, (0pt, -5pt), (9pt, 0pt), (0pt, 5pt))),
	text(size: 12pt, [#h(0pt)]))))
#let tag(body, col, size: 14pt) = box(fill: col.lighten(85%), radius: 4pt, inset: (x: 0.2cm, y: 0.12cm),
	text(size: size, weight: "semibold", fill: col.darken(20%), body))
#let stat(value, label, col: blue) = block(width: 100%, inset: (left: 0.4cm, y: 0.15cm),
	stroke: (left: 4pt + col))[
	#text(size: 44pt, weight: "bold", fill: col, value) \
	#text(size: 16pt, fill: muted, label)]
#let heads(w, ..names) = grid(columns: (w,) * names.pos().len(), align: center + bottom,
	..names.pos().map(t => text(size: 13pt, fill: muted, t)))

// ---------------------------------------------------------------- 1. Titre
#align(center + horizon)[
	#text(size: 44pt, weight: "bold")[Chiffreur latent réversible]
	#v(0.1cm)
	#text(size: 22pt, fill: muted)[Anonymiser un visage avec une clé… et le retrouver]
	#v(1cm)
	#grid(columns: 5, column-gutter: 0.3cm, align: center + horizon,
		pic(I("pipe_orig.png"), h: 4.5cm), arr(label: [🔑 $p$]),
		pic(I("pipe_chiffre.png"), h: 4.5cm, stroke: 3pt + blue), arr(label: [🔑 $p$]),
		pic(I("pipe_dechiffre.png"), h: 4.5cm, stroke: 3pt + green))
	#v(1cm)
	#text(size: 16pt, fill: muted)[Groupe 5 · HAI927I Projet Image · d'après RiDDLE (Li et al., 2023)]
]

// ---------------------------------------------------------------- 2. Pipeline
#slide[Pipeline][
	#let G = arr(label: [StyleGAN2], w: 2.9cm)
	#let out(path, label, col) = stack(spacing: 0.15cm, pic(path, stroke: 3pt + col), tag(label, col))
	#align(center, grid(
		columns: 9, align: center + horizon, row-gutter: 0.35cm,
		pic(I("pipe_orig.png")), arr(label: [e4e]), node[$w$ \ #text(size: 12pt, fill: muted)[$14 times 512$]],
		arr(), enc[$p$], arr(), node[$w_e$], G, out(I("pipe_chiffre.png"), [chiffré], blue),
		[], [], node[$w_e$], arr(), enc[$p$], arr(), node[$w_d$], G, out(I("pipe_dechiffre.png"), [bonne clé], green),
		[], [], node[$w_e$], arr(), enc(col: orange)[$q$], arr(), node[$w_q$], G, out(I("pipe_mauvaise.png"), [mauvaise clé], orange),
	))
	#place(right + top, dy: 0.25cm, stack(dir: ltr, spacing: 0.3cm,
		tag([entraîné : $E$ (transformer, 9,7 M param.)], blue),
		tag([figés : e4e, StyleGAN2], muted)))
]

// ---------------------------------------------------------------- 3. Meilleurs résultats
#slide[Meilleur modèle : `exp1`][
	#grid(columns: (auto, 1fr), column-gutter: 1.5cm, align: top,
		[
			#heads(2.3cm, [original], [clé 1], [clé 2], [déchiffré], [mauv. clé])
			#v(-0.4cm)
			#image(I("exp1_validation.png"), height: 11.5cm)
		],
		align(horizon, stack(spacing: 0.9cm,
			stat([0,92], [similarité original / déchiffré · RiDDLE : 0,91], col: green),
			stat([100 %], [chiffrés non reconnus], col: blue),
			stat([20 k], [itérations · RiDDLE : 90 k], col: muted),
		)),
	)
]

// ---------------------------------------------------------------- 4. Ce qui ne marche pas
#slide[Ce qui ne marche pas : la mauvaise clé][
	#grid(columns: (auto, 1fr), column-gutter: 1.2cm,
		grid(columns: (3.2cm, auto), column-gutter: 0.3cm,
			stack(v(0.75cm), ..(
				([`exp1` \ chiffré], blue), ([`exp1` \ mauvaise clé], orange),
				([RiDDLE \ chiffré], muted), ([RiDDLE \ mauvaise clé], muted),
			).map(((l, c)) => box(height: 2.875cm, align(horizon + right, text(size: 14pt, fill: c, weight: "semibold", l))))),
			[
				#heads(2.875cm, [photo], [clé 1], [clé 2], [clé 3], [clé 4], [clé 5])
				#v(-0.4cm)
				#image(I("exp1_multi_cle.png"), height: 11.5cm)
			],
		),
		align(horizon, stack(spacing: 0.9cm,
			stat([45 %], [mauvaises clés *reconnues* \ RiDDLE : 17 %], col: orange),
			stat([0,39], [même personne, deux clés \ RiDDLE : 0,22], col: orange),
		)),
	)
]

// ---------------------------------------------------------------- 5. Nos corrections
#slide[Corriger la fuite… sans perdre le visage][
	#let w = 2.45cm
	#grid(columns: (auto, 1fr), column-gutter: 0.8cm, align: top,
		[
			#grid(columns: (w,) * 6, align: center + bottom,
				..(([original], muted), ([`exp1`], blue), ([`exp3a`], orange), ([`exp4a`], orange),
					([`exp4a_bis`], orange), ([`exp4a_ter`], orange)).map(((t, c)) => text(size: 11pt, fill: c, weight: "semibold", t)))
			#v(-0.4cm)
			#image(I("exp34_mauvaises_cles.png"), width: 6 * w)
			#v(-0.4cm)
			#grid(columns: (w,) * 6, align: center,
				[], ..(([fuite], blue), ([taches], orange), ([taches], orange), ([diverge], orange), ([$D$ trompé], orange)).map(((t, c)) => tag(t, c, size: 12pt)))
		],
		align(horizon, image(I("compromis_mauvaises_cles.png"), width: 100%)),
	)
]

// ---------------------------------------------------------------- 6. Améliorer
#slide[Prochain essai : ancrer la mauvaise clé sur le chiffré][
	#let s = 4.4cm
	#let card(title, col, anchor, result, verdict, anchor-label, strong: false) = block(width: 100%, inset: 0.4cm,
		radius: 8pt, stroke: (if strong { 2.5pt } else { 1.5pt }) + col.lighten(if strong { 0% } else { 40% }),
		fill: col.lighten(93%), align(center, stack(spacing: 0.3cm,
			text(size: 18pt, weight: "bold", fill: col.darken(20%), title),
			grid(columns: 3, align: center + horizon, column-gutter: 0.25cm, row-gutter: 0.15cm,
				anchor, text(size: 26pt, fill: muted)[⟵], result,
				text(size: 13pt, fill: muted, anchor-label), [], text(size: 13pt, fill: muted)[mauvaise clé]),
			text(size: 18pt, weight: "semibold", fill: col.darken(20%), verdict))))
	#let empty-box = box(width: s, height: s, radius: 4pt, stroke: (dash: "dashed", paint: muted, thickness: 1.5pt),
		align(center + horizon, text(size: 30pt, fill: muted)[∅]))
	#let target = box(width: s, height: s, radius: 4pt, stroke: (dash: "dashed", paint: green, thickness: 2pt),
		fill: green.lighten(90%), align(center + horizon, text(size: 40pt, fill: green, weight: "bold")[?]))
	#grid(columns: (1fr,) * 3, column-gutter: 0.6cm,
		card([`exp1`], blue, pic(I("pipe_orig.png"), h: s), pic(I("pipe_mauvaise.png"), h: s),
			[✗ l'identité fuit], [ancre = original]),
		card([`exp3a` → `exp4a_ter`], orange, empty-box, pic(I("pipe_mauvaise_4a.png"), h: s),
			[✗ plus un visage], [pas d'ancre]),
		card([`exp5a`], green, pic(I("pipe_chiffre.png"), h: s), target,
			[✓ autre visage, même scène], [ancre = chiffré], strong: true),
	)
	#v(0.7cm)
	#align(center, text(size: 18pt, fill: muted)[
		$"LPIPS"(X_q, X_e) + alpha norm(w_q - w_e)$ #h(0.8cm) · #h(0.8cm) sans discriminateur #h(0.8cm) · #h(0.8cm)
		puis FaceNet → 10 k itérations])
]
