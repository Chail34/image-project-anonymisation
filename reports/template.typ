// ===========
// = Helpers =
// ===========


#let _heading_numbering(depth, counter) = {
 if counter == 0 {
 } else if depth == 1 {
  numbering("I.", counter)
 } else if depth == 2 {
  numbering("1.", counter)
 } else if depth == 3 {
  numbering("a.", counter)
 }
}


// ===========
// = Styling =
// ===========


#let _styling(body) = {
 show ref: it => {
  let element = it.element
  if element == none {
   return it
  }

  let number = counter(element.func()).at(element.location()).last()
  let number_content = numbering(element.numbering, number)


  let func = element.func()
  if func == math.equation {
   link(element.location(), number_content)
  } else { // TODO: add figure when not propbox
   it
  }
 }

 set heading(numbering: "I.1.a.")
 show heading: it => {
  let depth = it.depth
  let count = counter(heading).get().last()

  [
   #_heading_numbering(depth, count) #it.body
  ]
 }

 // Code formatting
 show raw.where(block: true): it => {
  set align(top + left)

  block(
   stroke: 1.2pt + black,
   breakable: true,
   inset: (top: 2pt, bottom: 2pt),
   width: 100%,
   radius: 3pt,
   grid(
    columns: 2,
    column-gutter: 12pt,
    inset: 0.3em,
    stroke: (x, y) => if x == 0 {(right: gray,)},

    ..it.lines.map(
     line => (
      align(top + right, text(fill: gray, [#line.number])),
      [#line.body],
     )
    ).flatten()
   )
  )
 }

 show raw.where(block: false): it => {
  box(
   fill: aqua.lighten(60%),
   radius: 1pt,
   inset: 1pt,
   baseline: 10%,
   [*#it.text*]
  )
 }

 // Link formatting
 show link: it => if type(it.dest) == str { text(fill: aqua.darken(15%), underline(it)) } else { it }

 // Figure formatting
 show figure.where(kind: raw): set figure(supplement: [#context _stl_tr("code")])
 show figure: it => {
  set par(spacing: 5pt)
  it
  v(2em)
 }

 // Show bibliography heading
 show bibliography: set heading(numbering: "I.")

 // Vector and matrices
 set math.vec(delim: "[")
 set math.mat(delim: "[")

 // Body
 body
}


// =====================
// = Template function =
// =====================


#let template(
 body,
 author: none,
 title: none,
 short_title: none,
 subtitle: none,
) = {
 // = Define styling =
 show: _styling
 set text(lang: "fr")

 // = Cover page =
 context counter(page).update(0)
 {
  set par(justify: false, spacing: 0.75em)

  v(1fr)

  align(center, {
   [
    #text(size: 28pt, title)

    #text(size: 16pt, emph(subtitle))

    #if author != none {
     v(7pt)
     text(size: 16pt, author.join(", ", last: ", et "))
    }
   ]
   v(2em)
   line(length: 60%, stroke: black + 0.1pt)
  })

  v(3em)
  {
   show outline.entry: it => {
    let heading_depth = it.element.depth
    let heading_count = counter(heading).at(it.element.location()).last()

    link(
     it.element.location(),
     it.indented(_heading_numbering(heading_depth, heading_count), it.inner()),
    )
   }
   outline(
    title: none,
   )
  }

  v(2fr)
 }
 pagebreak()

 // = Page Configuration =
 set page(
  header: context {
   set par(justify: false)
   let current_page = counter(page).at(here()).first()

   if current_page > 0 {
    align(left, {
     if short_title != none {
      short_title
     } else {
      title
     }
    })

    v(-0.8em)
    line(length: 100%)
   }
  },
  footer: context {
   set par(justify: false)
   let current_page = counter(page).at(here()).last()

   if current_page > 0 {
    line(length: 100%)
    v(-0.5em)
    context {
     let final_page = counter(page).final().last()
     align(center)[#(current_page) / #(final_page)]
    }
   }
  }
 )

 // = Show Document =
 body
}
