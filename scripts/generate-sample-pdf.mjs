import { readFile, mkdir, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { PDFDocument, StandardFonts, rgb } from 'pdf-lib'

const scriptDirectory = dirname(fileURLToPath(import.meta.url))
const projectRoot = resolve(scriptDirectory, '..')
const passages = JSON.parse(await readFile(resolve(projectRoot, 'src/data/sampleGeometry.json'), 'utf8'))

const pdf = await PDFDocument.create()
pdf.setTitle('Northstar Review Memorandum — Fictional Demonstration')
pdf.setAuthor('Office of Archival Studies (Fictional)')
pdf.setSubject('Fictional demonstration document for the Redacted Document LLM interface')
pdf.setKeywords(['fictional', 'demonstration', 'redacted'])

const serif = await pdf.embedFont(StandardFonts.TimesRoman)
const serifBold = await pdf.embedFont(StandardFonts.TimesRomanBold)
const sans = await pdf.embedFont(StandardFonts.Helvetica)
const sansBold = await pdf.embedFont(StandardFonts.HelveticaBold)
const pageWidth = 612
const pageHeight = 792

const pageCopy = [
  {
    title: 'OBSERVATION SUMMARY',
    intro: 'On 18 April, regional monitoring stations recorded an intermittent signal above the Northstar research corridor. The event was reviewed as part of a routine coordination exercise.',
    lines: [
      'The initial report compared the observation with scheduled activity in the area.',
      'After reviewing the available timing and weather information, analysts concluded:',
      '',
      'No operational response was initiated. The record remained open for administrative review.',
      'To support that review, the coordinating office issued the following handling guidance:',
    ],
  },
  {
    title: 'ANALYTICAL REVIEW',
    intro: 'The interoffice review considered sensor reliability, weather conditions, civilian schedules, and the consistency of reports received from three monitoring locations.',
    lines: [
      'The observations were internally consistent but limited in duration and detail.',
      'The review reached the following preliminary assessment:',
      '',
      'No additional collection was authorized. The working group instead focused on process.',
      'For future events, the group recommended the following coordination measure:',
    ],
  },
  {
    title: 'DISPOSITION AND NOTICE',
    intro: 'The exercise was closed after all participating offices confirmed that their records had been reconciled and that no unresolved safety or security issue remained.',
    lines: [
      'This document was created for a user-interface demonstration. Names, dates, offices,',
      'observations, and conclusions are invented and do not refer to any real-world activity.',
      '',
      'The final disposition statement reads:',
    ],
  },
]

const wrapText = (text, font, size, maxWidth) => {
  const words = text.split(' ')
  const lines = []
  let line = ''
  for (const word of words) {
    const candidate = line ? `${line} ${word}` : word
    if (font.widthOfTextAtSize(candidate, size) > maxWidth && line) {
      lines.push(line)
      line = word
    } else {
      line = candidate
    }
  }
  if (line) lines.push(line)
  return lines
}

for (let index = 0; index < pageCopy.length; index += 1) {
  const pageNumber = index + 1
  const page = pdf.addPage([pageWidth, pageHeight])
  const copy = pageCopy[index]

  page.drawText('OFFICE OF ARCHIVAL STUDIES', { x: 72, y: 733, font: sansBold, size: 9, color: rgb(0.16, 0.2, 0.24) })
  page.drawText('FICTIONAL DEMONSTRATION DOCUMENT', { x: 355, y: 733, font: sansBold, size: 8, color: rgb(0.33, 0.47, 0.58) })
  page.drawLine({ start: { x: 72, y: 720 }, end: { x: 540, y: 720 }, thickness: 0.7, color: rgb(0.72, 0.75, 0.77) })
  page.drawText('NORTHSTAR REVIEW MEMORANDUM', { x: 72, y: 680, font: serifBold, size: 18, color: rgb(0.12, 0.14, 0.16) })
  page.drawText(copy.title, { x: 72, y: 655, font: sansBold, size: 9.5, color: rgb(0.33, 0.37, 0.4) })
  page.drawText(`PAGE ${pageNumber} OF ${pageCopy.length}`, { x: 475, y: 655, font: sans, size: 8, color: rgb(0.38, 0.42, 0.45) })

  let y = 612
  for (const line of wrapText(copy.intro, serif, 11, 455)) {
    page.drawText(line, { x: 102, y, font: serif, size: 11, color: rgb(0.16, 0.17, 0.18) })
    y -= 18
  }
  y -= 22
  for (const paragraph of copy.lines) {
    if (!paragraph) {
      y -= 76
      continue
    }
    for (const line of wrapText(paragraph, serif, 11, 455)) {
      page.drawText(line, { x: 102, y, font: serif, size: 11, color: rgb(0.16, 0.17, 0.18) })
      y -= 18
    }
    y -= 12
  }

  for (const passage of passages.filter((item) => item.page === pageNumber)) {
    for (const rect of passage.rects) {
      page.drawRectangle({
        x: rect.x * pageWidth,
        y: (1 - rect.y - rect.height) * pageHeight,
        width: rect.width * pageWidth,
        height: rect.height * pageHeight,
        color: rgb(0.035, 0.04, 0.045),
      })
    }
  }

  page.drawLine({ start: { x: 72, y: 58 }, end: { x: 540, y: 58 }, thickness: 0.5, color: rgb(0.76, 0.78, 0.8) })
  page.drawText('UNCLASSIFIED // FICTIONAL TRAINING MATERIAL', { x: 72, y: 40, font: sansBold, size: 7.5, color: rgb(0.38, 0.42, 0.45) })
  page.drawText('RD-LLM-DEMO-001', { x: 460, y: 40, font: sans, size: 7.5, color: rgb(0.38, 0.42, 0.45) })
}

const outputPath = resolve(projectRoot, 'public/fictional-briefing.pdf')
await mkdir(dirname(outputPath), { recursive: true })
await writeFile(outputPath, await pdf.save())
console.log(`Generated ${outputPath}`)
