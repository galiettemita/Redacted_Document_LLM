import { randomUUID } from 'node:crypto'
import { copyFile, mkdir, readdir, rename, unlink } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

const source = fileURLToPath(new URL('../node_modules/pdfjs-dist/wasm/', import.meta.url))
const destination = fileURLToPath(new URL('../public/pdfjs/wasm/', import.meta.url))

async function copyDirectoryAtomically(sourceDirectory, destinationDirectory) {
  await mkdir(destinationDirectory, { recursive: true })

  for (const entry of await readdir(sourceDirectory, { withFileTypes: true })) {
    const sourcePath = join(sourceDirectory, entry.name)
    const destinationPath = join(destinationDirectory, entry.name)

    if (entry.isDirectory()) {
      await copyDirectoryAtomically(sourcePath, destinationPath)
      continue
    }
    if (!entry.isFile()) {
      throw new Error(`Unsupported PDF.js asset type: ${sourcePath}`)
    }

    const temporaryPath = `${destinationPath}.${process.pid}.${randomUUID()}.tmp`
    try {
      await copyFile(sourcePath, temporaryPath)
      await rename(temporaryPath, destinationPath)
    } finally {
      await unlink(temporaryPath).catch((error) => {
        if (error?.code !== 'ENOENT') throw error
      })
    }
  }
}

await copyDirectoryAtomically(source, destination)
