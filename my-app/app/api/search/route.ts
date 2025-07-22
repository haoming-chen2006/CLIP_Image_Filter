import { NextRequest, NextResponse } from 'next/server'
import { promisify } from 'util'
import { execFile } from 'child_process'
import path from 'path'

const execFileAsync = promisify(execFile)

export async function GET(req: NextRequest) {
  const query = req.nextUrl.searchParams.get('q')
  if (!query) {
    return NextResponse.json({ error: 'Missing query' }, { status: 400 })
  }
  try {
    const scriptPath = path.join(process.cwd(), '..', 'inference.py')
    const { stdout } = await execFileAsync('python3', [scriptPath, '--query', query, '--top', '1'])
    const data = JSON.parse(stdout.trim())
    return NextResponse.json({ image: data.matches[0] })
  } catch (err) {
    console.error('Search error', err)
    return NextResponse.json({ error: 'Inference failed' }, { status: 500 })
  }
}
