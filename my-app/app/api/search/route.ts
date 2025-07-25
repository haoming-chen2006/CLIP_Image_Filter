import { NextRequest, NextResponse } from 'next/server'

export async function GET(req: NextRequest) {
  const base = process.env.BACKEND_URL || 'http://localhost:8000'
  const query = req.nextUrl.searchParams.get('q')
  if (!query) {
    return NextResponse.json({ error: 'Missing query' }, { status: 400 })
  }
  try {
    const res = await fetch(`${base}/search?q=${encodeURIComponent(query)}`)
    const data = await res.json()
    return NextResponse.json(data)
  } catch (err) {
    console.error('Search error', err)
    return NextResponse.json({ error: 'Inference failed' }, { status: 500 })
  }
}
