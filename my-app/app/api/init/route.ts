import { NextRequest, NextResponse } from 'next/server'

export async function GET(req: NextRequest) {
  try {
    const res = await fetch('http://localhost:8000/init')
    const data = await res.json()
    return NextResponse.json(data)
  } catch (err) {
    console.error('Init error', err)
    return NextResponse.json({ error: 'Init failed' }, { status: 500 })
  }
}
