#!/bin/bash
# Start backend and frontend together
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/my-app"

pushd "$BACKEND_DIR" >/dev/null
node index.js &
BACKEND_PID=$!
popd >/dev/null

trap 'kill $BACKEND_PID' EXIT

pushd "$FRONTEND_DIR" >/dev/null
npm run dev
popd >/dev/null
