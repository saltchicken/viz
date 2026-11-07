#!/usr/bin/env bash

MODEL="qwen3.8:27b"
IMG_PATH="/tmp/ollama-shot-$(date +%s).png"
B64_PATH="${IMG_PATH}.b64"
JSON_PATH="${IMG_PATH}.json"
RESP_PATH="${IMG_PATH}.resp.json"

# 1. Select region and capture
grim -g "$(slurp)" "$IMG_PATH" || exit 1

# 2. Prompt for user input using Zenity
PROMPT=$(zenity --entry \
  --title="Ollama Vision" \
  --text="Ask Ollama about this screenshot:" \
  --width=500)

if [ -z "$PROMPT" ]; then
  rm -f "$IMG_PATH"
  exit 0
fi

# 3. Convert image to Base64 and save to a file
base64 -w 0 "$IMG_PATH" >"$B64_PATH"

# 4. Build JSON payload safely using jq
jq -n \
  --arg model "$MODEL" \
  --arg prompt "$PROMPT" \
  --rawfile img "$B64_PATH" \
  '{
    "model": $model,
    "prompt": $prompt,
    "images": [$img],
    "stream": false
  }' >"$JSON_PATH"

# 5. Run API request in the background and save to a temp file
curl -s http://localhost:11434/api/generate -d @"$JSON_PATH" >"$RESP_PATH" &
CURL_PID=$!

# 6. Show a pulsating Zenity progress window while waiting for Ollama
while kill -0 $CURL_PID 2>/dev/null; do
  sleep 0.2
done | zenity --progress \
  --title="Ollama Vision" \
  --text="Thinking: '$PROMPT'..." \
  --pulsate \
  --auto-close \
  --auto-kill \
  --width=400

# Ensure the background process is fully complete
wait $CURL_PID

# 7. Extract the response
RESPONSE=$(jq -r '.response' <"$RESP_PATH")

# 8. Output to clipboard
echo -n "$RESPONSE" | wl-copy

# 9. Display the final result
echo "$RESPONSE" | zenity --text-info \
  --title="Ollama Response" \
  --width=800 \
  --height=600 \
  --ok-label="Close"

# 10. Cleanup
rm -f "$IMG_PATH" "$B64_PATH" "$JSON_PATH" "$RESP_PATH"
