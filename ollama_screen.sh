#!/usr/bin/env bash

MODEL="qwen3.8:27b"
IMG_PATH="/tmp/ollama-shot-$(date +%s).png"
B64_PATH="${IMG_PATH}.b64"
JSON_PATH="${IMG_PATH}.json"

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

# 3. Notify user that generation has started
notify-send -u low "Ollama" "Thinking: '$PROMPT'..."

# 4. Convert image to Base64 and save to a file
base64 -w 0 "$IMG_PATH" >"$B64_PATH"

# 5. Build JSON payload safely using jq
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

# 6. Request generation via API reading directly from the JSON file
RESPONSE=$(curl -s http://localhost:11434/api/generate -d @"$JSON_PATH" | jq -r '.response')

# 7. Output to clipboard
echo -n "$RESPONSE" | wl-copy

# 8. Display the result
echo "$RESPONSE" | zenity --text-info \
  --title="Ollama Response" \
  --width=800 \
  --height=600 \
  --ok-label="Close"

# 9. Cleanup
rm -f "$IMG_PATH" "$B64_PATH" "$JSON_PATH"
