#!/usr/bin/env bash

MODEL="qwen3.8:27b"
IMG_PATH="/tmp/ollama-shot-$(date +%s).png"

# 1. Select region and capture
grim -g "$(slurp)" "$IMG_PATH" || exit 1

# 2. Prompt for user input
PROMPT=$(echo "" | wofi --show dmenu --prompt "Ask Ollama:" --lines 1)

if [ -z "$PROMPT" ]; then
  rm "$IMG_PATH"
  exit 0
fi

# 3. Notify user that generation has started (ephemeral)
notify-send -u low "Ollama" "Thinking: '$PROMPT'..."

# 4. Convert image to Base64
IMG_B64=$(base64 -w 0 "$IMG_PATH")

# 5. Request generation via local Ollama API
RESPONSE=$(curl -s http://localhost:11434/api/generate -d '{
  "model": "'"$MODEL"'",
  "prompt": "'"$PROMPT"'",
  "images": ["'"$IMG_B64"'"],
  "stream": false
}' | jq -r '.response')

# 6. Output to clipboard for immediate pasting (optional but handy)
echo -n "$RESPONSE" | wl-copy

# 7. Display the result in a persistent, selectable window
echo "$RESPONSE" | zenity --text-info \
  --title="Ollama Response" \
  --width=800 \
  --height=600 \
  --ok-label="Close"

# 8. Cleanup
rm "$IMG_PATH"
