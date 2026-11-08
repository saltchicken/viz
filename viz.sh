#!/usr/bin/env bash

# 1. Configurable Model (defaults to qwen3.8:27b if no argument provided)
MODEL="${1:-qwen3.8:27b}"

# 2. Dependency Checks
for cmd in grim slurp yad jq curl wl-copy base64 mktemp; do
  if ! command -v "$cmd" &>/dev/null; then
    yad --error --center --title="Missing Dependency" --text="Required command '$cmd' is not installed."
    exit 1
  fi
done

# ImageMagick check for scaling
if ! command -v magick &>/dev/null && ! command -v convert &>/dev/null; then
  yad --error --center --title="Missing Dependency" --text="ImageMagick is required for UI scaling.\nPlease install 'imagemagick'."
  exit 1
fi

# 3. Check if Ollama is running
if ! curl -s --connect-timeout 2 http://localhost:11434 >/dev/null; then
  yad --error --center --title="Ollama Not Found" --text="Ollama is not running on localhost:11434.\nPlease start the Ollama server and try again."
  exit 1
fi

# 4. Secure Temporary Directory & Cleanup Trap
TMP_DIR=$(mktemp -d -t ollama-vision-XXXXXX)
trap 'rm -rf "$TMP_DIR"' EXIT

IMG_PATH="$TMP_DIR/shot.png"
PREVIEW_PATH="$TMP_DIR/preview.png"
B64_PATH="$TMP_DIR/shot.b64"
JSON_PATH="$TMP_DIR/shot.json"
RESP_PATH="$TMP_DIR/shot.resp.json"
CHAT_LOG="$TMP_DIR/chat.txt"
MESSAGES_PATH="$TMP_DIR/messages.json"

echo "[]" >"$MESSAGES_PATH"

# 5. Select region and capture
if ! grim -g "$(slurp)" "$IMG_PATH"; then
  exit 0
fi

# 6. Scale image for UI Preview (Keep aspect ratio, max 500x500)
if command -v magick &>/dev/null; then
  magick "$IMG_PATH" -resize "500x500>" "$PREVIEW_PATH"
else
  convert "$IMG_PATH" -resize "500x500>" "$PREVIEW_PATH"
fi

# 7. Initial Prompt selection (Polished UI)
FORM_DATA=$(yad --title="Ollama Vision ($MODEL)" \
  --center \
  --borders=10 \
  --image="$PREVIEW_PATH" \
  --text="<big><b>Analyze Captured Image</b></big>\nSelect a preset or enter a custom prompt." \
  --form \
  --separator="|||" \
  --field="Select Preset:CB" "Custom!Describe this image in detail.!Extract all the text from this image.!Create a comprehensive, highly descriptive prompt optimized for the FLUX3 text-to-image model. Include the main subject, artistic style, camera angle, lighting, textures, and mood. Output ONLY the prompt." \
  --field="Custom Prompt (Multiline):TXT" "" \
  --width=650 --height=600 \
  --button="Cancel!gtk-cancel:1" --button="Ask Ollama!gtk-ok:0")

if [ $? -ne 0 ] || [ -z "$FORM_DATA" ]; then
  exit 0
fi

# Extract fields safely using bash parameter expansion based on our custom separator
PRESET="${FORM_DATA%%|||*}"
REMAINDER="${FORM_DATA#*|||}"
CUSTOM="${REMAINDER%%|||*}"

if [ "$PRESET" = "Custom" ]; then
  if [ -z "$CUSTOM" ]; then
    exit 0
  fi
  PROMPT="$CUSTOM"
else
  PROMPT="$PRESET"
fi

# 8. Convert the ORIGINAL HIGH-RES image to Base64 (not the UI preview)
base64 -w 0 "$IMG_PATH" >"$B64_PATH"

FIRST_TURN=true

while true; do
  echo -e "=== You ===\n$PROMPT\n" >>"$CHAT_LOG"

  # 9. Update messages history
  if [ "$FIRST_TURN" = true ]; then
    jq --arg prompt "$PROMPT" \
      --rawfile img "$B64_PATH" \
      '. + [{"role": "user", "content": $prompt, "images": [$img]}]' "$MESSAGES_PATH" >"$MESSAGES_PATH.tmp"
    mv "$MESSAGES_PATH.tmp" "$MESSAGES_PATH"
    FIRST_TURN=false
  else
    jq --arg prompt "$PROMPT" \
      '. + [{"role": "user", "content": $prompt}]' "$MESSAGES_PATH" >"$MESSAGES_PATH.tmp"
    mv "$MESSAGES_PATH.tmp" "$MESSAGES_PATH"
  fi

  # 10. Build JSON payload
  jq --arg model "$MODEL" \
    '{
      "model": $model,
      "messages": .,
      "stream": false
    }' "$MESSAGES_PATH" >"$JSON_PATH"

  # 11. Run API request in background
  curl -s --max-time 120 http://localhost:11434/api/chat -d @"$JSON_PATH" >"$RESP_PATH" &
  CURL_PID=$!

  # 12. Pulsating yad progress window (Polished UI)
  while kill -0 $CURL_PID 2>/dev/null; do
    sleep 0.2
  done | yad --progress \
    --center \
    --title="Ollama Vision" \
    --text="\n<b>Thinking...</b>\n<i>'${PROMPT:0:50}...'</i>\n" \
    --pulsate \
    --auto-close \
    --auto-kill \
    --width=450 \
    --borders=15 \
    --button="Cancel!gtk-cancel:1"

  wait $CURL_PID

  # 13. Extract and validate response
  if [ -s "$RESP_PATH" ]; then
    RESPONSE=$(jq -r '.message.content // empty' <"$RESP_PATH")
    ERROR_MSG=$(jq -r '.error // empty' <"$RESP_PATH")
  else
    RESPONSE=""
    ERROR_MSG="No response or connection timed out."
  fi

  if [ -n "$ERROR_MSG" ]; then
    yad --error --center --title="Ollama Error" --text="API Error: $ERROR_MSG"
    break
  fi

  if [ -z "$RESPONSE" ]; then
    yad --error --center --title="Ollama Error" --text="Received empty response from the model."
    break
  fi

  echo -e "=== Ollama ===\n$RESPONSE\n\n----------------------------------------\n" >>"$CHAT_LOG"

  # 14. Add response to history
  jq --arg resp "$RESPONSE" \
    '. + [{"role": "assistant", "content": $resp}]' "$MESSAGES_PATH" >"$MESSAGES_PATH.tmp"
  mv "$MESSAGES_PATH.tmp" "$MESSAGES_PATH"

  # 15. Clipboard
  echo -n "$RESPONSE" | wl-copy

  # 16. Display full chat log (Polished UI)
  yad --text-info \
    --center \
    --title="Ollama Vision Chat ($MODEL)" \
    --filename="$CHAT_LOG" \
    --width=800 \
    --height=600 \
    --wrap \
    --margins=15 \
    --button="End Chat!gtk-quit:1" \
    --button="Next Question!gtk-go-forward:0"

  if [ $? -ne 0 ]; then
    break
  fi

  # 17. Multiline prompt for the next question (Polished UI)
  NEXT_DATA=$(yad --title="Ollama Vision" \
    --center \
    --borders=10 \
    --text="<big><b>Follow-up Question</b></big>\nAsk something else about the image:" \
    --form \
    --separator="|||" \
    --field=":TXT" "" \
    --width=600 --height=350 \
    --button="End Chat!gtk-quit:1" --button="Ask!gtk-ok:0")

  if [ $? -ne 0 ] || [ -z "$NEXT_DATA" ]; then
    break
  fi

  # Extract follow-up prompt
  PROMPT="${NEXT_DATA%%|||*}"

  if [ -z "$PROMPT" ]; then
    break
  fi

done
