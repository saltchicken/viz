#!/usr/bin/env bash

# 1. Configurable Model
MODEL="${1:-qwen3.8:27b}"

# 2. Dependency Checks
for cmd in grim slurp yad jq curl wl-copy base64 mktemp; do
  if ! command -v "$cmd" &>/dev/null; then
    yad --center --title="Missing Dependency" --text="Required command '$cmd' is not installed." --button="OK:0"
    exit 1
  fi
done

if ! command -v magick &>/dev/null && ! command -v convert &>/dev/null; then
  yad --center --title="Missing Dependency" --text="ImageMagick is required for UI scaling.\nPlease install 'imagemagick'." --button="OK:0"
  exit 1
fi

# 3. Check if Ollama is running
if ! curl -s --connect-timeout 2 http://localhost:11434 >/dev/null; then
  yad --center --title="Ollama Not Found" --text="Ollama is not running on localhost:11434.\nPlease start the Ollama server and try again." --button="OK:0"
  exit 1
fi

# 4. Secure Temporary Directory & Setup
TMP_DIR=$(mktemp -d -t ollama-vision-XXXXXX)
trap 'rm -rf "$TMP_DIR"' EXIT

IMG_DIR="$TMP_DIR/images"
mkdir -p "$IMG_DIR"

PREVIEW_PATH="$TMP_DIR/preview.png"
JSON_PATH="$TMP_DIR/shot.json"
RESP_PATH="$TMP_DIR/shot.resp.json"
CHAT_LOG="$TMP_DIR/chat.txt"
MESSAGES_PATH="$TMP_DIR/messages.json"
IMAGES_JSON_FILE="$TMP_DIR/images_array.json"

echo "[]" >"$MESSAGES_PATH"

# ImageMagick wrappers to support both v6 and v7
im_convert() {
  if command -v magick &>/dev/null; then magick "$@"; else convert "$@"; fi
}
im_montage() {
  if command -v magick &>/dev/null; then magick montage "$@"; else montage "$@"; fi
}

# 5. Function to generate the UI preview
update_preview() {
  local count=$(ls -1q "$IMG_DIR"/*.png 2>/dev/null | wc -l)
  if [ "$count" -eq 0 ]; then
    # Generate a placeholder if no images are captured yet
    im_convert -size 600x150 xc:"#222222" -gravity center -pointsize 18 -fill "#EEEEEE" \
      -annotate +0+0 "No Images Captured\nClick 'Capture Image' or 'Add File' to attach" "$PREVIEW_PATH"
  else
    # Stitch all captured images into a tidy gallery preview
    im_montage -background transparent -geometry "150x150>+5+5" "$IMG_DIR"/*.png "$PREVIEW_PATH"
  fi
}

# 6. Main Preparation Loop
PRESETS="^Custom!Describe this image in detail.!Extract all the text from this image.!Create a comprehensive, highly descriptive prompt optimized for the FLUX3 text-to-image model. Include the main subject, artistic style, camera angle, lighting, textures, and mood. Output ONLY the prompt."
CURRENT_CUSTOM=""

while true; do
  update_preview
  IMG_COUNT=$(ls -1q "$IMG_DIR"/*.png 2>/dev/null | wc -l)

  FORM_DATA=$(yad --title="Ollama Vision ($MODEL)" \
    --center \
    --borders=10 \
    --image="$PREVIEW_PATH" \
    --text="<big><b>Prepare Vision Prompt</b></big>\nImages Attached: <b>$IMG_COUNT</b>" \
    --form \
    --separator="|||" \
    --field="Select Preset:CB" "$PRESETS" \
    --field="Custom Prompt (Multiline):TXT" "$CURRENT_CUSTOM" \
    --width=700 --height=600 \
    --button="Add File:4" \
    --button="Capture Image:2" \
    --button="Clear Images:3" \
    --button="Cancel:1" \
    --button="Ask Ollama:0")

  RET=$?

  # Save the typed custom prompt so it isn't wiped when taking a screenshot/adding file
  if [ -n "$FORM_DATA" ]; then
    SELECTED_PRESET="${FORM_DATA%%|||*}"
    REMAINDER="${FORM_DATA#*|||}"
    CURRENT_CUSTOM="${REMAINDER%%|||*}"
  fi

  if [ $RET -eq 1 ] || [ $RET -eq 252 ]; then
    # Cancel / Window closed
    exit 0
  elif [ $RET -eq 4 ]; then
    # Add Image via File Explorer (Fixed module name to --file)
    SELECTED_FILES=$(yad --file --multiple --separator="|" --title="Select Image(s)")

    if [ -n "$SELECTED_FILES" ]; then
      # Read separated files into an array and convert them to PNG in the temp dir
      IFS='|' read -ra FILES <<<"$SELECTED_FILES"
      for f in "${FILES[@]}"; do
        if [ -f "$f" ]; then
          IMG_FILE="$IMG_DIR/file_$(date +%s%N)_$RANDOM.png"
          im_convert "$f" "$IMG_FILE"
        fi
      done
    fi
  elif [ $RET -eq 2 ]; then
    # Capture Image (yad temporarily closes, screen dims, user selects, loop restarts)
    IMG_FILE="$IMG_DIR/shot_$(date +%s%N).png"
    grim -g "$(slurp)" "$IMG_FILE"
  elif [ $RET -eq 3 ]; then
    # Clear Images
    rm -f "$IMG_DIR"/*.png
  elif [ $RET -eq 0 ]; then
    # Ask Ollama (Break the loop and proceed)
    break
  fi
done

# 7. Finalize Prompt
if [ "$SELECTED_PRESET" = "Custom" ] || [ "$SELECTED_PRESET" = "^Custom" ]; then
  PROMPT="$CURRENT_CUSTOM"
else
  PROMPT="${SELECTED_PRESET#^}" # Remove the default indicator caret if preset used
fi

if [ -z "$PROMPT" ]; then
  yad --center --title="Error" --text="Prompt cannot be empty." --button="OK:0"
  exit 1
fi

# 8. Convert ALL captured images to Base64 and safely build JSON Array via files to bypass ARG_MAX
echo "[]" >"$IMAGES_JSON_FILE"
IMG_FILES=("$IMG_DIR"/*.png)

# Check if at least one file actually exists
if [ -e "${IMG_FILES[0]}" ]; then
  for f in "${IMG_FILES[@]}"; do
    B64_FILE="${f%.png}.b64"
    base64 -w 0 "$f" >"$B64_FILE"

    # Read the raw Base64 string directly from the file into jq to avoid command-line length limits
    jq --rawfile img "$B64_FILE" '. + [$img]' "$IMAGES_JSON_FILE" >"${IMAGES_JSON_FILE}.tmp"
    mv "${IMAGES_JSON_FILE}.tmp" "$IMAGES_JSON_FILE"
  done
fi

FIRST_TURN=true

# 9. Main Chat Loop
while true; do
  echo -e "=== You ===\n$PROMPT\n" >>"$CHAT_LOG"

  # Append to message history
  if [ "$FIRST_TURN" = true ]; then
    if [ "$(cat "$IMAGES_JSON_FILE")" != "[]" ]; then
      # --slurpfile reads the file into an array. Since the file is already an array, $imgs is a nested array.
      # We use $imgs[0] to get our flat array of base64 strings.
      jq --arg prompt "$PROMPT" \
        --slurpfile imgs "$IMAGES_JSON_FILE" \
        '. + [{"role": "user", "content": $prompt, "images": $imgs[0]}]' "$MESSAGES_PATH" >"$MESSAGES_PATH.tmp"
    else
      jq --arg prompt "$PROMPT" \
        '. + [{"role": "user", "content": $prompt}]' "$MESSAGES_PATH" >"$MESSAGES_PATH.tmp"
    fi
    mv "$MESSAGES_PATH.tmp" "$MESSAGES_PATH"
    FIRST_TURN=false
  else
    jq --arg prompt "$PROMPT" \
      '. + [{"role": "user", "content": $prompt}]' "$MESSAGES_PATH" >"$MESSAGES_PATH.tmp"
    mv "$MESSAGES_PATH.tmp" "$MESSAGES_PATH"
  fi

  # Build payload
  jq --arg model "$MODEL" \
    '{
      "model": $model,
      "messages": .,
      "stream": false
    }' "$MESSAGES_PATH" >"$JSON_PATH"

  # API Request
  curl -s --max-time 180 http://localhost:11434/api/chat -d @"$JSON_PATH" >"$RESP_PATH" &
  CURL_PID=$!

  # Static Processing Indicator (No Progress Bar, No Icon)
  yad --title="Ollama Vision" \
    --center --borders=20 \
    --text="<big><b>Thinking...</b></big>\n\nPlease wait while the model generates a response.\n\n<i>'${PROMPT:0:50}...'</i>" \
    --width=450 \
    --button="Cancel:1" &
  YAD_PID=$!

  # Wait loop: continues as long as BOTH processes are still running
  while kill -0 $CURL_PID 2>/dev/null && kill -0 $YAD_PID 2>/dev/null; do
    sleep 0.2
  done

  # Cleanup based on which process ended first
  if ! kill -0 $YAD_PID 2>/dev/null; then
    # User clicked Cancel (dialog closed before curl finished)
    kill $CURL_PID 2>/dev/null
  else
    # Curl finished successfully, so we close the dialog
    kill $YAD_PID 2>/dev/null
  fi

  wait $CURL_PID 2>/dev/null

  # Validate
  if [ -s "$RESP_PATH" ]; then
    RESPONSE=$(jq -r '.message.content // empty' <"$RESP_PATH")
    ERROR_MSG=$(jq -r '.error // empty' <"$RESP_PATH")
  else
    RESPONSE=""
    ERROR_MSG="No response or connection timed out."
  fi

  if [ -n "$ERROR_MSG" ]; then
    yad --center --title="Ollama Error" --text="API Error: $ERROR_MSG" --button="OK:0"
    break
  fi

  if [ -z "$RESPONSE" ]; then
    yad --center --title="Ollama Error" --text="Received empty response from the model." --button="OK:0"
    break
  fi

  echo -e "=== Ollama ===\n$RESPONSE\n\n----------------------------------------\n" >>"$CHAT_LOG"

  # Add assistant response to history
  jq --arg resp "$RESPONSE" \
    '. + [{"role": "assistant", "content": $resp}]' "$MESSAGES_PATH" >"$MESSAGES_PATH.tmp"
  mv "$MESSAGES_PATH.tmp" "$MESSAGES_PATH"

  # Clipboard
  echo -n "$RESPONSE" | wl-copy

  # Display Chat Log
  yad --text-info \
    --center --margins=15 \
    --title="Ollama Vision Chat ($MODEL)" \
    --filename="$CHAT_LOG" \
    --width=800 --height=600 --wrap \
    --button="End Chat:1" \
    --button="Next Question:0"

  if [ $? -ne 0 ]; then
    break
  fi

  # Follow-up Prompt
  NEXT_DATA=$(yad --title="Ollama Vision" \
    --center --borders=10 \
    --text="<big><b>Follow-up Question</b></big>\nAsk something else about the image(s):" \
    --form \
    --separator="|||" \
    --field=":TXT" "" \
    --width=600 --height=350 \
    --button="End Chat:1" --button="Ask:0")

  if [ $? -ne 0 ] || [ -z "$NEXT_DATA" ]; then
    break
  fi

  PROMPT="${NEXT_DATA%%|||*}"
  if [ -z "$PROMPT" ]; then
    break
  fi

done
