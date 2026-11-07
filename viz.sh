#!/usr/bin/env bash

# 1. Configurable Model (defaults to qwen3.8:27b if no argument provided)
MODEL="${1:-qwen3.8:27b}"

# 2. Dependency Checks
for cmd in grim slurp zenity jq curl wl-copy base64 mktemp; do
  if ! command -v "$cmd" &>/dev/null; then
    zenity --error --title="Missing Dependency" --text="Required command '$cmd' is not installed."
    exit 1
  fi
done

# 3. Check if Ollama is running
if ! curl -s --connect-timeout 2 http://localhost:11434 >/dev/null; then
  zenity --error --title="Ollama Not Found" --text="Ollama is not running on localhost:11434.\nPlease start the Ollama server and try again."
  exit 1
fi

# 4. Secure Temporary Directory & Cleanup Trap
TMP_DIR=$(mktemp -d -t ollama-vision-XXXXXX)
trap 'rm -rf "$TMP_DIR"' EXIT

IMG_PATH="$TMP_DIR/shot.png"
B64_PATH="$TMP_DIR/shot.b64"
JSON_PATH="$TMP_DIR/shot.json"
RESP_PATH="$TMP_DIR/shot.resp.json"
CHAT_LOG="$TMP_DIR/chat.txt"
MESSAGES_PATH="$TMP_DIR/messages.json"

# Initialize an empty messages array in a file (avoids "Argument list too long")
echo "[]" >"$MESSAGES_PATH"

# 5. Select region and capture
if ! grim -g "$(slurp)" "$IMG_PATH"; then
  # User canceled slurp or grim failed
  exit 0
fi

# 6. Initial Prompt selection
CHOICE=$(zenity --list \
  --title="Ollama Vision ($MODEL)" \
  --text="Select a predetermined prompt or choose 'Custom':" \
  --radiolist \
  --column="Select" --column="Prompt" \
  FALSE "Describe this image in detail." \
  FALSE "Extract all the text from this image." \
  FALSE "Create a comprehensive, highly descriptive prompt optimized for the FLUX3 text-to-image model. Include the main subject, artistic style, camera angle, lighting, textures, and mood. Output ONLY the prompt." \
  TRUE "Custom" \
  --width=500 --height=300)

if [ -z "$CHOICE" ]; then
  exit 0
fi

if [ "$CHOICE" = "Custom" ]; then
  PROMPT=$(zenity --entry \
    --title="Ollama Vision - Custom Prompt" \
    --text="Ask Ollama about this screenshot:" \
    --width=500)

  if [ -z "$PROMPT" ]; then
    exit 0
  fi
else
  PROMPT="$CHOICE"
fi

# 7. Convert image to Base64 (strip newlines)
base64 -w 0 "$IMG_PATH" >"$B64_PATH"

FIRST_TURN=true

while true; do
  # Log the user's prompt to the visible UI chat log
  echo -e "=== You ===\n$PROMPT\n" >>"$CHAT_LOG"

  # 8. Update the messages history file
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

  # 9. Build JSON payload directly from the history file
  jq --arg model "$MODEL" \
    '{
      "model": $model,
      "messages": .,
      "stream": false
    }' "$MESSAGES_PATH" >"$JSON_PATH"

  # 10. Run API request in the background (with a 120s timeout)
  curl -s --max-time 120 http://localhost:11434/api/chat -d @"$JSON_PATH" >"$RESP_PATH" &
  CURL_PID=$!

  # 11. Show a pulsating Zenity progress window
  while kill -0 $CURL_PID 2>/dev/null; do
    sleep 0.2
  done | zenity --progress \
    --title="Ollama Vision" \
    --text="Thinking: '${PROMPT:0:40}...' " \
    --pulsate \
    --auto-close \
    --auto-kill \
    --width=400

  wait $CURL_PID

  # 12. Extract and validate the response
  if [ -s "$RESP_PATH" ]; then
    RESPONSE=$(jq -r '.message.content // empty' <"$RESP_PATH")
    ERROR_MSG=$(jq -r '.error // empty' <"$RESP_PATH")
  else
    RESPONSE=""
    ERROR_MSG="No response or connection timed out."
  fi

  if [ -n "$ERROR_MSG" ]; then
    zenity --error --title="Ollama Error" --text="API Error: $ERROR_MSG"
    break
  fi

  if [ -z "$RESPONSE" ]; then
    zenity --error --title="Ollama Error" --text="Received empty response from the model."
    break
  fi

  # Log Ollama's response to the visible UI chat log
  echo -e "=== Ollama ===\n$RESPONSE\n\n----------------------------------------\n" >>"$CHAT_LOG"

  # 13. Add Ollama's response to the API context history file
  jq --arg resp "$RESPONSE" \
    '. + [{"role": "assistant", "content": $resp}]' "$MESSAGES_PATH" >"$MESSAGES_PATH.tmp"
  mv "$MESSAGES_PATH.tmp" "$MESSAGES_PATH"

  # 14. Output ONLY the latest response to clipboard
  echo -n "$RESPONSE" | wl-copy

  # 15. Display the full chat log
  zenity --text-info \
    --title="Ollama Vision Chat ($MODEL)" \
    --filename="$CHAT_LOG" \
    --width=800 \
    --height=600 \
    --auto-scroll \
    --ok-label="Next Question" \
    --cancel-label="End Chat"

  # If the user clicks "End Chat" (Cancel button) or closes the window, exit the loop
  if [ $? -ne 0 ]; then
    break
  fi

  # 16. Prompt for the next question
  PROMPT=$(zenity --entry \
    --title="Ollama Vision - Chat" \
    --text="Follow-up question (Leave blank or Cancel to exit):" \
    --width=500)

  if [ -z "$PROMPT" ]; then
    break
  fi

done

# Note: The EXIT trap will automatically handle cleanup of $TMP_DIR here.
