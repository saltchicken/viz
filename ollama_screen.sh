#!/usr/bin/env bash

MODEL="qwen3.8:27b"
IMG_PATH="/tmp/ollama-shot-$(date +%s).png"
B64_PATH="${IMG_PATH}.b64"
JSON_PATH="${IMG_PATH}.json"
RESP_PATH="${IMG_PATH}.resp.json"

# 1. Select region and capture
grim -g "$(slurp)" "$IMG_PATH" || exit 1

# 2. Initial Prompt selection
CHOICE=$(zenity --list \
  --title="Ollama Vision" \
  --text="Select a predetermined prompt or choose 'Custom':" \
  --radiolist \
  --column="Select" --column="Prompt" \
  FALSE "Describe this image in detail." \
  FALSE "Extract all the text from this image." \
  FALSE "Create a comprehensive, highly descriptive prompt optimized for the FLUX3 text-to-image model. Include the main subject, artistic style, camera angle, lighting, textures, and mood. Output ONLY the prompt." \
  TRUE "Custom" \
  --width=500 --height=300)

if [ -z "$CHOICE" ]; then
  rm -f "$IMG_PATH"
  exit 0
fi

if [ "$CHOICE" = "Custom" ]; then
  PROMPT=$(zenity --entry \
    --title="Ollama Vision - Custom Prompt" \
    --text="Ask Ollama about this screenshot:" \
    --width=500)

  if [ -z "$PROMPT" ]; then
    rm -f "$IMG_PATH"
    exit 0
  fi
else
  PROMPT="$CHOICE"
fi

# 3. Convert image to Base64
base64 -w 0 "$IMG_PATH" >"$B64_PATH"

# --- NEW CONVERSATIONAL LOGIC START ---

# Initialize an empty JSON array to hold the conversation history
MESSAGES='[]'
FIRST_TURN=true

while true; do
  if [ "$FIRST_TURN" = true ]; then
    # First turn: attach the image to the user's message
    MESSAGES=$(jq -n \
      --argjson msgs "$MESSAGES" \
      --arg prompt "$PROMPT" \
      --rawfile img "$B64_PATH" \
      '$msgs + [{"role": "user", "content": $prompt, "images": [$img]}]')
    FIRST_TURN=false
  else
    # Subsequent turns: Ask for follow-up question
    PROMPT=$(zenity --entry \
      --title="Ollama Vision - Chat" \
      --text="Follow-up question (Leave blank or Cancel to exit):" \
      --width=500)

    # Exit the loop if the user cancels or submits an empty string
    if [ -z "$PROMPT" ]; then
      break
    fi

    # Add the new text-only message to the history
    MESSAGES=$(jq -n \
      --argjson msgs "$MESSAGES" \
      --arg prompt "$PROMPT" \
      '$msgs + [{"role": "user", "content": $prompt}]')
  fi

  # 4. Build JSON payload safely using jq and the /api/chat format
  jq -n \
    --arg model "$MODEL" \
    --argjson messages "$MESSAGES" \
    '{
      "model": $model,
      "messages": $messages,
      "stream": false
    }' >"$JSON_PATH"

  # 5. Run API request in the background
  curl -s http://localhost:11434/api/chat -d @"$JSON_PATH" >"$RESP_PATH" &
  CURL_PID=$!

  # 6. Show a pulsating Zenity progress window
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

  # 7. Extract the response (Chat API nests this under .message.content)
  RESPONSE=$(jq -r '.message.content' <"$RESP_PATH")

  # 8. Add Ollama's response to the conversation history so it remembers context
  MESSAGES=$(jq -n \
    --argjson msgs "$MESSAGES" \
    --arg resp "$RESPONSE" \
    '$msgs + [{"role": "assistant", "content": $resp}]')

  # 9. Output to clipboard
  echo -n "$RESPONSE" | wl-copy

  # 10. Display the final result with a button to continue
  echo "$RESPONSE" | zenity --text-info \
    --title="Ollama Response" \
    --width=800 \
    --height=600 \
    --ok-label="Next Question" \
    --cancel-label="End Chat"

  # If the user clicks "End Chat" (Cancel button), exit the loop
  if [ $? -ne 0 ]; then
    break
  fi

done

# 11. Cleanup all temporary files upon exiting
rm -f "$IMG_PATH" "$B64_PATH" "$JSON_PATH" "$RESP_PATH"
