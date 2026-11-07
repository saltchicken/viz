#!/usr/bin/env bash

MODEL="qwen3.8:27b"
IMG_PATH="/tmp/ollama-shot-$(date +%s).png"
B64_PATH="${IMG_PATH}.b64"
JSON_PATH="${IMG_PATH}.json"
RESP_PATH="${IMG_PATH}.resp.json"
CHAT_LOG="${IMG_PATH}.chat.txt" # New: File to store the visible chat history

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

MESSAGES='[]'
FIRST_TURN=true

while true; do
  # Log the user's prompt to the visible UI chat log
  echo -e "=== You ===\n$PROMPT\n" >>"$CHAT_LOG"

  if [ "$FIRST_TURN" = true ]; then
    MESSAGES=$(jq -n \
      --argjson msgs "$MESSAGES" \
      --arg prompt "$PROMPT" \
      --rawfile img "$B64_PATH" \
      '$msgs + [{"role": "user", "content": $prompt, "images": [$img]}]')
    FIRST_TURN=false
  else
    MESSAGES=$(jq -n \
      --argjson msgs "$MESSAGES" \
      --arg prompt "$PROMPT" \
      '$msgs + [{"role": "user", "content": $prompt}]')
  fi

  # 4. Build JSON payload
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

  # 7. Extract the response
  RESPONSE=$(jq -r '.message.content' <"$RESP_PATH")

  # Log Ollama's response to the visible UI chat log
  echo -e "=== Ollama ===\n$RESPONSE\n\n----------------------------------------\n" >>"$CHAT_LOG"

  # 8. Add Ollama's response to the API context history
  MESSAGES=$(jq -n \
    --argjson msgs "$MESSAGES" \
    --arg resp "$RESPONSE" \
    '$msgs + [{"role": "assistant", "content": $resp}]')

  # 9. Output ONLY the latest response to clipboard
  echo -n "$RESPONSE" | wl-copy

  # 10. Display the full chat log
  cat "$CHAT_LOG" | zenity --text-info \
    --title="Ollama Chat History" \
    --width=800 \
    --height=600 \
    --auto-scroll \
    --ok-label="Next Question" \
    --cancel-label="End Chat"

  # If the user clicks "End Chat" (Cancel button), exit the loop
  if [ $? -ne 0 ]; then
    break
  fi

  # Prompt for the next question
  PROMPT=$(zenity --entry \
    --title="Ollama Vision - Chat" \
    --text="Follow-up question (Leave blank or Cancel to exit):" \
    --width=500)

  if [ -z "$PROMPT" ]; then
    break
  fi

done

# 11. Cleanup all temporary files (including the chat log)
rm -f "$IMG_PATH" "$B64_PATH" "$JSON_PATH" "$RESP_PATH" "$CHAT_LOG"
