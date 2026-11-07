#!/bin/bash

# Define the core instructions for FLUX3
INSTRUCTIONS="Create a comprehensive, highly descriptive prompt optimized for the FLUX3 text-to-image model. Include the main subject, artistic style, camera angle, lighting, textures, and mood. Output ONLY the prompt."

# Get available clipboard MIME types
CLIP_TYPES=$(wl-paste --list-types)

if echo "$CLIP_TYPES" | grep -q "^image/"; then
  # --- HANDLE IMAGE ---
  TEMP_IMG="/tmp/clipboard_reference.png"
  wl-paste --type image/png >"$TEMP_IMG"

  if [ ! -s "$TEMP_IMG" ]; then
    notify-send "Ollama Script" "Failed to extract image from clipboard."
    rm -f "$TEMP_IMG"
    exit 1
  fi

  # Note: Ensure the model used here supports vision (e.g., qwen2.5-vl or llava)
  ollama run qwen3.8:27b "Analyze this image in deep detail. $INSTRUCTIONS $TEMP_IMG"

  rm -f "$TEMP_IMG"

elif echo "$CLIP_TYPES" | grep -q "^text/"; then
  # --- HANDLE TEXT ---
  CLIP_TEXT=$(wl-paste --type text/plain)

  if [ -z "$CLIP_TEXT" ]; then
    notify-send "Ollama Script" "Clipboard text was empty."
    exit 1
  fi

  # Pass the text as the seed concept to your standard text model
  ollama run qwen3.8:27b "Based on this base concept: '${CLIP_TEXT}', $INSTRUCTIONS"

else
  # --- UNKNOWN TYPE ---
  notify-send "Ollama Script" "Clipboard contains neither text nor an image."
  exit 1
fi
