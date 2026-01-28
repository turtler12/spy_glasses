#!/usr/bin/env python3
"""
Run this script on the Raspberry Pi to stream video via UDP
"""
import subprocess
import sys

# Configure these settings
WIDTH = 640
HEIGHT = 480
FRAMERATE = 30
BITRATE = 1200000
DESTINATION_IP = "172.20.40.62"  # Change to your client machine's IP
DESTINATION_PORT = 1234

# Command to run the video server on Raspberry Pi
video_server_command = f'rpicam-vid -t 0 --width {WIDTH} --height {HEIGHT} --framerate {FRAMERATE} --codec h264 --profile baseline --bitrate {BITRATE} --inline --nopreview --libav-format h264 -o - | ffmpeg -f h264 -use_wallclock_as_timestamps 1 -fflags +genpts -i - -an -c:v copy -muxdelay 0 -muxpreload 0 -f mpegts udp://{DESTINATION_IP}:{DESTINATION_PORT}'

print(f"🚀 Starting video server...")
print(f"📡 Streaming to {DESTINATION_IP}:{DESTINATION_PORT}")
print(f"📹 Resolution: {WIDTH}x{HEIGHT} @ {FRAMERATE}fps")

try:
    subprocess.run(video_server_command, shell=True)
except KeyboardInterrupt:
    print("\n✅ Server stopped")
    sys.exit(0)
