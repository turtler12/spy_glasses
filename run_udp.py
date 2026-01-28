import cv2
import time

# Configuration
UDP_PORT = 1234
LISTEN_IP = "0.0.0.0"  # Listen on all interfaces

print("🎬 UDP Stream Receiver")
print(f"📡 Listening on port {UDP_PORT}")

# GStreamer pipeline for UDP H264 stream
pipeline = (
    f"udpsrc port={UDP_PORT} ! "
    "application/x-rtp,encoding-name=H264,payload=96 ! "
    "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
)

# Try to open the stream with retry logic
max_retries = 10
retry_count = 0

print("⏳ Waiting for stream...")

while retry_count < max_retries:
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if cap.isOpened():
        print("✅ Successfully connected to UDP stream!")
        break
    else:
        retry_count += 1
        if retry_count < max_retries:
            print(f"   Retry {retry_count}/{max_retries}... (waiting for Raspberry Pi to start streaming)")
            time.sleep(2)
        else:
            print("❌ Failed to connect to UDP stream after multiple retries")
            print("   Make sure:")
            print("   1. The Raspberry Pi server is running (run run_udp_server.py on the Pi)")
            print("   2. The IP address matches your Raspberry Pi's IP")
            print("   3. Both devices are on the same network")
            exit(1)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Stream interrupted - connection lost")
            break

        # Display the frame
        cv2.imshow('UDP Stream from Raspberry Pi', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("✅ Stream closed by user")
            break

except KeyboardInterrupt:
    print("\n✅ Stream closed")

finally:
    cap.release()
    cv2.destroyAllWindows()


