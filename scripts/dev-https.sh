#!/usr/bin/env bash
#
# Serve Argus over HTTPS with a self-signed certificate, so a PHONE can use its
# camera.
#
#     ./scripts/dev-https.sh && npm run dev
#
# Why this exists: browsers only expose getUserMedia in a secure context.
# `localhost` counts as one, which is why this laptop's own webcam works over
# plain http, but `http://192.168.1.x:3000` does not — so a phone opening the
# pair link is refused the camera with no way for the page to ask again.
#
# This script only GENERATES the certificate. server.ts picks it up on the next
# start and listens on HTTPS_PORT (3443) alongside the usual http port, so the
# /ws and /ws/cam upgrades keep working with no proxy in the way.
#
# The certificate is self-signed, so the phone shows a warning the first time.
# Accepting it is what makes the origin secure. If that is unacceptable, attach
# an IP-camera app's stream URL to the pairing code instead: that path needs no
# certificate at all, because the sidecar reads the phone's stream directly.
set -euo pipefail

CERT_DIR="${CERT_DIR:-.certs}"
PORT="${HTTPS_PORT:-3443}"

command -v openssl >/dev/null || { echo "openssl not found: sudo dnf install openssl"; exit 1; }

mkdir -p "$CERT_DIR"

# Every address a phone might reach this machine on has to be in the
# certificate, or the browser rejects it before the warning is even offered.
ips=$(ip -4 addr show scope global 2>/dev/null | grep -oP 'inet \K[\d.]+' \
      | grep -v '^172\.1[6-9]\.' | grep -v '^172\.2' || true)
alt="DNS:localhost,IP:127.0.0.1"
for ip in $ips; do alt="$alt,IP:$ip"; done

if [ -f "$CERT_DIR/cert.pem" ] && [ -f "$CERT_DIR/key.pem" ]; then
  echo "reusing $CERT_DIR/cert.pem"
else
  echo "generating a self-signed certificate for $alt"
  openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
    -keyout "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.pem" \
    -subj "/CN=argus-dev" -addext "subjectAltName=$alt" 2>/dev/null
fi

echo
echo "Certificate ready. Restart Argus and it will serve HTTPS itself:"
echo
echo "  npm run dev"
echo
for ip in $ips; do echo "  phone: https://$ip:$PORT/devices"; done
echo
echo "The phone will warn about the certificate. Accept it once — that warning"
echo "IS the step that makes the origin secure enough for the browser to hand"
echo "over the camera."
echo
echo "No certificate is needed for this laptop's own webcam (localhost is always"
echo "a secure context), nor for the stream-URL path, where an IP-camera app on"
echo "the phone serves the video and the sidecar reads it directly."
