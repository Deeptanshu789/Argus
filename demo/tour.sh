#!/usr/bin/env bash
# A scripted tour of the whole system, over real HTTP.
#
#   ./demo/tour.sh                     against the live pipeline
#   API=http://localhost:3000/api/mock ./demo/tour.sh
#
# Everything below is a real request to the running server. Nothing is
# pre-recorded, which is the point: if the pipeline is down, this fails loudly
# instead of showing a screenshot of a system that is not running.
set -u
API="${API:-http://localhost:3000/api}"
h() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
get() { curl -s --max-time 10 "$API/$1"; }

h "Health"
get health | jq -c .

h "Cameras"
get cameras | jq -r '.[] | "\(.id)  \(.name)  [\(.status)]  \(.lat),\(.lon)"'

h "Road graph — this is layer 3 of the association engine"
get cameras/links | jq -r '.[] | select(.from < .to)
  | "\(.from) <-> \(.to)   \(.distance_m) m   ~\(.travel_time_s)s expected"'

h "Cross-camera journeys — Module C output"
get 'trajectories?limit=8' | jq -r '.[] |
  "\(.plate_text // "(unidentified)")   \(.path | length) cameras   "
  + ([.hops[] | "\(.from_camera)->\(.to_camera) \(.method) \(.confidence)"] | join("  |  "))'

h "How each hop was confirmed"
get 'trajectories?limit=100' | jq -r '[.[].hops[].method] | group_by(.)
  | .[] | "\(.[0])  x\(length)"'

h "Recent tracks"
get 'tracks?limit=8' | jq -r '.[] | "\(.camera_id)  track \(.id)  \(.vehicle_type)  "
  + "\(.plate_text // "-")  \(.colour // "-")  speed \(.speed_kmh // "-")"'

h "Search by plate"
PLATE=$(get 'trajectories?limit=50' | jq -r '[.[] | select(.plate_text)][0].plate_text // empty')
if [ -n "$PLATE" ]; then
  echo "query: ${PLATE:0:4}"
  get "search?plate=${PLATE:0:4}" | jq -r '"sightings \(.sightings|length), journeys "
    + "\(.trajectories|length), last seen \(.last_seen.camera_id // "never") "
    + "\(.last_seen.ts // "")"'
else
  echo "no plate read yet — run the worker first"
fi

h "Analytics"
get 'analytics?window=1h' | jq -r '"buckets \(.series|length)   vehicles "
  + "\(.totals.vehicle_count)   avg speed \(.totals.avg_speed_kmh) km/h"'
get 'analytics?window=1h' | jq -r '.series[] | "  \(.ts)  \(.vehicle_count) vehicles  "
  + "congestion \(.congestion_score)  \(.by_type | to_entries | map("\(.key) \(.value)") | join(", "))"'
echo "(speed is uncalibrated on the demo clips: cameras have no metres_per_pixel,"
echo " so db.ts falls back to 0.05 and the km/h figure is a placeholder.)"

h "Alerts"
get 'alerts?limit=6' | jq -r '.[] | "\(.severity|ascii_upcase)  \(.kind)  "
  + "\(.camera_id // "-")  \(.plate_text // "no plate")  \(.detail)"'

h "Uploaded video — analysed separately from the live cameras"
get 'uploads?limit=5' | jq -r '.[] |
  "\(.status)  \(.label // "untitled")  \(.sources | length) video(s)  "
  + "\([.sources[].tracks] | add // 0) vehicles  "
  + "\([.sources[].plates] | add // 0) plates"'
echo "(uploaded cameras are excluded from every view above: they are the"
echo " operator's own footage, not the city's.)"

h "Phones and webcams paired as cameras"
get devices | jq -r '.[] |
  "\(.status)  \(.code)  \(.camera_id)  \(.label // "unnamed")  "
  + "\(.kind // "not connected yet")\(if .source_url then "  " + .source_url else "" end)"'
echo "(a paired device is a real camera: it is in the list above, on the map,"
echo " and in the analytics. Open its /cam/<code> link on the phone.)"

h "Live WebSocket"
echo "open http://localhost:3000  — the grid, map, analytics and search all"
echo "subscribe to ws://localhost:3000/ws and update without a refresh."
