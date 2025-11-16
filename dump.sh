set -euo pipefail

# ---- Config ----
# CSV output file
CSV_FILE="${CSV_FILE:-mds_metrics.csv}"

# Metrics command (override via env). Examples:
#   METRICS_CMD='ceph daemon mds.mds1 perf dump'
#   METRICS_CMD='ceph tell mds.0 perf dump'
METRICS_CMD="${METRICS_CMD:-ceph daemon mds.${MDS_ID:-a} perf dump}"

# Path to jq (needs jq installed)
JQ_BIN="${JQ_BIN:-jq}"

# ----------------

# Ensure header exists
if [[ ! -f "$CSV_FILE" ]]; then
	  echo "timestamp,msgr_recv_bytes,msgr_send_bytes,msgr_recv_messages,msgr_send_messages,mds_request,mds_reply,mds_reply_latency_s,getattr_latency_s,mkdir_latency_s,mds_journal_latency_s,objecter_op_latency_s,objecter_send_bytes,objecter_ops,mds_rss_bytes,mds_heap_bytes,sessions_open" \
		      > "$CSV_FILE"
fi

# Capture metrics JSON
JSON_OUT="$($METRICS_CMD)"

# Current timestamp (ISO 8601)
TS="$(date -Is)"

# Append one CSV line
# Sums across AsyncMessenger workers 0..2; uses 0 if a field/key is missing.
echo "$JSON_OUT" | "$JQ_BIN" -r --arg ts "$TS" '
  def wsum(field):
      ([
            (."AsyncMessenger::Worker-0"[field]),
	          (."AsyncMessenger::Worker-1"[field]),
		        (."AsyncMessenger::Worker-2"[field])
			    ] | map(select(. != null)) | add) // 0;

			      [
			          $ts,
				      wsum("msgr_recv_bytes"),
				          wsum("msgr_send_bytes"),
					      wsum("msgr_recv_messages"),
					          wsum("msgr_send_messages"),
						      (.mds.request // 0),
						          (.mds.reply // 0),
							      (.mds.reply_latency.avgtime // 0),
							          (.mds_server.req_getattr_latency.avgtime // 0),
								      (.mds_server.req_mkdir_latency.avgtime // 0),
								          (.mds_log.jlat.avgtime // 0),
									      (.objecter.op_latency.avgtime // 0),
									          (.objecter.op_send_bytes // 0),
										      (.objecter.op // 0),
										          (.mds_mem.rss // 0),
											      (.mds_mem.heap // 0),
											          (.mds_sessions.sessions_open // 0)
												    ] | @csv
												    ' >> "$CSV_FILE"
												    #!/usr/bin/env bash
												    set -euo pipefail

												    # ---- Config ----
												    # CSV output file
												    CSV_FILE="${CSV_FILE:-mds_metrics.csv}"

												    # Metrics command (override via env). Examples:
												    #   METRICS_CMD='ceph daemon mds.mds1 perf dump'
												    #   METRICS_CMD='ceph tell mds.0 perf dump'
												    METRICS_CMD="${METRICS_CMD:-ceph daemon mds.${MDS_ID:-0} perf dump}"

												    # Path to jq (needs jq installed)
												    JQ_BIN="${JQ_BIN:-jq}"

												    # ----------------

												    # Ensure header exists
												    if [[ ! -f "$CSV_FILE" ]]; then
													      echo "timestamp,msgr_recv_bytes,msgr_send_bytes,msgr_recv_messages,msgr_send_messages,mds_request,mds_reply,mds_reply_latency_s,getattr_latency_s,mkdir_latency_s,mds_journal_latency_s,objecter_op_latency_s,objecter_send_bytes,objecter_ops,mds_rss_bytes,mds_heap_bytes,sessions_open" \
														          > "$CSV_FILE"
												    fi

												    # Capture metrics JSON
												    JSON_OUT="$($METRICS_CMD)"

												    # Current timestamp (ISO 8601)
												    TS="$(date -Is)"

												    # Append one CSV line
												    # Sums across AsyncMessenger workers 0..2; uses 0 if a field/key is missing.
												    echo "$JSON_OUT" | "$JQ_BIN" -r --arg ts "$TS" '
												      def wsum(field):
												          ([
													        (."AsyncMessenger::Worker-0"[field]),
														      (."AsyncMessenger::Worker-1"[field]),
														            (."AsyncMessenger::Worker-2"[field])
															        ] | map(select(. != null)) | add) // 0;

																  [
																      $ts,
																          wsum("msgr_recv_bytes"),
																	      wsum("msgr_send_bytes"),
																	          wsum("msgr_recv_messages"),
																		      wsum("msgr_send_messages"),
																		          (.mds.request // 0),
																			      (.mds.reply // 0),
																			          (.mds.reply_latency.avgtime // 0),
																				      (.mds_server.req_getattr_latency.avgtime // 0),
																				          (.mds_server.req_mkdir_latency.avgtime // 0),
																					      (.mds_log.jlat.avgtime // 0),
																					          (.objecter.op_latency.avgtime // 0),
																						      (.objecter.op_send_bytes // 0),
																						          (.objecter.op // 0),
																							      (.mds_mem.rss // 0),
																							          (.mds_mem.heap // 0),
																								      (.mds_sessions.sessions_open // 0)
																								        ] | @csv
																									' >> "$CSV_FILE"
																									#!/usr/bin/env bash
																									set -euo pipefail

																									# ---- Config ----
																									# CSV output file
																									CSV_FILE="${CSV_FILE:-mds_metrics.csv}"

																									# Metrics command (override via env). Examples:
																									#   METRICS_CMD='ceph daemon mds.mds1 perf dump'
																									#   METRICS_CMD='ceph tell mds.0 perf dump'
																									METRICS_CMD="${METRICS_CMD:-ceph daemon mds.${MDS_ID:-0} perf dump}"

																									# Path to jq (needs jq installed)
																									JQ_BIN="${JQ_BIN:-jq}"

																									# ----------------

																									# Ensure header exists
																									if [[ ! -f "$CSV_FILE" ]]; then
																										  echo "timestamp,msgr_recv_bytes,msgr_send_bytes,msgr_recv_messages,msgr_send_messages,mds_request,mds_reply,mds_reply_latency_s,getattr_latency_s,mkdir_latency_s,mds_journal_latency_s,objecter_op_latency_s,objecter_send_bytes,objecter_ops,mds_rss_bytes,mds_heap_bytes,sessions_open" \
																											      > "$CSV_FILE"
																									fi

																									# Capture metrics JSON
																									JSON_OUT="$($METRICS_CMD)"

																									# Current timestamp (ISO 8601)
																									TS="$(date -Is)"

																									# Append one CSV line
																									# Sums across AsyncMessenger workers 0..2; uses 0 if a field/key is missing.
																									echo "$JSON_OUT" | "$JQ_BIN" -r --arg ts "$TS" '
																									  def wsum(field):
																									      ([
																									            (."AsyncMessenger::Worker-0"[field]),
																										          (."AsyncMessenger::Worker-1"[field]),
																											        (."AsyncMessenger::Worker-2"[field])
																												    ] | map(select(. != null)) | add) // 0;

																												      [
																												          $ts,
																													      wsum("msgr_recv_bytes"),
																													          wsum("msgr_send_bytes"),
																														      wsum("msgr_recv_messages"),
																														          wsum("msgr_send_messages"),
																															      (.mds.request // 0),
																															          (.mds.reply // 0),
																																      (.mds.reply_latency.avgtime // 0),
																																          (.mds_server.req_getattr_latency.avgtime // 0),
																																	      (.mds_server.req_mkdir_latency.avgtime // 0),
																																	          (.mds_log.jlat.avgtime // 0),
																																		      (.objecter.op_latency.avgtime // 0),
																																		          (.objecter.op_send_bytes // 0),
																																			      (.objecter.op // 0),
																																			          (.mds_mem.rss // 0),
																																				      (.mds_mem.heap // 0),
																																				          (.mds_sessions.sessions_open // 0)
																																					    ] | @csv
																																					    ' >> "$CSV_FILE"
