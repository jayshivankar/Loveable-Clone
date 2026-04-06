"""
Prometheus metrics.
"""

from prometheus_client import Counter, Gauge, Histogram

AGENT_LATENCY = Histogram(
    "codeforge_agent_duration_seconds",
    "Time spent in each agent node",
    labelnames=["node_name"],
)

ISSUE_COUNTER = Counter(
    "codeforge_issues_total",
    "Total number of issues found by reviewer",
    labelnames=["severity"],
)

REVIEW_SCORE = Gauge(
    "codeforge_review_score",
    "Reviewer score out of 10",
    labelnames=["session_id"],
)

JOBS_ACTIVE = Gauge(
    "codeforge_jobs_active_total",
    "Number of active generation jobs",
)
