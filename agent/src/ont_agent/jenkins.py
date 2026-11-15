"""Jenkins connectivity test script.

Usage: uv run python -m ont_agent.jenkins
Env: JENKINS_URL, JENKINS_USER, JENKINS_API_TOKEN
"""

import base64
import json
import os
import sys
import urllib.request

fusdkbfksdkjbfjdfjbkjbdsfdsvdsfdsfdbrghhgreggfggeggegeg
def main() -> int:
    url = os.environ.get("JENKINS_URL", "").rstrip("/")
    user = os.environ.get("JENKINS_USER", "")
    token = os.environ.get("JENKINS_API_TOKEN", "")
    if not url.startswith(("http://", "https://")):
        print("JENKINS_URL must be set to an http(s) URL")
        return 1

    req = urllib.request.Request(f"{url}/api/json?tree=mode,numExecutors,jobs[name]")  # noqa: S310
    if user and token:
        creds = base64.b64encode(f"{user}:{token}".encode()).decode()
        req.add_header("Authorization", f"Basic {creds}")

    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
        data = json.load(resp)

    jobs = [j["name"] for j in data.get("jobs", [])]
    print(f"Connected. mode={data.get('mode')} executors={data.get('numExecutors')}")
    print(f"Jobs ({len(jobs)}): {', '.join(jobs[:10])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
