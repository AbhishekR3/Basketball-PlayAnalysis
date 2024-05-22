import os
import requests
import json

# GitHub repository details
owner = os.getenv('GITHUB_OWNER')
repo = os.getenv('GITHUB_REPO')
gist_id = os.getenv('GIST_ID')
token = os.getenv('GITHUB_TOKEN')

print("GIST_ID")
print(gist_id)

print("TOKEN")
print(token)

# Workflows to check
workflows = ["object-tracking", "game-simulation"]

# GitHub API URL for workflow runs
api_url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs"

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github.v3+json"
}

def get_workflow_status(workflow_name):
    response = requests.get(f"{api_url}?workflow_id={workflow_name}&per_page=1", headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data["workflow_runs"]:
            return data["workflow_runs"][0]["conclusion"]
    return "unknown"

# Check statuses of all workflows
statuses = [get_workflow_status(workflow) for workflow in workflows]
print(statuses)

# Determine combined status
if all(status == "success" for status in statuses):
    combined_status = "passing"
    color = "brightgreen"
else:
    combined_status = "failing"
    color = "red"

# Badge JSON content
badge_json = {
    "schemaVersion": 1,
    "label": "CI/CD",
    "message": combined_status,
    "color": color
}

# Update the Gist with the badge JSON
gist_url = f"https://api.github.com/gists/{gist_id}"
headers_gist = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github.v3+json"
}
response = requests.patch(gist_url, headers=headers_gist, json={"files": {"badge.json": {"content": json.dumps(badge_json)}}})
print(f"Badge update response: {response.status_code}, {response.text}")