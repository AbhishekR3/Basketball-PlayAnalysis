import os
import requests
import json

# GitHub repository details
owner = os.getenv('GITHUB_OWNER')
repo = os.getenv('GITHUB_REPO')
gist_id = os.getenv('GIST_ID')
token = os.getenv('GITHUB_TOKEN')

# Workflows to check by their names
workflows = ["Run Basketball Object Tracking", "Run Game Simulation"]  # Ensure these match the 'name' fields in your workflow files

# GitHub API URL for workflows
api_url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows"

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github.v3+json"
}

def get_workflow_id(workflow_name):
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        for workflow in data['workflows']:
            print(f"Workflow found: {workflow['name']} with ID {workflow['id']}")
            if workflow['name'] == workflow_name:
                return workflow['id']
    print(f"Error fetching workflow ID for {workflow_name}: {response.text}")
    print('')
    return None

def get_workflow_status(workflow_id):
    workflow_runs_url = f"{api_url}/{workflow_id}/runs?per_page=1"
    response = requests.get(workflow_runs_url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data["workflow_runs"]:
            return data["workflow_runs"][0]["conclusion"]
    print(f"Error fetching workflow status for ID {workflow_id}: {response.text}")
    return "unknown"

# Get workflow IDs
workflow_ids = [get_workflow_id(name) for name in workflows]

# Check statuses of all workflows
statuses = [get_workflow_status(wf_id) for wf_id in workflow_ids if wf_id]
print(f"Workflow statuses: {statuses}")
print('')

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