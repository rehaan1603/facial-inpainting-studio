# Private online research demo

This deployment shares the running Windows GPU app through an HTTPS Cloudflare Quick Tunnel. Inference still runs on the host RTX 5070 Laptop GPU. It is a temporary private demo, not an independently hosted or always-on GPU service.

## Start and open

1. Keep the laptop awake, connected and available for GPU inference.
2. Double-click **Start Sharing.cmd**. The launcher starts or reuses the local studio, then starts the protected sharing gateway and tunnel.
3. Open the issued HTTPS URL. Enter the generated password from **.local-share/LOGIN.txt** into the sign-in page. No browser authentication popup is required. The same file includes the current URL. The secure browser session expires after eight hours or when the gateway restarts.
4. Upload your own photo and mask, or add three or four reference photos for reference-assisted completion. Uploaded images, references and results travel through Cloudflare to the host laptop and are retained under outputs/webapp_runs. Only upload photos you have permission to use.

The .local-share directory contains credentials, tunnel logs, the client executable and process state. It is excluded from GitHub. Do not copy LOGIN.txt, access.json or authorization headers into repository files or issue reports. Credentials are shared among the people the owner admits; this is not a multi-user account system. Admitted people can access runs created during that gateway session if they have the run URL.

## Stop or restart

Run `./Stop Sharing.ps1` in PowerShell from the project folder to close the sharing processes. It verifies that the recorded processes belong to this project's gateway/client before stopping them. The local studio on port 8765 remains running. After stopping, double-click Start Sharing.cmd for a new temporary address. Update the GitHub link if the address changes.

Closing the laptop, losing connectivity or stopping either sharing process makes the link unavailable. Quick Tunnels have no uptime guarantee and can have provisioning or DNS failures. They are intended for development and demonstrations. [Official Quick Tunnel documentation](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/).

## Deployment boundaries

- The original GPU server remains bound to 127.0.0.1:8765. The sharing gateway listens only on 127.0.0.1:8877; the tunnel supplies HTTPS.
- Every shared route requires a random password. Uploads also require a session nonce and same-origin checks. Do not expose port 8877 directly as an unencrypted public service.
- Dataset demo endpoints, earlier local jobs, references, logs, metadata and arbitrary local paths are not shared. Only the frontend and newly admitted job status/input/result/effective-mask routes are available.
- The GPU server's one-job-at-a-time guard is shared by local and online users. This gateway introduces no second inference process or bypass of the GPU admission guard.
- Shared mode accurately states where uploads are processed and disables automatic dataset sample loading. Local mode retains its existing sample workflow.
- This deployment does not change the models, frozen experimental protocols or measured reconstruction quality. Model and dataset restrictions still apply; this is a noncommercial research demonstration.

## Verification

`scripts/test_share_gateway.py` checks authentication, blocked data/history routes, backend-token isolation, cross-origin rejection, upload size limits and new-run admission. `scripts/test_webapp.py` retains the existing local HTTP checks.

After a working tunnel is issued, `scripts/check_shared_deployment.py` checks the public HTTPS page, access controls, a real LaMa GPU reconstruction and result/mask downloads. It uses a synthetic colour gradient, so no dataset photograph is transmitted to the public endpoint. Exact preservation outside the mask is checked. Its report is research/shared_deployment_check.json when the run succeeds; that report is a functional deployment check, not a face-quality measurement.

The Windows tunnel client is downloaded from the official Cloudflare GitHub release 2026.9.1. The launcher verifies SHA256 `2837888cc0f5d58f15b6dc478376de90b4d3ba5241c7947455d1e0a0df429712` before execution. [Official client downloads](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/).

For a permanent independently hosted version, the next deployment stage requires a GPU hosting account and budget, portable model/environment setup, persistent storage/retention controls and individual access management. Static GitHub Pages hosting cannot run this Python/CUDA backend.
