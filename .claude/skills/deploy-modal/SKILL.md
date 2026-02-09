---
name: deploy-modal
description: Deploy the CLaaS Modal serverless distillation API. Use when deploying the SDPO distillation service to production.
---

# Deploy Modal Worker

Deploy the Modal serverless distillation API for CLaaS.

## Instructions

When this skill is invoked, perform the following steps:

1. **Deploy the Modal app**:
   ```bash
   cd /Users/admin/Documents/CLaaS && modal deploy claas.api
   ```

2. **Extract the endpoint URL** from the deployment output (look for the line with the FastAPI endpoint URL)

3. **Report the deployment status** including:
   - The deployed endpoint URL
   - Link to the Modal dashboard for the deployment
   - Any warnings or errors from the deployment

## Deployed Services

The deployment includes:
- **DistillWorker**: L40S GPU worker with Qwen2.5-Coder-3B student model
- **TeacherService**: A100 GPU worker with Qwen2.5-32B teacher model (vLLM)
- **FastAPI endpoint**: REST API for distillation requests

## Health Check

After deployment, verify the services are healthy:

```bash
curl https://{your-modal-app-url}/v1/health
```

## Notes

- The Modal token must already be configured (run `modal token set` if not)
- First deployment takes longer due to model downloads (~5-10 minutes)
- Subsequent deployments are faster if the images haven't changed
- GPU memory snapshots enable sub-second cold starts after initial warmup
