"""Jobs HTTP endpoints."""

from fastapi import APIRouter

from .. import jobs as job_service, responses, store

router = APIRouter(prefix="/api/v1")


@router.get("/jobs", response_model=list[responses.Job])
def jobs():
    return store.jobs()


@router.get("/jobs/{job_id}", response_model=responses.Job)
def job(job_id: str):
    return job_service.detail(job_id)


@router.post("/jobs/{job_id}/cancel", response_model=responses.Job)
def cancel(job_id: str):
    return job_service.cancel(job_id)


@router.post("/jobs/{job_id}/rerun", response_model=responses.Job)
def rerun(job_id: str):
    return job_service.rerun(job_id)
