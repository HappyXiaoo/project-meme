from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="待检测的中文文本")


class PredictResponse(BaseModel):
    label: int = Field(..., description="预测标签，0 表示正常文本，1 表示疑似有害文本")
    label_name: str = Field(..., description="标签名称")
    score: float = Field(..., ge=0, le=1, description="模型置信度分数")
    message: str = Field(..., description="结果说明")


class HealthResponse(BaseModel):
    status: str
    mode: str
    data_path: str | None = None
    sample_count: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
