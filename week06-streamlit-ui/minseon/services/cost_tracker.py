"""
비용 & 시간 추적기 - Advanced RAG 파이프라인 전 과정 측정

[추적 대상]
  각 OpenAI API 호출마다:
    - 사용 토큰 (입력 / 출력)
    - 소요 시간 (초)
    - 비용 (USD, KRW)

[단계별 구분]
  pre        : Multi-query Generation (LLM 1회)
  embedding  : 쿼리 임베딩 (Embedding API, 쿼리 수만큼)
  reranking  : GPT Re-ranking (LLM 1회)
  compression: Context Compression (LLM, 청크 수만큼)
  generation : 최종 답변 생성 (LLM 스트리밍 1회)

[가격 기준] OpenAI 공식 가격 (2025년 기준)
  text-embedding-3-small : $0.020 / 1M tokens
  gpt-4o-mini input      : $0.150 / 1M tokens
  gpt-4o-mini output     : $0.600 / 1M tokens
"""

import time
from dataclasses import dataclass, field

# ── 모델별 가격 (USD per token) ────────────────────────────
PRICING: dict[str, dict[str, float]] = {
    "text-embedding-3-small": {
        "input":  0.020 / 1_000_000,
        "output": 0.0,
    },
    "gpt-4o-mini": {
        "input":  0.150 / 1_000_000,
        "output": 0.600 / 1_000_000,
    },
    "gpt-4o": {
        "input":  2.50 / 1_000_000,
        "output": 10.00 / 1_000_000,
    },
}

USD_TO_KRW = 1_380  # 환율 (고정 근사값)


@dataclass
class ApiCall:
    """단일 API 호출 기록"""
    stage:         str
    model:         str
    input_tokens:  int
    output_tokens: int
    elapsed:       float  # 초
    cost_usd:      float


class CostTracker:
    """
    파이프라인 전체의 API 호출 비용·시간 추적기

    사용법:
        tracker = CostTracker()
        tracker.start_stage("pre")
        # ... API 호출 ...
        tracker.record("pre", "gpt-4o-mini", in_tok, out_tok, elapsed)
        tracker.end_stage("pre")
        summary = tracker.get_summary()
    """

    def __init__(self):
        self.calls: list[ApiCall] = []
        self._stage_starts: dict[str, float] = {}
        self.stage_elapsed: dict[str, float] = {}

    # ── 단계 시간 측정 ─────────────────────────────────────

    def start_stage(self, stage: str) -> None:
        self._stage_starts[stage] = time.time()

    def end_stage(self, stage: str) -> float:
        """단계 종료, 소요 시간(초) 반환"""
        if stage in self._stage_starts:
            elapsed = time.time() - self._stage_starts.pop(stage)
            self.stage_elapsed[stage] = self.stage_elapsed.get(stage, 0) + elapsed
            return elapsed
        return 0.0

    # ── API 호출 기록 ──────────────────────────────────────

    def record(
        self,
        stage: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        elapsed: float,
    ) -> float:
        """API 호출 기록, 비용(USD) 반환"""
        pricing = PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost = input_tokens * pricing["input"] + output_tokens * pricing["output"]
        self.calls.append(ApiCall(stage, model, input_tokens, output_tokens, elapsed, cost))
        return cost

    # ── 집계 ──────────────────────────────────────────────

    def get_summary(self) -> dict:
        """
        전체 비용·시간 요약 반환

        Returns:
            {
                "total_cost_usd":      float,
                "total_cost_krw":      float,
                "total_input_tokens":  int,
                "total_output_tokens": int,
                "total_elapsed":       float,  # 초
                "by_stage": {
                    stage_name: {
                        "cost_usd":      float,
                        "cost_krw":      float,
                        "input_tokens":  int,
                        "output_tokens": int,
                        "calls":         int,
                        "elapsed":       float,
                    }
                },
                "api_calls": [...]  # 원시 기록
            }
        """
        # 단계별 집계
        by_stage: dict[str, dict] = {}
        for call in self.calls:
            if call.stage not in by_stage:
                by_stage[call.stage] = {
                    "cost_usd": 0.0, "cost_krw": 0.0,
                    "input_tokens": 0, "output_tokens": 0,
                    "calls": 0, "elapsed": 0.0,
                }
            s = by_stage[call.stage]
            s["cost_usd"]      += call.cost_usd
            s["cost_krw"]      += call.cost_usd * USD_TO_KRW
            s["input_tokens"]  += call.input_tokens
            s["output_tokens"] += call.output_tokens
            s["calls"]         += 1
            s["elapsed"]       += call.elapsed

        # 단계 타이머 시간 병합 (API 호출 시간과 다를 수 있음)
        for stage, elapsed in self.stage_elapsed.items():
            if stage not in by_stage:
                by_stage[stage] = {
                    "cost_usd": 0.0, "cost_krw": 0.0,
                    "input_tokens": 0, "output_tokens": 0,
                    "calls": 0, "elapsed": elapsed,
                }
            else:
                # 단계 타이머가 더 정확 (검색 등 비API 시간 포함)
                by_stage[stage]["elapsed"] = elapsed

        total_cost  = sum(c.cost_usd for c in self.calls)
        total_in    = sum(c.input_tokens for c in self.calls)
        total_out   = sum(c.output_tokens for c in self.calls)
        total_time  = sum(self.stage_elapsed.values()) or sum(c.elapsed for c in self.calls)

        return {
            "total_cost_usd":      round(total_cost, 6),
            "total_cost_krw":      round(total_cost * USD_TO_KRW, 2),
            "total_input_tokens":  total_in,
            "total_output_tokens": total_out,
            "total_elapsed":       round(total_time, 2),
            "by_stage":            by_stage,
            "api_calls": [
                {
                    "stage":         c.stage,
                    "model":         c.model,
                    "input_tokens":  c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "elapsed":       round(c.elapsed, 3),
                    "cost_usd":      round(c.cost_usd, 6),
                }
                for c in self.calls
            ],
        }

    def reset(self) -> None:
        self.calls.clear()
        self._stage_starts.clear()
        self.stage_elapsed.clear()
