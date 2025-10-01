from __future__ import annotations

import json
from pathlib import Path
from typing import List

from google import genai
from google.oauth2.service_account import Credentials


class LAVE:
    MODEL_NAME = "gemini-2.5-flash-lite"

    def __init__(self, json_path: str | Path, location: str = "us-central1", debug: bool = False) -> None:
        json_path = Path(json_path)

        with json_path.open("r", encoding="utf-8") as f:
            service_account_info = json.load(f)

        self.project_id: str = service_account_info.get("project_id", "")
        if not self.project_id:
            raise ValueError("Missing 'project_id' in service account JSON.")

        self.credentials = Credentials.from_service_account_file(
            str(json_path),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        self.client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=location,
            credentials=self.credentials,
        )

        self.debug = debug

    def compute_score(self, cand: str, refs: List[str], question: str) -> float:
        """Compute a normalized LAVE score between 0.0 and 1.0."""

        prompt = (
            "Bạn là giám khảo đánh giá chất lượng trả lời câu hỏi.\n"
            "Nhiệm vụ: So sánh câu trả lời ứng viên với tập câu trả lời tham chiếu "
            "và chấm điểm theo thang 1-3:\n"
            " - 1: Sai hoàn toàn, không phù hợp hoặc trái ngược với tham chiếu.\n"
            " - 2: Một phần đúng, mơ hồ, thiếu chi tiết, chưa trọn vẹn.\n"
            " - 3: Đúng hoàn toàn, nhất quán và bao quát với tham chiếu.\n\n"
            "Bạn phải suy luận ngắn gọn trước khi đưa ra kết quả cuối cùng.\n"
            "Chỉ xuất ra một số nguyên (1, 2, hoặc 3) theo đúng định dạng JSON.\n\n"
            f"Câu hỏi: {question}\n"
            f"Câu trả lời tham chiếu: {refs}\n"
            f"Câu trả lời ứng viên: {cand}\n\n"
            "Kết quả:"
        )

        try:
            response = self.client.models.generate_content(
                model=self.MODEL_NAME,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": int,
                },
            )
        except Exception as e:
            if self.debug:
                print("Generation failed:", e)
            return 0.0

        if self.debug:
            print("Prompt:\n", prompt)
            print("Response text:\n", getattr(response, "text", None))
            print("Response parsed:\n", getattr(response, "parsed", None))

        try:
            raw_score = int(response.parsed)
            if raw_score not in (1, 2, 3):
                raise ValueError(f"Unexpected score: {raw_score}")
            return (raw_score - 1) / 2.0
        except Exception as e:
            if self.debug:
                print("Parsing failed:", e)
            return 0.0
