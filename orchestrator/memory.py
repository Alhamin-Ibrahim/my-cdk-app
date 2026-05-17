from __future__ import annotations

import logging
import time
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

TABLE_NAME = "rag-conversation-history"
TTL_SECONDS = 86_400          # 24 hours
MAX_HISTORY_TURNS = 5         # how many turns to load per query
MAX_TURNS_TO_STORE = 50       # cap total turns per session


class ConversationMemory:

    def __init__(self, region: str = "eu-west-1") -> None:
        self._ddb = boto3.resource("dynamodb", region_name=region)
        self._table = self._ddb.Table(TABLE_NAME)

    def load_history(self, session_id: str) -> list[dict[str, str]]:
        try:
            response = self._table.query(
                KeyConditionExpression=boto3.dynamodb.conditions.Key(
                    "session_id"
                ).eq(session_id),
                ScanIndexForward=False,       # newest first
                Limit=MAX_HISTORY_TURNS * 2,  # *2 because each turn = 2 items (user + assistant)
                ProjectionExpression="#r, content",
                ExpressionAttributeNames={"#r": "role"},  # "role" is reserved in DDB
            )
        except ClientError as exc:
            logger.warning("DynamoDB load_history failed: %s", exc)
            return []

        items = response.get("Items", [])
        # Reverse so we get oldest-first for the LLM prompt
        items.reverse()
        return [{"role": item["role"], "content": item["content"]} for item in items]

    def save_turn(self, session_id: str, role: str, content: str) -> None:
        turn_number = int(time.time() * 1000)
        ttl = int(time.time()) + TTL_SECONDS

        try:
            self._table.put_item(
                Item={
                    "session_id": session_id,
                    "turn_number": turn_number,
                    "role": role,
                    "content": content,
                    "ttl": ttl,
                }
            )
        except ClientError as exc:
            logger.warning("DynamoDB save_turn failed: %s", exc)

    def format_for_prompt(self, history: list[dict[str, str]]) -> str:
        if not history:
            return ""

        lines = []
        for turn in history:
            role = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content']}")

        return "\n".join(lines)