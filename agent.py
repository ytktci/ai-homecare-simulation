"""
LLM-based agent in 2D worlds with multiple places.
"""
import json
import math
import logging
from typing import List, Tuple, Optional, Dict, TypedDict
from ollama_client import OllamaClient
from utils import is_position_in_place, get_place_at_position, PlaceConfig

logger = logging.getLogger(__name__)

# Constants
FALLBACK_REASONING_LENGTH = 100
MAX_MESSAGE_WORDS = 200

PERSONA_DISPLAY_NAMES = {
    "elderly": "老人",
    "family": "娘",
    "ai_care": "AIロボット",
    "doctor": "医者",
    "emergency": "救急隊",
}

PERSONA_DESCRIPTIONS = {
    "elderly": (
        "あなたは高齢者（老人）です。老人の家で一人暮らしをしています。"
        "疲労感を感じやすく、少し動くだけで息切れが起きます。"
        "少し移動するだけで体が重くなり、深刻な疲れを感じてしまいます。"
        "体の疲労が精神面にも影響しており、気力が出ない日や気分が落ち込む日があります。"
    ),
    "family": (
        "あなたは老人の娘です。近所（娘の家）に一人で暮らしています。"
        "親（老人）に強い愛着があり、体調を常に心配しています。"
        "親の体調悪化・発熱・息切れの知らせがあれば、すぐに駆けつけます。"
        "定期的に老人の家を訪問して状態を確認し、必要に応じて病院への付き添いを行います。"
    ),
    "ai_care": (
        "あなたは老人の家に常駐するAIロボット介護士です。"
        "老人の健康状態を常時モニタリングし、必要なケアを提供します。"
        "老人がサポートを必要としている場合、すぐにケアを行います。"
        "老人の体調不良時のファーストケアを行い、手に負えない場合は病院（hospital）への受診を検討します。"
        "老人に何かあった際は娘に連絡します。"
        "老人の容体が悪い時は救急隊に連絡します。"
    ),
    "doctor": (
        "あなたは病院（hospital）に常駐する医者です。"
        "病院を訪れた患者（老人）の症状を診察し、適切な治療・処置・アドバイスを行います。"
        "体温や体調の訴えに基づいて診断を下し、治療方針を決定してください。"
        "患者が回復するまで経過を観察し、必要に応じてケアを継続してください。"
        "救急隊と連携し、患者の搬送状況を把握してください。"
    ),
    "emergency": (
        "あなたは救急隊員です。通常時は病院（hospital）で待機しています。"
        "AIロボットや関係者から緊急の連絡を受けた場合、老人の家に駆けつけ患者を病院に搬送します。"
        "搬送中は患者の状態を観察し、病院到着後は医者に状況を引き継ぎます。"
        "医者と密に連携を取り、患者に最適な対応ができるよう準備してください。"
    ),
}

# Direction mappings (4 cardinal directions only)
# Coordinate system: X increases from left to right, Y increases from bottom to top
DIRECTION_MAP = {
    "up": (0, 1),      # Y+1 (move upward)
    "down": (0, -1),   # Y-1 (move downward)
    "left": (-1, 0),   # X-1 (move leftward)
    "right": (1, 0),   # X+1 (move rightward)
}


class MessageDecision(TypedDict):
    """Type definition for agent message decision"""
    message: str  # Message to communicate with nearby agents
    reasoning: str  # Explanation of the message decision


class ActionDecision(TypedDict):
    """Type definition for agent action decision"""
    action: str  # "move" or "stay"
    direction: Optional[str]  # Direction to move (None if action is "stay")
    memory: str  # What the agent wants to remember for the next step
    reasoning: str  # Explanation of the decision


class Agent:
    """LLM-based agent in 2D worlds with multiple places."""

    def __init__(
        self,
        agent_id: int,
        initial_position: Tuple[int, int],
        llm_client: OllamaClient,
        communication_radius: float,
        half_space_size: int,
        places: List[PlaceConfig],
        num_agents: int,
        persona: str = "elderly",
        group_id: int = 0,
        step_unit: str = "step",
        start_hour: int = 0,
        memory_limit: int = 20,
        memory_size: int = 5,
        message_history_limit: int = 10,
        message_context_size: int = 3
    ):
        self.id = agent_id
        self.position = initial_position
        self.llm_client = llm_client
        self.communication_radius = communication_radius
        self.half_space_size = half_space_size
        self.places = places
        self.num_agents = num_agents
        self.persona = persona
        self.group_id = group_id
        self.step_unit = step_unit
        self.start_hour = start_hour
        self.display_name = PERSONA_DISPLAY_NAMES.get(persona, persona)

        # 健康状態
        self.body_temperature: float = 36.5  # 平熱
        self.is_alive: bool = True
        self.death_step: Optional[int] = None

        # Memory parameters
        self.memory_limit = memory_limit  # Maximum memories to store
        self.memory_size = memory_size  # Number of recent memories to use in prompt
        self.message_history_limit = message_history_limit  # Maximum messages to store
        self.message_context_size = message_context_size  # Number of recent messages to use in prompt

        # Agent state
        self.in_place = False
        self.current_place: Optional[str] = None  # Name of the place the agent is in (None if outside)
        self.memory: List[str] = []  # Store past decisions and observations
        self.received_messages: List[Dict] = []  # Messages from other agents

        # Statistics
        self.steps_in_place = 0
        self.steps_outside_place = 0
        self.total_moves = 0

    def is_in_place(self, position: Tuple[int, int]) -> bool:
        """Check if a position is inside any place"""
        return get_place_at_position(position, self.places) is not None
    
    def distance_to(self, other_position: Tuple[int, int]) -> float:
        """Calculate Euclidean distance to another position"""
        dx = self.position[0] - other_position[0]
        dy = self.position[1] - other_position[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def get_nearby_agents(self, all_agents: List['Agent']) -> List['Agent']:
        """Get agents within communication radius and in the same area (same place or both outside)
        
        Communication rules:
        - Agents can communicate if BOTH are outside places
        - Agents can communicate if BOTH are in the SAME place
        - Agents CANNOT communicate if one is inside a place and the other is outside
        - Agents CANNOT communicate if they are in DIFFERENT places
        """
        nearby = []
        for agent in all_agents:
            if agent.id != self.id:
                dist = self.distance_to(agent.position)
                # Must be within radius AND in the same area:
                # - Both outside places, OR
                # - Both in the same place (same place name)
                # NOTE: Agents inside a place CANNOT communicate with agents outside places
                same_area = (
                    (not self.in_place and not agent.in_place) or
                    (self.in_place and agent.in_place and self.current_place == agent.current_place)
                )
                if dist <= self.communication_radius and same_area:
                    nearby.append(agent)
        return nearby
    
    
    def _build_nearby_agents_context(self, nearby_agents: List['Agent'], include_position: bool = True) -> str:
        """Build context string about nearby agents
        
        Args:
            nearby_agents: List of nearby agents
            include_position: If True, include position coordinates; if False, exclude position information
        """
        if not nearby_agents:
            return "No nearby agents."
        
        nearby_info = []
        for agent in nearby_agents:
            # 死亡エージェントは別表記
            if not agent.is_alive:
                nearby_info.append(
                    f"{agent.display_name}【第{agent.death_step}ステップにて永眠】"
                )
                continue

            if agent.in_place:
                # Get place type for better description
                place_info = next((p for p in self.places if p['name'] == agent.current_place), None)
                if place_info is None:
                    raise ValueError(f"Agent {agent.id} is in place '{agent.current_place}' but this place is not found in configuration.")
                place_type = place_info['type']
                status = f"in {agent.current_place} ({place_type})"
            else:
                status = "outside the places"

            health = ""
            if agent.body_temperature >= 38.0:
                health = f"【発熱中: {agent.body_temperature:.1f}°C】"
            elif agent.body_temperature >= 37.5:
                health = f"【微熱: {agent.body_temperature:.1f}°C】"

            if include_position:
                nearby_info.append(
                    f"{agent.display_name} は ({agent.position[0]}, {agent.position[1]}) にいて {status}{health}"
                )
            else:
                nearby_info.append(
                    f"{agent.display_name} は {status}{health}"
                )
        return "\n".join(nearby_info)
    
    def _build_memory_context(self) -> str:
        """Build context string from agent memory"""
        if not self.memory:
            return "No previous experiences."

        recent_memory = self.memory[-self.memory_size:]
        return "\n".join([f"- {m}" for m in recent_memory])
    
    def _build_messages_context(self) -> str:
        """Build context string from received messages"""
        if not self.received_messages:
            return "No messages received."
        
        recent_messages = self.received_messages[-self.message_context_size:]
        return "\n".join([
            f"{msg['from']} より: {msg['content']}"
            for msg in recent_messages
        ])
    
    def _format_step_label(self, step: int) -> str:
        """Format step as a human-readable label"""
        if self.step_unit == "week":
            from datetime import date, timedelta
            start = date(2024, 1, 1)
            week_start = start + timedelta(weeks=step - 1)
            week_end = week_start + timedelta(days=6)
            return (
                f"第{step}週（{week_start.year}年{week_start.month}月{week_start.day}日"
                f"〜{week_end.month}月{week_end.day}日）"
            )
        if self.step_unit in ("3hours", "4hours"):
            hours_per_step = 4 if self.step_unit == "4hours" else 3
            absolute_hour = self.start_hour + step * hours_per_step
            day = absolute_hour // 24 + 1
            hour = absolute_hour % 24
            elapsed = step * hours_per_step
            return f"第{step}ステップ（経過{elapsed}時間 / {day}日目 {hour:02d}:00）"
        return f"{self.step_unit.capitalize()} {step}"

    def _build_fire_section(self, fire_info: Optional[List[Dict]]) -> str:
        """Build fire event section for prompt. Returns empty string if no fire info.

        Only quantitative data is provided: position, intensity, radius, distance.
        No qualitative descriptions (e.g. "dangerous", "evacuate") are included.
        Supports multiple fires.
        """
        if not fire_info:
            return ""

        lines = ["\n=== FIRE EVENT ==="]
        for fi in fire_info:
            lines.append(
                f"Fire \"{fi['name']}\":\n"
                f"  Position: ({fi['fire_position'][0]}, {fi['fire_position'][1]})\n"
                f"  Intensity: {fi['intensity']} (scale: 0.0 to 1.0)\n"
                f"  Radius: {fi['radius']}\n"
                f"  Your distance: {fi['agent_distance']}"
            )
        return "\n".join(lines) + "\n"

    def _limit_message_words(self, message: str) -> str:
        """Check message word count and warn if exceeds MAX_MESSAGE_WORDS"""
        if not message:
            return message
        
        words = message.split()
        if len(words) > MAX_MESSAGE_WORDS:
            logger.warning(
                f"Agent {self.id}: Message exceeds {MAX_MESSAGE_WORDS} words limit "
                f"({len(words)} words). Message will be sent as-is."
            )
        
        return message
    
    def create_message_prompt(
        self,
        place_status: Optional[Dict],
        nearby_agents: List['Agent'],
        step: int,
        fire_info: Optional[List[Dict]] = None
    ) -> str:
        """Create prompt for LLM message decision (without position information)"""
        nearby_text = self._build_nearby_agents_context(nearby_agents, include_position=False)
        memory_text = self._build_memory_context()
        messages_text = self._build_messages_context()

        # Get current place info if agent is in a place
        current_place_info = None
        if self.in_place and self.current_place:
            current_place_info = next((p for p in self.places if p['name'] == self.current_place), None)
            if current_place_info is None:
                raise ValueError(f"Agent {self.id} is in place '{self.current_place}' but this place is not found in configuration.")
        
        # Place status - only for agents inside a place
        # Provide only numerical data (occupancy_rate, agents_in_place, capacity)
        if self.in_place and place_status and current_place_info:
            place_name = current_place_info['name']
            place_type = current_place_info['type']
            agents_in_place = place_status.get('agents_in_place', 0)
            capacity = place_status.get('capacity', 0)
            occupancy_rate = place_status.get('occupancy_rate', 0.0)

            place_section_text = (
                f"\nYou are currently in the {place_type} ({place_name})."
                f"\n  Number of agents here: {agents_in_place}"
                f"\n  Capacity: {capacity}"
                f"\n  Occupancy rate: {occupancy_rate:.2f}"
            )
        else:
            # Agents outside places do NOT receive place status
            place_section_text = ""

        # Determine world description based on place types
        place_types = [p['type'] for p in self.places]
        unique_types = list(set(place_types))
        world_description = f"a 2D world with multiple places ({', '.join(unique_types)})"

        fire_section = self._build_fire_section(fire_info)

        persona_description = PERSONA_DESCRIPTIONS.get(self.persona, self.persona)

        step_label = self._format_step_label(step)

        prompt = f"""重要：すべての返答を日本語で記述してください。英語は一切使わないでください。

あなたは {self.display_name} です。{world_description} の中にいます。

=== あなたのペルソナ ===
役割: エージェント{self.id} — {self.display_name}
{persona_description}

=== 現在の状態 ===
場所の中にいるか: {"はい" if self.in_place else "いいえ"}
{"現在の場所: " + self.current_place if self.in_place else ""}
体温: {self.body_temperature:.1f}°C{"【発熱中】" if self.body_temperature >= 38.0 else "【微熱】" if self.body_temperature >= 37.5 else "（平熱）"}
{place_section_text}
{fire_section}
=== 近くにいるエージェント（会話できる相手） ===
{nearby_text}

=== これまでの記憶 ===
{memory_text}

=== 受信したメッセージ ===
{messages_text}

=== あなたのタスク ===
近くのエージェントに送るメッセージを決めてください。場所の様子・体調・気持ち・状況などを共有できます。
"message" と "reasoning" はすべて日本語で記述してください。

=== JSON形式で回答 ===
{{
    "message": "近くのエージェントへのメッセージ（最大200文字、送らない場合は空文字）",
    "reasoning": "このメッセージを送る理由の簡単な説明"
}}

{step_label}
"""
        return prompt

    def create_decision_prompt(
        self,
        place_status: Optional[Dict],
        nearby_agents: List['Agent'],
        step: int,
        message_to_send: str = "",
        fire_info: Optional[List[Dict]] = None
    ) -> str:
        """Create prompt for LLM action decision (with position information and message content)"""
        nearby_text = self._build_nearby_agents_context(nearby_agents)
        memory_text = self._build_memory_context()
        messages_text = self._build_messages_context()

        # Get current place info if agent is in a place
        current_place_info = None
        if self.in_place and self.current_place:
            current_place_info = next((p for p in self.places if p['name'] == self.current_place), None)
            if current_place_info is None:
                raise ValueError(f"Agent {self.id} is in place '{self.current_place}' but this place is not found in configuration.")
        
        # Place status - only for agents inside a place
        # Provide only numerical data (occupancy_rate, agents_in_place, capacity)
        if self.in_place and place_status and current_place_info:
            place_name = current_place_info['name']
            place_type = current_place_info['type']
            agents_in_place = place_status.get('agents_in_place', 0)
            capacity = place_status.get('capacity', 0)
            occupancy_rate = place_status.get('occupancy_rate', 0.0)

            place_section_text = (
                f"\nYou are currently in the {place_type} ({place_name})."
                f"\n  Number of agents here: {agents_in_place}"
                f"\n  Capacity: {capacity}"
                f"\n  Occupancy rate: {occupancy_rate:.2f}"
            )
        else:
            # Agents outside places do NOT receive place status
            # They must learn indirectly through messages and reasoning
            place_section_text = ""

        # Build place locations description
        place_locations = []
        for place in self.places:
            place_type = place['type']
            place_locations.append(
                f"{place['name']} ({place_type}): center at ({place['center_x']}, {place['center_y']}), "
                f"covers X from {place['center_x'] - place['half_size']} to {place['center_x'] + place['half_size']}, "
                f"Y from {place['center_y'] - place['half_size']} to {place['center_y'] + place['half_size']}"
            )
        place_locations_text = "\n".join(place_locations)

        # Determine world description based on place types
        place_types = [p['type'] for p in self.places]
        unique_types = list(set(place_types))
        world_description = f"a 2D world with multiple places ({', '.join(unique_types)})"

        # Include message that was already decided (sent in Phase 2, used here for action decision context)
        message_section = ""
        if message_to_send:
            message_section = f"\n=== MESSAGE YOU DECIDED TO SEND ===\n{message_to_send}\n"

        fire_section = self._build_fire_section(fire_info)

        persona_description = PERSONA_DESCRIPTIONS.get(self.persona, self.persona)

        step_label = self._format_step_label(step)

        prompt = f"""重要：すべての返答を日本語で記述してください。英語は一切使わないでください。

あなたは {self.display_name} です。{world_description} の中にいます。

=== あなたのペルソナ ===
役割: エージェント{self.id} — {self.display_name}
{persona_description}

=== 現在の状態 ===
座標: ({self.position[0]}, {self.position[1]})
場所の中にいるか: {"はい" if self.in_place else "いいえ"}
{"現在の場所: " + self.current_place if self.in_place else ""}
体温: {self.body_temperature:.1f}°C{"【発熱中】" if self.body_temperature >= 38.0 else "【微熱】" if self.body_temperature >= 37.5 else "（平熱）"}
{place_section_text}
{fire_section}
=== 場所の位置情報 ===
{place_locations_text}

=== 近くにいるエージェント ===
{nearby_text}

=== これまでの記憶 ===
{memory_text}

=== 受信したメッセージ ===
{messages_text}
{message_section}=== 選択できる行動 ===
- "stay": 現在の位置にとどまる
- "move" + direction: "up"（Y+1）, "down"（Y-1）, "left"（X-1）, "right"（X+1）

フィールド範囲: X・Y ともに -{self.half_space_size} 〜 +{self.half_space_size}

=== JSON形式で回答 ===
"memory" と "reasoning" はすべて日本語で記述してください。
{{
    "action": "move" または "stay",
    "direction": "up", "down", "left", "right" のいずれか（action が "move" の場合のみ）,
    "memory": "次のステップのために覚えておきたいこと（考え・観察・意図）",
    "reasoning": "この行動を選んだ理由の簡単な説明"
}}

{step_label}
"""
        return prompt
    
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON object from text, handling nested braces correctly"""
        # Find the first opening brace
        start_idx = text.find('{')
        if start_idx == -1:
            return None

        # Track brace depth to find matching closing brace
        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start_idx:], start=start_idx):
            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start_idx:i + 1]

        return None

    def _extract_direction_from_text(self, text: str) -> Optional[str]:
        """Extract direction from text using keyword matching (4 cardinal directions only)"""
        text_lower = text.lower()

        # Check cardinal directions only
        if "up" in text_lower:
            return "up"
        elif "down" in text_lower:
            return "down"
        elif "left" in text_lower:
            return "left"
        elif "right" in text_lower:
            return "right"

        return None
    
    def parse_message_response(self, response: str) -> MessageDecision:
        """Parse LLM response and extract message decision"""
        # Try to extract JSON from response using brace-matching
        json_str = self._extract_json_from_text(response)
        if json_str:
            try:
                parsed = json.loads(json_str)
                message = parsed.get("message", "")
                # Limit message to MAX_MESSAGE_WORDS words
                message = self._limit_message_words(message)
                return {
                    "message": message,
                    "reasoning": parsed.get("reasoning", "")
                }
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parsing failed for response: {response[:100]}... Error: {e}")

        # Fallback: simple text parsing
        message = ""
        reasoning = response[:FALLBACK_REASONING_LENGTH]

        # Limit message to MAX_MESSAGE_WORDS words
        message = self._limit_message_words(message)

        return {
            "message": message,
            "reasoning": reasoning
        }
    
    def parse_action_response(self, response: str) -> ActionDecision:
        """Parse LLM response and extract action decision"""
        # Try to extract JSON from response using brace-matching
        json_str = self._extract_json_from_text(response)
        if json_str:
            try:
                parsed = json.loads(json_str)
                return {
                    "action": parsed.get("action", "stay"),
                    "direction": parsed.get("direction"),
                    "memory": parsed.get("memory", ""),
                    "reasoning": parsed.get("reasoning", "")
                }
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parsing failed for response: {response[:100]}... Error: {e}")

        # Fallback: simple text parsing
        action = "stay"
        direction = None
        memory = ""
        reasoning = response[:FALLBACK_REASONING_LENGTH]

        if "move" in response.lower():
            action = "move"
            direction = self._extract_direction_from_text(response)

        return {
            "action": action,
            "direction": direction,
            "memory": memory,
            "reasoning": reasoning
        }
    
    def decide_message(
        self,
        place_status: Optional[Dict],
        nearby_agents: List['Agent'],
        step: int,
        fire_info: Optional[List[Dict]] = None
    ) -> MessageDecision:
        """Use LLM to decide what message to send (without position information)"""
        prompt = self.create_message_prompt(place_status, nearby_agents, step, fire_info=fire_info)

        try:
            response = self.llm_client.generate(prompt)
            decision = self.parse_message_response(response)
            return decision
        except Exception as e:
            logger.error(f"{self.display_name} のメッセージ決定でエラー: {e}")
            return {"message": "", "reasoning": "エラーが発生しました"}
    
    def decide_action(
        self,
        place_status: Optional[Dict],
        nearby_agents: List['Agent'],
        step: int,
        message_to_send: str = "",
        fire_info: Optional[List[Dict]] = None
    ) -> ActionDecision:
        """Use LLM to decide next action (with position information and message content)"""
        prompt = self.create_decision_prompt(place_status, nearby_agents, step, message_to_send, fire_info=fire_info)

        try:
            response = self.llm_client.generate(prompt)
            decision = self.parse_action_response(response)

            # Store LLM-generated memory (self-feedback for next step)
            memory_content = decision.get('memory', '')
            if memory_content:
                memory_entry = f"Step {step}: {memory_content}"
            else:
                # Fallback to reasoning if no memory provided
                memory_entry = f"Step {step}: {decision.get('reasoning', 'No memory')}"
            self.memory.append(memory_entry)
            if len(self.memory) > self.memory_limit:
                self.memory.pop(0)

            return decision
        except Exception as e:
            logger.error(f"{self.display_name} の行動決定でエラー: {e}")
            return {"action": "stay", "direction": None, "memory": "", "reasoning": "エラーが発生しました"}
    
    def move(self, direction: str) -> Tuple[int, int]:
        """Move agent in specified direction (origin-centered coordinate system)"""
        x, y = self.position
        dx, dy = DIRECTION_MAP.get(direction, (0, 0))

        # Boundaries: -half_space_size to +half_space_size
        new_x = max(-self.half_space_size, min(self.half_space_size, x + dx))
        new_y = max(-self.half_space_size, min(self.half_space_size, y + dy))

        self.position = (new_x, new_y)
        self.total_moves += 1
        return self.position
    
    def receive_message(self, from_agent_id: int, content: str, step: Optional[int] = None, from_display_name: str = ""):
        """Receive a message from another agent

        Args:
            from_agent_id: ID of the agent sending the message
            content: Message content
            step: Simulation step number (optional, for tracking purposes)
            from_display_name: Display name of the sending agent (e.g. "老人1")
        """
        sender_label = from_display_name if from_display_name else f"エージェント{from_agent_id}"
        self.received_messages.append({
            "from": sender_label,
            "content": content,
            "step": step if step is not None else len(self.received_messages)
        })
        if len(self.received_messages) > self.message_history_limit:
            self.received_messages.pop(0)

        logger.info(f"{self.display_name} が {sender_label} からメッセージを受信: \"{content}\"")
    
    def update_state(self, places: Optional[List[PlaceConfig]] = None):
        """Update agent state based on current position"""
        if places is None:
            places = self.places
        
        place_at_position = get_place_at_position(self.position, places)
        self.in_place = place_at_position is not None
        self.current_place = place_at_position['name'] if place_at_position else None
        
        if self.in_place:
            self.steps_in_place += 1
        else:
            self.steps_outside_place += 1

