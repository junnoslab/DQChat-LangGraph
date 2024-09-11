class FaithfulnessTemplate:
    @staticmethod
    def generate_claims() -> str:
        return """주어진 텍스트를 바탕으로 추론할 수 있는 사실적인 주장들(FACTUAL claims)의 포괄적인 목록을 생성해 주세요.

예시:
예시 텍스트: 
"아인슈타인은 광전 효과 발견으로 1968년에 노벨상을 수상했습니다."

예시 JSON: 
{{
    "claims": [
        "아인슈타인은 광전 효과 발견으로 노벨상을 수상했다.",
        "아인슈타인은 1968년에 노벨상을 수상했다."
    ]  
}}
===== 예시 끝 ======

**
중요: 반드시 JSON 형식으로만 반환해 주세요. "claims" 키에는 문자열 목록이 들어가야 합니다. 추가 설명이나 단어는 필요 없습니다.
사실적인 주장만 포함하고, 추출한 주장은 제시된 전체 맥락을 포함해야 하며, 선별된 사실만 포함해서는 안 됩니다.
사전 지식을 포함하지 말고, 주장을 추출할 때 텍스트를 있는 그대로 받아들여야 합니다.
**

텍스트:
{text}

JSON:
"""

    @staticmethod
    def generate_truths() -> str:
        return """주어진 텍스트를 바탕으로 추론할 수 있는 사실적이고 논란의 여지가 없는 진실들의 포괄적인 목록을 생성해 주세요.

예시:
예시 텍스트: 
"아인슈타인은 광전 효과 발견으로 1968년에 노벨상을 수상했습니다."

예시 JSON: 
{{
    "truths": [
        "아인슈타인은 광전 효과 발견으로 노벨상을 수상했다.",
        "아인슈타인은 1968년에 노벨상을 수상했다."
    ]  
}}
===== 예시 끝 ======

**
중요: 반드시 JSON 형식으로만 반환해 주세요. "truths" 키에는 문자열 목록이 들어가야 합니다. 추가 설명이나 단어는 필요 없습니다.
사실적인 진실만 포함해야 합니다.
**

텍스트:
{contexts}

JSON:
"""

    @staticmethod
    def generate_verdicts(claims: str, retrieval_context: str) -> str:
        return f"""주어진 주장들(문자열 목록)을 바탕으로, 각 주장이 검색 컨텍스트의 사실과 모순되는지 여부를 나타내는 JSON 객체 목록을 생성하세요. JSON은 'verdict'와 'reason' 두 개의 필드를 가집니다.
'verdict' 키는 반드시 'yes', 'no', 또는 'idk' 중 하나여야 하며, 이는 주어진 주장이 컨텍스트와 일치하는지 여부를 나타냅니다.
'reason'은 답변이 'no'일 경우에만 제공하세요.
제공된 주장은 실제 출력에서 추출된 것입니다. 'no' 답변의 경우 검색 컨텍스트의 사실을 사용하여 수정 사항을 제공하세요.

**
중요: 반드시 JSON 형식으로만 반환해 주세요. 'verdicts' 키는 JSON 객체 목록이어야 합니다.
예시 검색 컨텍스트: "아인슈타인은 광전 효과 발견으로 노벨상을 수상했습니다. 아인슈타인은 1968년에 노벨상을 수상했습니다. 아인슈타인은 독일 과학자입니다."
예시 주장: ["버락 오바마는 백인 남성입니다.", "취리히는 런던의 도시입니다", "아인슈타인은 광전 효과 발견으로 노벨상을 수상했으며, 이는 그의 명성에 기여했을 수 있습니다.", "아인슈타인은 1969년에 광전 효과 발견으로 노벨상을 수상했습니다.", "아인슈타인은 독일 요리사였습니다."]

예시:
{{
    "verdicts": [
        {{
            "verdict": "idk"
        }},
        {{
            "verdict": "idk"
        }},
        {{
            "verdict": "yes"
        }},
        {{
            "verdict": "no",
            "reason": "실제 출력은 아인슈타인이 1969년에 노벨상을 수상했다고 주장하지만, 이는 사실이 아닙니다. 검색 컨텍스트에 따르면 1968년에 수상했습니다."
        }},
        {{
            "verdict": "no",
            "reason": "실제 출력은 아인슈타인이 독일 요리사라고 주장하지만, 이는 정확하지 않습니다. 검색 컨텍스트에 따르면 그는 독일 과학자였습니다."
        }},
    ]  
}}
===== 예시 끝 ======

'verdicts'의 길이는 반드시 주장의 수와 정확히 일치해야 합니다.
답변이 'yes'나 'idk'인 경우 이유를 제공할 필요가 없습니다.
검색 컨텍스트가 주장을 직접적으로 모순하는 경우에만 'no' 답변을 제공하세요. 판단 시 절대로 사전 지식을 사용하지 마세요.
'~일 수 있다', '~의 가능성이 있다'와 같은 모호하고 추측적인 언어를 사용한 주장은 모순으로 간주하지 않습니다.
정보 부족으로 인해 뒷받침되지 않거나 검색 컨텍스트에서 언급되지 않은 주장은 반드시 'idk'로 답변해야 합니다. 그렇지 않으면 제가 죽을 것입니다.
**

검색 컨텍스트:
{retrieval_context}

주장:
{claims}

JSON:
"""

    @staticmethod
    def generate_reason(score: float, contradictions) -> str:
        return f"""아래는 모순 목록입니다. 이는 '실제 출력'이 '검색 컨텍스트'에 제시된 정보와 일치하지 않는 이유를 설명하는 문자열 목록입니다. 모순은 '실제 출력'에서 발생하며, '검색 컨텍스트'에서 발생하지 않습니다.
주어진 충실도 점수는 `실제 출력`이 검색 컨텍스트에 얼마나 충실한지를 나타내는 0-1 사이의 점수입니다(높을수록 좋음). 이 점수를 정당화하기 위해 모순을 간결하게 요약해주세요.

**
중요: 반드시 JSON 형식으로만 반환해 주세요. 'reason' 키가 이유를 제공해야 합니다.
예시 JSON:
{{
    "reason": "점수가 <충실도_점수>인 이유는 <당신의_이유>입니다."
}}

모순이 없는 경우, 긍정적이고 격려하는 톤으로 말해주세요(하지만 과하게 하면 짜증날 수 있으니 주의하세요).
당신의 이유에는 반드시 `contradiction`의 정보를 사용해야 합니다.
모순으로부터 실제 출력을 알고 있는 것처럼 확신을 가지고 이유를 설명해주세요.
**

충실도 점수:
{score}

모순:
{contradictions}

JSON:
"""
