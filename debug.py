from transformers.configuration_utils import CONFIG_MAPPING
print("[디버깅] (사전) 등록된 config model_type 목록:", list(CONFIG_MAPPING.keys()))

ovis_config = AutoConfig.from_pretrained(VLM_MODEL_PATH, trust_remote_code=True)
print("[디버깅] (로드 후) ovis_config.model_type:", ovis_config.model_type)
ovis_config.model_type = "ovis_aimv2_custom_20250730"
print("[디버깅] (변경 후) ovis_config.model_type:", ovis_config.model_type)