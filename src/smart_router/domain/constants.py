from smart_router.domain.enums import RouteIntent
from smart_router.domain.models import CalibrationItem

# ANSI formatting and colors
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"

# Default calibration set for intent vector space
DEFAULT_CALIBRATION_SET: list[CalibrationItem] = [
    CalibrationItem("Qual o horário de funcionamento das agências?", RouteIntent.SIMPLE),
    CalibrationItem("Como faço para consultar meu saldo e extrato?", RouteIntent.SIMPLE),
    CalibrationItem("Onde vejo o código de rastreamento do pedido?", RouteIntent.SIMPLE),
    CalibrationItem("Vocês aceitam transferência via Pix para pagamento?", RouteIntent.SIMPLE),
    CalibrationItem("Consta uma compra internacional no cartão que eu não fiz.", RouteIntent.CRITICAL),
    CalibrationItem("Minha conta foi bloqueada por suspeita de fraude.", RouteIntent.CRITICAL),
    CalibrationItem("Cobrança duplicada no aplicativo e suspensão de serviço.", RouteIntent.CRITICAL),
    CalibrationItem("Caí em um golpe do Pix e transferi o valor por engano.", RouteIntent.CRITICAL),
]

# System prompts
SYSTEM_PROMPT_SIMPLE = "Você é um assistente de suporte operacional para dúvidas simples."
SYSTEM_PROMPT_CRITICAL = "Atendimento especializado em segurança, fraudes e ocorrências financeiras críticas."
SYSTEM_PROMPT_SLM_CLASSIFICATION = "Classifique a intenção em [SIMPLES] ou [CRITICO]. Responda apenas a tag."
