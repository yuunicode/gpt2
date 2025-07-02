# 블럭사이즈 128인데 문맥 유지하는지 확인용 
from transformers import PreTrainedTokenizerFast
import json

with open("config.json", "r") as f:
    config = json.load(f)

tokenizer = PreTrainedTokenizerFast(tokenizer_file=config["tokenizer_path"])

document = """재팬 오픈에서 4회 우승하였으며, 통산 단식 200승 이상을 거두었다. 1994년 생애 최초로 세계 랭킹 10위권에 진입하였다. 1992년에는 WTA로부터 '올해 가장 많은 향상을 보여준 선수상'(Most Improved Player Of The Year)을 수여받았으며, 일본 남자 패션 협회(Japan Men's Fashion Association)는 그녀를 '가장 패셔너블한 선수'(Most Fashionable)로 칭했다. 생애 두 번째 올림픽 참가 직후인 1996년 9월 24일 최초로 은퇴를 선언하였다. 이후 12년만인 2008년 4월에 예상치 못한 복귀 선언을 하고 투어에 되돌아왔다. 2008년 6월 15일 도쿄 아리아케 인터내셔널 여자 오픈에서 복귀 후 첫 우승을 기록했으며, 2009년 9월 27일에는 한국에서 열린 한솔 코리아 오픈 대회에서 우승하면서 복귀 후 첫 WTA 투어급 대회 우승을 기록했다. 한숨 좀 작작 쉬어!"""
question = "다테 기미코가 최초로 은퇴 선언을 한게 언제지"

# 전체 문장 토크나이즈
input_text = document + " " + question
tokens = tokenizer.encode(input_text)

print("총 토큰 수:", len(tokens))