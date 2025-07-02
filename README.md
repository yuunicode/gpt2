## 사용 방법
cli에서 wandb login
api key 발급받은거 넣어야함

'''cli
pip install torch transformers wandb
python train.py
'''

## 고려한 데이터셋
1. 주제별 텍스트 일상 대화 데이터 (보류)
https://aihub.or.kr/aihubdata/data/view.do?pageIndex=5&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%EB%AC%B8%EC%9E%A5&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&srchDataRealmCode=REALM002&aihubDataSe=data&dataSetSn=543

2. 일반상식 QA 데이터셋 (이거 사용함)
https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%ED%95%9C%EA%B5%AD%EC%96%B4+QA&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=106

## 데이터셋 특징
깡통모델이기 때문에 정말 간단한 QA를 넣고 (stage 1), 그 뒤에는 문맥을 같은 QA 위에 추가함 (stage 2).
stage 1에 간단한 한국어 문장 데이터셋을 찾으면 그걸 오히려 넣는 것이 나을 수 있음
피클이 있으니 데이터셋은 다음 에포크부터 바꿀 수 있기도 하니까 우선 홀드

## 파일 설명
train.py -> 모델 훈련
model.py -> 지피티2의 경량버전으로 짠 모델 지피티 ㄳ
generate.py -> 후에 성능테스트를 위해 넣어둠

convert.py -> dataset에 있던 json파일을 txt로 변환 뒤 data에 저장하는 함수
check_token.py -> 안중요함 블럭사이즈가 문맥 해칠까봐 확인용으로 넣음
test_run.py -> 피클 별 성능테스트 궁금해서 넣음