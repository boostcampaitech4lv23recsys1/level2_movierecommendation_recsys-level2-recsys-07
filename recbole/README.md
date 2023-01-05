# Docs

## install

```bash
pip install requirements.txt
```

## run

```bash
# general
python run_gen.py --model [Model_Name] --config_files [Config_Name.yaml]
```
```bash
# sequential
python run_seq.py --model [Model_Name] --config_files [Config_Name.yaml]
```

```bash
# 축약어도 가능
python run_gen.py -m [Model_Name] -cf [Config_Name.yaml]
python run_seq.py -m [Model_Name] -cf [Config_Name.yaml]

```

#### Default

+ model: EASE(gen) / SASRec(seq)

+ dataset: recbole

+ config_dir(required=True): /opt/ml/input/recbole/yamls


#### Required
+ config_files: 필수 입력

    /opt/ml/input/recbole/yamls + args.config_files 경로에 있는 yaml 파일을 기준으로 실행됨.


## Create submission.csv

```bash
# 가장 최근에 저장된 Model_Name.pth 불러옴
# → user별 탑 10개 아이템 목록으로 Save inference at ./output/Model_Name_Cuurent_Time.csv로 저장됨

python submission.py --model [Model_Name] --confing_files [Config_Name.yaml]
```
   



#### Default

+ dataset: recbole

+ config_dir: /opt/ml/input/recbole/yamls


#### Required

+ model : 필수 입력

+ config_files : 필수 입력



