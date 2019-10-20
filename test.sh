
git pull
python KD_distillation.py --distillation=composed,T-100.0 --resume
python KD_distillation.py --distillation=composed,T-1000.0 --resume
python KD_distillation.py --distillation=composed,T-10.0 --resume
python KD_distillation.py --distillation=composed,T-3.5 --resume

python KD_distillation.py --distillation=soft,T-100 --resume
python KD_distillation.py --distillation=soft,T-1000.0 --resume
python KD_distillation.py --distillation=soft,T-10.0 --resume
python KD_distillation.py --distillation=soft,T-3.5 --resume

python KD_distillation.py --distillation=soft,T-100.0 --student=MobileNet --resume
python KD_distillation.py --distillation=soft,T-1000.0 --student=MobileNet --resume
python KD_distillation.py --distillation=soft,T-10.0 --student=MobileNet --resume
python KD_distillation.py --distillation=soft,T-3.5 --student=MobileNet --resume

python KD_distillation.py --distillation=composed,T-100.0 --student=MobileNet --resume
python KD_distillation.py --distillation=composed,T-1000.0 --student=MobileNet --resume
python KD_distillation.py --distillation=composed,T-10.0 --student=MobileNet --resume
python KD_distillation.py --distillation=composed,T-3.5 --student=MobileNet --resume
