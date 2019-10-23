
git pull
python KD_distillation.py --distillation=T-100.0 --resume
python KD_distillation.py --distillation=T-1000.0 --resume
python KD_distillation.py --distillation=T-10.0 --resume
python KD_distillation.py --distillation=T-3.5 --resume

python KD_distillation.py --distillation=soft,T-100 --resume
python KD_distillation.py --distillation=T-1000.0 --resume
python KD_distillation.py --distillation=T-10.0 --resume
python KD_distillation.py --distillation=T-3.5 --resume

python KD_distillation.py --distillation=T-100.0 --student=MobileNet --resume
python KD_distillation.py --distillation=T-1000.0 --student=MobileNet --resume
python KD_distillation.py --distillation=T-10.0 --student=MobileNet --resume
python KD_distillation.py --distillation=T-3.5 --student=MobileNet --resume

python KD_distillation.py --distillation=T-100.0 --student=MobileNet --resume
python KD_distillation.py --distillation=T-1000.0 --student=MobileNet --resume
python KD_distillation.py --distillation=T-10.0 --student=MobileNet --resume
python KD_distillation.py --distillation=T-3.5 --student=MobileNet --resume
