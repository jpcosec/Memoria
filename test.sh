
git pull
python student_train.py --distillation=composed,T-100.0 --resume
python student_train.py --distillation=composed,T-1000.0 --resume
python student_train.py --distillation=composed,T-10.0 --resume
python student_train.py --distillation=composed,T-3.5 --resume

python student_train.py --distillation=soft,T-100 --resume
python student_train.py --distillation=soft,T-1000.0 --resume
python student_train.py --distillation=soft,T-10.0 --resume
python student_train.py --distillation=soft,T-3.5 --resume

python student_train.py --distillation=soft,T-100.0 --student=MobileNet --resume
python student_train.py --distillation=soft,T-1000.0 --student=MobileNet --resume
python student_train.py --distillation=soft,T-10.0 --student=MobileNet --resume
python student_train.py --distillation=soft,T-3.5 --student=MobileNet --resume

python student_train.py --distillation=composed,T-100.0 --student=MobileNet --resume
python student_train.py --distillation=composed,T-1000.0 --student=MobileNet --resume
python student_train.py --distillation=composed,T-10.0 --student=MobileNet --resume
python student_train.py --distillation=composed,T-3.5 --student=MobileNet --resume
