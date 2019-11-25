#python feat_distillation.py --distillation=att_max --layer=2
#python feat_distillation.py --distillation=att_max --layer=3
#python feat_distillation.py --distillation=att_max --layer=4
#python feat_distillation.py --distillation=att_max --layer=5

python feat_distillation.py --distillation=att_mean --layer=2
python feat_distillation.py --distillation=att_mean --layer=3
python feat_distillation.py --distillation=att_mean --layer=4
python feat_distillation.py --distillation=att_mean --layer=5

python feat_distillation.py --distillation=PKT --layer=2
python feat_distillation.py --distillation=PKT --layer=3
python feat_distillation.py --distillation=PKT --layer=4
python feat_distillation.py --distillation=PKT --layer=5

python feat_distillation.py --distillation=nst_poly --layer=2
python feat_distillation.py --distillation=nst_poly --layer=3 --train_batch_size=64 --test_batch_size=64
python feat_distillation.py --distillation=nst_poly --layer=4 --train_batch_size=64 --test_batch_size=64
python feat_distillation.py --distillation=nst_poly --layer=5 --train_batch_size=64 --test_batch_size=64

python feat_distillation.py --distillation=nst_gauss --layer=2 --train_batch_size=8 --test_batch_size=8
python feat_distillation.py --distillation=nst_gauss --layer=3 --train_batch_size=8 --test_batch_size=8
python feat_distillation.py --distillation=nst_gauss --layer=4 --train_batch_size=8 --test_batch_size=8
python feat_distillation.py --distillation=nst_gauss --layer=5 --train_batch_size=8 --test_batch_size=8

python feat_distillation.py --distillation=nst_linear --layer=2 --train_batch_size=64 --test_batch_size=64
python feat_distillation.py --distillation=nst_linear --layer=3 --train_batch_size=64 --test_batch_size=64
python feat_distillation.py --distillation=nst_linear --layer=4 --train_batch_size=64 --test_batch_size=64
python feat_distillation.py --distillation=nst_linear --layer=5 --train_batch_size=64 --test_batch_size=64