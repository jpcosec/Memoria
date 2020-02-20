import argparse


def fake_arg(**kwargs):
    args = argparse.Namespace()
    d = vars(args)

    def add_field(field,default):
        if field in kwargs:
            d[field] = kwargs[field]
        else:
            d[field] = default

    add_field('lr' ,0.01)
    add_field('epochs' ,50)
    add_field('train_batch_size',128)
    add_field('test_batch_size', 128)
    add_field('student' ,"ResNet18")
    add_field('teacher' ,"ResNet101")
    add_field('distillation' ,"nst_linear")
    add_field('last_layer',"KD")
    add_field("layer", 5)# Arreglar para caso multicapa
    add_field('pre',50)
    add_field("student_layer",5)
    add_field("teacher_layer",26)
    add_field("transform", "none,")#, help="ej. noise,0.1")
    add_field("dataset", "cifar10")#, help="ej. vae_sample")


    args.resume=True
    return args