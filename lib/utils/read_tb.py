from tensorboard.backend.event_processing import event_accumulator
import glob
import json
import numpy as np
import logging
import argparse

logging.basicConfig(filename='tb_collector.log', level=logging.INFO)


def epoch_summarize(di, info):
    # epoch_len=len(dict["train/acc"])/record['epoch']
    d = {}
    for k, v in di.items():
        if k.split("/")[-1] == 'wall_time':
            d[k] = [i[-1] - v[0] for i in np.split(v, info['epoch'])]

        else:
            d[k] = [np.mean(i) for i in np.split(v, info['epoch'])]

    return d


def collect_tbs(folder):
    tb_logs = glob.glob(folder + "/**/tb_logs/", recursive=True)

    try:
        total_dict =json.load(open(folder + '/epoch_summary.json', 'r'))
        print("Abriendo total_dict")
    except:
        print("Creando total_Dict")
        total_dict = {}

    for path in tb_logs:
        path=path.replace("\\","/")

        if path in list(total_dict.keys()):
            print(path, "ya leido")
            continue

        #print(path.split("/")
        if len(path.split("/")) > 4:
            print(len(path.split("/")),path)
            test_kw = ['test/acc',
                       'test/loss',
                       'test/eval']
            train_kw = ['train/acc',
                        'train/loss',
                        'train/eval']
            try:
                #info = json.load(open(path.replace('tb_logs/', "config.json"), 'r'))
                info = json.load(open(path.replace('tb_logs/', "record.json"), 'r'))

                filez = glob.glob(path + "/*")

                assert len(filez) == 1, 'more than one ran'

                file = filez[0].replace(path, "")
                ea = event_accumulator.EventAccumulator(path, file)
                ea.Reload()
                train_dict = dict([(kw, np.array([i.value for i in ea.Scalars(kw)])) for kw in train_kw])
                train_dict["train/wall_time"] = np.array([i.wall_time for i in ea.Scalars("train/acc")])

                train_dict = epoch_summarize(train_dict, info)

                test_dict = dict([(kw, np.array([i.value for i in ea.Scalars(kw)])) for kw in test_kw])
                test_dict["test/wall_time"] = np.array([i.wall_time for i in ea.Scalars("test/acc")])
                test_dict = epoch_summarize(test_dict, info)

                # logging.info"problemo en parse", path)
                # trd.update(ted)

                key = path.replace('/students', "").replace('tb_logs', '')

                # td[key]=trd
                train_dict.update(test_dict)
                total_dict[key] = train_dict

            except AssertionError:
                logging.info("mas de 1 wea en len"+ path)

            except FileNotFoundError as e:
                logging.info("[No se encuentra]"+str(e))

            except ValueError as e:
                logging.info(str(e) + " " + path)
                # logging.info(len(test_dict["test/wall_time"])/(info["epochs"]),len(test_dict["test/wall_time"]),info["epochs"])

            #except Exception as e:
            #    logging.info(str(e) + " " + path)
        else:
            test_kw = ['test/acc',
                       'test/loss_']
            train_kw = ['train/acc',
                        'train/loss_']
            try:
                # info = json.load(open(path.replace('tb_logs/', "config.json"), 'r'))
                info = json.load(open(path.replace('tb_logs/', "record.json"), 'r'))

                filez = glob.glob(path + "/*")

                assert len(filez) == 1, 'more than one ran'

                file = filez[0].replace(path, "")
                ea = event_accumulator.EventAccumulator(path, file)
                ea.Reload()
                train_dict = dict([(kw, np.array([i.value for i in ea.Scalars(kw)])) for kw in train_kw])
                train_dict["train/wall_time"] = np.array([i.wall_time for i in ea.Scalars("train/acc")])

                train_dict = epoch_summarize(train_dict, info)

                test_dict = dict([(kw, np.array([i.value for i in ea.Scalars(kw)])) for kw in test_kw])
                test_dict["test/wall_time"] = np.array([i.wall_time for i in ea.Scalars("test/acc")])
                test_dict = epoch_summarize(test_dict, info)

                # logging.info"problemo en parse", path)
                # trd.update(ted)

                key = path.replace('tb_logs', 'teacher')

                # td[key]=trd
                train_dict.update(test_dict)
                total_dict[key] = train_dict

                print(train_dict)

            except AssertionError:
                logging.info("mas de 1 wea en len" + path)

            except FileNotFoundError as e:
                logging.info("[No se encuentra]" + str(e))

            except ValueError as e:
                logging.info(str(e) + " " + path)
                # logging.info(len(test_dict["test/wall_time"])/(info["epochs"]),len(test_dict["test/wall_time"]),info["epochs"])

            # except Exception as e:
            #    logging.info(str(e) + " " + path)



    with open(folder + '/epoch_summary.json', 'w') as outfile:
        json.dump(total_dict, outfile, indent=4)

    print(total_dict)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tb logs collector')
    parser.add_argument('--folder', default="Imagenet", help='learning rate')
    args=parser.parse_args()
    import os
    os.chdir("../..")

    print(os.getcwd())

    collect_tbs(args.folder)

