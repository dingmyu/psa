srun -p Segmentation -n1 --gres=gpu:2 python3 -u infer_cls.py --infer_list voc12/train_aug.txt --voc12_root ../VOC2012 --out_cam result_cam/ --out_la_crf result_la_crf/ --out_ha_crf result_ha_crf/
