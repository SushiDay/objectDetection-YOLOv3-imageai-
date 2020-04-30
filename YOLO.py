from imageai.Detection.Custom import DetectionModelTrainer


trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="characters")
trainer.setTrainConfig(object_names_array=['aeolianDune', 'beach', 'crevasseRiverBedLagDeposit', 'crevasseRiverChannelBar', 'distalBar', 'distributaryChannelFill', 'distributaryMouthBar', 'floodPlain', 'floodSwamp', 'interdistributaryEstuary', 'majorRiverBedLagDeposit', 'majorRiverChannelBar', 'meanderPointBar', 'mouthBar', 'proximalDeepseaFan', 'underwaterDistributaryChannel', 'underwaterNaturalLevee'], batch_size=2, num_experiments=10, train_from_pretrained_model="pretrained-yolov3.h5")

trainer.trainModel()


metrics = trainer.evaluateModel(model_path="characters/models", json_path="characters/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)
print(metrics)