import tensorflow as tf
import os, sys
from utils.logger import Logger

# implementation without logger from https://github.com/inkyusa/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py

# def mergeLoss(logPath):
#     sess = tf.Session()
#     x = tf.placeholder(dtype=tf.float32)
#     summary = tf.summary.scalar('Losses', x)
#     merged = tf.summary.merge_all()
#     sess.run(tf.global_variables_initializer())
#     writerTrain = tf.summary.FileWriter(os.path.join('logs','tb_summary', 'train'))
#     writerValid = tf.summary.FileWriter(os.path.join('logs','tb_summary', 'valid'))
#     trainEpoch = 0
#     validEpoch = 0
#     for event in tf.train.summary_iterator(logPath):
#         for v in event.summary.value:
#             #print(v)
#             if v.tag == 'train_loss':
#                 #print(f"loss = {v.simple_value}, epoch = {epoch}")
#                 summaryTrain = sess.run(merged, {x: v.simple_value})
#                 writerTrain.add_summary(summaryTrain, trainEpoch)
#                 writerTrain.flush()
#                 trainEpoch += 1
#             if v.tag == 'valid_loss':
#                 #print(f"loss = {v.simple_value}, epoch = {epoch}")
#                 summaryValid = sess.run(merged, {x: v.simple_value})
#                 writerValid.add_summary(summaryValid, validEpoch)
#                 writerValid.flush()
#                 validEpoch += 1

#With logger https://github.com/inkyusa/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py

def mergeLossLogger(logPath):
    trainLogger = Logger(os.path.join('logs', 'tb_merged', 'train'))
    validLogger = Logger(os.path.join('logs', 'tb_merged', 'valid'))
    for event in tf.compat.v1.train.summary_iterator(logPath):
        for v in event.summary.value:
            #Loss
            if v.tag == 'train_loss':
                trainInfo = {'loss': v.simple_value, }
                for tag, value in trainInfo.items():
                    trainLogger.scalar_summary(tag, value, step=event.step)
            if v.tag == 'valid_loss':
                validInfo = {'loss': v.simple_value, }
                for tag, value in validInfo.items():
                    validLogger.scalar_summary(tag, value, step=event.step)
            #IOU
            if v.tag == 'best_train_iou':
                trainInfo = {'iou': v.simple_value, }
                for tag, value in trainInfo.items():
                    trainLogger.scalar_summary(tag, value, step=event.step)
            if v.tag == 'best_val_iou':
                validInfo = {'iou': v.simple_value, }
                for tag, value in validInfo.items():
                    validLogger.scalar_summary(tag, value, step=event.step)
            #Acc
            if v.tag == 'train_acc':
                trainInfo = {'acc': v.simple_value, }
                for tag, value in trainInfo.items():
                    trainLogger.scalar_summary(tag, value, step=event.step)
            if v.tag == 'valid_acc':
                validInfo = {'acc': v.simple_value, }
                for tag, value in validInfo.items():
                    validLogger.scalar_summary(tag, value, step=event.step)


if __name__ == "__main__":
    argLength = len(sys.argv)
    if argLength != 2:
    	print ("usage: python ./merge_plot.py tb_event_file_path")
    else:
        logPath = sys.argv[1]
        mergeLossLogger(logPath)