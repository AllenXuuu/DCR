import pickle as pkl
import os
import numpy as np

root = '.'

val1_flow = './EG+_SP1_FLOW.pt.valid1.pkl'
val2_flow = './EG+_SP2_FLOW.pt.valid2.pkl'
val3_flow = './EG+_SP3_FLOW.pt.valid3.pkl'
val1_rgb  = './EG+_SP1_RGB.pt.valid1.pkl'
val2_rgb  = './EG+_SP2_RGB.pt.valid2.pkl'
val3_rgb  = './EG+_SP3_RGB.pt.valid3.pkl'


val1_flow = pkl.load(open(os.path.join(root, val1_flow),'rb'))
val2_flow = pkl.load(open(os.path.join(root, val2_flow),'rb'))
val3_flow = pkl.load(open(os.path.join(root, val3_flow),'rb'))
val1_rgb = pkl.load(open(os.path.join(root, val1_rgb),'rb'))
val2_rgb = pkl.load(open(os.path.join(root, val2_rgb),'rb'))
val3_rgb = pkl.load(open(os.path.join(root, val3_rgb),'rb'))

assert np.all(val1_rgb['gt']['id'] == val1_flow['gt']['id'])
assert np.all(val1_rgb['gt']['action'] == val1_flow['gt']['action'])
assert np.all(val2_rgb['gt']['id'] == val2_flow['gt']['id'])
assert np.all(val2_rgb['gt']['action'] == val2_flow['gt']['action'])
assert np.all(val3_rgb['gt']['id'] == val3_flow['gt']['id'])
assert np.all(val3_rgb['gt']['action'] == val3_flow['gt']['action'])

gt1 = val1_rgb['gt']
gt2 = val2_rgb['gt']
gt3 = val3_rgb['gt']

val1_flow = val1_flow['pred']['action']
val2_flow = val2_flow['pred']['action']
val3_flow = val3_flow['pred']['action']
val1_rgb = val1_rgb['pred']['action']
val2_rgb = val2_rgb['pred']['action']
val3_rgb = val3_rgb['pred']['action']


def calc_t5_acc_rec(pred,gt):
    rank = np.argsort(pred,1)[:,::-1][:,:5]
    is_correct = np.sum(rank == np.expand_dims(gt,1),1)
    acc = np.mean(is_correct)
    
    classes = np.unique(gt)
    recalls = []
    for c in classes:
        is_correct_this_class = is_correct[gt == c]
        recalls.append(is_correct_this_class.mean())
    rec = np.nanmean(recalls)
    return acc,rec

val1_fuse = val1_rgb  + 0.15 * val1_flow
val2_fuse = val2_rgb  + 0.4  * val2_flow
val3_fuse = val3_rgb  + 0.15 * val3_flow

acc1,rec1 = calc_t5_acc_rec(val1_fuse , gt1['action'])
acc2,rec2 = calc_t5_acc_rec(val2_fuse , gt2['action'])
acc3,rec3 = calc_t5_acc_rec(val3_fuse , gt3['action'])


acc_avg = (acc1+acc2+acc3) / 3
rec_avg = (rec1+rec2+rec3) / 3

print('Split 1: T5-ACC: %.3f, RECALL@5: %.3f' % (acc1,rec1))
print('Split 2: T5-ACC: %.3f, RECALL@5: %.3f' % (acc2,rec2))
print('Split 3: T5-ACC: %.3f, RECALL@5: %.3f' % (acc3,rec3))
print('AVG    : T5-ACC: %.3f, RECALL@5: %.3f' % (acc_avg,rec_avg))