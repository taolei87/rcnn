
def evaluate_average(predictions, masks = None):
    if masks is None:
        sum_all = sum(sum(x.ravel()) for x in predictions)
        cnt_all = sum(len(x.ravel()) for x in predictions)
    else:
        #masked = predictions * masks
        masked = [ x*m for x,m in zip(predictions, masks) ]
        sum_all = sum(sum(x.ravel()) for x in masked)
        cnt_all = sum(sum(x.ravel()) for x in masks)
    return sum_all / cnt_all
