from datetime import datetime
from scipy.stats import wilcoxon

def get_dt_in_fmt(function=datetime.now, fmt="%m_%d_%Y__%H_%M_%S"):
    time = datetime.strftime(function(), fmt)
    return time

def wilcoxon_p_value(x, y):
    w_test = wilcoxon(x, y)
    return w_test.pvalue
